import os
import warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 指定使用的GPU设备ID

# 设置 DDP 相关环境变量来优化性能
os.environ["NCCL_DEBUG"] = "WARN"  # 减少 NCCL 调试信息
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # 优化内存分配

# 过滤 PyTorch 的警告
warnings.filterwarnings("ignore", category=UserWarning, message=".*TypedStorage is deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Grad strides do not match bucket view strides.*")

from pathlib import Path

import hydra
import torch
import wandb
import signal
from colorama import Fore
from jaxtyping import install_import_hook
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment
from omegaconf import DictConfig, OmegaConf

from src.misc.weight_modify import checkpoint_filter_fn
from src.model.distiller import get_distiller

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule
    from src.global_cfg import set_cfg
    from src.loss import get_losses
    from src.misc.LocalLogger import LocalLogger
    from src.misc.step_tracker import StepTracker
    from src.misc.wandb_tools import update_checkpoint_path
    from src.misc.save_initial_weights_callback import SaveInitialWeightsCallback
    from src.misc.rich_progress_callback import RichProgressBar, DatasetProgressCallback
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper


def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


@hydra.main(
    version_base=None,
    config_path="./config",
    config_name="main",
)
def train(cfg_dict: DictConfig):
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)

    # Set up the output directory.
    output_dir = cfg.train.output_path / f"exp_{cfg.wandb['name']}"
    output_dir = Path(output_dir)
    print(cyan(f"Saving outputs to {output_dir}."))

    # Set up logging with wandb.
    callbacks = []
    if cfg_dict.wandb.mode != "disabled":
        logger = WandbLogger(
            project=cfg_dict.wandb.project,
            mode=cfg_dict.wandb.mode,
            name=f"{cfg_dict.wandb.name} ({output_dir.parent.name}/{output_dir.name})",
            tags=cfg_dict.wandb.get("tags", None),
            log_model=False,
            save_dir=output_dir,
            config=OmegaConf.to_container(cfg_dict),
        )
        callbacks.append(LearningRateMonitor("step", True))

        # On rank != 0, wandb.run is None.
        if wandb.run is not None:
            wandb.run.log_code("src")
    else:
        logger = LocalLogger()

    # 添加专门的测试权重保存Callback - 每个epoch保存
    callbacks.append(
        ModelCheckpoint(
            output_dir / "checkpoints",
            every_n_epochs=1,  # 每个epoch保存一次
            save_last=True,  # 保存最后一个epoch的权重
            save_top_k=-1,  # 保存所有epoch的权重
            save_weights_only=True,
            filename="epoch_{epoch:02d}-test",  # 文件名格式
            auto_insert_metric_name=False,
        )
    )
    callbacks[-1].CHECKPOINT_EQUALS_CHAR = '_'

    # Add callback to save initial weights
    callbacks.append(SaveInitialWeightsCallback())

    # 添加Rich进度条Callbacks
    callbacks.append(RichProgressBar(
        refresh_rate=10,  # 每10个step更新一次进度条
    ))

    # 添加数据集信息Callback
    callbacks.append(DatasetProgressCallback())

    # Prepare the checkpoint for loading.
    checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)

    # This allows the current step to be shared with the data loader processes.
    step_tracker = StepTracker()

    trainer = Trainer(
        max_epochs=getattr(cfg.trainer, 'max_epochs', 10),  # 从配置读取max_epochs
        num_nodes=cfg.trainer.num_nodes,
        accelerator="gpu",
        logger=logger,
        devices="auto",
        strategy="auto",  # 简化策略，让 PyTorch Lightning 自动选择
        # 添加 DDP 性能优化配置
        # use_distributed_sampler=False,  # 如果有数据加载问题可以启用
        callbacks=callbacks,
        val_check_interval=cfg.trainer.val_check_interval,
        enable_progress_bar=True,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        # plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1)],  # Uncomment for SLURM auto resubmission.
        inference_mode=False if (cfg.mode == "test" and cfg.test.align_pose) else True,
        num_sanity_val_steps=getattr(cfg.trainer, 'num_sanity_val_steps', 2),
    )
    torch.manual_seed(cfg_dict.seed + trainer.global_rank)

    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)

    distiller = None

    # Load the encoder weights.
    if cfg.model.encoder.pretrained_weights and cfg.mode == "train":
        weight_path = cfg.model.encoder.pretrained_weights
        ckpt_weights = torch.load(weight_path, map_location='cpu')
        if 'model' in ckpt_weights:
            ckpt_weights = ckpt_weights['model']
            ckpt_weights = checkpoint_filter_fn(ckpt_weights, encoder)
            missing_keys, unexpected_keys = encoder.load_state_dict(ckpt_weights, strict=False)
        elif 'state_dict' in ckpt_weights:
            ckpt_weights = ckpt_weights['state_dict']
            ckpt_weights = {k[8:]: v for k, v in ckpt_weights.items() if k.startswith('encoder.')}
            missing_keys, unexpected_keys = encoder.load_state_dict(ckpt_weights, strict=False)
        else:
            raise ValueError(f"Invalid checkpoint format: {weight_path}")

    model_wrapper = ModelWrapper(
        cfg.optimizer,
        cfg.test,
        cfg.train,
        encoder,
        encoder_visualizer,
        get_decoder(cfg.model.decoder),
        get_losses(cfg.loss),
        step_tracker,
        distiller=distiller,
    )
    data_module = DataModule(
        cfg.dataset,
        cfg.data_loader,
        step_tracker,
        global_rank=trainer.global_rank,
    )

    if cfg.mode == "train":
        trainer.fit(model_wrapper, datamodule=data_module, ckpt_path=checkpoint_path)
    else:
        trainer.test(
            model_wrapper,
            datamodule=data_module,
            ckpt_path=checkpoint_path,
        )


if __name__ == "__main__":
    train()
