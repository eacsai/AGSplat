from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable, Any

from moviepy import *
import torch
import wandb
from einops import pack, rearrange, repeat
from jaxtyping import Float
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from tabulate import tabulate
from torch import Tensor, nn, optim
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms import functional as TF
import torch.nn.functional as F
import numpy as np

from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from ..global_cfg import get_cfg
from ..loss import Loss
from ..loss.loss_point import Regr3D
from ..loss.loss_ssim import ssim
from ..misc.benchmarker import Benchmarker
from ..misc.cam_utils import update_pose, get_pnp_pose
from ..misc.image_io import prep_image, save_image, save_video
from ..misc.LocalLogger import LOG_PATH, LocalLogger
from ..misc.nn_module_tools import convert_to_buffer
from ..misc.step_tracker import StepTracker
from ..misc.utils import inverse_normalize, vis_depth_map, confidence_map, get_overlap_tag
from ..visualization.annotation import add_label
from ..visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics,
)
from ..visualization.camera_trajectory.wobble import (
    generate_wobble,
    generate_wobble_transformation,
)
from ..visualization.color_map import apply_color_map_to_image
from ..visualization.layout import add_border, hcat, vcat
from ..visualization.validation_in_3d import render_cameras, render_projections
from .decoder.decoder import Decoder, DepthRenderingMode
from .decoder.cuda_splatting import render_cuda_orthographic, render_bevs, forward_project
from .encoder import Encoder
from .encoder.visualization.encoder_visualizer import EncoderVisualizer
from .encoder.utils import project_point_clouds
from .ply_export import export_ply, write_ply
from .vis import vis_bev, single_features_to_RGB_colormap
from .dino_dpt import DINO, DPT
from ..dataset.dataset_kitti import get_meter_per_pixel
import os
import scipy.io as scio
from PIL import Image, ImageDraw

import torchvision.transforms as transforms
import cv2
import os
to_pil_image = transforms.ToPILImage()

@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int
    backbone_lr_multiplier: float


@dataclass
class TestCfg:
    output_path: Path
    align_pose: bool
    pose_align_steps: int
    rot_opt_lr: float
    trans_opt_lr: float
    compute_scores: bool
    save_image: bool
    save_video: bool
    save_compare: bool


@dataclass
class TrainCfg:
    depth_mode: DepthRenderingMode | None
    output_path: Path
    extended_visualization: bool
    print_log_every_n_steps: int
    distiller: str
    distill_max_steps: int
    meter_per_pixel: float
    weakly_supervised: bool
    pts_type: str = 'all'  # 'grd', 'drone', 'all'


@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
        self,
        t: Float[Tensor, " t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        pass


class ModelWrapper(LightningModule):
    logger: Optional[WandbLogger]
    encoder: nn.Module
    encoder_visualizer: Optional[EncoderVisualizer]
    decoder: Decoder
    losses: nn.ModuleList
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    step_tracker: StepTracker | None

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        encoder: Encoder,
        encoder_visualizer: Optional[EncoderVisualizer],
        decoder: Decoder,
        losses: list[Loss],
        step_tracker: StepTracker | None,
        distiller: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker

        # Set up the model.
        self.encoder = encoder
        self.encoder_visualizer = encoder_visualizer
        self.decoder = decoder
        self.data_shim = get_data_shim(self.encoder)
        self.losses = nn.ModuleList(losses)
        self.dino_feat = DINO()
        self.dino_feat.eval()
        self.dpt = DPT(self.dino_feat.feat_dim)

        self.meter_per_pixel = train_cfg.meter_per_pixel  # downsampled by 4
        # This is used for testing.
        self.benchmarker = Benchmarker()

        # Initialize lists to store pred_u and pred_v during testing
        self.pred_lons = []
        self.pred_lats = []
        self.gt_lons = []
        self.gt_lats = []

        # 记录脏数据的列表
        self.dirty_data_log = []

    def log_dirty_data(self, batch, function_name, additional_info=None):
        """记录脏数据信息以便后续删除"""
        if batch is None:
            return

        dirty_info = {
            'function_name': function_name,
            'global_step': self.global_step,
            'timestamp': torch.tensor(self.global_step, dtype=torch.float32).item(),
        }

        # 如果有额外信息
        if additional_info:
            dirty_info.update(additional_info)

        # 打印信息
        print(f"=== DIRTY DATA DETECTED ===")
        print(f"Function: {function_name}")
        print(f"Global step: {self.global_step}")
        for key, value in dirty_info.items():
            if key not in ['function_name', 'global_step', 'timestamp']:
                print(f"{key}: {value}")
        print("==========================")

        # 添加到日志
        self.dirty_data_log.append(dirty_info)

        # 可选：保存到文件
        if len(self.dirty_data_log) % 10 == 1:  # 每10条脏数据记录保存一次
            self.save_dirty_data_log()

    def save_dirty_data_log(self):
        """保存脏数据日志到文件"""
        if not self.dirty_data_log:
            return

        import json
        save_path = self.train_cfg.output_path / "dirty_data_log.json"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w') as f:
            json.dump(self.dirty_data_log, f, indent=2, default=str)

        print(f"Dirty data log saved to {save_path}")

    def training_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        # Render Gaussians.
        feat_img = batch["context"]["feat_image"]
        grd_img = batch["context"]["image"]
        b, v, _, h, w = grd_img.shape
        feat_img = rearrange(feat_img, "b v c h w -> (b v) c h w")
        sat_img = batch["sat"]["sat_ref"]

        # extract grd and sat feature
        with torch.no_grad():
            # dino
            sat_feat = self.dino_feat(sat_img)
            grd_feat = self.dino_feat(feat_img)
            if isinstance(sat_feat, (tuple, list)):
                sat_feats = [_f.detach() for _f in sat_feat]
            if isinstance(grd_feat, (tuple, list)):
                grd_feats = [_f.detach() for _f in grd_feat]
        
        sat_feat, sat_conf = self.dpt(sat_feats)
        grd_feat, grd_conf = self.dpt(grd_feats)
        A = sat_feat.shape[-1]
        meter_per_pixel = self.meter_per_pixel * 512 / A
        
        gaussians, grd_camera = self.encoder(
            grd_feat.shape[-2:], 
            batch, 
            self.global_step,
        )
        
        gaussians.features = rearrange(grd_feat, "(b v) c h w -> b (v h w) c", b=b, v=v)
        gaussians.confidences = rearrange(grd_conf, "(b v) c h w -> b (v h w) c", b=b, v=v)
        
        # get gaussian bev
        if grd_camera is not None:
            # intrinsics = grd_camera.intrinsics
            # extrinsics = grd_camera.extrinsics
            # output = self.decoder.forward(
            #     gaussians,
            #     extrinsics,
            #     intrinsics,
            #     batch["target"]["near"],
            #     batch["target"]["far"],
            #     (int(A * 2 / 3), int(A * 2 / 3)),
            #     use_sh=False,
            # )
            # grd2sat_color = output.color.squeeze(1)
            # grd2sat_feature = output.feature.squeeze(1)
            # grd2sat_confidence = output.confidence.squeeze(1)
            grd2sat_color, grd2sat_feature, grd2sat_confidence = render_bevs(
                gaussians,
                (int(A * 2 / 3), int(A * 2 / 3)),
                look_axis=2,
                width=grd_camera.width,
                height=grd_camera.height
            )
        else:
            heading = torch.zeros([b, 1], dtype=torch.float32, requires_grad=True, device=batch["target"]["extrinsics"].device)
            grd2sat_color, grd2sat_feature, grd2sat_confidence = render_bevs(
                gaussians,
                (A, A),
                heading=heading, 
                width=torch.tensor([A * meter_per_pixel] * b, dtype=torch.float32, device=batch["target"]["extrinsics"].device), 
                height=torch.tensor([A * meter_per_pixel] * b, dtype=torch.float32, device=batch["target"]["extrinsics"].device)
            )

        # grd2sat_color = project_point_clouds(
        #     gaussians.means,
        #     gaussians.harmonics.squeeze(-1),
        #     grd_camera.intrinsics.squeeze(1),
        #     int(A * 2 / 3), int(A * 2 / 3)
        # )

        # grd2sat_feature = project_point_clouds(
        #     gaussians.means,
        #     gaussians.features,
        #     grd_camera.intrinsics.squeeze(1),
        #     int(A * 2 / 3), int(A * 2 / 3)
        # )
        # grd2sat_confidence = project_point_clouds(
        #     gaussians.means,
        #     gaussians.confidences,
        #     grd_camera.intrinsics.squeeze(1),
        #     int(A * 2 / 3), int(A * 2 / 3)
        # )
        # Weakly Supervised Loss Computation
        # crop_H = int(A - 20 * 3 / meter_per_pixel)
        # crop_W = int(A - 20 * 3 / meter_per_pixel)
        crop_H = (int(A * 2 / 3))
        crop_W = (int(A * 2 / 3))
        g2s_feat = TF.center_crop(grd2sat_feature, [crop_H, crop_W])
        g2s_feat = F.normalize(g2s_feat.reshape(b, -1)).reshape(b, -1, crop_H, crop_W)
        g2s_conf = TF.center_crop(grd2sat_confidence, [crop_H, crop_W])

        grd2sat_color = TF.center_crop(grd2sat_color, [crop_H, crop_W])
        rgb_bev = grd2sat_color[0]
        rgb_bev = TF.center_crop(rgb_bev, [crop_H, crop_W])
        test_img = to_pil_image(rgb_bev.clamp(min=0,max=1))
        test_img.save('splat_bev.png')

        test_img = to_pil_image(F.interpolate(batch["sat"]["sat_align_ref"], size=(A, A), mode='bilinear', align_corners=False)[0].clamp(min=0,max=1))
        test_img.save('sat_bev.png')

        # g2s_conf = torch.ones_like(g2s_feat[:, :1, :, :]).float()
        # g2s_conf = g2s_feat.any(dim=1, keepdim=True).float()
        # vis_bev(batch, gaussians)
        # single_features_to_RGB_colormap(grd2sat_feature, idx=0, img_name='bev_feature.png', cmap_name='rainbow')
        if self.train_cfg.weakly_supervised:
            corr = self.weakly_corr(sat_feat, g2s_feat, g2s_conf, batch)
        else:
            corr = self.supervise_corr(sat_feat, g2s_feat, g2s_conf, batch)

        # self.visualize_positions_on_satellite(
        #     corr, gaussians, batch
        # )

        # Compute and log loss.
        total_loss = 0
        for loss_fn in self.losses:
            if loss_fn.name == 'glue':
                loss = loss_fn.forward(
                    batch,
                    sat_feat,
                    g2s_feat,
                    self.meter_per_pixel,
                    weakly_supervised=self.train_cfg.weakly_supervised,
                )
            else:
                loss = loss_fn.forward(
                    batch,
                    corr,
                    meter_per_pixel,
                    weakly_supervised=self.train_cfg.weakly_supervised,
                )
            self.log(f"loss/{loss_fn.name}", loss)
            total_loss = total_loss + loss

        self.log("loss/total", total_loss.item())

        if (
            self.global_rank == 0
            and self.global_step % self.train_cfg.print_log_every_n_steps == 0
        ):
            # 获取当前学习率
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']

            # 计算梯度范数
            total_norm = 0
            param_count = 0
            for name, param in self.named_parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
            total_norm = total_norm ** (1. / 2)

            print(
                f"train step {self.global_step}; "
                f"loss = {total_loss.item():.6f}; "
                f"lr = {current_lr:.2e}; "
                f"grad_norm = {total_norm:.6f}"
            )
        self.log("info/global_step", self.global_step)  # hack for ckpt monitor

        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)

        return total_loss

    @rank_zero_only
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        batch: BatchedExample = self.data_shim(batch)

        b, v, _, h, w = batch["target"]["image"].shape
        if self.train_cfg.pts_type == 'grd':
            feat_img = batch["context"]["feat_image"][:, :1]
            grd_img = batch["context"]["image"][:, :1]
        elif self.train_cfg.pts_type == 'drone':
            feat_img = batch["context"]["feat_image"][:, 1:]
            grd_img = batch["context"]["image"][:, 1:]
        else:
            feat_img = batch["context"]["feat_image"]
            grd_img = batch["context"]["image"]

        b, v, _, h, w = grd_img.shape
        feat_img = rearrange(feat_img, "b v c h w -> (b v) c h w")
        sat_img = batch["sat"]["sat_ref"]

        if batch_idx % 100 == 0:
            print(f"Test step {batch_idx:0>6}.")

        # extract grd and sat feature
        with torch.no_grad():
            # dino
            sat_feat = self.dino_feat(sat_img)
            grd_feat = self.dino_feat(feat_img)
            if isinstance(sat_feat, (tuple, list)):
                sat_feats = [_f.detach() for _f in sat_feat]
            if isinstance(grd_feat, (tuple, list)):
                grd_feats = [_f.detach() for _f in grd_feat]

        # dpt
        sat_feat, sat_conf = self.dpt(sat_feats)
        grd_feat, grd_conf = self.dpt(grd_feats)
        A = sat_feat.shape[-1]
        meter_per_pixel = self.meter_per_pixel * 512 / A

        # Render Gaussians.
        with self.benchmarker.time("encoder"):
            gaussians, grd_camera = self.encoder(
                grd_feat.shape[-2:], 
                batch, 
                self.global_step,
            )
        gaussians.features = rearrange(grd_feat, "(b v) c h w -> b (v h w) c", b=b, v=v)
        gaussians.confidences = rearrange(grd_conf, "(b v) c h w -> b (v h w) c", b=b, v=v)

        # get gaussian bev
        if grd_camera is not None:
            # intrinsics = grd_camera.intrinsics
            # extrinsics = grd_camera.extrinsics
            # output = self.decoder.forward(
            #     gaussians,
            #     batch["target"]["extrinsics"],
            #     intrinsics,
            #     batch["target"]["near"],
            #     batch["target"]["far"],
            #     (int(A * 2 / 3), int(A * 2 / 3)),
            #     use_sh=False,
            # )
            # grd2sat_color = output.color.squeeze(1)
            # grd2sat_feature = output.feature.squeeze(1)
            # grd2sat_confidence = output.confidence.squeeze(1)
            grd2sat_color, grd2sat_feature, grd2sat_confidence = render_bevs(
                gaussians,
                (int(A * 2 / 3), int(A * 2 / 3)),
                look_axis=2,
                width=grd_camera.width,
                height=grd_camera.height
            )
        
        else:
            heading = torch.zeros([b, 1], dtype=torch.float32, requires_grad=True, device=batch["target"]["extrinsics"].device)
            grd2sat_color, grd2sat_feature, grd2sat_confidence = render_bevs(
                gaussians,
                (A, A),
                heading=heading,
                width=torch.tensor([A * meter_per_pixel] * b, dtype=torch.float32, device=batch["target"]["extrinsics"].device), 
                height=torch.tensor([A * meter_per_pixel] * b, dtype=torch.float32, device=batch["target"]["extrinsics"].device)
            )


        # Weakly Supervised Loss Computation
        crop_H = (int(A * 2 / 3))
        crop_W = (int(A * 2 / 3))
        g2s_feat = TF.center_crop(grd2sat_feature, [crop_H, crop_W])
        g2s_feat = F.normalize(g2s_feat.reshape(b, -1)).reshape(b, -1, crop_H, crop_W)
        g2s_conf = TF.center_crop(grd2sat_confidence, [crop_H, crop_W])

        rgb_bev = grd2sat_color[0]
        rgb_bev = TF.center_crop(rgb_bev, [crop_H, crop_W])
        test_img = to_pil_image(rgb_bev.clamp(min=0,max=1))
        test_img.save('splat_bev_t.png')
        test_img = to_pil_image(F.interpolate(batch["sat"]["sat_align_ref"], size=(A, A), mode='bilinear', align_corners=False)[0].clamp(min=0,max=1))
        test_img.save('sat_bev_t.png')
        ### compute correlation
        signal = sat_feat.reshape(1, -1, A, A)
        kernel = g2s_feat * g2s_conf.pow(2)
        corr = F.conv2d(signal, kernel, groups=b)[0]  # [B, H, W]

        sat_feat_pow = (sat_feat).pow(2).transpose(0, 1)  # [B, C, H, W]->[C, B, H, W]
        g2s_conf_pow = g2s_conf.pow(2)
        denominator_sat = F.conv2d(sat_feat_pow, g2s_conf_pow, groups=b).transpose(0, 1)  # [B, C, H, W]
        denominator_sat = torch.sqrt(torch.sum(denominator_sat, dim=1))  # [B, H, W]

        denom_grd = torch.linalg.norm((g2s_feat * g2s_conf).reshape(b, -1), dim=-1)  # [B]
        shape = denominator_sat.shape
        denominator_grd = denom_grd[:, None, None].repeat(1, shape[1], shape[2])
        denominator = denominator_sat * denominator_grd
        denominator = torch.maximum(denominator, torch.ones_like(denominator) * 1e-6)
        corr = corr / denominator  # [B, H, W]
        corr_H = int(20.0 * 3 / meter_per_pixel)
        corr_W = int(20.0 * 3 / meter_per_pixel)
        corr = TF.center_crop(corr[:, None], [corr_H, corr_W])[:, 0]

        # compute pred_u and pred_v
        max_index = torch.argmax(corr.reshape(b, -1), dim=1)

        pred_u = (max_index % corr_W - corr_W / 2 + 0.5) * meter_per_pixel  # / self.args.shift_range_lon
        pred_v = (max_index // corr_W - corr_H / 2 + 0.5) * meter_per_pixel  # / self.args.shift_range_lat

        gt_shift_u = batch['sat']['gt_shift_u'][:, 0] * 20.0 # 单位(m)
        gt_shift_v = batch['sat']['gt_shift_v'][:, 0] * 20.0 # 单位(m)
        # Store pred_u and pred_v values in the lists
        self.pred_lons.extend(pred_u.cpu().detach().numpy())
        self.pred_lats.extend(pred_v.cpu().detach().numpy())
        self.gt_lons.extend(gt_shift_u.cpu().detach().numpy())
        self.gt_lats.extend(gt_shift_v.cpu().detach().numpy())

        # Visualize positions on satellite image
        self.visualize_positions_on_satellite(
            corr, gaussians, batch, meter_per_pixel
        )

    def test_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)

        b, v, _, h, w = batch["target"]["image"].shape

        feat_img = batch["context"]["feat_image"]
        grd_img = batch["context"]["image"]
        b, v, _, h, w = grd_img.shape
        feat_img = rearrange(feat_img, "b v c h w -> (b v) c h w")
        sat_img = batch["sat"]["sat_ref"]

        if batch_idx % 100 == 0:
            print(f"Test step {batch_idx:0>6}.")

        # extract grd and sat feature
        with torch.no_grad():
            # dino
            sat_feat = self.dino_feat(sat_img)
            grd_feat = self.dino_feat(feat_img)
            if isinstance(sat_feat, (tuple, list)):
                sat_feats = [_f.detach() for _f in sat_feat]
            if isinstance(grd_feat, (tuple, list)):
                grd_feats = [_f.detach() for _f in grd_feat]

        # dpt
        sat_feat, sat_conf = self.dpt(sat_feats)
        grd_feat, grd_conf = self.dpt(grd_feats)
        A = sat_feat.shape[-1]
        meter_per_pixel = self.meter_per_pixel * 512 / A

        # Render Gaussians.
        with self.benchmarker.time("encoder"):
            gaussians, grd_camera = self.encoder(
                grd_feat.shape[-2:], 
                batch, 
                self.global_step,
            )
        gaussians.features = rearrange(grd_feat, "(b v) c h w -> b (v h w) c", b=b, v=v)
        gaussians.confidences = rearrange(grd_conf, "(b v) c h w -> b (v h w) c", b=b, v=v)

        # get gaussian bev
        if grd_camera is not None:
            # intrinsics = grd_camera.intrinsics
            # extrinsics = grd_camera.extrinsics
            # output = self.decoder.forward(
            #     gaussians,
            #     extrinsics,
            #     intrinsics,
            #     batch["target"]["near"],
            #     batch["target"]["far"],
            #     (int(A * 2 / 3), int(A * 2 / 3)),
            #     use_sh=False,
            # )
            # grd2sat_color = output.color.squeeze(1)
            # grd2sat_feature = output.feature.squeeze(1)
            # grd2sat_confidence = output.confidence.squeeze(1)
            grd2sat_color, grd2sat_feature, grd2sat_confidence = render_bevs(
                gaussians,
                (int(A * 2 / 3), int(A * 2 / 3)),
                look_axis=2,
                width=grd_camera.width,
                height=grd_camera.height
            )
        
        else:
            heading = torch.zeros([b, 1], dtype=torch.float32, requires_grad=True, device=batch["target"]["extrinsics"].device)
            grd2sat_color, grd2sat_feature, grd2sat_confidence = render_bevs(
                gaussians,
                (A, A),
                heading=heading,
                width=torch.tensor([A * meter_per_pixel] * b, dtype=torch.float32, device=batch["target"]["extrinsics"].device), 
                height=torch.tensor([A * meter_per_pixel] * b, dtype=torch.float32, device=batch["target"]["extrinsics"].device)
            )


        # Weakly Supervised Loss Computation
        crop_H = (int(A * 2 / 3))
        crop_W = (int(A * 2 / 3))
        g2s_feat = TF.center_crop(grd2sat_feature, [crop_H, crop_W])
        g2s_feat = F.normalize(g2s_feat.reshape(b, -1)).reshape(b, -1, crop_H, crop_W)
        g2s_conf = TF.center_crop(grd2sat_confidence, [crop_H, crop_W])

        rgb_bev = grd2sat_color[0]
        rgb_bev = TF.center_crop(rgb_bev, [crop_H, crop_W])
        test_img = to_pil_image(rgb_bev.clamp(min=0,max=1))
        test_img.save('splat_bev_t.png')
        test_img = to_pil_image(F.interpolate(batch["sat"]["sat_align_ref"], size=(A, A), mode='bilinear', align_corners=False)[0].clamp(min=0,max=1))
        test_img.save('sat_bev_t.png')
        ### compute correlation
        signal = sat_feat.reshape(1, -1, A, A)
        kernel = g2s_feat * g2s_conf.pow(2)
        corr = F.conv2d(signal, kernel, groups=b)[0]  # [B, H, W]

        sat_feat_pow = (sat_feat).pow(2).transpose(0, 1)  # [B, C, H, W]->[C, B, H, W]
        g2s_conf_pow = g2s_conf.pow(2)
        denominator_sat = F.conv2d(sat_feat_pow, g2s_conf_pow, groups=b).transpose(0, 1)  # [B, C, H, W]
        denominator_sat = torch.sqrt(torch.sum(denominator_sat, dim=1))  # [B, H, W]

        denom_grd = torch.linalg.norm((g2s_feat * g2s_conf).reshape(b, -1), dim=-1)  # [B]
        shape = denominator_sat.shape
        denominator_grd = denom_grd[:, None, None].repeat(1, shape[1], shape[2])
        denominator = denominator_sat * denominator_grd
        denominator = torch.maximum(denominator, torch.ones_like(denominator) * 1e-6)
        corr = corr / denominator  # [B, H, W]
        corr_H = int(20.0 * 3 / meter_per_pixel)
        corr_W = int(20.0 * 3 / meter_per_pixel)
        corr = TF.center_crop(corr[:, None], [corr_H, corr_W])[:, 0]

        # compute pred_u and pred_v
        max_index = torch.argmax(corr.reshape(b, -1), dim=1)

        pred_u = (max_index % corr_W - corr_W / 2 + 0.5) * meter_per_pixel  # / self.args.shift_range_lon
        pred_v = (max_index // corr_W - corr_H / 2 + 0.5) * meter_per_pixel  # / self.args.shift_range_lat

        gt_shift_u = batch['sat']['gt_shift_u'][:, 0] * 20.0 # 单位(m)
        gt_shift_v = batch['sat']['gt_shift_v'][:, 0] * 20.0 # 单位(m)
        # Store pred_u and pred_v values in the lists
        self.pred_lons.extend(pred_u.cpu().detach().numpy())
        self.pred_lats.extend(pred_v.cpu().detach().numpy())
        self.gt_lons.extend(gt_shift_u.cpu().detach().numpy())
        self.gt_lats.extend(gt_shift_v.cpu().detach().numpy())

        # Visualize positions on satellite image
        self.visualize_positions_on_satellite(
            corr, gaussians, batch, meter_per_pixel
        )

    def visualize_positions_on_satellite(self, corr, gaussians, batch, meter_per_pixel):
        """
        在卫星图上可视化预测位置和真实位置

        Args:
            gaussians: 高斯球的参数
            batch: 包含卫星图像的batch
        """
        # write_ply(gaussians.means[0].cpu().detach().numpy(), gaussians.harmonics[0,:,:,0].cpu().detach().numpy())

        corr_H, corr_W = corr.shape[-2:]
        max_index = torch.argmax(corr[0].reshape(-1)).data.cpu().numpy()
        pred_u = (max_index % corr_W - corr_W / 2 + 0.5) * meter_per_pixel
        pred_v = (max_index // corr_W - corr_H / 2 + 0.5) * meter_per_pixel

        gt_shift_u = batch['sat']['gt_shift_u'][0, 0] * 20.0 # 单位：（m）
        gt_shift_v = batch['sat']['gt_shift_v'][0, 0] * 20.0 # 单位：（m）

        # 获取卫星图像
        sat_img = batch['sat']['sat_ref'][0]  # 取第一个样本的卫星图像

        # 处理图像格式
        if isinstance(sat_img, torch.Tensor):
            sat_img_np = sat_img.detach().cpu().numpy()
            # 如果是CHW格式，转换为HWC
            if sat_img_np.shape[0] < sat_img_np.shape[2]:
                sat_img_np = np.transpose(sat_img_np, (1, 2, 0))
        else:
            sat_img_np = sat_img

        # 归一化到0-255范围
        if sat_img_np.max() <= 1.0:
            sat_img_np = (sat_img_np * 255).astype(np.uint8)
        else:
            sat_img_np = sat_img_np.astype(np.uint8)

        # 转换为BGR格式（OpenCV格式）
        if sat_img_np.shape[2] == 3:
            vis_img = cv2.cvtColor(sat_img_np, cv2.COLOR_RGB2BGR)
        else:
            vis_img = sat_img_np.copy()

        H, W = vis_img.shape[:2]
        center_x, center_y = W // 2, H // 2

        # 绘制预测位置（红色三角形）
        if isinstance(pred_u, torch.Tensor):
            pred_x = float(pred_u.cpu().detach().numpy())
        else:
            pred_x = float(pred_u)

        if isinstance(pred_v, torch.Tensor):
            pred_y = float(pred_v.cpu().detach().numpy())
        else:
            pred_y = float(pred_v)
        pixel_x_pred = int(center_x + pred_x / self.meter_per_pixel)
        pixel_y_pred = int(center_y + pred_y / self.meter_per_pixel)

        triangle_size = 15
        pred_triangle = np.array([
            [pixel_x_pred, pixel_y_pred - triangle_size],
            [pixel_x_pred - triangle_size//2, pixel_y_pred + triangle_size//2],
            [pixel_x_pred + triangle_size//2, pixel_y_pred + triangle_size//2]
        ], dtype=np.int32)

        cv2.fillPoly(vis_img, [pred_triangle], (0, 0, 255))  # 红色填充
        cv2.polylines(vis_img, [pred_triangle], True, (255, 255, 255), 2)  # 白色边框

        # 绘制真实位置（绿色三角形）
        gt_x = float(gt_shift_u)
        gt_y = float(gt_shift_v)

        pixel_x_gt = int(center_x + gt_x / self.meter_per_pixel)
        pixel_y_gt = int(center_y + gt_y / self.meter_per_pixel)

        gt_triangle = np.array([
            [pixel_x_gt, pixel_y_gt - triangle_size],
            [pixel_x_gt - triangle_size//2, pixel_y_gt + triangle_size//2],
            [pixel_x_gt + triangle_size//2, pixel_y_gt + triangle_size//2]
        ], dtype=np.int32)

        cv2.fillPoly(vis_img, [gt_triangle], (0, 255, 0))  # 绿色填充
        cv2.polylines(vis_img, [gt_triangle], True, (255, 255, 255), 2)  # 白色边框

        # 添加图例
        legend_y = 30
        cv2.rectangle(vis_img, (10, legend_y-20), (250, legend_y+40), (0, 0, 0), -1)  # 黑色背景
        cv2.putText(vis_img, "Position Visualization", (15, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.fillPoly(vis_img, [np.array([[25, legend_y+5], [35, legend_y-5], [45, legend_y+5]])], (0, 0, 255))
        cv2.putText(vis_img, "Predicted", (55, legend_y+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.fillPoly(vis_img, [np.array([[25, legend_y+20], [35, legend_y+10], [45, legend_y+20]])], (0, 255, 0))
        cv2.putText(vis_img, "Ground Truth", (55, legend_y+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # 创建保存目录
        save_dir = "./camera_position_visualization"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "camera_pos.png")

        # 保存图像
        cv2.imwrite(save_path, vis_img)

        ###### 保存原始卫星图 ######
        # 定义红点的半径（可以根据需要调整）
        radius = 5
        sat_img = F.interpolate(batch['sat']['sat_align_ref'], size=(H, W), mode='bilinear', align_corners=False)
        test_img = to_pil_image(sat_img[0])
        # 创建一个可以在图像上绘图的对象
        draw = ImageDraw.Draw(test_img)

        # 绘制一个红色圆形作为中心点
        # draw.ellipse((x1, y1, x2, y2), fill=color, outline=color)
        # x1, y1 是左上角坐标，x2, y2 是右下角坐标
        draw.ellipse((center_x - radius, center_y - radius,
                    center_x + radius, center_y + radius),
                    fill=(255, 0, 0), outline=(255, 0, 0)) # 填充和边框都设为红色
        test_img.save(os.path.join(save_dir, 'sat.png'))

        ###### 保存输入地面图 ######
        rgb_input = (batch['context']["image"][0,0] + 1) / 2
        test_img = to_pil_image(rgb_input.clamp(min=0,max=1))
        test_img.save(os.path.join(save_dir, 'input1.png'))
        rgb_input = (batch['context']["image"][0,1] + 1) / 2
        test_img = to_pil_image(rgb_input.clamp(min=0,max=1))
        test_img.save(os.path.join(save_dir, 'input2.png'))

    def weakly_corr(self, sat_feat, g2s_feat, g2s_conf, batch=None):
        # 这里实现弱监督相关性计算
        b = sat_feat.shape[0]
        # 检查输入是否包含 NaN 或 Inf
        if torch.isnan(sat_feat).any() or torch.isnan(g2s_feat).any() or torch.isnan(g2s_conf).any():
            print("Warning: Input contains NaN values in weakly_corr")

            # 记录脏数据信息
            additional_info = {
                'sat_feat_nan_count': torch.isnan(sat_feat).sum().item(),
                'g2s_feat_nan_count': torch.isnan(g2s_feat).sum().item(),
                'g2s_conf_nan_count': torch.isnan(g2s_conf).sum().item(),
                'dirty_line': batch['context']['line'],
            }

            # 记录脏数据信息
            self.log_dirty_data(batch, "weakly_corr", additional_info)

            # 用 0 替换 NaN 值
            sat_feat = torch.nan_to_num(sat_feat, nan=0.0)
            g2s_feat = torch.nan_to_num(g2s_feat, nan=0.0)
            g2s_conf = torch.nan_to_num(g2s_conf, nan=0.0)

        signal = sat_feat.repeat(1, b, 1, 1)  #[B(M), BC(NC), H, W]
        kernel = g2s_feat * g2s_conf.pow(2)
        corr = F.conv2d(signal, kernel, groups=b)
        denominator_sat = []
        sat_feat_pow = (sat_feat).pow(2)
        g2s_conf_pow = g2s_conf.pow(2)
        for i in range(0, b):
            denom_sat = torch.sum(F.conv2d(sat_feat_pow[i, :, None, :, :], g2s_conf_pow), dim=0)
            denominator_sat.append(denom_sat)
        denominator_sat = torch.sqrt(torch.stack(denominator_sat, dim=0))  #[B (M), B (N), H, W]
        denom_grd = torch.linalg.norm((g2s_feat * g2s_conf).reshape(b, -1), dim=-1) # [B]
        shape = denominator_sat.shape
        denominator_grd = denom_grd[None, :, None, None].repeat(shape[0], 1, shape[2], shape[3])
        denominator = denominator_sat * denominator_grd
        denominator = torch.maximum(denominator, torch.ones_like(denominator) * 1e-6)
        corr = 2 - 2 * corr / denominator  #[B, B, H, W]
        return corr

    def supervise_corr(self, sat_feat, g2s_feat, g2s_conf, batch=None):
        # 这里实现弱监督相关性计算
        B, c, crop_H, crop_W = g2s_feat.shape
        A = sat_feat.shape[-1]

        # 检查输入是否包含 NaN
        if torch.isnan(sat_feat).any() or torch.isnan(g2s_feat).any() or torch.isnan(g2s_conf).any():
            print("Warning: Input contains NaN values in supervise_corr")

            # 记录脏数据信息
            additional_info = {
                'sat_feat_nan_count': torch.isnan(sat_feat).sum().item(),
                'g2s_feat_nan_count': torch.isnan(g2s_feat).sum().item(),
                'g2s_conf_nan_count': torch.isnan(g2s_conf).sum().item(),
                'crop_size': f"{crop_H}x{crop_W}",
                'dirty_line': batch['context']['line'],
            }

            # 记录脏数据信息
            self.log_dirty_data(batch, "supervise_corr", additional_info)
            # 用 0 替换 NaN 值
            sat_feat = torch.nan_to_num(sat_feat, nan=0.0)
            g2s_feat = torch.nan_to_num(g2s_feat, nan=0.0)
            g2s_conf = torch.nan_to_num(g2s_conf, nan=0.0)
            raise ValueError(f"NaN detected in supervise_corr. Check the above printed data path information.")

        # numerator
        signal = sat_feat.reshape(1, -1, A, A)    # [B, C, H, W]->[1, B*C, H, W]
        kernel = g2s_feat * g2s_conf.pow(2)
        corr = F.conv2d(signal, kernel, groups=B)[0]   # [B, H, W]

        # denominator
        sat_feat_pow = (sat_feat).pow(2).transpose(0, 1)  # [B, C, H, W]->[C, B, H, W]
        g2s_conf_pow = g2s_conf.pow(2)
        denominator_sat = F.conv2d(sat_feat_pow, g2s_conf_pow, groups=B).transpose(0, 1)  # [B, C, H, W]
        denominator_sat = torch.sqrt(torch.sum(denominator_sat, dim=1))  # [B, H, W]

        denom_grd = torch.linalg.norm((g2s_feat * g2s_conf).reshape(B, -1), dim=-1) # [B]
        shape = denominator_sat.shape
        denominator_grd = denom_grd[:, None, None].repeat(1, shape[1], shape[2])
        denominator = denominator_sat * denominator_grd

        # if args.use_uncertainty:
        #     denominator = denominator * TF.center_crop(sat_uncer_dict[level], [corr.shape[1], corr.shape[2]])[:, 0]

        denominator = torch.maximum(denominator, torch.ones_like(denominator) * 1e-6)

        corr = corr / denominator
        corr_res = 2 - 2 * corr

        return corr_res


    def on_test_end(self) -> None:
        name = get_cfg()["wandb"]["name"]
        save_path = self.test_cfg.output_path / name
        save_path.mkdir(parents=True, exist_ok=True)

        self.benchmarker.dump(save_path / "benchmark.json")
        self.benchmarker.dump_memory(
            save_path / "peak_memory.json"
        )
        self.benchmarker.summarize()

        # Calculate and print evaluation metrics if we have predictions
        if self.pred_lons and self.pred_lats and self.gt_lons and self.gt_lats:
            try:
                # Convert lists to numpy arrays
                pred_lons = np.array(self.pred_lons)
                pred_lats = np.array(self.pred_lats)
                gt_lons = np.array(self.gt_lons)
                gt_lats = np.array(self.gt_lats)

                # Calculate evaluation metrics
                distance = np.sqrt((pred_lons - gt_lons) ** 2 + (pred_lats - gt_lats) ** 2)
                init_dis = np.sqrt(gt_lats ** 2 + gt_lons ** 2)
                diff_lats = np.abs(pred_lats - gt_lats)
                diff_lons = np.abs(pred_lons - gt_lons)

                # Get benchmark timing info
                # avg_time_per_image = self.benchmarker.get_avg_time("encoder") + self.benchmarker.get_avg_time("decoder")

                # Save results to .mat file
                scio.savemat(save_path / 'test_results.mat', {
                    'gt_lons': gt_lons,
                    'gt_lats': gt_lats,
                    'pred_lats': pred_lats,
                    'pred_lons': pred_lons
                })

                # Write results to text file (append mode)
                with open(save_path / 'test_results.txt', 'a') as f:
                    f.write('====================================\n')
                    f.write(f'       TEST RESULTS\n')
                    f.write('====================================\n')

                    print('====================================')
                    print('       TEST RESULTS')
                    print('====================================')

                    # Timing info
                    # line = f'Time per image (second): {avg_time_per_image:.4f}\n'
                    # print(line, end='')
                    # f.write(line)

                    # Distance metrics
                    line = f'Distance average: {np.mean(init_dis):.4f} (init) -> {np.mean(distance):.4f} (pred)\n'
                    print(line, end='')
                    f.write(line)

                    line = f'Distance median: {np.median(init_dis):.4f} (init) -> {np.median(distance):.4f} (pred)\n'
                    print(line, end='')
                    f.write(line)

                    # Lateral error
                    line = f'Lateral average: {np.mean(np.abs(gt_lats)):.4f} (init) -> {np.mean(diff_lats):.4f} (pred)\n'
                    print(line, end='')
                    f.write(line)

                    line = f'Lateral median: {np.median(np.abs(gt_lats)):.4f} (init) -> {np.median(diff_lats):.4f} (pred)\n'
                    print(line, end='')
                    f.write(line)

                    # Longitudinal error
                    line = f'Longitudinal average: {np.mean(np.abs(gt_lons)):.4f} (init) -> {np.mean(diff_lons):.4f} (pred)\n'
                    print(line, end='')
                    f.write(line)

                    line = f'Longitudinal median: {np.median(np.abs(gt_lons)):.4f} (init) -> {np.median(diff_lons):.4f} (pred)\n'
                    print(line, end='')
                    f.write(line)

                    print('\n-------------------------')
                    f.write('\n-------------------------\n')

                    # Accuracy metrics
                    metrics = [1, 3, 5]
                    for threshold in metrics:
                        pred_acc = np.sum(distance < threshold) / distance.shape[0] * 100
                        init_acc = np.sum(init_dis < threshold) / init_dis.shape[0] * 100
                        line = f'distance within {threshold}m: {init_acc:.2f}% (init) -> {pred_acc:.2f}% (pred)\n'
                        print(line, end='')
                        f.write(line)

                    print('\n-------------------------')
                    f.write('\n-------------------------\n')

                    # Lateral accuracy
                    for threshold in metrics:
                        pred_acc = np.sum(diff_lats < threshold) / diff_lats.shape[0] * 100
                        init_acc = np.sum(np.abs(gt_lats) < threshold) / gt_lats.shape[0] * 100
                        line = f'lateral within {threshold}m: {init_acc:.2f}% (init) -> {pred_acc:.2f}% (pred)\n'
                        print(line, end='')
                        f.write(line)

                    print('\n-------------------------')
                    f.write('\n-------------------------\n')

                    # Longitudinal accuracy
                    for threshold in metrics:
                        pred_acc = np.sum(diff_lons < threshold) / diff_lons.shape[0] * 100
                        init_acc = np.sum(np.abs(gt_lons) < threshold) / gt_lons.shape[0] * 100
                        line = f'longitudinal within {threshold}m: {init_acc:.2f}% (init) -> {pred_acc:.2f}% (pred)\n'
                        print(line, end='')
                        f.write(line)

                    print('\n-------------------------')
                    f.write('\n-------------------------\n')

                print(f"Test results saved to {save_path}")

            except Exception as e:
                print(f"Error calculating test metrics: {e}")

            # Reset lists for next test run
            self.pred_lons = []
            self.pred_lats = []
            self.gt_lons = []
            self.gt_lats = []

    def on_validation_epoch_end(self) -> None:
        """在每个验证epoch结束时执行，与on_test_end相同的操作"""
        # 获取当前epoch
        current_epoch = self.current_epoch

        # 获取输出目录
        name = get_cfg()["wandb"]["name"]
        # 使用验证特有的目录
        save_path = self.train_cfg.output_path / f"exp_{name}"
        save_path.mkdir(parents=True, exist_ok=True)

        # Calculate and print evaluation metrics if we have predictions
        if self.pred_lons and self.pred_lats and self.gt_lons and self.gt_lats:
            try:
                print(f"\n=== Validation Epoch {current_epoch + 1} Results ===")

                # Convert lists to numpy arrays
                pred_lons = np.array(self.pred_lons)
                pred_lats = np.array(self.pred_lats)
                gt_lons = np.array(self.gt_lons)
                gt_lats = np.array(self.gt_lats)

                # Calculate evaluation metrics
                distance = np.sqrt((pred_lons - gt_lons) ** 2 + (pred_lats - gt_lats) ** 2)
                init_dis = np.sqrt(gt_lats ** 2 + gt_lons ** 2)
                diff_lats = np.abs(pred_lats - gt_lats)
                diff_lons = np.abs(pred_lons - gt_lons)

                # Save results to .mat file
                scio.savemat(save_path / 'val_results.mat', {
                    'gt_lons': gt_lons,
                    'gt_lats': gt_lats,
                    'pred_lats': pred_lats,
                    'pred_lons': pred_lons
                })

                # Write results to text file
                with open(save_path / f"{name}_val_epoch_{current_epoch + 1}_results.txt", 'w') as f:
                    f.write('====================================\n')
                    f.write(f'  VALIDATION EPOCH {current_epoch + 1} RESULTS\n')
                    f.write('====================================\n')

                    print('====================================')
                    print(f'  VALIDATION EPOCH {current_epoch + 1} RESULTS')
                    print('====================================')

                    # Distance metrics
                    line = f'Distance average: {np.mean(init_dis):.4f} (init) -> {np.mean(distance):.4f} (pred)\n'
                    print(line, end='')
                    f.write(line)

                    line = f'Distance median: {np.median(init_dis):.4f} (init) -> {np.median(distance):.4f} (pred)\n'
                    print(line, end='')
                    f.write(line)

                    # Lateral error
                    line = f'Lateral average: {np.mean(np.abs(gt_lats)):.4f} (init) -> {np.mean(diff_lats):.4f} (pred)\n'
                    print(line, end='')
                    f.write(line)

                    line = f'Lateral median: {np.median(np.abs(gt_lats)):.4f} (init) -> {np.median(diff_lats):.4f} (pred)\n'
                    print(line, end='')
                    f.write(line)

                    # Longitudinal error
                    line = f'Longitudinal average: {np.mean(np.abs(gt_lons)):.4f} (init) -> {np.mean(diff_lons):.4f} (pred)\n'
                    print(line, end='')
                    f.write(line)

                    line = f'Longitudinal median: {np.median(np.abs(gt_lons)):.4f} (init) -> {np.median(diff_lons):.4f} (pred)\n'
                    print(line, end='')
                    f.write(line)

                    print('\n-------------------------')
                    f.write('\n-------------------------\n')

                    # Accuracy metrics
                    metrics = [1, 3, 5]
                    for threshold in metrics:
                        pred_acc = np.sum(distance < threshold) / distance.shape[0] * 100
                        init_acc = np.sum(init_dis < threshold) / init_dis.shape[0] * 100
                        line = f'distance within {threshold}m: {init_acc:.2f}% (init) -> {pred_acc:.2f}% (pred)\n'
                        print(line, end='')
                        f.write(line)

                    print('\n-------------------------')
                    f.write('\n-------------------------\n')

                print(f"Validation epoch {current_epoch + 1} results saved to {save_path}")

            except Exception as e:
                print(f"Error calculating validation metrics: {e}")

            # Reset lists for next validation run
            self.pred_lons = []
            self.pred_lats = []
            self.gt_lons = []
            self.gt_lats = []

    def configure_optimizers(self):
        new_params, new_param_names = [], []
        pretrained_params, pretrained_param_names = [], []
        params, param_names = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            
            params.append(param)
            param_names.append(name)

            if "gaussian_param_head" in name or "intrinsic_encoder" in name:
                new_params.append(param)
                new_param_names.append(name)
            else:
                pretrained_params.append(param)
                pretrained_param_names.append(name)

        # param_dicts = [
        #     {
        #         "params": new_params,
        #         "lr": self.optimizer_cfg.lr,
        #      },
        #     {
        #         "params": pretrained_params,
        #         "lr": self.optimizer_cfg.lr,
        #     },
        # ]

        param_dicts = [
            {
                "params": params,
                "lr": self.optimizer_cfg.lr,
            },
        ]

        optimizer = torch.optim.AdamW(param_dicts, lr=self.optimizer_cfg.lr, weight_decay=0.05, betas=(0.9, 0.95))
        warm_up_steps = self.optimizer_cfg.warm_up_steps
        warm_up = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            1 / warm_up_steps,
            1,
            total_iters=warm_up_steps,
        )

        max_steps = get_cfg()["trainer"]["max_steps"] * get_cfg()["trainer"]["max_epochs"] / get_cfg()["data_loader"]["train"]["batch_size"]
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=self.optimizer_cfg.lr * 0.1)
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warm_up, lr_scheduler], milestones=[warm_up_steps])

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
