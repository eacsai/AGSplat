from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_only
import time
import sys


class RichProgressBar(Callback):
    """自定义Rich进度条，显示更多信息"""

    def __init__(self, refresh_rate: int = 10):
        super().__init__()
        self.refresh_rate = refresh_rate
        self.train_start_time = None
        self.val_start_time = None

    @rank_zero_only
    def on_train_epoch_start(self, trainer, pl_module):
        if self.train_start_time is None:
            self.train_start_time = time.time()

        current_epoch = trainer.current_epoch + 1
        max_epochs = trainer.max_epochs if trainer.max_epochs > 0 else "∞"

        print(f"\n🚀 Starting Epoch {current_epoch}/{max_epochs}")

    @rank_zero_only
    def on_validation_epoch_start(self, trainer, pl_module):
        if self.val_start_time is None:
            self.val_start_time = time.time()

        current_epoch = trainer.current_epoch + 1
        print(f"\n🔍 Starting Validation for Epoch {current_epoch}")

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch + 1
        max_epochs = trainer.max_epochs if trainer.max_epochs > 0 else "∞"

        if self.train_start_time is not None:
            elapsed_time = time.time() - self.train_start_time
            avg_time_per_epoch = elapsed_time / current_epoch

            remaining_epochs = max_epochs - current_epoch if isinstance(max_epochs, int) else "Unknown"
            estimated_remaining = avg_time_per_epoch * remaining_epochs if isinstance(remaining_epochs, int) else "Unknown"

            print(f"✅ Epoch {current_epoch}/{max_epochs} completed!")
            print(f"⏱️  Avg time per epoch: {avg_time_per_epoch:.2f}s")
            if isinstance(remaining_epochs, int):
                print(f"📊 Estimated time remaining: {estimated_remaining:.2f}s ({estimated_remaining/60:.1f}m)")

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch + 1

        if self.val_start_time is not None:
            elapsed_time = time.time() - self.val_start_time
            print(f"✅ Validation for Epoch {current_epoch} completed in {elapsed_time:.2f}s")

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        max_epochs = trainer.max_epochs if trainer.max_epochs > 0 else "∞"
        max_steps = trainer.max_steps if trainer.max_steps > 0 else "∞"

        print("=" * 80)
        print("🎯 TRAINING STARTED")
        print("=" * 80)
        print(f"📊 Max Epochs: {max_epochs}")
        print(f"📈 Max Steps: {max_steps}")
        print(f"🖥️  Devices: {trainer.num_devices} x {trainer.accelerator}")

        # 安全获取batch size
        try:
            if hasattr(trainer, 'datamodule') and trainer.datamodule:
                if hasattr(trainer.datamodule, 'train_dataloader') and trainer.datamodule.train_dataloader():
                    train_loader = trainer.datamodule.train_dataloader()
                    if hasattr(train_loader, 'batch_size'):
                        print(f"💾 Batch Size: {train_loader.batch_size}")
        except:
            print("💾 Batch Size: Unknown")

        print(f"📁 Output Directory: {trainer.default_root_dir}")
        print("=" * 80)

    @rank_zero_only
    def on_train_end(self, trainer, pl_module):
        if self.train_start_time is not None:
            total_time = time.time() - self.train_start_time
            current_epoch = trainer.current_epoch + 1

            print("=" * 80)
            print("🎉 TRAINING COMPLETED")
            print("=" * 80)
            print(f"⏱️  Total training time: {total_time:.2f}s ({total_time/3600:.2f}h)")
            print(f"📊 Total epochs completed: {current_epoch}")
            print(f"🚀 Training finished successfully!")
            print("=" * 80)

    @rank_zero_only
    def on_test_start(self, trainer, pl_module):
        print("=" * 80)
        print("🧪 TESTING STARTED")
        print("=" * 80)
        print(f"🖥️  Devices: {trainer.num_devices} x {trainer.accelerator}")

        # 安全获取test batch size
        try:
            if hasattr(trainer, 'datamodule') and trainer.datamodule:
                if hasattr(trainer.datamodule, 'test_dataloader') and trainer.datamodule.test_dataloader():
                    test_loader = trainer.datamodule.test_dataloader()
                    if hasattr(test_loader, 'batch_size'):
                        print(f"💾 Batch Size: {test_loader.batch_size}")
        except:
            print("💾 Batch Size: Unknown")

        print("=" * 80)

    @rank_zero_only
    def on_test_end(self, trainer, pl_module):
        print("=" * 80)
        print("✅ TESTING COMPLETED")
        print("=" * 80)
        print("🧪 All tests finished successfully!")
        print("=" * 80)


class DatasetProgressCallback(Callback):
    """显示数据集信息的Callback"""

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        try:
            datamodule = trainer.datamodule
            if datamodule is None:
                return

            print("\n📊 Dataset Information:")

            if hasattr(datamodule, 'train_dataloader') and datamodule.train_dataloader():
                train_loader = datamodule.train_dataloader()
                if hasattr(train_loader, 'dataset') and train_loader.dataset:
                    print(f"   📁 Training samples: {len(train_loader.dataset)}")
                if hasattr(train_loader, 'batch_size'):
                    print(f"   💾 Training batch size: {train_loader.batch_size}")
                print(f"   🔄 Training batches: {len(train_loader)}")

            if hasattr(datamodule, 'val_dataloader') and datamodule.val_dataloader():
                val_loader = datamodule.val_dataloader()
                if hasattr(val_loader, 'dataset') and val_loader.dataset:
                    print(f"   🔍 Validation samples: {len(val_loader.dataset)}")
                if hasattr(val_loader, 'batch_size'):
                    print(f"   💾 Validation batch size: {val_loader.batch_size}")
                print(f"   🔄 Validation batches: {len(val_loader)}")

            if hasattr(datamodule, 'test_dataloader') and datamodule.test_dataloader():
                test_loader = datamodule.test_dataloader()
                if hasattr(test_loader, 'dataset') and test_loader.dataset:
                    print(f"   🧪 Test samples: {len(test_loader.dataset)}")
                if hasattr(test_loader, 'batch_size'):
                    print(f"   💾 Test batch size: {test_loader.batch_size}")
                print(f"   🔄 Test batches: {len(test_loader)}")

        except Exception as e:
            print(f"   ⚠️  Could not get dataset information: {e}")
        print("-" * 50)