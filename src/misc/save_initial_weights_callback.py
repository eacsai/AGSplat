from lightning.pytorch.callbacks import Callback
from pathlib import Path
from colorama import Fore


def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


class SaveInitialWeightsCallback(Callback):
    """Callback to save model weights at the beginning of training."""

    def __init__(self, filename="initial_weights.ckpt"):
        super().__init__()
        self.filename = filename
        self.saved = False

    def on_train_start(self, trainer, pl_module):
        """Save initial weights when training starts."""
        if not self.saved:
            # Get the output directory from trainer logger
            if hasattr(trainer.logger, 'experiment') and hasattr(trainer.logger.experiment, 'dir'):
                output_dir = Path(trainer.logger.experiment.dir)
            elif hasattr(trainer, 'default_root_dir'):
                output_dir = Path(trainer.default_root_dir)
            else:
                # Fallback to current directory
                output_dir = Path(".")

            checkpoint_dir = output_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            checkpoint_path = checkpoint_dir / self.filename

            # Save checkpoint using Lightning's built-in save function
            trainer.save_checkpoint(str(checkpoint_path), weights_only=False)
            print(cyan(f"Initial weights saved to {checkpoint_path}"))

            self.saved = True