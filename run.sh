HYDRA_FULL_ERROR=1 python main.py +experiment=kitti wandb.mode=disabled wandb.name=kitti

python main.py +experiment=aer_grd_drone wandb.mode=disabled wandb.name=aer_grd_drone

HYDRA_FULL_ERROR=1 CUDA_LAUNCH_BLOCKING=1 python main.py +experiment=aer_grd_drone_pi3 wandb.mode=disabled wandb.name=aer_grd_drone_pi3_642

python main.py +experiment=aer_grd_drone_pi3 wandb.mode=disabled wandb.name=aer_grd_drone_pi3_rot0
python main.py +experiment=aer_grd_drone_pi3 wandb.mode=disabled wandb.name=aer_grd_drone_vggt_rot30

python main.py mode=test +experiment=aer_grd_drone_pi3 wandb.mode=disabled wandb.name=aer_grd_drone_pi3_vggt checkpointing.load='outputs/train/exp_aer_grd_drone_pi3_vggt/checkpoints/epoch_04-test.ckpt'