HYDRA_FULL_ERROR=1 python main.py +experiment=kitti wandb.mode=disabled wandb.name=kitti

python main.py +experiment=aer_grd_drone wandb.mode=disabled wandb.name=aer_grd_drone

python main.py +experiment=aer_grd_drone_pi3 wandb.mode=disabled wandb.name=aer_grd_drone_pi3
