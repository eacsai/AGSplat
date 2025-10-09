import json
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Literal

import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset
from torchvision import transforms

from ..geometry.projection import get_fov
from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler
from ..misc.cam_utils import camera_normalization
import numpy as np
import os

train_file = './dataLoader/train_files.txt'
test1_file = './dataLoader/test1_files.txt'
test2_file = './dataLoader/test2_files.txt'

test_csv_file_name = 'test.csv'
ignore_csv_file_name = 'ignore.csv'
satmap_dir = 'satmap'
grdimage_dir = 'depth_data'
grd_depth_dir = 'image_02/grd_depth'  # 'image_02\\data' #
left_color_camera_dir = 'image_02/grd_no_sky'  # 'image_02\\data' #
left_color_camera_dir_original = 'image_02/data'  # 'image_02\\data' #
right_color_camera_dir = 'image_03/data'  # 'image_03\\data' #
oxts_dir = 'oxts/data'  # 'oxts\\data' #

GrdImg_H = 64  # 256 # original: 375 #224, 256
GrdImg_W = 256  # 1024 # original:1242 #1248, 1024

GrdOriImg_H = 375
GrdOriImg_W = 1242

@dataclass
class DatasetKittiCfg(DatasetCfgCommon):
    name: str
    roots: list[Path]
    file: str


@dataclass
class DatasetKittiCfgWrapper:
    kitti: DatasetKittiCfg


class DatasetKitti(IterableDataset):
    cfg: DatasetKittiCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    chunks: list[Path]
    near: float = 0.1
    far: float = 100.0

    def __init__(
        self,
        cfg: DatasetKittiCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.to_tensor = tf.ToTensor()        
        self.root = str(cfg.roots[0])
        self.pro_grdimage_dir = 'depth_data'
        with open(cfg.file, 'r') as f:
            file_name = f.readlines()
        self.final_h = self.cfg.input_image_shape[0]
        self.final_w = self.cfg.input_image_shape[1]

        self.padding_top = (self.final_h - GrdImg_H) // 2
        self.padding_left = (self.final_w - GrdImg_W) // 2
        self.grdimage_transform = transforms.Compose([
            transforms.Resize(size=[GrdImg_H, GrdImg_W]),
            transforms.Pad(padding=(self.padding_left, self.padding_top, self.padding_left, self.padding_top), fill=0),
            transforms.ToTensor(),
        ])

        # self.grdimage_transform = transforms.Compose([
        #     transforms.Resize(size=[256, 1024]),
        #     transforms.ToTensor(),
        # ])

        self.file_name = [file[:-1] for file in file_name[: int(len(file_name))]]

    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]

    def __iter__(self):
        # Chunks must be shuffled here (not inside __init__) for validation to show
        # random chunks.
        if self.stage in ("train", "val"):
            self.file_name = self.shuffle(self.file_name)

        # When testing, the data loaders alternate chunks.
        worker_info = torch.utils.data.get_worker_info()
        if self.stage == "test" and worker_info is not None:
            self.file_name = self.file_name

        for file_name in self.file_name:
            # Load the chunk.
            day_dir = file_name[:10]
            drive_dir = file_name[:38]
            image_no = file_name[38:]


            grd_left_imgs = torch.tensor([])
            grd_left_depths = torch.tensor([])
            grd_depth_imgs = torch.tensor([])
            grd_left_imgs_ori = torch.tensor([])
            image_no = file_name[38:]

            # oxt: such as 0000000000.txt
            oxts_file_name = os.path.join(self.root, grdimage_dir, drive_dir, oxts_dir, image_no.lower().replace('.png', '.txt'))
            
            with open(oxts_file_name, 'r') as f:
                content = f.readline().split(' ')
                # get heading
                heading = float(content[5])
                heading = torch.from_numpy(np.asarray(heading))

                # ------ Added for weakly-supervised training ------
                GPS = torch.from_numpy(np.asarray([float(content[0]), float(content[1])]))
                # --------------------------------------------------
                grd_depth = os.path.join(self.root, self.pro_grdimage_dir, drive_dir, grd_depth_dir,
                                    image_no.lower().replace('.png', '_grd_depth.pt'))
                # read ground depth
                grd_depth_left = torch.load(grd_depth, map_location=torch.device('cpu'), weights_only=True)
                grd_depth_imgs = torch.cat([grd_depth_imgs, grd_depth_left.unsqueeze(0)], dim=0)

                left_img_name = os.path.join(self.root, self.pro_grdimage_dir, drive_dir, left_color_camera_dir,
                                            image_no.lower())
                            
                left_img_name_original = os.path.join(self.root, 'raw_data', drive_dir, left_color_camera_dir_original,
                                            image_no.lower())
                
                with Image.open(left_img_name, 'r') as GrdImg:
                    grd_img_left = GrdImg.convert('RGB')
                    if self.grdimage_transform is not None:
                        grd_img_left = self.grdimage_transform(grd_img_left)

                grd_left_imgs = torch.cat([grd_left_imgs, grd_img_left.unsqueeze(0)], dim=0)

                with Image.open(left_img_name_original, 'r') as GrdImg:
                    grd_img_left_ori = GrdImg.convert('RGB')
                    if self.grdimage_transform is not None:
                        grd_img_left_ori = self.grdimage_transform(grd_img_left_ori)

                grd_left_imgs_ori = torch.cat([grd_left_imgs_ori, grd_img_left_ori.unsqueeze(0)], dim=0)
            
            # =================== read camera intrinsice for left and right cameras ====================
            calib_file_name = os.path.join(self.root, grdimage_dir, day_dir, 'calib_cam_to_cam.txt')
            with open(calib_file_name, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    # left color camera k matrix
                    if 'P_rect_02' in line:
                        # get 3*3 matrix from P_rect_**:
                        items = line.split(':')
                        valus = items[1].strip().split(' ')
                        fx = float(valus[0]) * GrdImg_W / GrdOriImg_W / self.final_w
                        cx = (float(valus[2]) * GrdImg_W / GrdOriImg_W + self.padding_left) / self.final_w 
                        fy = float(valus[5]) * GrdImg_H / GrdOriImg_H / self.final_h
                        cy = (float(valus[6]) * GrdImg_H / GrdOriImg_H + self.padding_top) / self.final_h
                        
                        # fx = float(valus[0]) / GrdOriImg_W / 2
                        # cx = float(valus[2]) / GrdOriImg_W / 2
                        # fy = float(valus[5]) / GrdOriImg_H / 2
                        # cy = float(valus[6]) / GrdOriImg_H / 2
                        left_camera_k = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
                        left_camera_k = torch.from_numpy(np.asarray(left_camera_k, dtype=np.float32))
                        # if not self.stereo:
                        break

            extrinsics = torch.eye(4, dtype=torch.float32)

            example = {
                "context": {
                    "extrinsics": torch.stack((extrinsics, extrinsics), dim = 0),
                    "intrinsics": torch.stack((left_camera_k, left_camera_k), dim = 0),
                    "image": torch.cat((grd_left_imgs, grd_left_imgs), dim = 0),
                    "near": self.get_bound("near", 1),
                    "far": self.get_bound("far", 1),
                    "index": torch.tensor([0], dtype=torch.int64),
                    "overlap": torch.tensor([0.5], dtype=torch.float32),
                },
                "target": {
                    "extrinsics": extrinsics[None],
                    "intrinsics": left_camera_k[None],
                    "image": grd_left_imgs,
                    "near": self.get_bound("near", 1),
                    "far": self.get_bound("far", 1),
                    "index": torch.tensor([0], dtype=torch.int64),
                },
                "scene": 'kitti',
            }
            yield apply_crop_shim(example, tuple(self.cfg.input_image_shape))

    def convert_images(
        self,
        images: list[UInt8[Tensor, "..."]],
    ) -> Float[Tensor, "batch 3 height width"]:
        torch_images = []
        for image in images:
            image = Image.open(BytesIO(image.numpy().tobytes()))
            torch_images.append(self.to_tensor(image))
        return torch.stack(torch_images)

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    @property
    def data_stage(self) -> Stage:
        if self.cfg.overfit_to_scene is not None:
            return "test"
        if self.stage == "val":
            return "test"
        return self.stage

    @cached_property
    def index(self) -> dict[str, Path]:
        merged_index = {}
        data_stages = [self.data_stage]
        if self.cfg.overfit_to_scene is not None:
            data_stages = ("test", "train")
        for data_stage in data_stages:
            for root in self.cfg.roots:
                # Load the root's index.
                with (root / data_stage / "index.json").open("r") as f:
                    index = json.load(f)
                index = {k: Path(root / data_stage / v) for k, v in index.items()}

                # The constituent datasets should have unique keys.
                assert not (set(merged_index.keys()) & set(index.keys()))

                # Merge the root's index into the main index.
                merged_index = {**merged_index, **index}
        return merged_index

    # def __len__(self) -> int:
    #     return len(self.index.keys())
