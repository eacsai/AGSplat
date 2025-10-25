import json
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Literal

import torch
import torchvision.transforms as tf
import torchvision.transforms.functional as TF
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image, ImageOps
from torch import Tensor
from torch.utils.data import IterableDataset
from torchvision import transforms
import torch.nn.functional as F

from .dataset import DatasetCfgCommon
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler
import numpy as np
import os

# ----------- 全局配置 -----------
ROOT_DIR   = '/data/zhongyao/aer-grd-map'   
GrdOriImg_H = 1080
GrdOriImg_W = 1920
GrdImg_H, GrdImg_W = 256, 512             # 地面/航拍图 resize 后尺寸，原始数据：1080*1920
SatMap_SIDE = 1024                        # 卫星图输出边长，原始数据：2700*2700
# --------------------------------
satmap_dir = 'satmap'


@dataclass
class DatasetAerGrdDroneCfg(DatasetCfgCommon):
    name: str


@dataclass
class DatasetAerGrdDroneCfgWrapper:
    aer_grd_drone: DatasetAerGrdDroneCfg

Default_lat = 49.015
Satmap_zoom = 18
SatMap_original_sidelength = 512 # 0.2 m per pixel
SatMap_process_sidelength = 512 # 0.2 m per pixel

# 将 COLMAP 坐标系 (Z向上) 转换为 OpenCV 坐标系 (Z向前)
# 转换矩阵: 绕 X 轴旋转 π，将 Y 轴翻转，Z 轴从向上变为向前
# colmap_to_opencv = np.array([
#     [1, 0, 0, 0],
#     [0, 0, -1, 0],
#     [0, 1, 0, 0],
#     [0, 0, 0, 1]
# ], dtype=np.float32)

colmap_to_opencv = np.array([
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1]
], dtype=np.float32)

def get_meter_per_pixel(lat=Default_lat, zoom=Satmap_zoom, scale=SatMap_process_sidelength/SatMap_original_sidelength):
    meter_per_pixel = 156543.03392 * np.cos(lat * np.pi/180.) / (2**zoom)	
    meter_per_pixel /= 2 # because use scale 2 to get satmap 
    meter_per_pixel /= scale
    return meter_per_pixel

class DatasetAerGrdDrone(IterableDataset):
    cfg: DatasetAerGrdDroneCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    chunks: list[Path]
    near: float = 0.1
    far: float = 100.0

    def __init__(
        self,
        cfg: DatasetAerGrdDroneCfg,
        stage: Stage,
        view_sampler: ViewSampler,
        shift_range_lat=20, shift_range_lon=20, data_amount=1.0
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.to_tensor = tf.ToTensor()        
        self.pro_grdimage_dir = 'depth_data'
        if self.stage in ("train"):
            with open('/data/zhongyao/aer-grd-map/train_files_1024.txt', 'r') as f:
                lines = f.readlines()
                self.file_name = [l.rstrip() for l in lines][:int(len(lines) * data_amount)]
        else:
            with open('/data/zhongyao/aer-grd-map/test_files_1024.txt', 'r') as f:
                lines = f.readlines()
                self.file_name = [l.rstrip() for l in lines][:int(len(lines) * data_amount)]
        self.final_h = GrdImg_W   # 1024
        self.final_w = GrdImg_W   # 1024
        self.rotation_range = 0
        self.padding_top = (self.final_h - GrdImg_H) // 2
        self.padding_left = (self.final_w - GrdImg_W) // 2
        self.grdimage_transform = transforms.Compose([
            transforms.Resize(size=[GrdImg_H, GrdImg_W]),
            transforms.Pad(padding=(self.padding_left, self.padding_top, self.padding_left, self.padding_top), fill=0),
            transforms.ToTensor(),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize(size=[GrdImg_H, GrdImg_W]),
            transforms.Pad(padding=(self.padding_left, self.padding_top, self.padding_left, self.padding_top), fill=1),
            transforms.ToTensor(),
        ])
        self.satmap_transform = transforms.Compose([
            transforms.Resize(size=[SatMap_process_sidelength, SatMap_process_sidelength]),
            transforms.ToTensor(),
        ])

        self.meter_per_pixel = 0.32307    # zoom=18；210 m / 650 px
        # self.grdimage_transform = transforms.Compose([
        #     transforms.Resize(size=[256, 1024]),
        #     transforms.ToTensor(),
        # ])

        self.shift_range_meters_lat = shift_range_lat  # in terms of meters
        self.shift_range_meters_lon = shift_range_lon  # in terms of meters
        self.shift_range_pixels_lat = shift_range_lat / self.meter_per_pixel  # shift range is in terms of meters
        self.shift_range_pixels_lon = shift_range_lon / self.meter_per_pixel  # shift range is in terms of meters

    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]

    def _satellite_path(self, sub, branch, name):
        sat_dir = 'satellite_ground' if branch == 'ground' else 'satellite_drone'
        return os.path.join(ROOT_DIR, sub, sat_dir, name + '_map_new.png')

    def _drone_path(self, sub, name):
        drone_dir = 'drone'
        return os.path.join(ROOT_DIR, sub, drone_dir, name + '_map_new.png')

    def _parse_name(self, line):
        """/data/zhongyao/aer-grd-map/0000/ground/0000_000.jpeg.jpg"""
        parts = line.split('/')
        sub, branch, fname = parts[-3], parts[-2], parts[-1]
        name = fname.replace('.jpeg.jpg', '')
        return sub, branch, name

    def __iter__(self):
        # Chunks must be shuffled here (not inside __init__) for validation to show
        # random chunks.
        if self.stage in ("train"):
            self.file_name = self.shuffle(self.file_name)

        for line in self.file_name:

            if self.stage in ("train"):
                test_line = line.split(' ')
                grd_path, drone_path, sat_path, grd_mask_path, drone_mask_path = test_line[0], test_line[1], test_line[2], test_line[3], test_line[4]
                gt_shift_x = np.random.uniform(-1, 1)  # --> right as positive, parallel to the heading direction
                gt_shift_y = np.random.uniform(-1, 1)  # --> up as positive, vertical to the heading direction
                theta = np.random.uniform(-1, 1)
            else:
                test_line = line.split(' ')
                grd_path, drone_path, sat_path, grd_mask_path, drone_mask_path, gt_shift_x, gt_shift_y, theta = test_line[0], test_line[1], test_line[2], test_line[3], test_line[4], float(test_line[5]), float(test_line[6]), float(test_line[7])

            # 1. 读取图像
            grd_left_feat = Image.open(grd_path).convert('RGB')
            drone_left_feat = Image.open(drone_path).convert('RGB')
            sat_img = Image.open(sat_path).convert('RGB')
            grd_mask = Image.open(grd_mask_path).convert('L')
            drone_mask = Image.open(drone_mask_path).convert('L')

            # 对卫星图进行4倍下采样
            new_size = (650, 650)
            sat_img = sat_img.resize(new_size, Image.BILINEAR)

            # 2. 内参与位姿
            npz_path = grd_path.replace('.jpeg.jpg', '.jpeg.npz')
            npz = np.load(npz_path)
            K = torch.from_numpy(npz['intrinsics'].astype(np.float32))  # (3,3)
            fx = K[0, 0] * GrdImg_W / GrdOriImg_W / self.final_w
            fy = K[1, 1] * GrdImg_H / GrdOriImg_H / self.final_h
            cx = (K[0, 2] * GrdImg_W / GrdOriImg_W + self.padding_left) / self.final_w
            cy = (K[1, 2] * GrdImg_H / GrdOriImg_H + self.padding_top) / self.final_h
            left_camera_k = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)
            cam2world = npz['cam2world']
            cam2world = colmap_to_opencv @ cam2world
            cam2world = torch.from_numpy(cam2world.astype(np.float32))

            # 计算相机朝向(+Z轴方向)相对于世界坐标系Z轴方向(正东)的偏转角
            # 自定义坐标系: Z正东, X正南, Y朝下
            # 相机朝向是+Z轴方向，cam2world[2, 2]表示相机Z轴在世界坐标系Z方向(正东)的分量
            # cam2world[0, 2]表示相机Z轴在世界坐标系X方向(正南)的分量
            heading = np.arctan2(cam2world[0, 2], cam2world[2, 2])

            # 转换为度数制，便于理解
            heading_degrees = np.degrees(heading)

            # 确保角度在 [0, 360) 范围内
            if heading_degrees < 0:
                heading_degrees += 360 

            # 3. randomly generate shift
            # gt_shift_x负数的时候向右移动，正数的时候向左移动
            # gt_shift_y负数的时候向下移动，正数的时候向上移动
            # theta负数的时候顺时针旋转，正数的时候逆时针旋转
            dx_p = gt_shift_x * self.shift_range_meters_lon / self.meter_per_pixel
            dy_p = gt_shift_y * self.shift_range_meters_lat / self.meter_per_pixel
            sat_rand_shift = sat_img.transform(
                sat_img.size, Image.AFFINE,
                (1, 0, dx_p, 0, 1, -dy_p), resample=Image.BILINEAR)

            # randomly generate roation
            sat_rand_shift_rand_rot = \
                sat_rand_shift.rotate(theta * self.rotation_range)

            sat_rand_shift_rand_rot_central_crop = TF.center_crop(sat_rand_shift_rand_rot, SatMap_process_sidelength)
            sat_align_cam_central_crop = TF.center_crop(sat_img, SatMap_process_sidelength)

            if self.satmap_transform is not None:
                sat_rand_shift_rand_rot_central_crop = self.satmap_transform(sat_rand_shift_rand_rot_central_crop)
                sat_align_cam_central_crop = self.satmap_transform(sat_align_cam_central_crop)

            if self.grdimage_transform is not None:
                grd_left_feat = self.grdimage_transform(grd_left_feat).unsqueeze(0)
                drone_left_feat = self.grdimage_transform(drone_left_feat).unsqueeze(0)
                grd_mask = ~self.mask_transform(grd_mask).unsqueeze(0).bool()
                drone_mask = ~self.mask_transform(drone_mask).unsqueeze(0).bool()

            extrinsics = torch.eye(4, dtype=torch.float32)
            # 添加 batch 维度以适应 F.interpolate，然后移除

            grd_img = F.interpolate(grd_left_feat, size=self.cfg.input_image_shape, mode='bilinear', align_corners=False)
            drone_img = F.interpolate(drone_left_feat, size=self.cfg.input_image_shape, mode='bilinear', align_corners=False)
            grd_mask = F.interpolate(grd_mask.float(), size=self.cfg.input_image_shape, mode='bilinear', align_corners=False).squeeze(1)
            drone_mask = F.interpolate(drone_mask.float(), size=self.cfg.input_image_shape, mode='bilinear', align_corners=False).squeeze(1)

            example = {
                "context": {
                    "extrinsics": torch.stack((cam2world, cam2world), dim=0),
                    "intrinsics": torch.stack((left_camera_k, left_camera_k), dim=0),
                    "image": torch.cat((grd_img, drone_img), dim=0),
                    "mask": torch.cat((grd_mask, drone_mask), dim=0),
                    "feat_image": torch.cat((grd_left_feat, grd_left_feat), dim=0),
                    "near": self.get_bound("near", 1),
                    "far": self.get_bound("far", 1),
                    "index": torch.tensor([0], dtype=torch.int64),
                    "overlap": torch.tensor([0.5], dtype=torch.float32),
                    "grd_path": grd_path,
                    "drone_path": drone_path,
                },
                "target": {
                    "extrinsics": extrinsics[None],
                    "intrinsics": left_camera_k[None],
                    "image": grd_img,
                    "near": self.get_bound("near", 1),
                    "far": self.get_bound("far", 1),
                    "index": torch.tensor([0], dtype=torch.int64),
                },
                "sat": {
                    "sat_align": sat_align_cam_central_crop,
                    "sat": sat_rand_shift_rand_rot_central_crop,
                    "gt_shift_u": torch.tensor([-gt_shift_x], dtype=torch.float32),
                    "gt_shift_v": torch.tensor([gt_shift_y], dtype=torch.float32),
                    "gt_heading": torch.tensor([heading], dtype=torch.float32),
                    "gt_loc": torch.tensor([[-dy_p, dx_p]], dtype=torch.float32),
                    "gt_r": torch.zeros((2,2), dtype=torch.float32)
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

    def __len__(self) -> int:
        return len(self.file_name)
