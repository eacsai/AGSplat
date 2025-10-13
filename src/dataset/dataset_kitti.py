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
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset
from torchvision import transforms
import torch.nn.functional as F

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

GrdImg_H = 256  # 256 # original: 375 #224, 256
GrdImg_W = 1024  # 1024 # original:1242 #1248, 1024

GrdOriImg_H = 375
GrdOriImg_W = 1242

CameraGPS_shift_left = [1.08, 0.26]
CameraGPS_shift_right = [1.08, 0.8]  # 0.26 + 0.54
satmap_dir = 'satmap'


@dataclass
class DatasetKittiCfg(DatasetCfgCommon):
    name: str
    roots: list[Path]
    file: str


@dataclass
class DatasetKittiCfgWrapper:
    kitti: DatasetKittiCfg

Default_lat = 49.015
Satmap_zoom = 18
SatMap_original_sidelength = 512 # 0.2 m per pixel
SatMap_process_sidelength = 512 # 0.2 m per pixel

def get_meter_per_pixel(lat=Default_lat, zoom=Satmap_zoom, scale=SatMap_process_sidelength/SatMap_original_sidelength):
    meter_per_pixel = 156543.03392 * np.cos(lat * np.pi/180.) / (2**zoom)	
    meter_per_pixel /= 2 # because use scale 2 to get satmap 
    meter_per_pixel /= scale
    return meter_per_pixel

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
        shift_range_lat=20, shift_range_lon=20,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.to_tensor = tf.ToTensor()        
        self.root = str(cfg.roots[0])
        self.pro_grdimage_dir = 'depth_data'
        if self.stage in ("train"):
            with open('./kitti/train_files.txt', 'r') as f:
                file_name = f.readlines()
        else:
            with open('./kitti/test1_files.txt', 'r') as f:
                file_name = f.readlines()
        self.final_h = self.cfg.input_image_shape[0] * 4
        self.final_w = self.cfg.input_image_shape[1] * 4
        self.rotation_range = 0
        self.padding_top = (self.final_h - GrdImg_H) // 2
        self.padding_left = (self.final_w - GrdImg_W) // 2
        self.grdimage_transform = transforms.Compose([
            transforms.Resize(size=[GrdImg_H, GrdImg_W]),
            transforms.Pad(padding=(self.padding_left, self.padding_top, self.padding_left, self.padding_top), fill=0),
            transforms.ToTensor(),
        ])
        self.satmap_transform = transforms.Compose([
            transforms.Resize(size=[SatMap_process_sidelength, SatMap_process_sidelength]),
            transforms.ToTensor(),
        ])

        self.meter_per_pixel = get_meter_per_pixel(scale=1)
        # self.grdimage_transform = transforms.Compose([
        #     transforms.Resize(size=[256, 1024]),
        #     transforms.ToTensor(),
        # ])

        self.shift_range_meters_lat = shift_range_lat  # in terms of meters
        self.shift_range_meters_lon = shift_range_lon  # in terms of meters
        self.shift_range_pixels_lat = shift_range_lat / self.meter_per_pixel  # shift range is in terms of meters
        self.shift_range_pixels_lon = shift_range_lon / self.meter_per_pixel  # shift range is in terms of meters

        self.file_name = [file[:-1] for file in file_name[: int(len(file_name))]]

    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]

    def __iter__(self):
        # Chunks must be shuffled here (not inside __init__) for validation to show
        # random chunks.
        if self.stage in ("train"):
            self.file_name = self.shuffle(self.file_name)

        for line in self.file_name:
            if self.stage in ("train"):
                file_name = line
                gt_shift_x = np.random.uniform(-1, 1)  # --> right as positive, parallel to the heading direction
                gt_shift_y = np.random.uniform(-1, 1)  # --> up as positive, vertical to the heading direction
                theta = np.random.uniform(-1, 1)
            else:
                file_name, gt_shift_x, gt_shift_y, theta = line.split(' ')
                gt_shift_x = -float(gt_shift_x)
                gt_shift_y = -float(gt_shift_y)
                theta = float(theta)
            # Load the chunk.
            day_dir = file_name[:10]
            drive_dir = file_name[:38]
            image_no = file_name[38:]

            # =================== read satellite map ===================================
            SatMap_name = os.path.join(self.root, satmap_dir, file_name)
            with Image.open(SatMap_name, 'r') as SatMap:
                sat_map = SatMap.convert('RGB')

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

            sat_rot = sat_map.rotate(-heading / np.pi * 180)
            sat_align_cam = sat_rot.transform(sat_rot.size, Image.AFFINE,
                                            (1, 0, CameraGPS_shift_left[0] / self.meter_per_pixel,
                                            0, 1, CameraGPS_shift_left[1] / self.meter_per_pixel),
                                            resample=Image.BILINEAR)
            # randomly generate shift
            # gt_shift_x负数的时候向右移动，正数的时候向左移动
            # gt_shift_y负数的时候向下移动，正数的时候向上移动
            # theta负数的时候顺时针旋转，正数的时候逆时针旋
            sat_rand_shift = \
                sat_align_cam.transform(
                    sat_align_cam.size, Image.AFFINE,
                    (1, 0, gt_shift_x * self.shift_range_pixels_lon,
                    0, 1, -gt_shift_y * self.shift_range_pixels_lat),
                    resample=Image.BILINEAR)

            # randomly generate roation
            sat_rand_shift_rand_rot = \
                sat_rand_shift.rotate(theta * self.rotation_range)

            sat_rand_shift_rand_rot_central_crop = TF.center_crop(sat_rand_shift_rand_rot, SatMap_process_sidelength)
            sat_align_cam_central_crop = TF.center_crop(sat_align_cam, SatMap_process_sidelength)

            if self.satmap_transform is not None:
                sat_rand_shift_rand_rot_central_crop = self.satmap_transform(sat_rand_shift_rand_rot_central_crop)
                sat_align_cam_central_crop = self.satmap_transform(sat_align_cam_central_crop)

            extrinsics = torch.eye(4, dtype=torch.float32)
            grd_imgs = F.interpolate(grd_left_imgs, size=self.cfg.input_image_shape, mode='bilinear', align_corners=False)
            grd_sky_imgs = F.interpolate(grd_left_imgs_ori, size=self.cfg.input_image_shape, mode='bilinear', align_corners=False)

            mask = rearrange(grd_imgs, 'v c h w -> v h w c').any(dim=-1).float()
            example = {
                "context": {
                    "extrinsics": torch.stack((extrinsics, extrinsics), dim = 0),
                    "intrinsics": torch.stack((left_camera_k, left_camera_k), dim = 0),
                    "image": torch.cat((grd_imgs, grd_imgs), dim = 0),
                    "mask": torch.cat((mask, mask), dim = 0),
                    "feat_image": torch.cat((grd_left_imgs, grd_left_imgs), dim = 0),
                    "near": self.get_bound("near", 1),
                    "far": self.get_bound("far", 1),
                    "index": torch.tensor([0], dtype=torch.int64),
                    "overlap": torch.tensor([0.5], dtype=torch.float32),
                },
                "target": {
                    "extrinsics": extrinsics[None],
                    "intrinsics": left_camera_k[None],
                    "image": grd_imgs,
                    "near": self.get_bound("near", 1),
                    "far": self.get_bound("far", 1),
                    "index": torch.tensor([0], dtype=torch.int64),
                },
                "sat": {
                    "sat_align": sat_align_cam_central_crop,
                    "sat": sat_rand_shift_rand_rot_central_crop,
                    "gt_shift_u": torch.tensor([-gt_shift_x], dtype=torch.float32),
                    "gt_shift_v": torch.tensor([gt_shift_y], dtype=torch.float32),
                    "gt_heading": torch.tensor([-heading], dtype=torch.float32),
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
