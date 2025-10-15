import os
import json
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# ----------- 全局配置 -----------
ROOT_DIR   = '/data/zhongyao/aer-grd-map'   
GrdImg_H, GrdImg_W = 512, 1024             # 地面/航拍图 resize 后尺寸，原始数据：1080*1920
SatMap_SIDE = 1024                        # 卫星图输出边长，原始数据：2700*2700
# --------------------------------


class AerGrdDroneDataset(Dataset):
    """训练集：随机 lat/lon/rot 扰动"""
    def __init__(self, txt, transform=None,
                 shift_range_lat=20, shift_range_lon=20, rot_range=10, data_amount=1.0):
        super().__init__()
        with open(txt, 'r') as f:
            self.files = [l.rstrip() for l in f][:int(len(f) * data_amount)]
        self.shift_m_lat = shift_range_lat
        self.shift_m_lon = shift_range_lon
        self.rot_deg     = rot_range
        self.transform   = transform
        self.meter_per_pixel = 0.293          # zoom=20；150 m / 512 px
        self.json_cache = {}                

    def __len__(self):
        return len(self.files)

    def _parse_name(self, line):
        """/data/zhongyao/aer-grd-map/0000/ground/0000_000.jpeg.jpg"""
        parts = line.split('/')
        sub, branch, fname = parts[-3], parts[-2], parts[-1]
        name = fname.replace('.jpeg.jpg', '')
        return sub, branch, name

    def _satellite_path(self, sub, branch, name):
        sat_dir = 'satellite_ground' if branch == 'ground' else 'satellite_drone'
        return os.path.join(ROOT_DIR, sub, sat_dir, name + '_map_new.png')

    def _load_json(self, sub, branch):
        if (sub, branch) not in self.json_cache:
            json_path = os.path.join(ROOT_DIR, sub, f'{branch}.json')
            with open(json_path, 'r', encoding='utf-8') as f:
                self.json_cache[(sub, branch)] = json.load(f)
        return self.json_cache[(sub, branch)]

    def _get_lat_lon_from_json(self, sub, branch, name):
        """
        name: 1589_024 -> id=24
        """
        js = self._load_json(sub, branch)
        frame_id = int(name.split('_')[-1])          # 024 -> 24
        # 线性查找 id 匹配项
        for frame in js['cameraFrames']:
            if frame['id'] == frame_id:
                coord = frame['coordinate']
                return coord['latitude'], coord['longitude']
        raise ValueError(f"id={frame_id} not found in {sub}/{branch}.json")

    def __getitem__(self, idx):
        line = self.files[idx]
        sub, branch, name = self._parse_name(line)

        # 1. 图像
        grd_path = line
        grd_img = Image.open(grd_path).convert('RGB')
        sat_path = self._satellite_path(sub, branch, name)
        sat_img = Image.open(sat_path).convert('RGB')

        # 2. 内参与位姿
        npz_path = line.replace('.jpeg.jpg', '.jpeg.npz')
        npz = np.load(npz_path)
        K = torch.from_numpy(npz['intrinsics'].astype(np.float32))  # (3,3)
        cam2world = npz['cam2world']                                # (4,4)

        # 3. 经纬度查表 | heading 用矩阵
        lat, lon = self._get_lat_lon_from_json(sub, branch, name)
        heading = np.arctan2(cam2world[1, 0], cam2world[0, 0])      # rad
        GPS = torch.tensor([lat, lon], dtype=torch.float32)

        # 4. 随机扰动
        gt_shift_x = np.random.uniform(-1, 1)
        gt_shift_y = np.random.uniform(-1, 1)
        theta = np.random.uniform(-1, 1)

        # 5. 卫星图对齐 + 扰动
        sat_rot = sat_img.rotate(-np.rad2deg(heading))
        dx_p = gt_shift_x * self.shift_m_lon / self.meter_per_pixel
        dy_p = gt_shift_y * self.shift_m_lat / self.meter_per_pixel
        sat_shift = sat_rot.transform(
            sat_rot.size, Image.AFFINE,
            (1, 0, dx_p, 0, 1, -dy_p), resample=Image.BILINEAR)
        sat_final = sat_shift.rotate(np.rad2deg(theta * self.rot_deg))

        # 6. 中心裁剪 + tensor
        sat_Crop = TF.center_crop(sat_final, SatMap_SIDE)
        sat_Align = TF.center_crop(sat_rot, SatMap_SIDE)
        if self.transform:
            sat_Crop = self.transform(sat_Crop)
            sat_Align = self.transform(sat_Align)
            grd_img = self.transform(grd_img)

        return sat_Align, sat_Crop, K, grd_img, \
               torch.tensor(-gt_shift_x, dtype=torch.float32).view(1), \
               torch.tensor(-gt_shift_y, dtype=torch.float32).view(1), \
               torch.tensor(theta, dtype=torch.float32).view(1), \
               line, GPS


class AerGrdDroneDatasetTest(Dataset):
    """测试集：使用 txt 后三列作为真值扰动"""
    def __init__(self, txt, transform=None, **kwargs):
        super().__init__()
        self.samples = [l.rstrip().split() for l in open(txt)]
        self.transform = transform
        self.shift_m_lat = kwargs.get('shift_range_lat', 20)
        self.shift_m_lon = kwargs.get('shift_range_lon', 20)
        self.rot_deg = kwargs.get('rotation_range', 10)
        self.meter_per_pixel = 0.293        #与训练集一致
        self.json_cache = {}

    def __len__(self):
        return len(self.samples)

    def _parse_name(self, line0):
        parts = line0.split('/')
        sub, branch, fname = parts[-3], parts[-2], parts[-1]
        name = fname.replace('.jpeg.jpg', '')
        return sub, branch, name

    def _satellite_path(self, sub, branch, name):
        sat_dir = 'satellite_ground' if branch == 'ground' else 'satellite_drone'
        return os.path.join(ROOT_DIR, sub, sat_dir, name + '_map_new.png')

    def _load_json(self, sub, branch):
        if (sub, branch) not in self.json_cache:
            json_path = os.path.join(ROOT_DIR, sub, f'{branch}.json')
            with open(json_path, 'r', encoding='utf-8') as f:
                self.json_cache[(sub, branch)] = json.load(f)
        return self.json_cache[(sub, branch)]

    def _get_lat_lon_from_json(self, sub, branch, name):
        js = self._load_json(sub, branch)
        frame_id = int(name.split('_')[-1])
        coord = js['cameraFrames'][frame_id - 1]['coordinate']
        return coord['latitude'], coord['longitude']

    def __getitem__(self, idx):
        parts = self.samples[idx]
        line, gt_shift_x, gt_shift_y, theta = parts[0], float(parts[1]), float(parts[2]), float(parts[3])
        sub, branch, name = self._parse_name(line)

        grd_path = line
        grd_img = Image.open(grd_path).convert('RGB')
        sat_path = self._satellite_path(sub, branch, name)
        sat_img = Image.open(sat_path).convert('RGB')

        npz_path = line.replace('.jpeg.jpg', '.jpeg.npz')
        npz = np.load(npz_path)
        K = torch.from_numpy(npz['intrinsics'].astype(np.float32))
        cam2world = npz['cam2world']
        lat, lon = self._get_lat_lon_from_json(sub, branch, name)
        heading = np.arctan2(cam2world[1, 0], cam2world[0, 0])
        GPS = torch.tensor([lat, lon], dtype=torch.float32)

        # 给定扰动
        sat_rot = sat_img.rotate(-np.rad2deg(heading))
        dx_p = gt_shift_x * self.shift_m_lon / self.meter_per_pixel
        dy_p = gt_shift_y * self.shift_m_lat / self.meter_per_pixel
        sat_shift = sat_rot.transform(
            sat_rot.size, Image.AFFINE,
            (1, 0, dx_p, 0, 1, -dy_p), resample=Image.BILINEAR)
        sat_final = sat_shift.rotate(np.rad2deg(theta * self.rot_deg))

        sat_Crop = TF.center_crop(sat_final, SatMap_SIDE)
        sat_Align = TF.center_crop(sat_rot, SatMap_SIDE)
        if self.transform:
            sat_Crop = self.transform(sat_Crop)
            sat_Align = self.transform(sat_Align)
            grd_img = self.transform(grd_img)

        return sat_Align, sat_Crop, K, grd_img, \
               torch.tensor(-gt_shift_x, dtype=torch.float32).view(1), \
               torch.tensor(-gt_shift_y, dtype=torch.float32).view(1), \
               torch.tensor(theta, dtype=torch.float32).view(1), \
               line


# ---------------- 工厂函数 ----------------
def get_transform(is_sat):
    side = SatMap_SIDE if is_sat else (GrdImg_H, GrdImg_W)
    return T.Compose([T.Resize(side), T.ToTensor()])

def load_train_data(batch_size, txt, **kwargs):
    ds = AerGrdDroneDataset(txt, transform=(get_transform(True), get_transform(False)), **kwargs)
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=8, pin_memory=True, drop_last=False)

def load_test1_data(batch_size, txt, **kwargs):
    ds = AerGrdDroneDatasetTest(txt, transform=(get_transform(True), get_transform(False)), **kwargs)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=8, pin_memory=True, drop_last=False)