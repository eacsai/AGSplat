# -*- coding: utf-8 -*-
"""
批量生成BEV图像 - Y轴垂直(向下)→BEV投影到XZ平面 (兼容OpenCV坐标系)
"""
import torch
import numpy as np
from PIL import Image
import OpenEXR, Imath, os
from tqdm import tqdm
import glob

# ---------------- 用户配置区 ----------------
# 输入数据根目录
DATA_ROOT = r'/data/zhongyao/aer-grd-map/0015/ground'
# BEV输出根目录  
OUTPUT_ROOT = r'/home/wangqw/CVPR26/NoPoSplat/camera_position_visualization'

METER_PER_PIXEL = 0.2          # 0.2 m/px
RADIUS_M = 102.4               # BEV图像的半径 (米)
# 【修改】地面 Y 范围（米），Y轴现在是垂直轴，向下为正
GROUND_Y_RANGE = (-300.0, 500.0)
# ----------------------------------------

# -------------- 复用函数 --------------
def load_rgb(path):
    rgb = Image.open(path).convert('RGB')
    rgb = np.array(rgb, dtype=np.float32)/255.0
    return torch.from_numpy(rgb), rgb.shape[0], rgb.shape[1]

def load_depth_exr(path):
    exr = OpenEXR.InputFile(path)
    header = exr.header()
    w = int(header['dataWindow'].max.x + 1)
    h = int(header['dataWindow'].max.y + 1)
    depth = np.frombuffer(exr.channel('Y'), dtype=np.float32).reshape(h, w)
    exr.close()
    return torch.from_numpy(depth.copy())

def load_camera_params(npz_path):
    """从npz文件加载相机参数"""
    data = np.load(npz_path)
    intrinsics = data['intrinsics'].astype(np.float32)
    cam2world = data['cam2world'].astype(np.float32)
    return intrinsics, cam2world

def backproject_to_world(rgb, depth, K, T_cw):
    """反投影像素到相机坐标系，然后转换到世界坐标系"""
    H, W = depth.shape
    device = depth.device
    v, u = torch.meshgrid(torch.arange(H, dtype=torch.float32, device=device),
                          torch.arange(W, dtype=torch.float32, device=device), indexing='ij')
    uv1 = torch.stack([u, v, torch.ones_like(u)], -1)          # H,W,3
    Kinv = torch.inverse(torch.as_tensor(K, device=device))
    xyz_cam = (uv1 @ Kinv.T) * depth.unsqueeze(-1)             # H,W,3

    # 转换到世界坐标系
    xyz_world = xyz_cam.reshape(-1, 3) @ T_cw[:3, :3].T + T_cw[:3, 3].unsqueeze(0)
    rgb = rgb.reshape(-1, 3)
    
    # 过滤无效深度值
    valid = (depth.reshape(-1) > 0.1) & (depth.reshape(-1) < 1000)
    return xyz_world[valid], rgb[valid]

def make_xz_bev(xyz_world, rgb_world, cam_center_world,
                meter_per_pix, y_range, radius_m):
    """
    【核心修改】生成XZ平面的BEV图像 (Y轴垂直)
    """
    # 1. 地面高度过滤 (Y 垂直轴)
    mask = (xyz_world[:, 1] > y_range[0]) & (xyz_world[:, 1] < y_range[1])
    xyz, rgb = xyz_world[mask], rgb_world[mask]
    side = int(2 * radius_m / meter_per_pix)
    if xyz.shape[0] == 0:
        return np.zeros((side, side, 3), dtype=np.uint8)

    # 2. 局部世界坐标（相机光心为原点）
    xyz_local = xyz - cam_center_world
    xz = xyz_local[:, [0, 2]]  # 【修改】只取 X/Z 平面

    # 3. ROI：|X|<radius, |Z|<radius
    in_roi = (torch.abs(xz[:, 0]) < radius_m) & (torch.abs(xz[:, 1]) < radius_m)
    xz, rgb = xz[in_roi], rgb[in_roi]
    if xz.shape[0] == 0:
        return np.zeros((side, side, 3), dtype=np.uint8)

    # 4. 像素坐标：X→列, -Z→行（相机前方Z+在图像上方）
    side_px = int(2 * radius_m / meter_per_pix)
    cx = cy = side_px // 2
    px = (xz[:, 0] / meter_per_pix + cx).long()
    py = (-xz[:, 1] / meter_per_pix + cy).long() # Z越大, py越小, 图像越靠上
    in_map = (px >= 0) & (px < side_px) & (py >= 0) & (py < side_px)
    px, py, rgb = px[in_map], py[in_map], rgb[in_map]

    # 5. 【修改】高度排序（从高到低画点），Y值小的更高
    # 在Y轴朝下的坐标系中，Y值越小代表海拔越高。我们先画高处的点。
    order = torch.argsort(xyz_local[in_roi][in_map][:, 1], descending=False)
    px, py, rgb = px[order], py[order], rgb[order]

    # 6. 栅格化
    bev = np.zeros((side_px, side_px, 3), dtype=np.uint8)
    # 使用Numpy进行高效栅格化，避免循环
    # 注意：同一个像素位置，后面的点会覆盖前面的点，因为已经排序，这是正确的
    bev[py.numpy(), px.numpy()] = (rgb.numpy() * 255).astype(np.uint8)
    
    return bev

# -------------- 批量处理函数 --------------
def process_single_image(rgb_path, output_dir):
    """处理单个图像"""

    # 构建相关文件路径
    base_name = os.path.splitext(os.path.splitext(rgb_path)[0])[0]
    depth_path = base_name + '.jpeg.exr'
    
    # 加载数据
    rgb, h, w = load_rgb(rgb_path)
    if 'grd_drone_pair' in depth_path:
        depth_path = depth_path.replace('grd_drone_pair', 'drone')
        depth_name = depth_path.split('/')[-1]
        grd_name = depth_name.split('_')[1] + '_'
        depth_path = depth_path.replace(grd_name, '')
    depth = load_depth_exr(depth_path)
    npz_path = depth_path.replace('.exr', '.npz')
    K_np, T_cw_zup_np = load_camera_params(npz_path) # 这是Z轴朝上的外参
    
    # 【核心修改】定义从 "Z轴朝上" 到 "Y轴朝下" 的世界坐标系变换矩阵
    # 原始坐标系(A): X-?, Y-?, Z-Up
    # 目标坐标系(B): X->X, Y-> -Z, Z->Y  (即绕X轴顺时针旋转90度)
    T_zup_to_ydown = torch.tensor([
        [1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32)

    # 将相机外参从Z-up世界变换到Y-down世界
    T_cw_ydown = T_zup_to_ydown @ torch.tensor(T_cw_zup_np, dtype=torch.float32)
    
    print(f'处理: {os.path.basename(rgb_path)}')
    
    # 反投影到Z-up世界坐标系
    xyz_w_zup, rgb_w = backproject_to_world(rgb, depth, K_np, torch.tensor(T_cw_zup_np, dtype=torch.float32))

    # 【核心修改】将所有世界点从Z-up变换到Y-down
    xyz_w_zup_h = torch.cat([xyz_w_zup, torch.ones(xyz_w_zup.shape[0], 1, device=xyz_w_zup.device)], dim=-1)
    xyz_w_ydown_h = xyz_w_zup_h @ T_zup_to_ydown.T
    xyz_w_ydown = xyz_w_ydown_h[:, :3]

    print(f'有效世界点: {xyz_w_ydown.shape[0]}')
    
    # 生成BEV图像 (在新的Y-down坐标系下)
    cam_center_ydown = T_cw_ydown[:3, 3]  # 新坐标系下的相机光心
    bev_img = make_xz_bev(xyz_w_ydown, rgb_w, cam_center_ydown,
                            METER_PER_PIXEL, GROUND_Y_RANGE, RADIUS_M)
    
    # 在BEV图像中心标记相机位置
    cx = cy = bev_img.shape[0] // 2
    mark_size = int(bev_img.shape[0] * 0.02) # 标记大小为图像尺寸的2%
    bev_img[cy-mark_size:cy+mark_size+1, cx] = [255, 0, 0]
    bev_img[cy, cx-mark_size:cx+mark_size+1] = [255, 0, 0]

    # 在BEV图像中心画一个尺寸为图像尺寸40%的红色边框
    box_size = int(bev_img.shape[0] * 0.4)  # 边框尺寸为图像尺寸的40%
    half_box = box_size // 2
    # 计算边框的坐标范围
    y1, y2 = cy - half_box, cy + half_box
    x1, x2 = cx - half_box, cx + half_box

    # 确保坐标在图像范围内
    y1, x1 = max(0, y1), max(0, x1)
    y2, x2 = min(bev_img.shape[0], y2), min(bev_img.shape[1], x2)

    # 画红色边框（上、下、左、右四条边）
    bev_img[y1:y1+3, x1:x2] = [255, 0, 0]      # 上边框
    bev_img[y2-3:y2, x1:x2] = [255, 0, 0]      # 下边框
    bev_img[y1:y2, x1:x1+3] = [255, 0, 0]      # 左边框
    bev_img[y1:y2, x2-3:x2] = [255, 0, 0]      # 右边框

    # 保存BEV图像
    rgb_name = rgb_path.split('/')[-1]
    bev_save_path = os.path.join(output_dir, rgb_name)
    Image.fromarray(bev_img).save(bev_save_path)
    print(f'BEV保存至: {bev_save_path}')
    
    return True
        

def batch_process_bev():        
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    rgb_path = '/data/zhongyao/aer-grd-map/0070/grd_drone_pair/0070_003_228.jpg'
    process_single_image(rgb_path, OUTPUT_ROOT)
    
    print("处理完成! 成功生成BEV图像。")

# -------------- 主程序 --------------
if __name__ == '__main__':
    batch_process_bev()
