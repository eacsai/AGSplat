import os
import torch
import torch.nn.functional as F
import numpy as np
import math
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, Optional, Dict, List, Any
from einops import einsum, rearrange, repeat
from functools import partial
from scipy.optimize import least_squares

def normalized_view_plane_uv(width: int, height: int, aspect_ratio: float = None,
                           dtype: torch.dtype = None, device: torch.device = None) -> torch.Tensor:
    """
    Generate normalized UV coordinates with left-top corner as (-width/diagonal, -height/diagonal)
    and right-bottom corner as (width/diagonal, height/diagonal).

    Args:
        width: Image width
        height: Image height
        aspect_ratio: Optional aspect ratio (defaults to width/height)
        dtype: Tensor data type
        device: Tensor device

    Returns:
        UV coordinates tensor of shape (H, W, 2)
    """
    if aspect_ratio is None:
        aspect_ratio = width / height

    span_x = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5
    span_y = 1 / (1 + aspect_ratio ** 2) ** 0.5

    u = torch.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, width, dtype=dtype, device=device)
    v = torch.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height, height, dtype=dtype, device=device)
    u, v = torch.meshgrid(u, v, indexing='xy')
    uv = torch.stack([u, v], dim=-1)
    return uv


def solve_optimal_focal_shift(uv: np.ndarray, xyz: np.ndarray) -> Tuple[float, float]:
    """
    Solve `min |focal * xy / (z + shift) - uv|` with respect to shift and focal.

    Args:
        uv: UV coordinates
        xyz: 3D points

    Returns:
        Tuple of (optimal_shift, optimal_focal)
    """
    uv, xy, z = uv.reshape(-1, 2), xyz[..., :2].reshape(-1, 2), xyz[..., 2].reshape(-1)

    def fn(uv: np.ndarray, xy: np.ndarray, z: np.ndarray, shift: np.ndarray):
        xy_proj = xy / (z + shift)[:, None]
        f = (xy_proj * uv).sum() / np.square(xy_proj).sum()
        err = (f * xy_proj - uv).ravel()
        return err

    solution = least_squares(partial(fn, uv, xy, z), x0=0, ftol=1e-3, method='lm')
    optim_shift = solution['x'].squeeze().astype(np.float32)

    xy_proj = xy / (z + optim_shift)[:, None]
    optim_focal = (xy_proj * uv).sum() / np.square(xy_proj).sum()

    return optim_shift, optim_focal


def solve_optimal_shift(uv: np.ndarray, xyz: np.ndarray, focal: float) -> float:
    """
    Solve `min |focal * xy / (z + shift) - uv|` with respect to shift.

    Args:
        uv: UV coordinates
        xyz: 3D points
        focal: Known focal length

    Returns:
        Optimal shift value
    """
    uv, xy, z = uv.reshape(-1, 2), xyz[..., :2].reshape(-1, 2), xyz[..., 2].reshape(-1)

    def fn(uv: np.ndarray, xy: np.ndarray, z: np.ndarray, shift: np.ndarray):
        xy_proj = xy / (z + shift)[:, None]
        err = (focal * xy_proj - uv).ravel()
        return err

    solution = least_squares(partial(fn, uv, xy, z), x0=0, ftol=1e-3, method='lm')
    optim_shift = solution['x'].squeeze().astype(np.float32)

    return optim_shift

def recover_focal_shift(points: torch.Tensor, mask: torch.Tensor = None,
                       focal: torch.Tensor = None,
                       downsample_size: Tuple[int, int] = (64, 64)) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Recover the depth map and FoV from a point map with unknown z shift and focal.

    Args:
        points: Point map tensor of shape (..., H, W, 3)
        mask: Optional mask tensor
        focal: Optional focal length tensor
        downsample_size: Size for downsampling for efficient processing

    Returns:
        Tuple of (focal, shift) tensors
    """
    shape = points.shape
    height, width = points.shape[-3], points.shape[-2]

    points = points.reshape(-1, *shape[-3:])
    mask = None if mask is None else mask.reshape(-1, *shape[-3:-1])
    focal = focal.reshape(-1) if focal is not None else None
    uv = normalized_view_plane_uv(width, height, dtype=points.dtype, device=points.device)

    # Downsample for efficient processing
    points_lr = F.interpolate(points.permute(0, 3, 1, 2), downsample_size, mode='nearest').permute(0, 2, 3, 1)
    uv_lr = F.interpolate(uv.unsqueeze(0).permute(0, 3, 1, 2), downsample_size, mode='nearest').squeeze(0).permute(1, 2, 0)
    mask_lr = None if mask is None else F.interpolate(mask.to(torch.float32).unsqueeze(1), downsample_size, mode='nearest').squeeze(1) > 0

    # Convert to numpy for optimization
    uv_lr_np = uv_lr.cpu().numpy()
    points_lr_np = points_lr.detach().cpu().numpy()
    focal_np = focal.cpu().numpy() if focal is not None else None
    mask_lr_np = None if mask is None else mask_lr.cpu().numpy()

    optim_shift, optim_focal = [], []
    for i in range(points.shape[0]):
        points_lr_i_np = points_lr_np[i] if mask is None else points_lr_np[i][mask_lr_np[i]]
        uv_lr_i_np = uv_lr_np if mask is None else uv_lr_np[mask_lr_np[i]]

        if uv_lr_i_np.shape[0] < 2:
            optim_focal.append(1)
            optim_shift.append(0)
            continue

        if focal is None:
            optim_shift_i, optim_focal_i = solve_optimal_focal_shift(uv_lr_i_np, points_lr_i_np)
            optim_focal.append(float(optim_focal_i))
        else:
            optim_shift_i = solve_optimal_shift(uv_lr_i_np, points_lr_i_np, focal_np[i])
        optim_shift.append(float(optim_shift_i))

    optim_shift = torch.tensor(optim_shift, device=points.device, dtype=points.dtype).reshape(shape[:-3])

    if focal is None:
        optim_focal = torch.tensor(optim_focal, device=points.device, dtype=points.dtype).reshape(shape[:-3])
    else:
        optim_focal = focal.reshape(shape[:-3])

    return optim_focal, optim_shift

def intrinsics_from_focal_center(fx, fy, cx, cy):
    """
    Create camera intrinsic matrix from focal lengths and principal points.

    Args:
        fx: focal length in x direction
        fy: focal length in y direction
        cx: principal point x coordinate
        cy: principal point y coordinate

    Returns:
        3x3 camera intrinsic matrix (or batch of matrices if inputs are batched)
    """
    if isinstance(fx, torch.Tensor):
        device = fx.device
        dtype = fx.dtype
        # Create batch of intrinsic matrices if inputs are batched
        batch_shape = fx.shape if fx.dim() > 0 else ()

        # Ensure all inputs have the same batch shape
        if isinstance(fy, torch.Tensor):
            fy = fy.to(device=device, dtype=dtype)
            if fy.shape != batch_shape and fy.dim() > 0:
                fy = fy.expand(batch_shape)
        else:
            fy = torch.full(batch_shape, fy, device=device, dtype=dtype)

        if isinstance(cx, torch.Tensor):
            cx = cx.to(device=device, dtype=dtype)
            if cx.shape != batch_shape and cx.dim() > 0:
                cx = cx.expand(batch_shape)
        else:
            cx = torch.full(batch_shape, cx, device=device, dtype=dtype)

        if isinstance(cy, torch.Tensor):
            cy = cy.to(device=device, dtype=dtype)
            if cy.shape != batch_shape and cy.dim() > 0:
                cy = cy.expand(batch_shape)
        else:
            cy = torch.full(batch_shape, cy, device=device, dtype=dtype)

        K = torch.zeros(*batch_shape, 3, 3, device=device, dtype=dtype)
        K[..., 0, 0] = fx
        K[..., 1, 1] = fy
        K[..., 0, 2] = cx
        K[..., 1, 2] = cy
        K[..., 2, 2] = 1.0
    else:
        # Create numpy intrinsic matrix
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]], dtype=np.float32)

    return K

def project_point_clouds(
    point_clouds: torch.Tensor,
    point_color: torch.Tensor,
    normalized_intrinsics: torch.Tensor, # <--- 参数名变化，表示接收归一化内参
    image_height: int = 256,
    image_width: int = 256,
    background_color: tuple[float] = (0.0) # 黑色背景
) -> torch.Tensor:
    """
    将一批点云投影到图像平面上（使用归一化的内参）。

    Args:
        point_clouds (torch.Tensor): 形状为 (B, N, 3) 的点云，坐标在相机坐标系下。
        point_color (torch.Tensor): 形状为 (B, N, 3) 的点云颜色，范围 [0, 1]。
        normalized_intrinsics (torch.Tensor): 形状为 (B, 1, 3, 3) 或 (B, 3, 3) 的归一化相机内参。
        image_height (int): 输出图像的高度。
        image_width (int): 输出图像的宽度。
        background_color (tuple[float, float, float]): 图像的背景色。

    Returns:
        torch.Tensor: 形状为 (B, 3, H, W) 的渲染图像张量。
    """
    # --- 1. 准备工作 ---
    B, N, _ = point_clouds.shape
    device = point_clouds.device

    if normalized_intrinsics.dim() == 4:
        K_norm = normalized_intrinsics[:,0,:,:]
    else:
        K_norm = normalized_intrinsics

    # --- 2. 几何变换：3D到2D投影 ---
    points_transposed = torch.transpose(point_clouds, 1, 2)
    points_2d_homogeneous = K_norm @ points_transposed
    
    # --- 3. 透视除法 ---
    depths = points_2d_homogeneous[:, 2:3, :] 
    eps = 1e-8
    
    # 这里得到的是归一化坐标 (u_norm, v_norm)
    normalized_coords = points_2d_homogeneous[:, :2, :] / (depths + eps)

    # --- 4. 反归一化：将归一化坐标转换为像素坐标 --- # <<< 关键修改步骤
    u_norm = normalized_coords[:, 0, :]
    v_norm = normalized_coords[:, 1, :]
    
    u_pix = u_norm * image_width
    v_pix = v_norm * image_height
    
    # 将 u_pix 和 v_pix 重新组合成一个张量，方便后续处理
    pixel_coords = torch.stack([u_pix, v_pix], dim=1)
    
    # --- 5. 过滤无效点 (现在基于像素坐标进行过滤) ---
    u_coords = pixel_coords[:, 0, :]
    v_coords = pixel_coords[:, 1, :]
    
    valid_mask = (depths.squeeze(1) > 0) & \
                 (u_coords >= 0) & (u_coords < image_width) & \
                 (v_coords >= 0) & (v_coords < image_height)
    
    # --- 6. 渲染/着色 (Splatting) ---
    bg_color_tensor = torch.tensor(background_color, device=device, dtype=torch.float32).view(1, 1, 1, 1)
    C = point_color.shape[-1]
    rendered_images = bg_color_tensor.expand(B, C, image_height, image_width).clone()
    
    for i in range(B):
        mask_i = valid_mask[i]
        if mask_i.sum() == 0:
            continue
            
        valid_coords = pixel_coords[i, :, mask_i]
        valid_colors = point_color[i, mask_i, :]

        u_indices = valid_coords[0, :].long()
        v_indices = valid_coords[1, :].long()
        
        rendered_images[i, :, v_indices, u_indices] = valid_colors.transpose(0, 1)

    return rendered_images