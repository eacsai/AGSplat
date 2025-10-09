from math import isqrt
from typing import Literal, Union

import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from einops import einsum, rearrange, repeat
from jaxtyping import Float
from torch import Tensor
import numpy as np

from ...geometry.projection import get_fov, homogenize_points
from ...model.types import Gaussians
from ...visualization.drawing.cameras import compute_equal_aabb_with_margin

def adjust_intrinsics_for_crop(
    intrinsics: Tensor,          # 形状为 (B, 3, 3) 的原始内参
    canonical_shape: tuple[int, int], # (H_old, W_old) - 原始内参对应的尺寸
    target_shape: tuple[int, int]    # (H_new, W_new) - 想要裁剪到的目标尺寸
) -> Tensor:
    """
    为中心裁剪直接调整归一化的相机内参矩阵。
    输入和输出的内参都是相对于其对应尺寸归一化的。
    """
    H_old, W_old = canonical_shape
    H_new, W_new = target_shape
    
    # 检查目标尺寸是否不大于基准尺寸
    assert H_new <= H_old and W_new <= W_old, "Target shape must be smaller than canonical shape for cropping."

    # 复制一份新的内参进行修改
    intrinsics_cropped_normalized = intrinsics.clone()
    
    # 计算从旧尺寸到新尺寸的缩放比例
    scale_w = W_old / W_new
    scale_h = H_old / H_new

    # --- 调整归一化的焦距 (fx, fy) ---
    # K[..., 0, 0] 是 fx_norm, K[..., 1, 1] 是 fy_norm
    intrinsics_cropped_normalized[..., 0, 0] *= scale_w
    intrinsics_cropped_normalized[..., 1, 1] *= scale_h

    # --- 调整归一化的主点 (cx, cy) ---
    # K[..., 0, 2] 是 cx_norm, K[..., 1, 2] 是 cy_norm
    # 公式: c_new = (c_old - 0.5) * scale + 0.5
    intrinsics_cropped_normalized[..., 0, 2] = (intrinsics_cropped_normalized[..., 0, 2] - 0.5) * scale_w + 0.5
    intrinsics_cropped_normalized[..., 1, 2] = (intrinsics_cropped_normalized[..., 1, 2] - 0.5) * scale_h + 0.5
    
    return intrinsics_cropped_normalized


def get_projection_matrix(
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    fov_x: Float[Tensor, " batch"],
    fov_y: Float[Tensor, " batch"],
) -> Float[Tensor, "batch 4 4"]:
    """Maps points in the viewing frustum to (-1, 1) on the X/Y axes and (0, 1) on the Z
    axis. Differs from the OpenGL version in that Z doesn't have range (-1, 1) after
    transformation and that Z is flipped.
    """
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    top = tan_fov_y * near
    bottom = -top
    right = tan_fov_x * near
    left = -right

    (b,) = near.shape
    result = torch.zeros((b, 4, 4), dtype=torch.float32, device=near.device)
    result[:, 0, 0] = 2 * near / (right - left)
    result[:, 1, 1] = 2 * near / (top - bottom)
    result[:, 0, 2] = (right + left) / (right - left)
    result[:, 1, 2] = (top + bottom) / (top - bottom)
    result[:, 3, 2] = 1
    result[:, 2, 2] = far / (far - near)
    result[:, 2, 3] = -(far * near) / (far - near)
    return result


def render_cuda(
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    image_shape: tuple[int, int],
    background_color: Float[Tensor, "batch 3"],
    gaussian_means: Float[Tensor, "batch gaussian 3"],
    gaussian_covariances: Float[Tensor, "batch gaussian 3 3"],
    gaussian_sh_coefficients: Float[Tensor, "batch gaussian 3 d_sh"],
    gaussian_opacities: Float[Tensor, "batch gaussian"],
    scale_invariant: bool = True,
    use_sh: bool = True,
    cam_rot_delta: Float[Tensor, "batch 3"] | None = None,
    cam_trans_delta: Float[Tensor, "batch 3"] | None = None,
) -> tuple[Float[Tensor, "batch 3 height width"], Float[Tensor, "batch height width"]]:
    assert use_sh or gaussian_sh_coefficients.shape[-1] == 1

    # Make sure everything is in a range where numerical issues don't appear.
    if scale_invariant:
        scale = 1 / near
        extrinsics = extrinsics.clone()
        extrinsics[..., :3, 3] = extrinsics[..., :3, 3] * scale[:, None]
        gaussian_covariances = gaussian_covariances * (scale[:, None, None, None] ** 2)
        gaussian_means = gaussian_means * scale[:, None, None]
        near = near * scale
        far = far * scale

    _, _, _, n = gaussian_sh_coefficients.shape
    degree = isqrt(n) - 1
    shs = rearrange(gaussian_sh_coefficients, "b g xyz n -> b g n xyz").contiguous()

    b, _, _ = extrinsics.shape
    h, w = image_shape

    intrinsics = adjust_intrinsics_for_crop(
        intrinsics, canonical_shape=(256,256), target_shape=image_shape
    )

    fov_x, fov_y = get_fov(intrinsics).unbind(dim=-1)
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    projection_matrix = get_projection_matrix(near, far, fov_x, fov_y)
    projection_matrix = rearrange(projection_matrix, "b i j -> b j i")
    view_matrix = rearrange(extrinsics.inverse(), "b i j -> b j i")
    full_projection = view_matrix @ projection_matrix

    all_images = []
    all_radii = []
    all_depths = []
    for i in range(b):
        # Set up a tensor for the gradients of the screen-space means.
        mean_gradients = torch.zeros_like(gaussian_means[i], requires_grad=True)
        try:
            mean_gradients.retain_grad()
        except Exception:
            pass

        settings = GaussianRasterizationSettings(
            image_height=h,
            image_width=w,
            tanfovx=tan_fov_x[i].item(),
            tanfovy=tan_fov_y[i].item(),
            bg=background_color[i],
            scale_modifier=1.0,
            viewmatrix=view_matrix[i],
            projmatrix=full_projection[i],
            projmatrix_raw=projection_matrix[i],
            sh_degree=degree,
            campos=extrinsics[i, :3, 3],
            prefiltered=False,  # This matches the original usage.
            debug=False,
        )
        rasterizer = GaussianRasterizer(settings)

        row, col = torch.triu_indices(3, 3)

        image, radii, depth, opacity, n_touched = rasterizer(
            means3D=gaussian_means[i],
            means2D=mean_gradients,
            shs=shs[i] if use_sh else None,
            colors_precomp=None if use_sh else shs[i, :, 0, :],
            opacities=gaussian_opacities[i, ..., None],
            cov3D_precomp=gaussian_covariances[i, :, row, col],
            theta=cam_rot_delta[i] if cam_rot_delta is not None else None,
            rho=cam_trans_delta[i] if cam_trans_delta is not None else None,
        )
        all_images.append(image)
        all_radii.append(radii)
        all_depths.append(depth.squeeze(0))
    return torch.stack(all_images), torch.stack(all_depths)


def render_cuda_orthographic(
    extrinsics: Float[Tensor, "batch 4 4"],
    width: Float[Tensor, " batch"],
    height: Float[Tensor, " batch"],
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    image_shape: tuple[int, int],
    background_color: Float[Tensor, "batch 3"],
    gaussian_means: Float[Tensor, "batch gaussian 3"],
    gaussian_covariances: Float[Tensor, "batch gaussian 3 3"],
    gaussian_sh_coefficients: Float[Tensor, "batch gaussian 3 d_sh"],
    gaussian_opacities: Float[Tensor, "batch gaussian"],
    fov_degrees: float = 0.1,
    use_sh: bool = True,
    dump: dict | None = None,
) -> Float[Tensor, "batch 3 height width"]:
    b, _, _ = extrinsics.shape
    h, w = image_shape
    assert use_sh or gaussian_sh_coefficients.shape[-1] == 1

    _, _, _, n = gaussian_sh_coefficients.shape
    degree = isqrt(n) - 1
    shs = rearrange(gaussian_sh_coefficients, "b g xyz n -> b g n xyz").contiguous()

    # Create fake "orthographic" projection by moving the camera back and picking a
    # small field of view.
    fov_x = torch.tensor(fov_degrees, device=extrinsics.device).deg2rad()
    tan_fov_x = (0.5 * fov_x).tan()
    distance_to_near = (0.5 * width) / tan_fov_x
    tan_fov_y = 0.5 * height / distance_to_near
    fov_y = (2 * tan_fov_y).atan()
    near = near + distance_to_near
    far = far + distance_to_near
    move_back = torch.eye(4, dtype=torch.float32, device=extrinsics.device)
    move_back[2, 3] = -distance_to_near
    extrinsics = extrinsics @ move_back

    # Escape hatch for visualization/figures.
    if dump is not None:
        dump["extrinsics"] = extrinsics
        dump["fov_x"] = fov_x
        dump["fov_y"] = fov_y
        dump["near"] = near
        dump["far"] = far

    projection_matrix = get_projection_matrix(
        near, far, repeat(fov_x, "-> b", b=b), fov_y
    )
    projection_matrix = rearrange(projection_matrix, "b i j -> b j i")
    view_matrix = rearrange(extrinsics.inverse(), "b i j -> b j i")
    full_projection = view_matrix @ projection_matrix

    all_images = []
    all_radii = []
    for i in range(b):
        # Set up a tensor for the gradients of the screen-space means.
        mean_gradients = torch.zeros_like(gaussian_means[i], requires_grad=True)
        try:
            mean_gradients.retain_grad()
        except Exception:
            pass

        settings = GaussianRasterizationSettings(
            image_height=h,
            image_width=w,
            tanfovx=tan_fov_x,
            tanfovy=tan_fov_y,
            bg=background_color[i],
            scale_modifier=1.0,
            viewmatrix=view_matrix[i],
            projmatrix=full_projection[i],
            projmatrix_raw=projection_matrix[i],
            sh_degree=degree,
            campos=extrinsics[i, :3, 3],
            prefiltered=False,  # This matches the original usage.
            debug=False,
        )
        rasterizer = GaussianRasterizer(settings)

        row, col = torch.triu_indices(3, 3)

        image, radii, depth, opacity, n_touched = rasterizer(
            means3D=gaussian_means[i],
            means2D=mean_gradients,
            shs=shs[i] if use_sh else None,
            colors_precomp=None if use_sh else shs[i, :, 0, :],
            opacities=gaussian_opacities[i, ..., None],
            cov3D_precomp=gaussian_covariances[i, :, row, col],
        )
        all_images.append(image)
        all_radii.append(radii)
    return torch.stack(all_images)

def render_bevs(
    gaussians: Gaussians,
    resolution: tuple[int, int],
    margin: float = 0.1,
    heading: Union[Tensor, None] = None,
    look_axis = 1,
    rot_range = 10.0,
    width = 101.0 / 2,
    height = 101.0 / 2,
) -> Float[Tensor, "batch 3 height width"]:
    device = gaussians.means.device
    B, _, _ = gaussians.means.shape
    if heading == None:
        heading = torch.zeros([B, 1], dtype=torch.float32, device=gaussians.means.device)
    color_out = []
    feature_out = []
    confidence_out = []

    for b in range(B):
        # Compute the minima and maxima of the scene.
        minima = gaussians.means[b:b+1].min(dim=1).values
        maxima = gaussians.means[b:b+1].max(dim=1).values
        scene_minima, scene_maxima = compute_equal_aabb_with_margin(
            minima, maxima, margin=margin / 2
        )

        # look = ["x", "y", "z"]
        # for look_axis in range(3):
        # look_axis = 0
        right_axis = (look_axis + 1) % 3
        down_axis = (look_axis + 2) % 3

        # Define the extrinsics for rendering.
        extrinsics = torch.zeros((1, 4, 4), dtype=torch.float32, device=device)
        extrinsics[:, right_axis, 0] = 1
        extrinsics[:, down_axis, 1] = 1
        extrinsics[:, look_axis, 2] = 1
        # extrinsics[:, right_axis, 3] = 0.5 * (
        #     scene_minima[:, right_axis] + scene_maxima[:, right_axis]
        # )
        # extrinsics[:, down_axis, 3] = 0.5 * (
        #     scene_minima[:, down_axis] + scene_maxima[:, down_axis]
        # )

        extrinsics[:, look_axis, 3] = scene_minima[:, look_axis]
        extrinsics[:, 3, 3] = 1
        real_heading = heading[b] * rot_range / 180 * np.pi
        cos = torch.cos(-real_heading)
        sin = torch.sin(-real_heading)
        zeros = torch.zeros_like(cos)
        ones = torch.ones_like(cos)
        R = torch.cat([cos, zeros, -sin, zeros, ones, zeros, sin, zeros, cos], dim=-1)  # shape = [B,9]
        R = R.view(1, 3, 3)  # shape = [B,3,3]
        # 将 R 扩展为 4x4 矩阵，形状为 [B, 4, 4]
        R_4x4 = torch.eye(4, device=device).unsqueeze(0)  # [1,4,4]
        R_4x4[:, :3, :3] = R  # 替换上半部分为旋转矩阵
        
        extrinsics_rotated = torch.bmm(R_4x4, extrinsics)  # [1,4,4]
        # Define the intrinsics for rendering.
        extents = scene_maxima - scene_minima
        far = extents[:, look_axis]
        near = torch.zeros_like(far)
        # width = extents[:, right_axis]
        # height = extents[:, down_axis]
        # extrinsics[:, right_axis, 3] = 0
        # extrinsics[:, down_axis, 3] = 0

        render_out = render_cuda_orthographic(
            extrinsics_rotated,
            torch.tensor([width], device=device, dtype=torch.float32),
            torch.tensor([height], device=device, dtype=torch.float32),
            near,
            far,
            resolution,
            torch.zeros((1, 3), dtype=torch.float32, device=device),
            gaussians.means[b:b+1],
            gaussians.covariances[b:b+1],
            gaussians.harmonics[b:b+1],
            gaussians.opacities[b:b+1],
            fov_degrees=0.1,
            use_sh=True,
        )
        color = render_out
        # feature = render_out.feature
        # confidence = render_out.confidence
        color_out.append(color)
        # feature_out.append(feature)
        # confidence_out.append(confidence)
    return torch.cat(color_out, dim=0)

def forward_project(image_tensor, xyz_grd, meter_per_pixel=0.2, sat_width=512):
    B, N_points, C = image_tensor.shape
    # xyz_grd = xyz_grd.long()
    # xyz_grd[:,:,:,0:1] += sat_width // 2
    # xyz_grd[:,:,:,2:3] += sat_width // 2
    # B, H, W, C = xyz_grd.shape
    mask = image_tensor.any(dim=-1).float().unsqueeze(-1)
    xyz_grd = (xyz_grd * mask).reshape(B*N_points, -1)
    image_tensor = rearrange(image_tensor, 'b n c -> (b n) c')
    
    # min_depth = torch.maximum(xyz_grd[:, 2].min(), torch.tensor(0, device=xyz_grd.device))
    xyz_grd[:, 0] = xyz_grd[:, 0] / meter_per_pixel
    # xyz_grd[:, 2] = (xyz_grd[:, 2] - min_depth) / meter_per_pixel
    xyz_grd[:, 2] = (xyz_grd[:, 2]) / meter_per_pixel
    xyz_grd[:, 0] = xyz_grd[:, 0].long()
    xyz_grd[:, 2] = xyz_grd[:, 2].long()

    batch_ix = torch.cat([torch.full([N_points, 1], ix, device=image_tensor.device) for ix in range(B)], dim=0)
    xyz_grd = torch.cat([xyz_grd, batch_ix], dim=-1)

    kept = (xyz_grd[:,0] >= -(sat_width // 2)) & (xyz_grd[:,0] <= (sat_width // 2) - 1) & (xyz_grd[:,2] >= -(sat_width // 2)) & (xyz_grd[:,2] <= (sat_width // 2) - 1)

    xyz_grd_kept = xyz_grd[kept]
    image_tensor_kept = image_tensor[kept]

    max_height = xyz_grd_kept[:,1].max()

    xyz_grd_kept[:,0] = xyz_grd_kept[:,0] + sat_width // 2
    xyz_grd_kept[:,1] = max_height - xyz_grd_kept[:,1]
    xyz_grd_kept[:,2] = xyz_grd_kept[:,2] + sat_width // 2
    xyz_grd_kept = xyz_grd_kept[:,[2,0,1,3]]
    rank = torch.stack((xyz_grd_kept[:, 0] * sat_width * B + (xyz_grd_kept[:, 1] + 1) * B + xyz_grd_kept[:, 3], xyz_grd_kept[:, 2]), dim=1)
    sorts_second = torch.argsort(rank[:, 1])
    xyz_grd_kept = xyz_grd_kept[sorts_second]
    image_tensor_kept = image_tensor_kept[sorts_second]
    sorted_rank = rank[sorts_second]
    sorts_first = torch.argsort(sorted_rank[:, 0], stable=True)
    xyz_grd_kept = xyz_grd_kept[sorts_first]
    image_tensor_kept = image_tensor_kept[sorts_first]
    sorted_rank = sorted_rank[sorts_first]
    kept = torch.ones_like(sorted_rank[:, 0])
    kept[:-1] = sorted_rank[:, 0][:-1] != sorted_rank[:, 0][1:]
    res_xyz = xyz_grd_kept[kept.bool()]
    res_image = image_tensor_kept[kept.bool()]
    
    # grd_image_index = torch.cat((-res_xyz[:,1:2] + grd_image_width - 1,-res_xyz[:,0:1] + grd_image_height - 1), dim=-1)
    final = torch.zeros(B,sat_width,sat_width,C).to(torch.float32).to('cuda')
    sat_height = torch.zeros(B,sat_width,sat_width,1).to(torch.float32).to('cuda')
    final[res_xyz[:,3].long(),res_xyz[:,1].long(),res_xyz[:,0].long(),:] = res_image

    res_xyz[:,2][res_xyz[:,2] < 1e-1] = 1e-1
    sat_height[res_xyz[:,3].long(),res_xyz[:,1].long(),res_xyz[:,0].long(),:] = res_xyz[:,2].unsqueeze(-1)
    sat_height = sat_height.permute(0,3,1,2)
    # img_num = 0
    # project_grd_img = to_pil_image(final[img_num].permute(2, 0, 1))
    # project_grd_img.save('sat_feat.png')

    # project_grd_img = to_pil_image(origin_image_tensor[img_num])
    # project_grd_img.save('grd_feat.png')

    return final.permute(0,3,1,2)

def project_point_clouds(
    point_clouds: torch.Tensor,
    point_color: torch.Tensor,
    normalized_intrinsics: torch.Tensor, # <--- 参数名变化，表示接收归一化内参
    image_height: int = 256,
    image_width: int = 256,
    background_color: tuple[float, float, float] = (0.0, 0.0, 0.0) # 黑色背景
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
    bg_color_tensor = torch.tensor(background_color, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    rendered_images = bg_color_tensor.expand(B, 3, image_height, image_width).clone()
    
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

DepthRenderingMode = Literal["depth", "disparity", "relative_disparity", "log"]
