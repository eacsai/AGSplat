import torch
from jaxtyping import Float, Shaped
from torch import Tensor, Union

from ..model.decoder.cuda_splatting import render_cuda_orthographic
from ..model.types import Gaussians
from ..visualization.annotation import add_label
from ..visualization.drawing.cameras import draw_cameras
from .drawing.cameras import compute_equal_aabb_with_margin


def pad(images: list[Shaped[Tensor, "..."]]) -> list[Shaped[Tensor, "..."]]:
    shapes = torch.stack([torch.tensor(x.shape) for x in images])
    padded_shape = shapes.max(dim=0)[0]
    results = [
        torch.ones(padded_shape.tolist(), dtype=x.dtype, device=x.device)
        for x in images
    ]
    for image, result in zip(images, results):
        slices = [slice(0, x) for x in image.shape]
        result[slices] = image[slices]
    return results


def render_projections(
    gaussians: Gaussians,
    resolution: tuple[int, int],
    margin: float = 0.1,
    heading: Union[Tensor, None] = None,
    look_axis = 1,
    rot_range = 10.0,
    width = 101.0 / 2,
    height = 101.0 / 2,
) -> Float[Tensor, "batch 3 3 height width"]:
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
            width,
            height,
            near,
            far,
            resolution,
            torch.zeros((1, 3), dtype=torch.float32, device=device),
            gaussians.means[b:b+1],
            gaussians.covariances[b:b+1],
            gaussians.color_harmonics[b:b+1] if hasattr(gaussians, 'color_harmonics') else None,
            gaussians.opacities[b:b+1],
            gaussians.features[b:b+1],
            gaussians.confidence[b:b+1],
            gaussians.rgbs[b:b+1] if hasattr(gaussians, 'rgbs') else None,
            fov_degrees=0.1,
            use_sh=True,
        )
        color = render_out.color
        feature = render_out.feature
        confidence = render_out.confidence
        color_out.append(color)
        feature_out.append(feature)
        confidence_out.append(confidence)
    return torch.cat(color_out, dim=0), torch.cat(feature_out, dim=0), torch.cat(confidence_out, dim=0)


def render_cameras(batch: dict, resolution: int) -> Float[Tensor, "3 3 height width"]:
    # Define colors for context and target views.
    num_context_views = batch["context"]["extrinsics"].shape[1]
    num_target_views = batch["target"]["extrinsics"].shape[1]
    color = torch.ones(
        (num_target_views + num_context_views, 3),
        dtype=torch.float32,
        device=batch["target"]["extrinsics"].device,
    )
    color[num_context_views:, 1:] = 0

    return draw_cameras(
        resolution,
        torch.cat(
            (batch["context"]["extrinsics"][0], batch["target"]["extrinsics"][0])
        ),
        torch.cat(
            (batch["context"]["intrinsics"][0], batch["target"]["intrinsics"][0])
        ),
        color,
        torch.cat((batch["context"]["near"][0], batch["target"]["near"][0])),
        torch.cat((batch["context"]["far"][0], batch["target"]["far"][0])),
    )
