import torch        
from .decoder.cuda_splatting import render_cuda_orthographic, render_bevs, forward_project, project_point_clouds
from einops import pack, rearrange, repeat
import torch.nn.functional as F
from PIL import Image, ImageDraw

import numpy as np
from sklearn.decomposition import PCA
import os
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
to_pil_image = transforms.ToPILImage()

def vis_bev(batch, gaussians, output):
    b,v,c,h,w = batch["context"]["image"].shape
    point_color = (rearrange(batch["context"]["image"], 'b v c h w -> b (v h w) c') + 1) / 2
    point_clouds = gaussians.means
    grd2sat_direct_color = forward_project(
        point_color,
        point_clouds,
        meter_per_pixel=0.2
    )

    rgb_bev = grd2sat_direct_color[0]
    test_img = to_pil_image(rgb_bev.clamp(min=0,max=1))
    # 获取图像的宽度和高度
    width, height = test_img.size

    # 计算中心点的坐标
    center_x = width // 2
    center_y = height // 2

    # 创建一个可以在图像上绘图的对象
    draw = ImageDraw.Draw(test_img)

    # 定义红点的半径（可以根据需要调整）
    radius = 5

    # 绘制一个红色圆形作为中心点
    # draw.ellipse((x1, y1, x2, y2), fill=color, outline=color)
    # x1, y1 是左上角坐标，x2, y2 是右下角坐标
    draw.ellipse((center_x - radius, center_y - radius,
                center_x + radius, center_y + radius),
                fill=(255, 0, 0), outline=(255, 0, 0)) # 填充和边框都设为红色
    # 保存图片
    test_img.save('direct_bev.png')

    project_img = project_point_clouds(
        point_clouds,
        point_color,
        batch["target"]["intrinsics"]
    )

    test_img = to_pil_image(project_img[0].clamp(min=0,max=1))
    test_img.save('direct_project.png')
    # write_ply(gaussians.means[0].cpu().detach().numpy(), point_color[0].cpu().detach().numpy())

    rgb_input = (batch['context']["image"][0,0] + 1) / 2
    test_img = to_pil_image(rgb_input.clamp(min=0,max=1))
    test_img.save('input.png')

    rgb_output = output.color[0,0]
    test_img = to_pil_image(rgb_output.clamp(min=0,max=1))
    test_img.save('output.png')

    sat_img = F.interpolate(batch['sat']['sat'], size=(512, 512), mode='bilinear', align_corners=False)
    test_img = to_pil_image(sat_img[0])

    # 获取图像的宽度和高度
    width, height = test_img.size

    # 计算中心点的坐标
    center_x = width // 2
    center_y = height // 2

    # 创建一个可以在图像上绘图的对象
    draw = ImageDraw.Draw(test_img)

    # 定义红点的半径（可以根据需要调整）
    radius = 5

    # 绘制一个红色圆形作为中心点
    # draw.ellipse((x1, y1, x2, y2), fill=color, outline=color)
    # x1, y1 是左上角坐标，x2, y2 是右下角坐标
    draw.ellipse((center_x - radius, center_y - radius,
                center_x + radius, center_y + radius),
                fill=(255, 0, 0), outline=(255, 0, 0)) # 填充和边框都设为红色
    test_img.save('sat.png')

def single_features_to_RGB_colormap(sat_features, idx=0, img_name='test_img_cmap_zeros_black.png', cmap_name='viridis', zero_threshold=1e-6):
    """
    Visualizes features using the first PCA component and a colormap.
    Pixels where original features are all close to zero are set to black.

    Args:
        sat_features (torch.Tensor or np.ndarray): Feature tensor of shape [B, C, H, W].
        idx (int): Batch index to visualize.
        img_name (str): Output image file name.
        cmap_name (str): Name of the matplotlib colormap to use.
        zero_threshold (float): Threshold below which feature absolute values are considered zero.
    """
    # Helper functions (assuming they exist or define them)
    def reshape_normalize(features):
        """Reshapes [B, C, H, W] to [B*H*W, C] and normalizes features."""
        B, C, H, W = features.shape
        features_reshaped = features.transpose(0, 2, 3, 1).reshape(-1, C)
        # Example normalization (adapt if needed)
        mean = np.mean(features_reshaped, axis=0, keepdims=True)
        std = np.std(features_reshaped, axis=0, keepdims=True)
        std[std == 0] = 1e-6
        normalized = (features_reshaped - mean) / std
        return normalized

    # --- Ensure NumPy array on CPU ---
    if hasattr(sat_features, 'data') and hasattr(sat_features, 'cpu'):
        sat_feat_batch = sat_features.data.cpu().numpy()
    elif isinstance(sat_features, np.ndarray):
        sat_feat_batch = sat_features
    else:
        raise TypeError("Input must be a PyTorch tensor or NumPy array")

    sat_feat = sat_feat_batch[idx:idx+1, :, :, :] # Shape [1, C, H, W]
    B, C, H, W = sat_feat.shape
    assert B == 1

    # --- 0. Identify "Zero" Feature Locations BEFORE Normalization/PCA ---
    # Find pixels where the sum of absolute feature values is below the threshold
    # Reshape to [H, W, C] for easier spatial masking
    sat_feat_spatial = sat_feat[0].transpose(1, 2, 0) # Shape [H, W, C]
    # Check if *all* channels are close to zero for a pixel
    is_zero_mask = np.all(np.abs(sat_feat_spatial) < zero_threshold, axis=-1) # Shape [H, W]
    # Alternatively, check if the norm is close to zero:
    # feature_norm = np.linalg.norm(sat_feat_spatial, axis=-1)
    # is_zero_mask = feature_norm < zero_threshold * np.sqrt(C) # Adjust threshold based on norm


    # --- 1. Prepare data for PCA (Using only non-zero pixels might be better) ---
    # Option A: Use all data (simpler)
    flatten_slice = reshape_normalize(sat_feat)
    # Option B: Use only non-zero data for fitting (potentially more robust PCA)
    # sat_feat_reshaped_orig = sat_feat.transpose(0, 2, 3, 1).reshape(-1, C)
    # non_zero_features = sat_feat_reshaped_orig[~is_zero_mask.reshape(-1)]
    # if non_zero_features.shape[0] < 2: # Need at least 2 samples for PCA
    #     print("Warning: Too few non-zero features for PCA. Saving black image.")
    #     img = Image.fromarray(np.zeros((H,W,3), dtype=np.uint8))
    #     img.save(img_name)
    #     return
    # flatten_slice_nonzero_normalized = reshape_normalize(non_zero_features[np.newaxis,:,:,:]) # Requires adapting reshape_normalize

    # --- 2. PCA (only need 1 component) ---
    pca = PCA(n_components=1)
    # pca.fit(flatten_slice_nonzero_normalized) # Fit on non-zero data if using Option B
    pca.fit(flatten_slice) # Fit on all data (Option A)

    # Transform *all* original slice data (even zeros, though their transform might be less meaningful)
    sat_feat_reshaped = sat_feat.transpose(0, 2, 3, 1).reshape(-1, C)
    pca_transformed_1d = pca.transform(sat_feat_reshaped) # Shape [H*W, 1]

    # --- 3. Normalize the first component to [0, 1] ---
    pc1 = pca_transformed_1d.reshape(H, W) # Reshape to [H, W] first
    # Normalize using only the non-zero pixels' range for better contrast
    pc1_non_zero = pc1[~is_zero_mask]
    if pc1_non_zero.size == 0: # Handle case where all pixels were zero
         normalized_pc1_image = np.zeros((H,W)) + 0.5
    else:
        pc1_min = np.min(pc1_non_zero)
        pc1_max = np.max(pc1_non_zero)
        if pc1_max == pc1_min:
            # If all non-zero pixels map to the same PC1 value, assign a mid-value
             normalized_pc1_image = np.zeros((H,W)) # Start with zeros
             normalized_pc1_image[~is_zero_mask] = 0.5 # Set non-zero pixels to 0.5
        else:
            # Normalize PC1 values based on the range of non-zero pixels
            normalized_pc1 = (pc1 - pc1_min) / (pc1_max - pc1_min)
            # Clamp values potentially outside [0,1] due to extrapolation on zero pixels
            normalized_pc1_image = np.clip(normalized_pc1, 0.0, 1.0)
            # Ensure originally zero pixels don't affect normalization scaling visibly
            normalized_pc1_image[is_zero_mask] = 0.0 # Or assign a value reflecting "background" like 0 or 0.5


    # --- 4. Apply Colormap ---
    try:
        cmap = plt.get_cmap(cmap_name)
        # Apply colormap - cmap expects values in [0, 1]
        colored_image = cmap(normalized_pc1_image)[:, :, :3] # Shape [H, W, 3], range [0, 1]
    except ValueError:
        print(f"Warning: Colormap '{cmap_name}' not found. Using 'viridis'.")
        cmap = plt.get_cmap('viridis')
        colored_image = cmap(normalized_pc1_image)[:, :, :3]

    # --- 5. Apply Zero Mask ---
    # Where the original features were zero, set the color to black
    # Need to broadcast is_zero_mask [H, W] to [H, W, 3]
    colored_image[is_zero_mask] = 0.0 # Set RGB to (0, 0, 0)

    # --- 6. Convert to uint8 and Save ---
    final_image_uint8 = (colored_image * 255).astype(np.uint8)
    img = Image.fromarray(final_image_uint8)
    # img = img.resize((512, 512)) # Optional resize
    img.save(img_name)
    print(f"Saved colormapped feature visualization (zeros as black) to {img_name}")