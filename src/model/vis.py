import torch        
from .decoder.cuda_splatting import render_cuda_orthographic, render_bevs, forward_project, project_point_clouds
from einops import pack, rearrange, repeat

import torchvision.transforms as transforms
to_pil_image = transforms.ToPILImage()

def vis_bev(batch, gaussians, output):
    b,v,c,h,w = batch["context"]["image"].shape
    heading = torch.zeros([b, 1], dtype=torch.float32, requires_grad=True, device=batch["target"]["extrinsics"].device)
    grd2sat_gaussian_color = render_bevs( 
        gaussians, 
        (128,128),
        heading=heading, 
        width=100.0, 
        height=100.0
    )

    rgb_bev = grd2sat_gaussian_color[0]
    test_img = to_pil_image(rgb_bev)
    test_img.save('splat_bev.png')

    point_color = (rearrange(batch["context"]["image"], 'b v c h w -> b (v h w) c') + 1) / 2
    point_clouds = gaussians.means
    grd2sat_direct_color = forward_project(
        point_color,
        point_clouds,
        meter_per_pixel=0.2
    )

    rgb_bev = grd2sat_direct_color[0]
    test_img = to_pil_image(rgb_bev.clamp(min=0,max=1))
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