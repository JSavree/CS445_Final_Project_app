import numpy as np
import os 
import argparse
from PIL import Image, ImageFilter
import cv2
import imageio.v2 as imageio
from tqdm import tqdm
import glob
import skimage
import json


os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def downsample_image(image, new_size):
    img = Image.fromarray(image)
    if len(image.shape) == 3:
        # img = img.filter(ImageFilter.GaussianBlur(radius=2))  # anti-aliasing for rgb image
        img = img.resize(new_size, resample=Image.BILINEAR)
    else:
        img = img.resize(new_size, Image.NEAREST)   # use nearest neighbour for depth map
    return np.array(img)


def generate_video_from_frames(frame_series, video_name, fps=30):
    # frame_series = [np.array(Image.open(frame_path)) for frame_path in frames_path]  # return (0~255 in uint8)
    # reshape the size of the frames to be divisible by 2 for video rendering
    h, w = frame_series[0].shape[:2]
    new_h = h if h % 2 == 0 else h - 1
    new_w = w if w % 2 == 0 else w - 1
    frame_series = [(skimage.transform.resize(frame, (new_h, new_w)) * 255.).astype(np.uint8) for frame in frame_series]
    # generate video with proper quality
    imageio.mimsave(video_name,
        frame_series,
        fps=fps,
        macro_block_size=1
    )
    # generate video with high quality
    # imageio.mimsave(video_name,
    #     frame_series,
    #     fps=fps, 
    #     codec='libx264', 
    #     macro_block_size=None, 
    #     quality=10,
    #     pixelformat='yuv444p'
    # )
    print("Video saved at: {}".format(video_name))


def load_rgb(path):
    if not os.path.exists(path):
        return None
    else:
        return np.array(Image.open(path).convert("RGBA"))


def load_depth(path):
    if not os.path.exists(path):
        return None
    else:
        return np.load(path)


def load_depth_exr(path):
    if not os.path.exists(path):
        return None
    else:
        d = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        return d[:, :, 0]


def depth_check(depth1, depth2, option='naive', d_tol=0.1):
    '''
    Determine whether depth1 is closer than depth2 with a tolerance d_tol
    '''
    if option == 'naive':
        return depth1 <= depth2
    elif option == 'tolerance':
        return np.abs(depth1 - depth2) < d_tol
    elif option == 'naive_or_tolerance':
        return np.logical_or(depth1 <= depth2, np.abs(depth1 - depth2) < d_tol)
    else:
        raise ValueError('Invalid option: {}'.format(option))
    

root_dir = '/home/haoyuyh3/Documents/maxhsu/CS445_Final_Project_app/data/custom_camera_path/transforms_001'
blend_results_dir = '/home/haoyuyh3/Documents/maxhsu/CS445_Final_Project_app/output'
out_img_dir = os.path.join(blend_results_dir, 'frames')
os.makedirs(out_img_dir, exist_ok=True)

bg_rgb = sorted(glob.glob(os.path.join(root_dir, 'images', '*.png')))
bg_depth = sorted(glob.glob(os.path.join(root_dir, 'depth', '*.npy')))

rgb_all_img_path = glob.glob(os.path.join(blend_results_dir, 'rgb_all', '*.png'))
n_frame = len(rgb_all_img_path)

frames = []
# TODO: this part is for N frames, could reduce to single frame for testing
for i in tqdm(range(n_frame)):

    # Get the paths for each frame
    obj_rgb_path = os.path.join(blend_results_dir, 'rgb_obj', '{:0>3d}.png'.format(i+1))
    obj_depth_path = os.path.join(blend_results_dir, 'depth_obj', '{:0>3d}'.format(i+1), 'Image{:0>4d}.exr'.format(i+1))
    shadow_rgb_path = os.path.join(blend_results_dir, 'rgb_shadow', '{:0>3d}.png'.format(i+1))
    shadow_depth_path = os.path.join(blend_results_dir, 'depth_shadow', '{:0>3d}'.format(i+1), 'Image{:0>4d}.exr'.format(i+1))
    all_rgb_path = os.path.join(blend_results_dir, 'rgb_all', '{:0>3d}.png'.format(i+1))
    all_depth_path = os.path.join(blend_results_dir, 'depth_all', '{:0>3d}'.format(i+1), 'Image{:0>4d}.exr'.format(i+1))

    bg_c = load_rgb(bg_rgb[i])                    # bg_c: background image
    bg_d = load_depth(bg_depth[i])                # bg_d: background depth map
    o_c = load_rgb(obj_rgb_path)                  # o_c: object image from Blender
    o_d = load_depth_exr(obj_depth_path)          # o_d: object depth map from Blender
    s_c = load_rgb(shadow_rgb_path)               # s_c: shadow catcher image from Blender
    s_d = load_depth_exr(shadow_depth_path)       # s_d: shadow catcher depth map from Blender
    o_s_c = load_rgb(all_rgb_path)                # o_s_c: object with shadow catcher image from Blender
    o_s_d = load_depth_exr(all_depth_path)        # o_s_d: object with shadow catcher depth map from Blender

    # anti-aliasing
    new_size = (bg_c.shape[1], bg_c.shape[0])
    o_c = downsample_image(o_c, new_size)
    o_d = downsample_image(o_d, new_size)
    s_c = downsample_image(s_c, new_size)
    s_d = downsample_image(s_d, new_size)
    o_s_c = downsample_image(o_s_c, new_size)
    o_s_d = downsample_image(o_s_d, new_size)

    bg_c = bg_c.astype(np.float32)
    o_c = o_c.astype(np.float32)
    s_c = s_c.astype(np.float32)
    o_s_c = o_s_c.astype(np.float32)

    frame = bg_c.copy()

    ############################################################
    ##### Step 1: blend shadow into background image #####
    ############################################################
    obj_alpha = o_c[..., 3] / 255.
    depth_mask = depth_check(o_d, s_d, option='naive', d_tol=0.1)

    obj_mask = obj_alpha > 0.0
    mask = np.logical_and(obj_mask, depth_mask)
    obj_alpha[~mask] = 0.0
    non_object_alpha = 1. - obj_alpha

    fg_alpha = o_s_c[..., 3] / 255.
    shadow_catcher_alpha = non_object_alpha * fg_alpha
    shadow_catcher_mask = shadow_catcher_alpha > 0.0

    color_diff = np.ones_like(o_c)
    color_diff[shadow_catcher_mask, 0:3] = o_s_c[shadow_catcher_mask, :3] / (s_c[shadow_catcher_mask, :3] + 1e-6)
    color_diff = np.clip(color_diff, 0, 1)
    shadow_mask = np.logical_not(np.all(np.abs(color_diff - 1) < 0.01, axis=-1))
    mask = shadow_mask

    frame[mask] = frame[mask] * color_diff[mask] * shadow_catcher_alpha[mask, None] + frame[mask] * (1 - shadow_catcher_alpha[mask, None])

    ############################################################
    ##### Step 2: blend object and 3DGS object into background image #####
    ############################################################
    frame_tmp = frame.copy()
    mask = np.logical_and(obj_mask, depth_mask)
    frame[:, :, :3][mask] = o_c[:, :, :3][mask] * obj_alpha[mask, None] + frame_tmp[:, :, :3][mask] * (1 - obj_alpha[mask, None])

    # convert frame to uint8
    frame = np.clip(frame, 0, 255)
    frame = frame.astype(np.uint8)

    frames.append(frame)
    path = os.path.join(out_img_dir, '{:0>4d}.png'.format(i))
    Image.fromarray(frame).save(path)

generate_video_from_frames(np.array(frames), os.path.join(blend_results_dir, 'blended.mp4'), fps=15)
