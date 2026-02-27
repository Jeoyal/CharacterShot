import os, oss2, io
import numpy as np
import subprocess
import torch
import random, time
import cv2
import decord

from diffusers import (
    CogVideoXImageToVideoPipeline,
)
from safetensors.torch import load_file, save_file
from typing import Tuple, List
from torchvision import transforms
from PIL import Image
from pathlib import Path
from copy import deepcopy
from dwpose.dwpose_detector import dwpose_detector as dwprocessor
from tqdm import tqdm
import re
from collections import defaultdict


def prepare_dataset(vid_file, video_root_path, pose_root_path, sample_rate, vae, resolution):
    save_dir = "data/i2v/CharacterShot/cache_multiview/"
    
    n_sample_frames = 25
    data_lines = open(vid_file, 'r').read().splitlines()
    data_lines = [data_line.split('.')[0] for data_line in data_lines]
    trans = transforms.Compose(
        [transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)]
    )
    def get_save_prefix_counts(save_dir):
        raw_pose_dir = Path(save_dir) / "video_latent"
        prefix_counts = defaultdict(int)

        pattern = re.compile(r"^(.*?)_start_idx_\d+\.safetensors$")

        for file in os.listdir(raw_pose_dir):
            match = pattern.match(file)
            if match:
                prefix = match.group(1)
                prefix_counts[prefix] += 1

        return dict(prefix_counts)
    count_results = get_save_prefix_counts(save_dir)
    video_path_lines = [os.path.join(video_root_path, data_line) for data_line in data_lines]
    pose_path_lines = [os.path.join(pose_root_path, data_line) for data_line in data_lines]
    IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    for video_path_line, pose_path_line in tqdm(zip(video_path_lines, pose_path_lines), total=len(video_path_lines), desc=f"Processing {video_root_path}"):
        try:
            if not os.path.exists(video_path_line) or not os.path.exists(pose_path_line):
                continue
            random_white_bg = random.random()
            use_white_bg = False
            if random_white_bg > 0.7:
                use_white_bg = True

            video_dirs, pose_frames = [os.path.join(video_path_line, file) for file in os.listdir(video_path_line) if "view" in file], [os.path.join(pose_path_line, file) for file in os.listdir(pose_path_line) if os.path.splitext(file)[1] in IMAGE_EXTENSIONS]

            video_dirs, pose_frames = sorted(video_dirs), sorted(pose_frames, key=lambda p: int(Path(p).stem))
            if len(video_dirs) < 5 or "view0" not in video_dirs[0]:
                continue

            seed = int(time.time() * 1000000) + os.getpid()
            random.seed(seed)
            local_views = [video_dirs[0]] + random.sample(video_dirs[1:], 4)
            video_frames = []
            lengths = [len(pose_frames)]

            for view in local_views:
                video_frame = [os.path.join(view, file) for file in os.listdir(view) if os.path.splitext(file)[1] in IMAGE_EXTENSIONS]
                video_frame = sorted(video_frame, key=lambda p: int(Path(p).stem))
                video_frames.append(video_frame)
                lengths.append(len(video_frame))

            video_length = min(lengths)

            clip_length = min(
                video_length, (n_sample_frames - 1) * sample_rate + 1
            )


            start_idx = random.randint(0, video_length - clip_length)
            batch_index = np.linspace(
                start_idx, start_idx + clip_length - 1, n_sample_frames, dtype=int
            ).tolist()

            videos, poses = [], []

            parts = video_path_line.strip(os.sep).split(os.sep)
            save_prefix = f"{parts[-1]}"
            
            if save_prefix in count_results:
                pre_count = count_results[save_prefix]
            else:
                pre_count = 0
            if random.random() <= (pre_count / 8.0):
                print(save_prefix, pre_count)
                continue
            pose_info = []
            raw_image_info = []

            for video_index, index in enumerate(batch_index):
                def process_image(image_path, resolution, use_white_bg, process_type="image", trans=trans):
                    image = Image.open(image_path).resize((resolution[1], resolution[1]))
                    image = image.convert("RGB")
                    image = torch.from_numpy(np.array(image))
                    if process_type == "image":
                        canvas = torch.ones(resolution[1], resolution[0], 3, dtype=image.dtype) * 255
                    else:
                        canvas = torch.zeros(resolution[1], resolution[0], 3, dtype=image.dtype)

                    offset_x = (resolution[0] - resolution[1]) // 2
                    canvas[:, offset_x:offset_x+resolution[1], :] = image
                    return trans(canvas).permute(2, 0, 1)
                
                pose = process_image(pose_frames[index], resolution, use_white_bg, process_type="pose")
                pose_info.append(pose_frames[index])

                view_videos = []
                view0_video = process_image(video_frames[0][index], resolution, use_white_bg, process_type="image")
                if video_index == 0:
                    raw_image_info.append(video_frames[0][index])
                view_videos.append(view0_video)
                for video_frame in video_frames[1:]:
                    view_video = process_image(video_frame[index], resolution, use_white_bg, process_type="image")
                    view_videos.append(view_video)
                    if video_index == 0:
                        raw_image_info.append(video_frame[index])


                videos.append(torch.stack(view_videos))
                poses.append(pose)

            videos, poses = torch.stack(videos), torch.stack(poses)

            videos = videos.permute(1, 0, 2, 3, 4) 
            raw_pose_dir = (
                Path(save_dir) / "raw_pose"
            )
            raw_pose_dir.mkdir(parents=True, exist_ok=True)
            raw_pose_path = raw_pose_dir / f"{save_prefix}_start_idx_{start_idx}.txt"
            with open(raw_pose_path, 'w') as f:
                for line in pose_info:
                    f.write(str(line) + '\n')
   

            raw_image_dir = (
                Path(save_dir) / "raw_image"
            )
            raw_image_dir.mkdir(parents=True, exist_ok=True)
            raw_image_path = raw_image_dir / f"{save_prefix}_start_idx_{start_idx}.txt"
            with open(raw_image_path, 'w') as f:
                for line in raw_image_info:
                    f.write(str(line) + '\n')

            def process_vae(images, vae):
                images = images.unsqueeze(0)
                images = images.to(vae.device, dtype=vae.dtype)
                images = images.permute(0, 2, 1, 3, 4).contiguous()
                with torch.no_grad():
                    latent = vae.encode(images).latent_dist
                latent = latent.sample() * vae.config.scaling_factor
                return latent
            p_latent = process_vae(poses, vae)
            v_latents = []
            for video in videos:
                v_latent = process_vae(video, vae)
                v_latents.append(v_latent)
            v_latents = torch.cat(v_latents)

            v_latent_dir = (
                Path(save_dir) / "video_latent"
            )
            v_latent_dir.mkdir(parents=True, exist_ok=True)
            v_latent_path = v_latent_dir / f"{save_prefix}_start_idx_{start_idx}.safetensors"
            save_file({"video_latent": v_latents.to("cpu").contiguous()}, v_latent_path)

            p_latent_dir = (
                Path(save_dir) / "pose_latent"
            )
            p_latent_dir.mkdir(parents=True, exist_ok=True)
            p_latent_path = p_latent_dir / f"{save_prefix}_start_idx_{start_idx}.safetensors"
            save_file({"pose_latent": p_latent[0].to("cpu").contiguous()}, p_latent_path)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    pipe = CogVideoXImageToVideoPipeline.from_pretrained("THUDM/CogVideoX-5b-I2V", torch_dtype=torch.bfloat16)
    vae = pipe.vae
    del pipe
    vae = vae.to("cuda")

    # video path
    vid_files = [
        'Gaojunyao/Character4D/train.txt',
    ]
    video_root_paths = [
        'Gaojunyao/Character4D/render_results/binded_motion/',
    ]
    pose_root_paths = [
        'Gaojunyao/Character4D/render_results/binded_pose/',
    ]
    sample_rates =[
        1,
    ]
    resolutions = [
        (720, 480),
    ]
    for vid_file, video_root, pose_root, sample_rate, resolution in zip(vid_files, video_root_paths, pose_root_paths, sample_rates, resolutions):
        prepare_dataset(vid_file, video_root, pose_root, sample_rate, vae, resolution)