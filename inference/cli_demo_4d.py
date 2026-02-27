"""
Running the Script:
To run the script, use the following command with appropriate arguments:

```bash
$ python -m inference.cli_demo_4d
```

You can change `pipe.enable_sequential_cpu_offload()` to `pipe.enable_model_cpu_offload()` to speed up inference, but this will use more GPU memory

Additional options are available to specify the model path, guidance scale, number of inference steps, video generation type, and output paths.

"""
from einops import rearrange
from collections import OrderedDict
import argparse
import logging
import math
from typing import Literal, Optional, Union
import os
import torch
import numpy as np
import random
import re
import cv2, imageio
from torchvision import transforms
from pathlib import Path
from diffusers import CogVideoXDPMScheduler, CogVideoXDDIMScheduler, AutoencoderKLCogVideoX, CogVideoXImageToVideoPipeline
from transformers import T5EncoderModel, T5Tokenizer
from diffusers.utils import export_to_video, load_image, load_video
from safetensors.torch import load_file, save_file
from charactershot import CogVideoXTransformer4DModel, CameraGuider, CogVideoXImageToVideo4DPipeline, CogVideoXImageToVideo2DPipeline
from inference.dwpose.dwpose_detector import dwpose_detector as dwprocessor
from inference.utils import get_video_pose, get_image_pose

logging.basicConfig(level=logging.INFO)

def generate_video(
    prompt: str,
    model_path: str,
    func_type: str,
    num_frames: int = 81,
    width: Optional[int] = None,
    height: Optional[int] = None,
    output_dir: str = "results",
    image_path: str = "",
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 42,
    fps: int = 16,
):
    """
    Generates a multi-view video based on the given image, pose squence and saves it to the specified path.

    Parameters:
    - prompt (str): The description of the video to be generated.
    - model_path (str): The path of the pre-trained model to be used.
    - func_type (str): 4dfinetune or 2dpretrain
    - output_path (str): The path where the generated video will be saved.
    - num_inference_steps (int): Number of steps for the inference process. More steps can result in better quality.
    - num_frames (int): Number of frames to generate. CogVideoX1.0 generates 49 frames for 6 seconds at 8 fps, while CogVideoX1.5 produces either 81 or 161 frames, corresponding to 5 seconds or 10 seconds at 16 fps.
    - width (int): The width of the generated video, applicable only for CogVideoX1.5-5B-I2V
    - height (int): The height of the generated video, applicable only for CogVideoX1.5-5B-I2V
    - guidance_scale (float): The scale for classifier-free guidance. Higher values can lead to better alignment with the prompt.
    - num_videos_per_prompt (int): Number of videos to generate per prompt.
    - dtype (torch.dtype): The data type for computation (default is torch.bfloat16).
    - seed (int): The seed for reproducibility.
    - fps (int): The frames per second for the generated video.
    """

    # 1.  Load the pre-trained CogVideoX pipeline with the specified precision (bfloat16).
    # add device_map="balanced" in the from_pretrained function and remove the enable_model_cpu_offload()
    # function to use Multi GPUs.
    IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    random.seed(222)
    
    if func_type == "4dfinetune":
        tokenizer = T5Tokenizer.from_pretrained(model_path, subfolder="tokenizer")
        text_encoder = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder", torch_dtype=dtype)
        vae = AutoencoderKLCogVideoX.from_pretrained(model_path, subfolder="vae", torch_dtype=dtype)
        transformer = CogVideoXTransformer4DModel.from_pretrained(model_path, subfolder="transformer", torch_dtype=dtype)
        scheduler = CogVideoXDDIMScheduler.from_pretrained(model_path, subfolder="scheduler")

        pipe = CogVideoXImageToVideo4DPipeline(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
        ).to("cuda", dtype=dtype)

        camera_guider = CameraGuider(out_dim=3072)
        camera_guider.load_state_dict(torch.load(os.path.join(model_path, f"camera_guider.pt"), map_location="cpu"), strict=False)
    elif func_type == "2dpretrain":
        pipe = CogVideoXImageToVideo2DPipeline.from_pretrained(model_path, torch_dtype=dtype).to("cuda")
    else:
        raise ValueError(f"Invalid type: {func_type}")
        
    pipe.scheduler = CogVideoXDPMScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )

    # 3. Enable CPU offload for the model.
    # turn off if you have multiple GPUs or enough GPU memory(such as H100) and it will cost less time in inference
    # and enable to("cuda")

    # pipe.enable_model_cpu_offload()
    # pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    
    trans = transforms.Compose(
        [transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)]
    )

    if func_type == "4dfinetune":
        image_dirs = sorted([os.path.join(image_path, file) for file in os.listdir(image_path) if os.path.isdir(os.path.join(image_path, file)) and "checkpoints" not in file])
        video_generate = []
        for image_dir in image_dirs:
            view_idxs = [[0, 1, 6, 11, 16], [0, 2, 7, 12, 17], [0, 3, 8, 13, 18], [0, 4, 9, 14, 19], [0, 5, 10, 15, 20]]
            images = sorted([os.path.join(image_dir, file) for file in os.listdir(image_dir) if os.path.splitext(file)[1] in IMAGE_EXTENSIONS])
            images = sorted(images, key=lambda x: int(re.search(r'view(\d+)', x).group(1)))
            view_images = [[images[i] for i in group] for group in view_idxs]

            save_path = os.path.join(output_dir, model_path.strip('/').split('/')[-1])
            os.makedirs(save_path, exist_ok=True)
            name = os.path.basename(image_dir)
            output_path = os.path.join(save_path, f"{name}_{height}_{width}_cfg_{guidance_scale}.mp4")

            pose_dir = image_dir.replace("images", "poses")
            poses = [os.path.join(pose_dir, pose_img) for pose_img in os.listdir(pose_dir) if os.path.splitext(pose_img)[1] in IMAGE_EXTENSIONS]
            poses = sorted(poses, key=lambda p: int(Path(p).stem))[0:num_frames]
            poses = [np.array(load_image(image=pose_img).resize((width,height))) for pose_img in poses]

            poses = np.stack(poses).transpose(0, 3, 1, 2)
            ref_image = load_image(image=images[0]).resize((width,height))
            ref_pose = get_image_pose(np.array(ref_image))
            ref_pose = torch.stack([trans(torch.from_numpy(ref_pose)) for pose in np.expand_dims(ref_pose, axis=0)])
            poses = torch.stack([trans(torch.from_numpy(pose)) for pose in poses])

            poses = poses.unsqueeze(0)
            ref_pose = ref_pose.unsqueeze(0)

            poses = poses.to("cuda", dtype=pipe.vae.dtype)
            poses = poses.permute(0, 2, 1, 3, 4).contiguous()
            ref_pose = ref_pose.to("cuda", dtype=pipe.vae.dtype)
            ref_pose = ref_pose.permute(0, 2, 1, 3, 4).contiguous()

            with torch.no_grad():
                p_latent = pipe.vae.encode(poses).latent_dist
                rp_latent = pipe.vae.encode(ref_pose).latent_dist

            p_latent = p_latent.sample() * pipe.vae.config.scaling_factor
            rp_latent = rp_latent.sample() * pipe.vae.config.scaling_factor
            p_latent = torch.cat([rp_latent, p_latent], dim=2)
            for images in view_images:
                load_images = []
                rads = []
                for img in images:
                    image = load_image(image=img).resize((width,height))
                    load_images.append(image)
                    match_view = re.search(r'view(\d+)', str(img))
                    if match_view:
                        view = int(match_view.group(1))
                    else:
                        view = None
                    rads.append(torch.load(os.path.join(model_path, f"camera_embs/{view}.pt")))
                rads = torch.stack(rads)
                device = next(camera_guider.parameters()).device
                dtype = next(camera_guider.parameters()).dtype
                rads = rads.to(device, dtype)

                with torch.no_grad():
                    camera_latents = camera_guider(rads)

                pipe.transformer.patch_embed.use_learned_positional_embeddings = False
                camera_latents = rearrange(camera_latents, 'b c h w -> b (h w) c').contiguous()

                result = pipe(
                    height=height,
                    width=width,
                    pose=p_latent.permute(0, 2, 1, 3, 4).unsqueeze(1).repeat(1, len(load_images), 1, 1, 1, 1),
                    prompt=prompt,
                    camera_latents=camera_latents.unsqueeze(0),
                    negative_prompt="low quality, blurry, distorted face, deformed hands, extra fingers, missing limbs, cropped, bad anatomy, bad proportions, watermark",
                    image=load_images,
                    num_videos_per_prompt=num_videos_per_prompt,  # Number of videos to generate per prompt
                    num_inference_steps=num_inference_steps,  # Number of inference steps
                    num_frames=num_frames,  # Number of frames to generate
                    use_dynamic_cfg=True,  # This id used for DPM scheduler, for DDIM scheduler, it should be False
                    guidance_scale=guidance_scale,
                    generator=torch.Generator().manual_seed(seed),  # Set the seed for reproducibility
                ).frames
                video_generate.append([item for sublist in result for item in sublist])
            final_results = []
            for i in [5, 4, 3, 2, 1]:
                final_results = final_results + video_generate[-i]
            export_to_video(final_results, output_path , fps=fps)  
                
    elif func_type == "2dpretrain":
        images = sorted([os.path.join(image_path, file) for file in os.listdir(image_path) if os.path.splitext(file)[1] in IMAGE_EXTENSIONS])
        for img in images:
            video_generate = []
            image = load_image(image=img).resize((width,height))
            placeholder = ".png" if ".png" in img else ".jpg"
            save_path = os.path.join(output_dir, model_path.strip('/').split('/')[-1])
            os.makedirs(save_path, exist_ok=True)
            name = os.path.basename(img)
            output_path = os.path.join(save_path, f"{name[:-4]}_{height}_{width}_cfg_{guidance_scale}.mp4")

            pose_dir = str(Path(img.replace("images", "poses")).with_suffix(""))
            poses = [os.path.join(pose_dir, pose_img) for pose_img in os.listdir(pose_dir) if os.path.splitext(pose_img)[1] in IMAGE_EXTENSIONS]
            poses = sorted(poses, key=lambda p: int(Path(p).stem))[0:num_frames]
            poses = [np.array(load_image(image=pose_img).resize((width,height))) for pose_img in poses]

            poses = np.stack(poses).transpose(0, 3, 1, 2)
            ref_image = load_image(image=images[0]).resize((width,height))
            ref_pose = get_image_pose(np.array(ref_image))
            ref_pose = torch.stack([trans(torch.from_numpy(ref_pose)) for pose in np.expand_dims(ref_pose, axis=0)])
            poses = torch.stack([trans(torch.from_numpy(pose)) for pose in poses])

            poses = poses.unsqueeze(0)
            ref_pose = ref_pose.unsqueeze(0)

            poses = poses.to("cuda", dtype=pipe.vae.dtype)
            poses = poses.permute(0, 2, 1, 3, 4).contiguous()
            ref_pose = ref_pose.to("cuda", dtype=pipe.vae.dtype)
            ref_pose = ref_pose.permute(0, 2, 1, 3, 4).contiguous()

            with torch.no_grad():
                p_latent = pipe.vae.encode(poses).latent_dist
            p_latent = p_latent.sample() * pipe.vae.config.scaling_factor

            ref_pose = ref_pose.unsqueeze(0)
            ref_pose = ref_pose.to("cuda", dtype=pipe.vae.dtype)
            ref_pose = ref_pose.permute(0, 2, 1, 3, 4).contiguous()
            with torch.no_grad():
                rp_latent = pipe.vae.encode(ref_pose).latent_dist
            rp_latent = rp_latent.sample() * pipe.vae.config.scaling_factor
            pipe.transformer.patch_embed.use_learned_positional_embeddings = False

            p_latent = torch.cat([rp_latent, p_latent], dim=2)
            result = pipe(
                height=height,
                width=width,
                pose=p_latent.permute(0, 2, 1, 3, 4),
                prompt=prompt,
                negative_prompt="low quality, blurry, distorted face, deformed hands, extra fingers, missing limbs, cropped, bad anatomy, bad proportions, watermark",
                image=image,
                # The path of the image, the resolution of video will be the same as the image for CogVideoX1.5-5B-I2V, otherwise it will be 720 * 480
                num_videos_per_prompt=num_videos_per_prompt,  # Number of videos to generate per prompt
                num_inference_steps=num_inference_steps,  # Number of inference steps
                num_frames=num_frames,  # Number of frames to generate
                use_dynamic_cfg=True,  # This id used for DPM scheduler, for DDIM scheduler, it should be False
                guidance_scale=guidance_scale,
                generator=torch.Generator().manual_seed(seed),  # Set the seed for reproducibility
            ).frames[0]
            video_generate.append(result)
            export_to_video(video_generate[0], output_path , fps=fps)
    else:
        raise ValueError(f"Invalid type: {func_type}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a video from a text prompt using CogVideoX"
    )
    parser.add_argument(
        "--prompt", type=str, default="smooth motion or in the wind", help="The description of the video to be generated"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="inference/examples/4d/images/",
        help="The path of the image to be used as the background of the video",
    )
    parser.add_argument(
        "--func_type",
        type=str,
        default="4dfinetune",
        help="type of model, including 4dfinetune and 2dpretrain",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="Gaojunyao/CharacterShot/",
        help="Path of the pre-trained model use, Gaojunyao/Character2D for 2dpretrain and Gaojunyao/CharacterShot for 4dfinetune.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="inference/results/", help="The path save generated video"
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=6, help="The scale for classifier-free guidance"
    )
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Inference steps")
    parser.add_argument(
        "--num_frames", type=int, default=25, help="Number of steps for the inference process"
    )
    parser.add_argument("--width", type=int, default=720, help="The width of the generated video")
    parser.add_argument(
        "--height", type=int, default=480, help="The height of the generated video"
    )
    parser.add_argument(
        "--fps", type=int, default=10, help="The frames per second for the generated video"
    )
    parser.add_argument(
        "--num_videos_per_prompt",
        type=int,
        default=1,
        help="Number of videos to generate per prompt",
    )
    parser.add_argument(
        "--generate_type", type=str, default="t2v", help="The type of video generation"
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", help="The data type for computation"
    )
    parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility")
    
    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    generate_video(
        prompt=args.prompt,
        model_path=args.model_path,
        output_dir=args.output_dir,
        num_frames=args.num_frames,
        width=args.width,
        func_type=args.func_type,
        height=args.height,
        image_path=args.image_path,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_videos_per_prompt=args.num_videos_per_prompt,
        dtype=dtype,
        seed=args.seed,
        fps=args.fps,
    )
