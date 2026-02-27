import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import os
import re
import random
import torch
import numpy as np
from accelerate.logging import get_logger
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset
from torchvision import transforms
from typing_extensions import override
from PIL import Image
from finetune.constants import LOG_LEVEL, LOG_NAME
from .utils import (
    load_images,
    load_images_from_videos,
    load_prompts,
    load_videos,
    preprocess_image_with_resize,
    preprocess_video_with_buckets,
    preprocess_video_with_resize,
)


if TYPE_CHECKING:
    from finetune.trainer import Trainer

# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip

decord.bridge.set_bridge("torch")

logger = get_logger(LOG_NAME, LOG_LEVEL)

class CharacterDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        video_column: str,
        image_column: str,
        pose_column: str,
        func_type: str,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        
        self.data_root = Path(data_root)
        video_lines = open(data_root / video_column, 'r').read().splitlines()
        pose_lines = open(data_root / pose_column, 'r').read().splitlines()
        image_lines = open(data_root / image_column, 'r').read().splitlines()
        self.func_type = func_type

        self.encoded_videos, self.poses, self.images = [], [], []
        for pose_line, video_line, image_line in zip(pose_lines, video_lines, image_lines):
            self.encoded_videos.append(data_root / video_line)
            self.poses.append(data_root / pose_line)
            self.images.append(data_root / image_line)
            
        self.trans = transforms.Compose(
            [transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)]
        )
    
    def __len__(self) -> int:
        return len(self.encoded_videos)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        if isinstance(index, dict) or isinstance(index, list):
            # Here, index is actually a list of data objects that we need to return.
            # The BucketSampler should ideally return indices. But, in the sampler, we'd like
            # to have information about num_frames, height and width. Since this is not stored
            # as metadata, we need to read the video to get this information. You could read this
            # information without loading the full video in memory, but we do it anyway. In order
            # to not load the video twice (once to get the metadata, and once to return the loaded video
            # based on sampled indices), we cache it in the BucketSampler. When the sampler is
            # to yield, we yield the cache data instead of indices. So, this special check ensures
            # that data is not loaded a second time. PRs are welcome for improvements.
            return index
        cache_dir = self.data_root
        prompt_embeddings_dir = cache_dir / "prompt_embeddings"
        prompt_embedding_path = prompt_embeddings_dir / ("prompts.safetensors")

        prompt_embedding = load_file(prompt_embedding_path)["prompt_embedding"]
        logger.debug(
            f"Loaded prompt embedding from {prompt_embedding_path}",
            main_process_only=False,
        )

        encoded_video_path = self.encoded_videos[index]
        pose_path = self.poses[index]
        image_path = self.images[index]

        encode_videos = load_file(encoded_video_path)["video_latent"]
        IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
        
        if self.func_type == "2dpretrain":
            poses = load_file(pose_path)["pose_latent"]
            raw_image_paths = open(image_path, 'r').read().splitlines()
            ref_idx = random.randint(0, len(raw_image_paths) - 1)
            raw_image_path = raw_image_paths[ref_idx]
            
            pose_dir_path = raw_image_path.replace("images", "poses")
            
            pose_frames =  [os.path.join(pose_dir_path, file) for file in os.listdir(pose_dir_path) if os.path.splitext(file)[1] in IMAGE_EXTENSIONS]
            pose_frames = sorted(pose_frames, key=lambda p: int(Path(p).stem))
            
            pose_final_path = pose_frames[ref_idx]

            resolution = (encode_videos.shape[3]*8, encode_videos.shape[2]*8)
                
            raw_image = Image.open(raw_image_path).resize(resolution).convert("RGB")
            raw_pose = Image.open(pose_final_path).resize(resolution).convert("RGB")

            raw_pose = self.trans(torch.from_numpy(np.array(raw_pose)).permute(2, 0, 1))
            raw_image = self.trans(torch.from_numpy(np.array(raw_image)).permute(2, 0, 1))

            return {
                "image": torch.stack([raw_image, raw_pose]),
                "prompt_embedding": prompt_embedding,
                "encoded_video": encode_videos,
                "pose": poses,
                "video_metadata": {
                    "num_frames": encode_videos.shape[1],
                    "height": encode_videos.shape[2],
                    "width": encode_videos.shape[3],
                },
            }
    
        elif self.func_type == "4dfinetune":
            poses = load_file(pose_path)["pose_latent"]
            raw_image_paths = open(image_path, 'r').read().splitlines()

            video_lengths = []
            video_frames = []
    
            for raw_image_path in raw_image_paths:
                video_frame = [os.path.join(os.path.dirname(raw_image_path), file) for file in os.listdir(os.path.dirname(raw_image_path)) if os.path.splitext(file)[1] in IMAGE_EXTENSIONS]
                video_frame = sorted(video_frame, key=lambda p: int(Path(p).stem))
                video_frames.append(video_frame)
                video_lengths.append(len(video_frame))
            
            video_length = min(video_lengths)
            
            ref_idx = random.randint(0, video_length - 1)
            raw_image_paths = []
            for video_frame in video_frames:
                raw_image_paths.append(video_frame[ref_idx])
                
            pose_final_path = re.sub(r'view\d+/', '', raw_image_paths[0].replace("binded_motion", "binded_pose"))

            resolution = (encode_videos.shape[4]*8, encode_videos.shape[3]*8)
            
            def process_image(image_path, resolution, use_white_bg):
                image = Image.open(image_path).resize((resolution[1], resolution[1]))
                image = image.convert("RGB")
                image = torch.from_numpy(np.array(image))
                if use_white_bg:
                    canvas = torch.ones(resolution[1], resolution[0], 3, dtype=image.dtype) * 255
                else:
                    canvas = torch.zeros(resolution[1], resolution[0], 3, dtype=image.dtype)

                offset_x = (resolution[0] - resolution[1]) // 2
                canvas[:, offset_x:offset_x+resolution[1], :] = image
                return self.trans(canvas).permute(2, 0, 1)
            
            sample_index = random.randint(0, encode_videos.shape[0] - 1)
            raw_images = process_image(raw_image_paths[sample_index], resolution, True)
            pose_image = process_image(pose_final_path, resolution, False)
            
            cameras = []
            raw_images = []
            for raw_image_path in raw_image_paths:
                raw_images.append(process_image(raw_image_path, resolution, True))
                match_view = re.search(r'view(\d+)', str(raw_image_path))
                if match_view:
                    view = int(match_view.group(1))
                else:
                    view = None
                cameras.append(torch.load(f"Gaojunyao/CharacterShot/camera_embs/{view}.pt"))
            raw_images = torch.stack(raw_images)
            pose_image = process_image(pose_final_path, resolution, False)
            cameras = torch.stack(cameras)
            
            return {
                "image": torch.cat([raw_images, pose_image.unsqueeze(0)]),
                "prompt_embedding": prompt_embedding,
                "encoded_video": encode_videos,
                "pose": poses,
                "camera": cameras,
                "video_metadata": {
                    "num_frames": encode_videos.shape[2],
                    "height": encode_videos.shape[3],
                    "width": encode_videos.shape[4],
                },
            }


class BaseI2VDataset(Dataset):
    """
    Base dataset class for Image-to-Video (I2V) training.

    This dataset loads prompts, videos and corresponding conditioning images for I2V training.

    Args:
        data_root (str): Root directory containing the dataset files
        caption_column (str): Path to file containing text prompts/captions
        video_column (str): Path to file containing video paths
        image_column (str): Path to file containing image paths
        device (torch.device): Device to load the data on
        encode_video_fn (Callable[[torch.Tensor], torch.Tensor], optional): Function to encode videos
    """

    def __init__(
        self,
        data_root: str,
        caption_column: str,
        video_column: str,
        image_column: str | None,
        device: torch.device,
        trainer: "Trainer" = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        data_root = Path(data_root)
        self.prompts = load_prompts(data_root / caption_column)
        self.videos = load_videos(data_root / video_column)
        if image_column is not None:
            self.images = load_images(data_root / image_column)
        else:
            self.images = load_images_from_videos(self.videos)
        self.trainer = trainer

        self.device = device
        self.encode_video = trainer.encode_video
        self.encode_text = trainer.encode_text
        
        self.max_num_frames = max_num_frames
        self.height = height
        self.width = width

        self.__frame_transforms = transforms.Compose(
            [transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)]
        )
        self.__image_transforms = self.__frame_transforms
        # Check if number of prompts matches number of videos and images
        if not (len(self.videos) == len(self.prompts) == len(self.images)):
            raise ValueError(
                f"Expected length of prompts, videos and images to be the same but found {len(self.prompts)=}, {len(self.videos)=} and {len(self.images)=}. Please ensure that the number of caption prompts, videos and images match in your dataset."
            )

        # Check if all video files exist
        if any(not path.is_file() for path in self.videos):
            raise ValueError(
                f"Some video files were not found. Please ensure that all video files exist in the dataset directory. Missing file: {next(path for path in self.videos if not path.is_file())}"
            )

        # Check if all image files exist
        if any(not path.is_file() for path in self.images):
            raise ValueError(
                f"Some image files were not found. Please ensure that all image files exist in the dataset directory. Missing file: {next(path for path in self.images if not path.is_file())}"
            )

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if isinstance(index, list):
            # Here, index is actually a list of data objects that we need to return.
            # The BucketSampler should ideally return indices. But, in the sampler, we'd like
            # to have information about num_frames, height and width. Since this is not stored
            # as metadata, we need to read the video to get this information. You could read this
            # information without loading the full video in memory, but we do it anyway. In order
            # to not load the video twice (once to get the metadata, and once to return the loaded video
            # based on sampled indices), we cache it in the BucketSampler. When the sampler is
            # to yield, we yield the cache data instead of indices. So, this special check ensures
            # that data is not loaded a second time. PRs are welcome for improvements.
            return index

        prompt = self.prompts[index]
        video = self.videos[index]
        image = self.images[index]
        train_resolution_str = "x".join(str(x) for x in self.trainer.args.train_resolution)

        cache_dir = self.trainer.args.data_root / "cache"
        video_latent_dir = (
            cache_dir / "video_latent" / self.trainer.args.model_name / train_resolution_str
        )
        prompt_embeddings_dir = cache_dir / "prompt_embeddings"
        video_latent_dir.mkdir(parents=True, exist_ok=True)
        prompt_embeddings_dir.mkdir(parents=True, exist_ok=True)

        prompt_hash = str(hashlib.sha256(prompt.encode()).hexdigest())
        prompt_embedding_path = prompt_embeddings_dir / (prompt_hash + ".safetensors")
        encoded_video_path = video_latent_dir / (video.stem + ".safetensors")

        if prompt_embedding_path.exists():
            prompt_embedding = load_file(prompt_embedding_path)["prompt_embedding"]
            logger.debug(
                f"process {self.trainer.accelerator.process_index}: Loaded prompt embedding from {prompt_embedding_path}",
                main_process_only=False,
            )
        else:
            prompt_embedding = self.encode_text(prompt)
            prompt_embedding = prompt_embedding.to("cpu")
            # [1, seq_len, hidden_size] -> [seq_len, hidden_size]
            prompt_embedding = prompt_embedding[0]
            save_file({"prompt_embedding": prompt_embedding}, prompt_embedding_path)
            logger.info(
                f"Saved prompt embedding to {prompt_embedding_path}", main_process_only=False
            )

        if encoded_video_path.exists():
            encoded_video = load_file(encoded_video_path)["encoded_video"]
            logger.debug(f"Loaded encoded video from {encoded_video_path}", main_process_only=False)
            # shape of image: [C, H, W]
            _, image = self.preprocess(None, self.images[index])
            image = self.image_transform(image)
        else:
            frames, image = self.preprocess(video, image)
            frames = frames.to(self.device)
            image = image.to(self.device)
            image = self.image_transform(image)
            # Current shape of frames: [F, C, H, W]
            frames = self.video_transform(frames)

            # Convert to [B, C, F, H, W]
            frames = frames.unsqueeze(0)
            frames = frames.permute(0, 2, 1, 3, 4).contiguous()
            encoded_video = self.encode_video(frames)

            # [1, C, F, H, W] -> [C, F, H, W]
            encoded_video = encoded_video[0]
            encoded_video = encoded_video.to("cpu")
            image = image.to("cpu")
            save_file({"encoded_video": encoded_video}, encoded_video_path)
            logger.info(f"Saved encoded video to {encoded_video_path}", main_process_only=False)

        # shape of encoded_video: [C, F, H, W]
        # shape of image: [C, H, W]
        return {
            "image": image,
            "prompt_embedding": prompt_embedding,
            "encoded_video": encoded_video,
            "video_metadata": {
                "num_frames": encoded_video.shape[1],
                "height": encoded_video.shape[2],
                "width": encoded_video.shape[3],
            },
        }
    @override
    def preprocess(
        self, video_path: Path | None, image_path: Path | None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if video_path is not None:
            video = preprocess_video_with_resize(
                video_path, self.max_num_frames, self.height, self.width
            )
        else:
            video = None
        if image_path is not None:
            image = preprocess_image_with_resize(image_path, self.height, self.width)
        else:
            image = None
        return video, image

    @override
    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.__frame_transforms(f) for f in frames], dim=0)

    @override
    def image_transform(self, image: torch.Tensor) -> torch.Tensor:
        return self.__image_transforms(image)



class I2VDatasetWithResize(BaseI2VDataset):
    """
    A dataset class for image-to-video generation that resizes inputs to fixed dimensions.

    This class preprocesses videos and images by resizing them to specified dimensions:
    - Videos are resized to max_num_frames x height x width
    - Images are resized to height x width

    Args:
        max_num_frames (int): Maximum number of frames to extract from videos
        height (int): Target height for resizing videos and images
        width (int): Target width for resizing videos and images
    """

    def __init__(self, max_num_frames: int, height: int, width: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.max_num_frames = max_num_frames
        self.height = height
        self.width = width

        self.__frame_transforms = transforms.Compose(
            [transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)]
        )
        self.__image_transforms = self.__frame_transforms

    @override
    def preprocess(
        self, video_path: Path | None, image_path: Path | None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if video_path is not None:
            video = preprocess_video_with_resize(
                video_path, self.max_num_frames, self.height, self.width
            )
        else:
            video = None
        if image_path is not None:
            image = preprocess_image_with_resize(image_path, self.height, self.width)
        else:
            image = None
        return video, image

    @override
    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.__frame_transforms(f) for f in frames], dim=0)

    @override
    def image_transform(self, image: torch.Tensor) -> torch.Tensor:
        return self.__image_transforms(image)


class I2VDatasetWithBuckets(BaseI2VDataset):
    def __init__(
        self,
        video_resolution_buckets: List[Tuple[int, int, int]],
        vae_temporal_compression_ratio: int,
        vae_height_compression_ratio: int,
        vae_width_compression_ratio: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.video_resolution_buckets = [
            (
                int(b[0] / vae_temporal_compression_ratio),
                int(b[1] / vae_height_compression_ratio),
                int(b[2] / vae_width_compression_ratio),
            )
            for b in video_resolution_buckets
        ]
        self.__frame_transforms = transforms.Compose(
            [transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)]
        )
        self.__image_transforms = self.__frame_transforms

    @override
    def preprocess(self, video_path: Path, image_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        video = preprocess_video_with_buckets(video_path, self.video_resolution_buckets)
        image = preprocess_image_with_resize(image_path, video.shape[2], video.shape[3])
        return video, image

    @override
    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.__frame_transforms(f) for f in frames], dim=0)

    @override
    def image_transform(self, image: torch.Tensor) -> torch.Tensor:
        return self.__image_transforms(image)
