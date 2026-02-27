from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch, random
import math
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXTransformer3DModel,
)
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from PIL import Image
from numpy import dtype
from transformers import AutoTokenizer, T5EncoderModel
from typing_extensions import override
from einops import rearrange
from finetune.schemas import Components
from finetune.trainer import Trainer
from finetune.utils import unwrap_model
from collections import OrderedDict
from ..utils import register
from torchvision import transforms
from safetensors.torch import load_file
from charactershot import CogVideoXTransformer4DModel, CameraGuider, CogVideoXImageToVideo4DPipeline

    
class CombinedModel(torch.nn.Module):
    def __init__(self, transformer, camera_guider):
        super().__init__()
        self.transformer = transformer
        self.camera_guider = camera_guider
        self.config = transformer.config
    
    
class CogVideoXI2VLoraTrainer(Trainer):
    UNLOAD_LIST = ["text_encoder"]

    @override
    def load_components(self) -> Dict[str, Any]:
        components = Components()
        model_path = str(self.args.model_path)

        components.pipeline_cls = CogVideoXImageToVideoPipeline

        components.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")

        components.text_encoder = T5EncoderModel.from_pretrained(
            model_path, subfolder="text_encoder"
        )
        if self.args.func_type == "2dpretrain":
            components.transformer = CogVideoXTransformer3DModel.from_pretrained(
                model_path, subfolder="transformer"
            )
            components.transformer.patch_embed.use_learned_positional_embeddings = False
            
        elif self.args.func_type == "4dfinetune":

            DiT4D_config = CogVideoXTransformer4DModel.load_config(model_path, subfolder="transformer")

            local_transformer = CogVideoXTransformer4DModel.from_config(DiT4D_config)
            state_dict_3d = local_transformer.state_dict()
            if self.args.pose_model_path is not None: # 2d pretrained model
                checkpoint = torch.load(self.args.pose_model_path, map_location="cpu")
                state_dict_pose_3d = checkpoint.get("module", checkpoint.get("state_dict", checkpoint))
            else:
                state_dict_pose_3d = CogVideoXTransformer3DModel.from_pretrained(model_path, subfolder="transformer").state_dict()
            
            matched_params = OrderedDict()
            for name, param in state_dict_pose_3d.items():
                sub_name = name.replace("transformer.", "")
                if sub_name in state_dict_pose_3d and state_dict_3d[name].shape == param.shape:
                    matched_params[sub_name] = param
                
                # for new parameters
                if "attn1" in sub_name:
                    matched_params[sub_name.replace("attn1", "attn2")] = param
                if "norm1" in sub_name:
                    matched_params[sub_name.replace("norm1", "view_norm1")] = param            
                if "norm2" in sub_name:
                    matched_params[sub_name.replace("norm2", "view_norm2")] = param   

            missing_keys, unexpected_keys = local_transformer.load_state_dict(matched_params, strict=False)
            camera_encoder = CameraGuider(out_dim=3072)
            
            components.transformer = CombinedModel(local_transformer, camera_encoder) 
            print(f"Missing keys:{missing_keys} while loading 4D from 3D")

            components.transformer.transformer.patch_embed.use_learned_positional_embeddings = False

        components.vae = AutoencoderKLCogVideoX.from_pretrained(model_path, subfolder="vae")
        components.scheduler = CogVideoXDPMScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )
        print("Loading Components complete")
        return components

    @override
    def initialize_pipeline(self) -> CogVideoXImageToVideoPipeline:
        pipe = CogVideoXImageToVideoPipeline(
            tokenizer=self.components.tokenizer,
            text_encoder=self.components.text_encoder,
            vae=self.components.vae,
            transformer=unwrap_model(self.accelerator, self.components.transformer.transformer if self.args.func_type != "level1" else self.components.transformer),
            scheduler=self.components.scheduler,
        )
        return pipe

    @override
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        # shape of input video: [B, C, F, H, W]
        vae = self.components.vae
        video = video.to(vae.device, dtype=vae.dtype)
        latent_dist = vae.encode(video).latent_dist
        latent = latent_dist.sample() * vae.config.scaling_factor
        return latent

    @override
    def encode_text(self, prompt: str) -> torch.Tensor:
        prompt_token_ids = self.components.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.state.transformer_config.max_text_seq_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        prompt_token_ids = prompt_token_ids.input_ids
        prompt_embedding = self.components.text_encoder(
            prompt_token_ids.to(self.accelerator.device)
        )[0]
        return prompt_embedding

    @override
    def collate_fn(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        ret = {"encoded_videos": [], "prompt_embedding": [], "images": [], "poses": [], "cameras": []}
        try:
            for sample in samples:
                encoded_video = sample["encoded_video"]
                prompt_embedding = sample["prompt_embedding"]
                image = sample["image"]
                if "pose" in sample.keys():
                    pose = sample["pose"]
                    ret["poses"].append(pose)
                if "camera" in sample.keys():
                    camera = sample["camera"]
                    ret["cameras"].append(camera)
                ret["encoded_videos"].append(encoded_video)
                ret["prompt_embedding"].append(prompt_embedding)
                ret["images"].append(image)
                     
            ret["encoded_videos"] = torch.stack(ret["encoded_videos"])
            ret["prompt_embedding"] = torch.stack(ret["prompt_embedding"])
            ret["images"] = torch.stack(ret["images"])
            if len(ret["poses"]) != 0:
                ret["poses"] = torch.stack(ret["poses"])
            if len(ret["cameras"]) != 0:
                ret["cameras"] = torch.stack(ret["cameras"])
        except Exception as e:
            print(e)
        return ret
    
    @override
    def compute_loss(self, batch) -> torch.Tensor:
        prompt_embedding = batch["prompt_embedding"]
        latent = batch["encoded_videos"]
        images = batch["images"]
        # Shape of prompt_embedding: [B, seq_len, hidden_size]
        # Shape of latent: [B, C, F, H, W]
        # Shape of images: [B, C, H, W]
        latent = latent.to(dtype=self.components.transformer.dtype)
        images = images.to(dtype=self.components.transformer.dtype)
        patch_size_t = self.state.transformer_config.patch_size_t
        if patch_size_t is not None:
            ncopy = latent.shape[2] % patch_size_t
            # Copy the first frame ncopy times to match patch_size_t
            first_frame = latent[:, :, :1, :, :]  # Get first frame [B, C, 1, H, W]
            latent = torch.cat([first_frame.repeat(1, 1, ncopy, 1, 1), latent], dim=2)
            assert latent.shape[2] % patch_size_t == 0

        batch_size, num_channels, num_frames, height, width = latent.shape

        # Get prompt embeddings
        _, seq_len, _ = prompt_embedding.shape
        prompt_embedding = prompt_embedding.view(batch_size, seq_len, -1).to(dtype=latent.dtype)

        # Add frame dimension to images [B,C,H,W] -> [B,C,F,H,W]
        images = images.unsqueeze(2)
        # Add noise to images

        image_noise_sigma = torch.normal(
            mean=-3.0, std=0.5, size=(1,), device=self.accelerator.device
        )
        image_noise_sigma = torch.exp(image_noise_sigma).to(dtype=images.dtype)
        noisy_images = (
            images + torch.randn_like(images) * image_noise_sigma[:, None, None, None, None]
        )
        image_latent_dist = self.components.vae.encode(
            noisy_images.to(dtype=self.components.vae.dtype)
        ).latent_dist
        image_latents = image_latent_dist.sample() * self.components.vae.config.scaling_factor

        # Sample a random timestep for each sample
        timesteps = torch.randint(
            0,
            self.components.scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.accelerator.device,
        )
        timesteps = timesteps.long()

        # from [B, C, F, H, W] to [B, F, C, H, W]
        latent = latent.permute(0, 2, 1, 3, 4)
        image_latents = image_latents.permute(0, 2, 1, 3, 4)
        assert (latent.shape[0], *latent.shape[2:]) == (
            image_latents.shape[0],
            *image_latents.shape[2:],
        )

        # Padding image_latents to the same frame number as latent
        padding_shape = (latent.shape[0], latent.shape[1] - 1, *latent.shape[2:])
        latent_padding = image_latents.new_zeros(padding_shape)
        image_latents = torch.cat([image_latents, latent_padding], dim=1)

        # Add noise to latent
        noise = torch.randn_like(latent)
        latent_noisy = self.components.scheduler.add_noise(latent, noise, timesteps)

        # Concatenate latent and image_latents in the channel dimension
        latent_img_noisy = torch.cat([latent_noisy, image_latents], dim=2)

        # Prepare rotary embeds
        vae_scale_factor_spatial = 2 ** (len(self.components.vae.config.block_out_channels) - 1)
        transformer_config = self.state.transformer_config
        rotary_emb = (
            self.prepare_rotary_positional_embeddings(
                height=height * vae_scale_factor_spatial,
                width=width * vae_scale_factor_spatial,
                num_frames=num_frames,
                transformer_config=transformer_config,
                vae_scale_factor_spatial=vae_scale_factor_spatial,
                device=self.accelerator.device,
            )
            if transformer_config.use_rotary_positional_embeddings
            else None
        )

        # Predict noise, For CogVideoX1.5 Only.
        ofs_emb = (
            None
            if self.state.transformer_config.ofs_embed_dim is None
            else latent.new_full((1,), fill_value=2.0)
        )
        
        predicted_noise = self.components.transformer(
            hidden_states=latent_img_noisy,
            encoder_hidden_states=prompt_embedding,
            timestep=timesteps,
            ofs=ofs_emb,
            image_rotary_emb=rotary_emb,
            return_dict=False,
        )[0]

        # Denoise
        latent_pred = self.components.scheduler.get_velocity(
            predicted_noise, latent_noisy, timesteps
        )

        alphas_cumprod = self.components.scheduler.alphas_cumprod[timesteps]
        weights = 1 / (1 - alphas_cumprod)
        while len(weights.shape) < len(latent_pred.shape):
            weights = weights.unsqueeze(-1)

        loss = torch.mean((weights * (latent_pred - latent) ** 2).reshape(batch_size, -1), dim=1)
        loss = loss.mean()

        return loss


    def compute_loss_2d_pretrain(self, batch) -> torch.Tensor:
        prompt_embedding = batch["prompt_embedding"]
        latent = batch["encoded_videos"]
        pose = batch["poses"]
        images = batch["images"]
        latent = latent.to(dtype=self.components.transformer.dtype)
        images = images.to(dtype=self.components.transformer.dtype)
        raw_pose = images[:,1]
        images = images[:,0]
        
        pose = pose.to(dtype=self.components.transformer.dtype)
        # Shape of prompt_embedding: [B, seq_len, hidden_size]
        # Shape of latent: [B, V, C, F, H, W]
        # Shape of images: [B, V, C, 1, H, W]
        # Shape of poses: [B, V, F, C, H, W]
        patch_size_t = self.state.transformer_config.patch_size_t

        if patch_size_t is not None:
            ncopy = latent.shape[3] % patch_size_t0
            # Copy the first frame ncopy times to match patch_size_t
            first_frame = latent[:, :, :, :1, :, :]  # Get first frame [B, C, 1, H, W]
            latent = torch.cat([first_frame.repeat(1, 1, 1, ncopy, 1, 1), latent], dim=3)
            assert latent.shape[3] % patch_size_t == 0

        batch_size, num_channels, num_frames, height, width = latent.shape
        
        # Get prompt embeddings
        _, seq_len, _ = prompt_embedding.shape

        prompt_embedding = prompt_embedding.view(batch_size, seq_len, -1).to(dtype=latent.dtype)
        
        # Add frame dimension to images [B,C,H,W] -> [B,C,F,H,W]
        images = images.unsqueeze(2)
        image_noise_sigma = torch.normal(
            mean=-3.0, std=0.5, size=(1,), device=self.accelerator.device
        )
        image_noise_sigma = torch.exp(image_noise_sigma).to(dtype=images.dtype)
        noisy_images = (
            images + torch.randn_like(images) * image_noise_sigma[:, None, None, None, None]
        )
        image_latent_dist = self.components.vae.encode(
            noisy_images.to(dtype=self.components.vae.dtype)
        ).latent_dist
        image_latents = image_latent_dist.sample() * self.components.vae.config.scaling_factor
        
        raw_pose = raw_pose.unsqueeze(2)
        raw_pose_latent_dist = self.components.vae.encode(
            raw_pose.to(dtype=self.components.vae.dtype)
        ).latent_dist
        raw_pose_latents = raw_pose_latent_dist.sample() * self.components.vae.config.scaling_factor
        
        
        # Sample a random timestep for each sample
        timesteps = torch.randint(
            0,
            self.components.scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.accelerator.device,
        )
        timesteps = timesteps.long()

        # from [B, C, F, H, W] to [B, F, C, H, W]

        latent = latent.permute(0, 2, 1, 3, 4)
        image_latents = image_latents.permute(0, 2, 1, 3, 4)
        pose = pose.permute(0, 2, 1, 3, 4)
        raw_pose_latents = raw_pose_latents.permute(0, 2, 1, 3, 4)
        # pose[:,0:1] = raw_pose_latents
        pose = torch.cat([raw_pose_latents, pose], dim=1)
        assert (latent.shape[0], *latent.shape[2:]) == (
            image_latents.shape[0],
            *image_latents.shape[2:],
        )

        # Add noise to latent
        noise = torch.randn_like(latent)
        latent_noisy = self.components.scheduler.add_noise(latent, noise, timesteps)
        # if p < 0.15:
        #     pose = pose * 0.0
        # Concatenate latent and image_latents in the channel dimension

        latent_img_noisy = torch.cat([image_latents, latent_noisy], dim=1)
        latent_img_noisy = torch.cat([latent_img_noisy, pose], dim=2)
        # Prepare rotary embeds
        vae_scale_factor_spatial = 2 ** (len(self.components.vae.config.block_out_channels) - 1)
        transformer_config = self.state.transformer_config
        rotary_emb = (
            self.prepare_rotary_positional_embeddings(
                height=height * vae_scale_factor_spatial,
                width=width * vae_scale_factor_spatial,
                num_frames=num_frames+1,
                transformer_config=transformer_config,
                vae_scale_factor_spatial=vae_scale_factor_spatial,
                device=self.accelerator.device,
            )
            if transformer_config.use_rotary_positional_embeddings
            else None
        )

        # Predict noise, For CogVideoX1.5 Only.
        ofs_emb = (
            None
            if self.state.transformer_config.ofs_embed_dim is None
            else latent.new_full((1,), fill_value=2.0)
        )

        predicted_noise = self.components.transformer(
            hidden_states=latent_img_noisy,
            encoder_hidden_states=prompt_embedding,
            timestep=timesteps,
            ofs=ofs_emb,
            image_rotary_emb=rotary_emb,
            return_dict=False,
        )[0]

        # Denoise
        predicted_noise = predicted_noise[:, 1:]
        latent_pred = self.components.scheduler.get_velocity(
            predicted_noise, latent_noisy, timesteps
        )

        alphas_cumprod = self.components.scheduler.alphas_cumprod[timesteps]
        weights = 1 / (1 - alphas_cumprod)
        while len(weights.shape) < len(latent_pred.shape):
            weights = weights.unsqueeze(-1)

        loss = torch.mean((weights * (latent_pred - latent) ** 2).reshape(batch_size, -1), dim=1)
        loss = loss.mean()

        return loss

    def compute_loss_4d_finetune(self, batch) -> torch.Tensor:
        prompt_embedding = batch["prompt_embedding"]
        latent = batch["encoded_videos"]
        pose = batch["poses"]
        images = batch["images"]
        latent = latent.to(dtype=self.components.transformer.transformer.dtype)
        images = images.to(dtype=self.components.transformer.transformer.dtype)
        raw_pose = images[:,-1:]
        images = images[:,:-1]

        cameras = batch["cameras"]
        cameras = rearrange(cameras, 'b v c h w -> (b v) c h w')
        cameras = cameras.to(dtype=self.components.transformer.transformer.dtype)
        camera_latents = self.components.transformer.camera_guider(cameras.to(self.accelerator.device))
        
        pose = pose.to(dtype=self.components.transformer.transformer.dtype)
        # Shape of prompt_embedding: [B, seq_len, hidden_size]
        # Shape of latent: [B, V, C, F, H, W]
        # Shape of images: [B, V, C, 1, H, W]
        # Shape of poses: [B, V, F, C, H, W]
        patch_size_t = self.state.transformer_config.patch_size_t

        if patch_size_t is not None:
            ncopy = latent.shape[3] % patch_size_t0
            # Copy the first frame ncopy times to match patch_size_t
            first_frame = latent[:, :, :, :1, :, :]  # Get first frame [B, C, 1, H, W]
            latent = torch.cat([first_frame.repeat(1, 1, 1, ncopy, 1, 1), latent], dim=3)
            assert latent.shape[3] % patch_size_t == 0

        batch_size, views, num_channels, num_frames, height, width = latent.shape
        
        # Get prompt embeddings
        _, seq_len, _ = prompt_embedding.shape

        prompt_embedding = prompt_embedding.view(batch_size, seq_len, -1).to(dtype=latent.dtype)
        
        # Add frame dimension to images [B,C,H,W] -> [B,C,F,H,W]
        images = rearrange(images, 'b v c h w -> (b v) c h w').unsqueeze(2)
        image_noise_sigma = torch.normal(
            mean=-3.0, std=0.5, size=(1,), device=self.accelerator.device
        )
        image_noise_sigma = torch.exp(image_noise_sigma).to(dtype=images.dtype)
        noisy_images = (
            images + torch.randn_like(images) * image_noise_sigma[:, None, None, None, None]
        )
        noisy_images = rearrange(noisy_images, '(b v) c f h w -> v b c f h w', b=batch_size)
        image_latents = []
        
        for noisy_image in noisy_images:
            image_latent_dist = self.components.vae.encode(
                noisy_image.to(dtype=self.components.vae.dtype)
            ).latent_dist
            image_latents.append(image_latent_dist.sample() * self.components.vae.config.scaling_factor)
        image_latents = torch.stack(image_latents)

        raw_pose = rearrange(raw_pose, 'b v c h w -> (b v) c h w').unsqueeze(2)
        raw_pose_latent_dist = self.components.vae.encode(
            raw_pose.to(dtype=self.components.vae.dtype)
        ).latent_dist
        raw_pose_latents = raw_pose_latent_dist.sample() * self.components.vae.config.scaling_factor
        
        image_latents = rearrange(image_latents, 'v b c f h w -> b v c f h w')
        raw_pose_latents = rearrange(raw_pose_latents, '(b v) c f h w -> b v c f h w', b=batch_size)

        # Sample a random timestep for each sample
        timesteps = torch.randint(
            0,
            self.components.scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.accelerator.device,
        )
        timesteps = timesteps.long()
        # from [B, C, F, H, W] to [B, F, C, H, W]
        latent = latent.permute(0, 1, 3, 2, 4, 5)
        image_latents = image_latents.permute(0, 1, 3, 2, 4, 5)
        pose = pose.unsqueeze(1).permute(0, 1, 3, 2, 4, 5)
        raw_pose_latents = raw_pose_latents.permute(0, 1, 3, 2, 4, 5)

        pose = torch.cat([raw_pose_latents, pose], dim=2)
        assert (latent.shape[0], *latent.shape[3:]) == (
            image_latents.shape[0],
            *image_latents.shape[3:],
        )

        # Add noise to latent
        latent = rearrange(latent, 'b v f c h w -> v b f c h w')
        latent_noisy = []
        for view_latent in latent:
            noise = torch.randn_like(view_latent)
            latent_noisy.append(self.components.scheduler.add_noise(view_latent, noise, timesteps))
        latent_noisy = torch.stack(latent_noisy)
            
        latent_noisy = rearrange(latent_noisy, 'v b f c h w -> b v f c h w')
        latent = rearrange(latent, 'v b f c h w -> b v f c h w', b=batch_size)
        
        latent_img_noisy = torch.cat([image_latents, latent_noisy], dim=2)
        pose = pose.repeat(1,views,1,1,1,1)
        latent_img_noisy = torch.cat([latent_img_noisy, pose], dim=3)
        # Prepare rotary embeds
        vae_scale_factor_spatial = 2 ** (len(self.components.vae.config.block_out_channels) - 1)
        transformer_config = self.state.transformer_config
        rotary_emb = (
            self.prepare_rotary_positional_embeddings(
                height=height * vae_scale_factor_spatial,
                width=width * vae_scale_factor_spatial,
                num_frames=num_frames+1,
                transformer_config=transformer_config,
                vae_scale_factor_spatial=vae_scale_factor_spatial,
                device=self.accelerator.device,
            )
            if transformer_config.use_rotary_positional_embeddings
            else None
        )
        view_rotaty_emb = (
            self.prepare_rotary_positional_embeddings(
                height=height * vae_scale_factor_spatial,
                width=width * vae_scale_factor_spatial,
                num_frames=views,
                transformer_config=transformer_config,
                vae_scale_factor_spatial=vae_scale_factor_spatial,
                device=self.accelerator.device,
            )
            if transformer_config.use_rotary_positional_embeddings
            else None
        )
        # Predict noise, For CogVideoX1.5 Only.
        ofs_emb = (
            None
            if self.state.transformer_config.ofs_embed_dim is None
            else latent.new_full((1,), fill_value=2.0)
        )
        prompt_embedding = prompt_embedding.unsqueeze(1)
        prompt_embedding = prompt_embedding.repeat(1, latent_img_noisy.shape[1], 1, 1)
        camera_latents = rearrange(camera_latents, 'b c h w -> b (h w) c').contiguous()
        camera_latents = rearrange(camera_latents, '(b v) n c -> b v n c', b=batch_size, v=views)
        predicted_noise = self.components.transformer.transformer(
            hidden_states=latent_img_noisy,
            encoder_hidden_states=prompt_embedding,
            timestep=timesteps,
            ofs=ofs_emb,
            image_rotary_emb=rotary_emb,
            view_rotary_emb=view_rotaty_emb,
            return_dict=False,
            camera_latents=camera_latents,
        )[0]
        # Denoise
        predicted_noise = predicted_noise[:, :, 1:]
        
        predicted_noise = rearrange(predicted_noise, 'b v f c h w -> v b f c h w')
        latent_noisy = rearrange(latent_noisy, 'b v f c h w -> v b f c h w')
        latent_pred = []
        for p_n, l_n in zip(predicted_noise, latent_noisy):
            latent_pred.append(
                self.components.scheduler.get_velocity(
                    p_n, l_n, timesteps
                )
            )

        latent_pred = torch.stack(latent_pred)
        latent_pred = rearrange(latent_pred, 'v b f c h w -> b v f c h w')
        alphas_cumprod = self.components.scheduler.alphas_cumprod[timesteps]
        weights = 1 / (1 - alphas_cumprod)
        while len(weights.shape) < len(latent_pred.shape):
            weights = weights.unsqueeze(-1)

        loss = torch.mean((weights * (latent_pred - latent) ** 2).reshape(batch_size*views, -1), dim=1)
        loss = loss.mean()

        return loss


    @override
    def validation_step(
        self, eval_data: Dict[str, Any], pipe: CogVideoXImageToVideoPipeline
    ) -> List[Tuple[str, Image.Image | List[Image.Image]]]:
        """
        Return the data that needs to be saved. For videos, the data format is List[PIL],
        and for images, the data format is PIL
        """
        prompt, image, video = eval_data["prompt"], eval_data["image"], eval_data["video"]

        video_generate = pipe(
            num_frames=self.state.train_frames,
            height=self.state.train_height,
            width=self.state.train_width,
            prompt=prompt,
            image=image,
            generator=self.state.generator,
        ).frames[0]
        return [("video", video_generate)]

    def prepare_rotary_positional_embeddings(
        self,
        height: int,
        width: int,
        num_frames: int,
        transformer_config: Dict,
        vae_scale_factor_spatial: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        grid_height = height // (vae_scale_factor_spatial * transformer_config.patch_size)
        grid_width = width // (vae_scale_factor_spatial * transformer_config.patch_size)

        if transformer_config.patch_size_t is None:
            base_num_frames = num_frames
        else:
            base_num_frames = (
                num_frames + transformer_config.patch_size_t - 1
            ) // transformer_config.patch_size_t

        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=transformer_config.attention_head_dim,
            crops_coords=None,
            grid_size=(grid_height, grid_width),
            temporal_size=base_num_frames,
            grid_type="slice",
            max_size=(grid_height, grid_width),
            device=device,
        )

        return freqs_cos, freqs_sin


register("cogvideox-i2v", "lora", CogVideoXI2VLoraTrainer)
