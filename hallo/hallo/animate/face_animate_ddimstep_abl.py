# pylint: disable=R0801
"""
This module is responsible for animating faces in videos using a combination of deep learning techniques.
It provides a pipeline for generating face animations by processing video frames and extracting face features. 
The module utilizes various schedulers and utilities for efficient face animation and supports different types 
    of latents for more control over the animation process.

Functions and Classes:
- FaceAnimatePipeline: A class that extends the DiffusionPipeline class from the diffusers library to handle face animation tasks.
  - __init__: Initializes the pipeline with the necessary components (VAE, UNets, face locator, etc.).
  - prepare_latents: Generates or loads latents for the animation process, scaling them according to the scheduler's requirements.
  - prepare_extra_step_kwargs: Prepares extra keyword arguments for the scheduler step, ensuring compatibility with different schedulers.
  - decode_latents: Decodes the latents into video frames, ready for animation.

Usage:
- Import the necessary packages and classes.
- Create a FaceAnimatePipeline instance with the required components.
- Prepare the latents for the animation process.
- Use the pipeline to generate the animated video.

Note:
- This module is designed to work with the diffusers library, which provides the underlying framework for face animation using deep learning.
- The module is intended for research and development purposes, and further optimization and customization may be required for specific use cases.
"""

import inspect
from dataclasses import dataclass
from typing import Callable, List, Optional, Union
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import (
    DDIMScheduler, DiffusionPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
    LMSDiscreteScheduler, PNDMScheduler,
    DDIMInverseScheduler,
)
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange, repeat
from tqdm import tqdm

from hallo.models.mutual_self_attention import ReferenceAttentionControl


@dataclass
class FaceAnimatePipelineOutput(BaseOutput):
    """
    FaceAnimatePipelineOutput is a custom class that inherits from BaseOutput and represents the output of the FaceAnimatePipeline.
    """
    videos: Union[torch.Tensor, np.ndarray]


class FaceAnimatePipeline(DiffusionPipeline):
    """
    FaceAnimatePipeline is a custom DiffusionPipeline for animating faces.
    """

    def __init__(
        self,
        vae,
        reference_unet,
        denoising_unet,
        face_locator,
        image_proj,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
            DDIMInverseScheduler,
        ],
        use_guidance: Optional[bool] = False,
        use_sdedit: Optional[bool] = False,
        beta: Optional[float] = 1.0,
    ) -> None:
        super().__init__()

        self.register_modules(
            vae=vae,
            reference_unet=reference_unet,
            denoising_unet=denoising_unet,
            face_locator=face_locator,
            scheduler=scheduler,
            image_proj=image_proj,
        )

        self.vae_scale_factor: int = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.ref_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True,
        )
        self.use_guidance = use_guidance
        self.beta = beta
        if self.use_guidance:
            self.first_frame_cache = []
        self.use_sdedit = use_sdedit

        # --- cross-attention feature capture (hooks) ---
        self._attn_hooked = False
        self._attn_cache: List[torch.Tensor] = []
        self._attn_hook_handles = []

    def _reset_attn_cache(self):
        self._attn_cache.clear()

    def _ensure_attn_hook(self):
        """
        Install forward hooks on cross-attention modules (attn2) to capture attention features.
        We do NOT rely on UNet returning (noise_pred, attn_feat).
        """
        if self._attn_hooked:
            return

        def hook_fn(module, inp, out):
            # out could be tensor / tuple / ModelOutput-like
            x = out[0] if isinstance(out, (tuple, list)) else out
            if hasattr(x, "sample"):  # ModelOutput
                x = x.sample
            if torch.is_tensor(x):
                self._attn_cache.append(x.detach())

        # Prefer cross-attn blocks named 'attn2'
        targets = []
        for name, m in self.denoising_unet.named_modules():
            if name.endswith("attn2") or ".attn2" in name:
                targets.append((name, m))

        # Hook the last attn2 (often most semantically useful)
        if len(targets) > 0:
            _, m = targets[-1]
            self._attn_hook_handles.append(m.register_forward_hook(hook_fn))
            self._attn_hooked = True
            return

        # Fallback: any attention-like module that has to_k
        for name, m in self.denoising_unet.named_modules():
            cls = m.__class__.__name__.lower()
            if ("attn" in cls or "attention" in cls) and hasattr(m, "to_k"):
                self._attn_hook_handles.append(m.register_forward_hook(hook_fn))
                self._attn_hooked = True
                return

        raise RuntimeError("Could not find a cross-attention module to hook for attn_feat.")

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        width: int,
        height: int,
        video_length: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None
    ):
        shape = (
            batch_size,
            num_channels_latents,
            video_length,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            latents = latents.to(device)

        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_extra_step_kwargs(self, generator, eta):
        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def decode_latents(self, latents, output_type="numpy"):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")

        video = []
        for frame_idx in tqdm(range(latents.shape[0])):
            video.append(self.vae.decode(
                latents[frame_idx: frame_idx + 1]).sample)
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)

        if output_type == "tensor":
            video = video.cpu().float()
        elif output_type == "numpy":
            video = video.cpu().float().numpy()
        elif output_type == "pil":
            pil_images = []
            for i in range(video.shape[2]):
                frame = video[0, :, i, :, :].permute(1, 2, 0).cpu().numpy()
                frame = (frame * 255).astype(np.uint8)
                pil_images.append(Image.fromarray(frame))
            return pil_images

        return video

    @torch.no_grad()
    def __call__(
        self,
        ref_image,
        face_emb,
        audio_tensor,
        face_mask,
        pixel_values_full_mask,
        pixel_values_face_mask,
        pixel_values_lip_mask,
        width,
        height,
        video_length,
        num_inference_steps,
        guidance_scale,
        num_images_per_prompt=1,
        eta: float = 0.0,
        motion_scale: Optional[List[torch.Tensor]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        src_latents_frames: Optional[torch.Tensor] = None,
        inversion: Optional[bool] = False,
        reconstruction: Optional[bool] = False,
        no_image: Optional[bool] = False,
        rand_ref: Optional[bool] = False,
        bank_feature: Optional[List] = None,
        is_first_frame: Optional[bool] = False,
        step_ratio: Optional[float] = 0.5,
        return_cross_attention: Optional[bool] = False,
        **kwargs,
    ):
        attn_feat = None

        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        device = self._execution_device

        do_classifier_free_guidance = guidance_scale > 1.0
        do_classifier_free_guidance = False

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        batch_size = 1

        # prepare clip image embeddings
        clip_image_embeds = face_emb
        clip_image_embeds = clip_image_embeds.to(self.image_proj.device, self.image_proj.dtype)

        encoder_hidden_states = self.image_proj(clip_image_embeds)
        uncond_encoder_hidden_states = self.image_proj(torch.zeros_like(clip_image_embeds))

        if do_classifier_free_guidance:
            encoder_hidden_states = torch.cat([uncond_encoder_hidden_states, encoder_hidden_states], dim=0)

        reference_control_writer = ReferenceAttentionControl(
            self.reference_unet,
            do_classifier_free_guidance=do_classifier_free_guidance,
            mode="write",
            batch_size=batch_size,
            fusion_blocks="full",
        )
        reference_control_reader = ReferenceAttentionControl(
            self.denoising_unet,
            do_classifier_free_guidance=do_classifier_free_guidance,
            mode="read",
            batch_size=batch_size,
            fusion_blocks="full",
        )

        num_channels_latents = self.denoising_unet.in_channels

        # replaced with source latent frames!
        latent_size = int(ref_image.shape[-1] / 8)
        if inversion:
            assert src_latents_frames is not None, "src_latents_frames not given!"
            if src_latents_frames.shape[-1] != latent_size:
                frame_len = src_latents_frames.shape[2]
                src_frames_tensor = src_latents_frames.permute(0, 2, 1, 3, 4)
                src_frames_tensor = rearrange(src_frames_tensor, "b f c h w -> (b f) c h w")
                src_frames_tensor = self.ref_image_processor.preprocess(src_frames_tensor, height=height, width=width)
                src_frames_tensor = src_frames_tensor.to(dtype=self.vae.dtype, device=self.vae.device)
                latents = self.vae.encode(src_frames_tensor).latent_dist.mean
                latents = rearrange(latents, "(b f) c h w -> b c f h w", f=frame_len)
                latents = latents * 0.18215
                z0 = latents.detach().clone()

        if reconstruction:
            latents = src_latents_frames

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Prepare ref image latents z0
        ref_image_tensor = rearrange(ref_image, "b f c h w -> (b f) c h w")
        ref_image_tensor = self.ref_image_processor.preprocess(ref_image_tensor, height=height, width=width)
        ref_image_tensor = ref_image_tensor.to(dtype=self.vae.dtype, device=self.vae.device)
        ref_image_latents = self.vae.encode(ref_image_tensor).latent_dist.mean
        ref_image_latents = ref_image_latents * 0.18215

        face_mask = face_mask.unsqueeze(1).to(dtype=self.face_locator.dtype, device=self.face_locator.device)
        face_mask = repeat(face_mask, "b f c h w -> b (repeat f) c h w", repeat=video_length)
        face_mask = face_mask.transpose(1, 2)
        face_mask = self.face_locator(face_mask)
        face_mask = torch.cat([torch.zeros_like(face_mask), face_mask], dim=0) if do_classifier_free_guidance else face_mask

        pixel_values_full_mask = ([torch.cat([mask] * 2) for mask in pixel_values_full_mask] if do_classifier_free_guidance else pixel_values_full_mask)
        pixel_values_face_mask = ([torch.cat([mask] * 2) for mask in pixel_values_face_mask] if do_classifier_free_guidance else pixel_values_face_mask)
        pixel_values_lip_mask = ([torch.cat([mask] * 2) for mask in pixel_values_lip_mask] if do_classifier_free_guidance else pixel_values_lip_mask)

        pixel_values_face_mask_ = []
        for mask in pixel_values_face_mask:
            pixel_values_face_mask_.append(mask.to(device=self.denoising_unet.device, dtype=self.denoising_unet.dtype))
        pixel_values_face_mask = pixel_values_face_mask_

        pixel_values_lip_mask_ = []
        for mask in pixel_values_lip_mask:
            pixel_values_lip_mask_.append(mask.to(device=self.denoising_unet.device, dtype=self.denoising_unet.dtype))
        pixel_values_lip_mask = pixel_values_lip_mask_

        pixel_values_full_mask_ = []
        for mask in pixel_values_full_mask:
            pixel_values_full_mask_.append(mask.to(device=self.denoising_unet.device, dtype=self.denoising_unet.dtype))
        pixel_values_full_mask = pixel_values_full_mask_

        if do_classifier_free_guidance:
            uncond_audio_tensor = torch.zeros_like(audio_tensor)
            audio_tensor = torch.cat([uncond_audio_tensor, audio_tensor], dim=0)
            audio_tensor = audio_tensor.to(dtype=self.denoising_unet.dtype, device=self.denoising_unet.device)
        else:
            audio_tensor = audio_tensor

        # denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        half_steps = int(num_inference_steps * step_ratio)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            z0_list = []
            latent_list = []
            for i, t in enumerate(timesteps):
                is_last_timestep = i == len(timesteps) - 1

                # Forward reference image
                if i == 0 and not no_image:
                    if bank_feature:
                        reference_control_reader.update_bank(bank_feature)
                    else:
                        self.reference_unet(
                            ref_image_latents.repeat((2 if do_classifier_free_guidance else 1), 1, 1, 1),
                            torch.zeros_like(t),
                            encoder_hidden_states=encoder_hidden_states,
                            return_dict=False,
                        )
                        reference_control_reader.update(reference_control_writer)

                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # ---- robust cross-attention capture (hooks), do NOT unpack UNet output ----
                if inversion and return_cross_attention and i == 0:
                    self._ensure_attn_hook()
                    self._reset_attn_cache()

                    out = self.denoising_unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=encoder_hidden_states,
                        mask_cond_fea=face_mask,
                        full_mask=pixel_values_full_mask,
                        face_mask=pixel_values_face_mask,
                        lip_mask=pixel_values_lip_mask,
                        audio_embedding=audio_tensor,
                        motion_scale=motion_scale,
                        return_dict=True,
                        return_cross_attention=False,  # don't rely on UNet returning it
                    )

                    if hasattr(out, "sample"):
                        noise_pred = out.sample
                    elif isinstance(out, (tuple, list)):
                        noise_pred = out[0]
                    else:
                        noise_pred = out

                    if len(self._attn_cache) == 0:
                        raise RuntimeError("return_cross_attention=True but no attn_feat captured by hooks.")
                    attn_feat = self._attn_cache[-1]

                else:
                    out = self.denoising_unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=encoder_hidden_states,
                        mask_cond_fea=face_mask,
                        full_mask=pixel_values_full_mask,
                        face_mask=pixel_values_face_mask,
                        lip_mask=pixel_values_lip_mask,
                        audio_embedding=audio_tensor,
                        motion_scale=motion_scale,
                        return_dict=False,
                        return_cross_attention=False,
                    )
                    noise_pred = out[0] if isinstance(out, (tuple, list)) else (out.sample if hasattr(out, "sample") else out)

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.use_guidance and not is_first_frame:
                    latents_prev = latents.detach().clone()

                # compute x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # progress / callback
                if i == len(timesteps) - 1 or (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0:
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

                latent_list.append(latents)
                if inversion:
                    z0_list.append(z0)

                if self.use_guidance:
                    if is_first_frame:
                        self.first_frame_cache.append(
                            latents[:, :, 0:1].clone()
                        )
                    else:
                        if i <= half_steps:
                            src = latents.float().requires_grad_(True)
                            anc = self.first_frame_cache[i].float().requires_grad_(True).repeat(
                                1, 1, src.shape[2], 1, 1
                            )
                            with torch.enable_grad():
                                loss = F.mse_loss(anc, src, reduction="mean")
                                cond_grad = -torch.autograd.grad(loss, src)[0]

                            alpha = torch.linalg.norm(latents_prev - latents) / torch.linalg.norm(cond_grad)
                            cond_grad = cond_grad * alpha * self.beta

                            cond_grad = cond_grad.to(
                                device=self.denoising_unet.device,
                                dtype=self.denoising_unet.dtype
                            )
                            latents = latents + cond_grad

            reference_control_reader.clear()
            reference_control_writer.clear()

        if inversion:
            return latent_list, z0_list, attn_feat
        else:
            return latent_list
