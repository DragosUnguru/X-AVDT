from __future__ import annotations

# pylint: disable=E1101
# scripts/inference.py
import math
import argparse
import os
import sys
from pathlib import Path

import logging
logging.basicConfig(level=logging.ERROR)

import torch
from torch import nn

import glob
from PIL import Image
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import tempfile
import shutil

HALLO_ROOT = Path(__file__).resolve().parent
if str(HALLO_ROOT) not in sys.path:
    sys.path.insert(0, str(HALLO_ROOT))


def load_runtime_dependencies():
    global AutoencoderKL, DDIMScheduler, DDIMInverseScheduler, OmegaConf
    global FaceAnimatePipeline, AudioProcessor, ImageProcessor
    global AudioProjModel, FaceLocator, ImageProjModel
    global UNet2DConditionModel, UNet3DConditionModel, filter_non_none

    from diffusers import AutoencoderKL, DDIMInverseScheduler, DDIMScheduler
    from omegaconf import OmegaConf

    from hallo.animate.face_animate import FaceAnimatePipeline
    from hallo.datasets.audio_processor import AudioProcessor
    from hallo.datasets.image_processor import ImageProcessor
    from hallo.models.audio_proj import AudioProjModel
    from hallo.models.face_locator import FaceLocator
    from hallo.models.image_proj import ImageProjModel
    from hallo.models.unet_2d_condition import UNet2DConditionModel
    from hallo.models.unet_3d import UNet3DConditionModel
    from hallo.utils.config import filter_non_none


def _resolve_path(value):
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((HALLO_ROOT / path).resolve())


def resolve_config_paths(config):
    config.audio_ckpt_dir = _resolve_path(config.audio_ckpt_dir)
    config.base_model_path = _resolve_path(config.base_model_path)
    config.motion_module_path = _resolve_path(config.motion_module_path)
    config.save_path = _resolve_path(config.save_path)
    config.face_analysis.model_path = _resolve_path(config.face_analysis.model_path)
    config.wav2vec.model_path = _resolve_path(config.wav2vec.model_path)
    config.audio_separator.model_path = _resolve_path(config.audio_separator.model_path)
    config.vae.model_path = _resolve_path(config.vae.model_path)
    return config


def collect_clip_pairs(image_root, audio_root):
    image_root = Path(image_root)
    audio_root = Path(audio_root)
    if not image_root.is_dir():
        raise FileNotFoundError(f"frame directory does not exist: {image_root}")
    if not audio_root.is_dir():
        raise FileNotFoundError(f"wav directory does not exist: {audio_root}")

    image_folders = sorted(folder for folder in image_root.rglob("*") if folder.is_dir() and any(folder.glob("*.png")))
    audio_by_rel = {path.relative_to(audio_root).with_suffix(""): path for path in sorted(audio_root.rglob("*.wav"))}
    audio_by_stem = {}
    for path in sorted(audio_root.rglob("*.wav")):
        audio_by_stem.setdefault(path.stem, []).append(path)

    pairs = []
    missing_audio = []
    for folder in image_folders:
        rel_clip = folder.relative_to(image_root)
        audio_path = audio_by_rel.get(rel_clip)
        if audio_path is None and len(audio_by_stem.get(folder.name, [])) == 1:
            audio_path = audio_by_stem[folder.name][0]
        if audio_path is None:
            missing_audio.append(str(rel_clip))
            continue
        pairs.append((str(folder), str(audio_path), str(rel_clip)))

    if not pairs:
        audio_files = sorted(audio_root.rglob("*.wav"))
        if len(audio_files) != len(image_folders):
            raise RuntimeError(
                "Could not match image folders and wav files by clip id, and sorted fallback "
                f"is unsafe because counts differ: images={len(image_folders)} wavs={len(audio_files)}"
            )
        pairs = [(str(folder), str(audio), str(folder.relative_to(image_root))) for folder, audio in zip(image_folders, audio_files)]

    if missing_audio:
        print(f"[Warn] skipped {len(missing_audio)} image folders with no matching wav.")

    return pairs

class Net(nn.Module):
    def __init__(self, reference_unet: UNet2DConditionModel, denoising_unet: UNet3DConditionModel, 
                face_locator: FaceLocator, imageproj, audioproj):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.face_locator = face_locator
        self.imageproj = imageproj
        self.audioproj = audioproj

    def forward(self):
        pass

    def get_modules(self):
        return {
            "reference_unet": self.reference_unet,
            "denoising_unet": self.denoising_unet,
            "face_locator": self.face_locator,
            "imageproj": self.imageproj,
            "audioproj": self.audioproj,
        }


def process_audio_emb(audio_emb):
    concatenated_tensors = []

    for i in range(audio_emb.shape[0]):
        vectors_to_concat = [
            audio_emb[max(min(i + j, audio_emb.shape[0]-1), 0)]for j in range(-2, 3)]
        concatenated_tensors.append(torch.stack(vectors_to_concat, dim=0))

    audio_emb = torch.stack(concatenated_tensors, dim=0)

    return audio_emb

def save_attn_map(attn_feat, output_dir):
    save_dir = output_dir + "/attn_map"
    os.makedirs(save_dir, exist_ok=True)

    def visualize_attention_map(data_tensor, save_path_prefix):
        attn_map = data_tensor.mean(dim=-1).reshape(64, 64).cpu().numpy()

        plt.figure(figsize=(4, 4))
        plt.imshow(attn_map, cmap='viridis')  # cmap: 'hot', 'plasma', 'magma' 
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"{save_path_prefix}.png", bbox_inches='tight', pad_inches=0)
        plt.close()

    for i in range(attn_feat.shape[0]):
        frame = attn_feat[i]  # shape: [4096, 320]
        save_prefix = os.path.join(save_dir, f"{i:04d}")
        visualize_attention_map(frame, save_prefix)

def save_video_from_images(images, output_path):
    images = list(images)
    if not images:
        raise ValueError(f"No images were provided for video output: {output_path}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, img in enumerate(images):
            if not isinstance(img, Image.Image):
                with Image.open(img) as opened:
                    img = opened.convert("RGB")
            else:
                img = img.convert("RGB")
            img.save(os.path.join(tmp_dir, f"{i:04d}.png"))

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-framerate", "25",
            "-i", os.path.join(tmp_dir, "%04d.png"),
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output_path),
        ]
        completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            if output_path.exists() and output_path.stat().st_size == 0:
                output_path.unlink()
            raise RuntimeError(f"ffmpeg failed to save {output_path}:\n{completed.stderr.strip()}")
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise RuntimeError(f"ffmpeg produced an empty video: {output_path}")

def reconstruction_process(config, inverted_latents, device, weight_dtype, source_image_path, audio_emb, audio_length, vae, net):
    clip_length = config.data.n_sample_frames
    motion_scale = [config.pose_weight, config.face_weight, config.lip_weight]

    # 3.2 prepare source image, face mask, face embeddings
    img_size = (config.data.source_image.width, config.data.source_image.height)
    clip_length = config.data.n_sample_frames
    face_analysis_model_path = config.face_analysis.model_path
    with ImageProcessor(img_size, face_analysis_model_path) as image_processor:
        source_image_pixels, \
        source_image_face_region, \
        source_image_face_emb, \
        source_image_full_mask, \
        source_image_face_mask, \
        source_image_lip_mask = image_processor.preprocess(source_image_path[0], save_path, config.face_expand_ratio)

        # 3.2.5 load timestep latents
        source_latents_all = inverted_latents # -> [inference_step, 4, frames, 64, 64]  (e.g. [40, 4, 188, 64, 64])

        # first -> timestep 0 !
        source_latents_all = source_latents_all[-1].unsqueeze(0) # (e.g. [1, 4, 188, 64, 64])

        H, W = source_latents_all.shape[-2:]
        Channel = source_latents_all.shape[1]
        _times = math.ceil(source_latents_all.shape[2] / 16)
        _pad = _times * 16 - source_latents_all.shape[2]

        if _pad > 0:
            source_latents_all = torch.cat([source_latents_all, source_latents_all[:,:,-1:,:,:].repeat(1,1,_pad,1,1)], dim=2)
        source_latents_all = source_latents_all.reshape(1, Channel, _times, 16, H, W)

    # 4. build modules (reverse)
    sched_kwargs = OmegaConf.to_container(config.noise_scheduler_kwargs)
    if config.enable_zero_snr:
        sched_kwargs.update(rescale_betas_zero_snr=True, timestep_spacing="trailing", prediction_type="v_prediction")
    val_noise_scheduler = DDIMScheduler(**sched_kwargs)
    sched_kwargs.update({"beta_schedule": "scaled_linear"})

    # 5. inference
    pipeline = FaceAnimatePipeline(
        vae=vae,
        reference_unet=net.reference_unet,
        denoising_unet=net.denoising_unet,
        face_locator=net.face_locator,
        scheduler=val_noise_scheduler,
        image_proj=net.imageproj,
    )
    pipeline.to(device=device, dtype=weight_dtype)

    source_image_pixels = source_image_pixels.unsqueeze(0)
    source_image_face_region = source_image_face_region.unsqueeze(0)
    source_image_face_emb = source_image_face_emb.reshape(1, -1)
    source_image_face_emb = torch.tensor(source_image_face_emb)

    source_image_full_mask = [(mask.repeat(clip_length, 1)) for mask in source_image_full_mask]
    source_image_face_mask = [(mask.repeat(clip_length, 1)) for mask in source_image_face_mask]
    source_image_lip_mask = [(mask.repeat(clip_length, 1)) for mask in source_image_lip_mask]

    #times = audio_emb.shape[0] // clip_length
    times = inverted_latents.shape[2] // clip_length
    recon_result = []
    generator = torch.manual_seed(42)
    for t in range(times):
        print(f"[{t+1}/{times}]")
        source_latent_frames = source_latents_all[:,:,t,:,:]
        source_latent_frames = source_latent_frames.to(device=net.denoising_unet.device, dtype=net.denoising_unet.dtype)

        if len(recon_result) == 0:
            # The first iteration
            motion_zeros = source_image_pixels.repeat(config.data.n_motion_frames, 1, 1, 1)
            motion_zeros = motion_zeros.to(dtype=source_image_pixels.dtype, device=source_image_pixels.device)
            pixel_values_ref_img = torch.cat([source_image_pixels, motion_zeros], dim=0)  # concat the ref image and the first motion frames
        else:
            #motion_frames = recon_result[-1][0] # ([3, 16, 512, 512])
            motion_frames = motion_frames.permute(1, 0, 2, 3)
            motion_frames = motion_frames[0-config.data.n_motion_frames:]
            motion_frames = motion_frames * 2.0 - 1.0
            motion_frames = motion_frames.to(dtype=source_image_pixels.dtype, device=source_image_pixels.device)
            pixel_values_ref_img = torch.cat([source_image_pixels, motion_frames], dim=0)  # concat the ref image and the motion frames

        pixel_values_ref_img = pixel_values_ref_img.unsqueeze(0)

        audio_tensor = audio_emb[t * clip_length: min((t + 1) * clip_length, audio_emb.shape[0])]
        audio_tensor = audio_tensor.unsqueeze(0)
        audio_tensor = audio_tensor.to(device=net.audioproj.device, dtype=net.audioproj.dtype)
        audio_tensor = net.audioproj(audio_tensor)

        pipeline_output = pipeline(
            ref_image=pixel_values_ref_img,
            audio_tensor=audio_tensor,
            face_emb=source_image_face_emb,
            face_mask=source_image_face_region,
            pixel_values_full_mask=source_image_full_mask,
            pixel_values_face_mask=source_image_face_mask,
            pixel_values_lip_mask=source_image_lip_mask,
            width=img_size[0],
            height=img_size[1],
            video_length=clip_length,
            num_inference_steps=config.inference_steps,
            guidance_scale=config.cfg_scale,
            generator=generator,
            motion_scale=motion_scale,
            src_latents_frames=source_latent_frames, ##### -> input 16 source latent frames 
            reconstruction=True,
            #return_cross_attention=True,
        )

        pipeline_output = torch.vstack(pipeline_output) # [inference_steps, 4, frame_number, 64, 64]
        recon_result.append(pipeline_output)
        
        # convert latent to image for next frame generation
        motion_frames = pipeline.decode_latents(pipeline_output[-1:])[0] #input size = b c f h w
        motion_frames = torch.from_numpy(motion_frames)
    
    recon_result = torch.cat(recon_result, dim=2)
    reconstructed_latents = recon_result[:, :, :audio_length] # [inference_steps, 4, audio_length, 64, 64]
    z0_ = reconstructed_latents[-1:]

    reconstructed_images = pipeline.decode_latents(z0_, output_type='pil')
    return reconstructed_latents, z0_, reconstructed_images

def inversion_process(config, device, weight_dtype, source_image_path, audio_emb, audio_length, vae, net,):
    clip_length = config.data.n_sample_frames
    motion_scale = [config.pose_weight, config.face_weight, config.lip_weight]
    
    # 3.2 prepare source image, face mask, face embeddings
    img_size = (config.data.source_image.width, config.data.source_image.height)
    face_analysis_model_path = config.face_analysis.model_path
    with ImageProcessor(img_size, face_analysis_model_path) as image_processor:
        source_image_pixels, \
        source_image_face_region, \
        source_image_face_emb, \
        source_image_full_mask, \
        source_image_face_mask, \
        source_image_lip_mask = image_processor.preprocess(source_image_path[0], save_path, config.face_expand_ratio)

        # 3.2.5 load timestep latents
        source_image_files = source_image_path
        src_image_pil = [Image.open(img).convert("RGB") for img in source_image_files]
        source_latents_all = torch.stack([image_processor.pixel_transform(img) for img in src_image_pil])
        source_latents_all = source_latents_all.permute(1,0,2,3)[None] # -> [inference_step, 3, frames, 512, 512] 

        H, W = source_latents_all.shape[-2:]
        Channel = source_latents_all.shape[1]
        _times = math.ceil(source_latents_all.shape[2] / 16)
        _pad = _times * 16 - source_latents_all.shape[2]

        if _pad > 0:
            source_latents_all = torch.cat([source_latents_all, source_latents_all[:,:,-1:,:,:].repeat(1,1,_pad,1,1)], dim=2)
        source_latents_all = source_latents_all.reshape(1, Channel, _times, 16, H, W)
        
    # 4. build modules (inverse)
    sched_kwargs = OmegaConf.to_container(config.noise_scheduler_kwargs)
    if config.enable_zero_snr:
        sched_kwargs.update(rescale_betas_zero_snr=True, timestep_spacing="trailing", prediction_type="v_prediction")
    inverse_noise_scheduler = DDIMInverseScheduler(**sched_kwargs)
    sched_kwargs.update({"beta_schedule": "scaled_linear"})

    # 5. inference
    pipeline = FaceAnimatePipeline(
        vae=vae,
        reference_unet=net.reference_unet,
        denoising_unet=net.denoising_unet,
        face_locator=net.face_locator,
        scheduler=inverse_noise_scheduler,
        image_proj=net.imageproj,
    )
    pipeline.to(device=device, dtype=weight_dtype)

    source_image_pixels = source_image_pixels.unsqueeze(0)
    source_image_face_region = source_image_face_region.unsqueeze(0)
    source_image_face_emb = torch.tensor(source_image_face_emb.reshape(1, -1))

    source_image_full_mask = [(mask.repeat(clip_length, 1)) for mask in source_image_full_mask]
    source_image_face_mask = [(mask.repeat(clip_length, 1)) for mask in source_image_face_mask]
    source_image_lip_mask = [(mask.repeat(clip_length, 1)) for mask in source_image_lip_mask]

    times_a = audio_emb.shape[0] // clip_length
    times_v = len(src_image_pil) // clip_length
    times = min(times_a, times_v)

    inv_result = []
    z0_result = []
    attn_feat_list = []
    generator = torch.manual_seed(42)

    for t in range(times): # --> frames
        print(f"[{t+1}/{times}]")
        source_latent_frames = source_latents_all[:,:,t,:,:]
        source_latent_frames = source_latent_frames.to(device=net.denoising_unet.device, dtype=net.denoising_unet.dtype)
        
        if len(inv_result) == 0:
            # The first iteration
            motion_zeros = source_image_pixels.repeat(config.data.n_motion_frames, 1, 1, 1)
            motion_zeros = motion_zeros.to(dtype=source_image_pixels.dtype, device=source_image_pixels.device)
            pixel_values_ref_img = torch.cat([source_image_pixels, motion_zeros], dim=0)  # concat the ref image and the first motion frames
        else:
            # motion_frames = inv_result[-1][0] # ([3, 16, 512, 512])
            motion_frames = motion_frames.permute(1, 0, 2, 3)
            motion_frames = motion_frames[0-config.data.n_motion_frames:]
            motion_frames = motion_frames * 2.0 - 1.0
            motion_frames = motion_frames.to(dtype=source_image_pixels.dtype, device=source_image_pixels.device)
            pixel_values_ref_img = torch.cat([source_image_pixels, motion_frames], dim=0)  # concat the ref image and the motion frames

        pixel_values_ref_img = pixel_values_ref_img.unsqueeze(0)

        audio_tensor = audio_emb[t * clip_length: min((t + 1) * clip_length, audio_emb.shape[0])]
        audio_tensor = audio_tensor.unsqueeze(0)
        audio_tensor = audio_tensor.to(device=net.audioproj.device, dtype=net.audioproj.dtype)
        audio_tensor = net.audioproj(audio_tensor)

        # list[torch.Tensor]!
        pipeline_output, z0, attn_feat = pipeline(
            ref_image=pixel_values_ref_img,
            audio_tensor=audio_tensor,
            face_emb=source_image_face_emb,
            face_mask=source_image_face_region,
            pixel_values_full_mask=source_image_full_mask,
            pixel_values_face_mask=source_image_face_mask,
            pixel_values_lip_mask=source_image_lip_mask,
            width=img_size[0],
            height=img_size[1],
            video_length=clip_length,
            num_inference_steps=config.inference_steps,
            guidance_scale=config.cfg_scale,
            generator=generator,
            motion_scale=motion_scale,
            src_latents_frames=source_latent_frames, ##### -> input 16 source latent frames 
            inversion=True,
            return_cross_attention=True,
        )

        pipeline_output = torch.vstack(pipeline_output) # [inference_steps, 4, frame_number, 64, 64]
        inv_result.append(pipeline_output)
        z0 = torch.vstack(z0)
        z0_result.append(z0)
        attn_feat_list.append(attn_feat)

        # convert latent to image for next frame generation
        motion_frames = pipeline.decode_latents(pipeline_output[-1:])[0] #input size = b c f h w
        motion_frames = torch.from_numpy(motion_frames)

    inv_result = torch.cat(inv_result, dim=2)
    inverted_latents = inv_result[:, :, :audio_length] #torch.Size([40, 4, 80, 64, 64])
    zt = inverted_latents[-1:] #torch.Size([1, 4, 80, 64, 64])
    attn_feat = torch.cat(attn_feat_list, dim=0)

    z0_result = torch.cat(z0_result, dim=2)
    z0_result = z0_result[:, :, :audio_length] #torch.Size([40, 4, 80, 64, 64])
    z0 = z0_result[-1:]
    inverted_images = pipeline.decode_latents(zt, output_type='pil')
    
    return inverted_latents, zt, z0, inverted_images, attn_feat

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(HALLO_ROOT / "configs/inference/default.yaml"))
    parser.add_argument("--frames_dir", type=str, required=True, help="Root directory containing frame/ and wav/ subdirectories. ex) ./MMDF_frames")
    parser.add_argument("--pose_weight", type=float, help="weight of pose")
    parser.add_argument("--face_weight", type=float, help="weight of face")
    parser.add_argument("--lip_weight", type=float, help="weight of lip")
    parser.add_argument("--face_expand_ratio", type=float, help="face region")
    parser.add_argument("--audio_ckpt_dir", type=str, help="specific checkpoint dir", required=False)
    parser.add_argument("--output_dir", type=str, required=True, help="Feature output directory. ex) ./MMDF_features")
    args = parser.parse_args()

    load_runtime_dependencies()

    # 1. init config
    cli_args = filter_non_none(vars(args))
    config = OmegaConf.load(args.config)
    config = OmegaConf.merge(config, cli_args)
    config = resolve_config_paths(config)
    save_path = config.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 2. runtime variables
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if config.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif config.weight_dtype == "bf16":
        weight_dtype = torch.bfloat16
    elif config.weight_dtype == "fp32":
        weight_dtype = torch.float32
    else:
        weight_dtype = torch.float32

    vae = AutoencoderKL.from_pretrained(config.vae.model_path)
    reference_unet = UNet2DConditionModel.from_pretrained(config.base_model_path, subfolder="unet")
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        config.base_model_path,
        config.motion_module_path,
        subfolder="unet",
        unet_additional_kwargs=OmegaConf.to_container(
            config.unet_additional_kwargs),
        use_landmark=False,
    )
    face_locator = FaceLocator(conditioning_embedding_channels=320)
    image_proj = ImageProjModel(cross_attention_dim=denoising_unet.config.cross_attention_dim, clip_embeddings_dim=512, clip_extra_context_tokens=4)
    audio_proj = AudioProjModel(seq_len=5, blocks=12, channels=768, intermediate_dim=512, output_dim=768, context_tokens=32).to(device=device, dtype=weight_dtype)
    audio_ckpt_dir = config.audio_ckpt_dir

    # Freeze
    vae.requires_grad_(False)
    image_proj.requires_grad_(False)
    reference_unet.requires_grad_(False)
    denoising_unet.requires_grad_(False)
    face_locator.requires_grad_(False)
    audio_proj.requires_grad_(False)

    reference_unet.enable_gradient_checkpointing()
    denoising_unet.enable_gradient_checkpointing()
    net = Net(reference_unet, denoising_unet, face_locator, image_proj, audio_proj,)
    m,u = net.load_state_dict(torch.load(os.path.join(audio_ckpt_dir, "net.pth"), map_location="cpu"))
    assert len(m) == 0 and len(u) == 0, "Fail to load correct checkpoint."
    print("loaded weight from ", os.path.join(audio_ckpt_dir, "net.pth"))

    frames_dir = Path(args.frames_dir)
    clip_pairs = collect_clip_pairs(frames_dir / "frame", frames_dir / "wav")

    for frames, audio, rel_clip in tqdm(clip_pairs, total=len(clip_pairs), desc="Getting features"):
        print("Processing", frames)
        source_image_path = sorted(glob.glob(os.path.join(frames, "*.png")))

        # 3.1 prepare audio embeddings
        sample_rate = config.data.driving_audio.sample_rate
        assert sample_rate == 16000, "audio sample rate must be 16000"
        fps = config.data.export_video.fps
        clip_length = config.data.n_sample_frames
        wav2vec_model_path = config.wav2vec.model_path
        wav2vec_only_last_features = config.wav2vec.features == "last"
        audio_separator_model_file = config.audio_separator.model_path
        with AudioProcessor(
            sample_rate,
            fps,
            wav2vec_model_path,
            wav2vec_only_last_features,
            os.path.dirname(audio_separator_model_file),
            os.path.basename(audio_separator_model_file),
            os.path.join(save_path, "audio_preprocess")
        ) as audio_processor:
            audio_emb, audio_length = audio_processor.preprocess(audio, clip_length)
        audio_emb = process_audio_emb(audio_emb)
        
        output_dir = os.path.join(args.output_dir, rel_clip)
        os.makedirs(output_dir, exist_ok=True)

        print('[inversion]'*10)
        inverted_latents, zt, z0, inverted_images, attn_feat = inversion_process(config, device, weight_dtype, source_image_path, audio_emb, audio_length, vae, net)

        print('[reconstruction]'*10)
        reconstructed_latents, z0_, reconstructed_images = reconstruction_process(config, inverted_latents, device, weight_dtype, source_image_path, audio_emb, audio_length, vae, net)

        min_len = min(len(source_image_path), len(inverted_images), len(reconstructed_images), attn_feat.shape[0])
        features = {
            "original": source_image_path[:min_len],
            "inverted": inverted_images[:min_len],
            "reconstructed": reconstructed_images[:min_len],
            "original_feat": z0[:,:,:min_len,:],
            "inverted_feat": zt[:,:,:min_len,:],
            "reconstructed_feat": z0_[:,:,:min_len,:],
            "residual_feat": z0[:,:,:min_len,:] - z0_[:,:,:min_len,:],
            }

        for feat in ["original", "inverted", "reconstructed"]:
            images = features[feat]
            video_path = os.path.join(output_dir, feat + ".mp4")
            print(f"Saving {feat} as video: {video_path}")
            save_video_from_images(images, video_path)  

        for feat in ["original_feat", "inverted_feat", "reconstructed_feat", "residual_feat"]:
            latents = features[feat]
            print(f"Saving {feat} as features: {os.path.join(output_dir, f'{feat[:-5]}.pt')}")
            torch.save(latents.cpu(),os.path.join(output_dir, f'{feat[:-5]}.pt'))

        # cross attention
        torch.save(attn_feat.cpu(), os.path.join(output_dir, "attn_feat.pt"))
        save_attn_map(attn_feat, output_dir)
        video_path = os.path.join(output_dir, "attn_map.mp4")
        save_video_from_images(sorted(glob.iglob(os.path.join(output_dir, "attn_map", "*.png"))), video_path)
        shutil.rmtree(os.path.join(output_dir, "attn_map"))

        # residual
        residual_dir = os.path.join(output_dir, "residual")
        os.makedirs(residual_dir, exist_ok=True)
        with tqdm(total=min_len, desc="Processing residual") as pbar:
            for i in range(min_len):
                original = features["original"][i]
                reconstructed = features["reconstructed"][i]
                
                original = Image.open(original) if not isinstance(original, Image.Image) else original
                reconstructed = Image.open(reconstructed) if not isinstance(reconstructed, Image.Image) else reconstructed
                residual = np.abs(np.asarray(reconstructed, dtype=np.float32) - np.asarray(original, dtype=np.float32))
                residual = np.clip(residual, 0, 255).astype(np.uint8)
        
                Image.fromarray(residual).save(os.path.join(output_dir, "residual", f'{i:04d}.png'))
                pbar.update()
        residual_video_path = os.path.join(output_dir, "residual.mp4")
        print(f"Saving residual as video: {residual_video_path}")
        save_video_from_images(sorted(glob.iglob(os.path.join(residual_dir, "*.png"))), residual_video_path)
        shutil.rmtree(residual_dir)
