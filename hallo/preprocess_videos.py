import argparse
import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")
FEATURE_VIDEOS = ("original", "inverted", "reconstructed", "residual")


def run_command(command):
    subprocess.run(command, check=True)


def require_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for video frame/audio extraction.")


def relative_clip_path(video_path, video_root):
    return video_path.relative_to(video_root).with_suffix("")


def extract_frames_and_audio(video_path, video_root, frame_out, wav_out, fps, size, duration):
    require_ffmpeg()
    clip_path = relative_clip_path(video_path, video_root)
    frame_dir = frame_out / clip_path
    audio_path = (wav_out / clip_path).with_suffix(".wav")
    frame_dir.mkdir(parents=True, exist_ok=True)
    audio_path.parent.mkdir(parents=True, exist_ok=True)

    if any(frame_dir.glob("*.png")) and audio_path.exists():
        return

    for old_frame in frame_dir.glob("*.png"):
        old_frame.unlink()
    if audio_path.exists():
        audio_path.unlink()

    video_filter = f"fps={fps}"
    if size is not None:
        width, height = size
        video_filter = f"{video_filter},scale={width}:{height}"

    image_cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video_path)]
    if duration is not None:
        image_cmd.extend(["-t", str(duration)])
    image_cmd.extend(["-vf", video_filter, "-start_number", "0", str(frame_dir / "%04d.png")])

    audio_cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video_path)]
    if duration is not None:
        audio_cmd.extend(["-t", str(duration)])
    audio_cmd.extend(["-ac", "2", "-ar", "16000", "-vn", str(audio_path)])

    run_command(image_cmd)
    run_command(audio_cmd)


def extract_raw_videos(args):
    video_root = Path(args.video_dir)
    if not video_root.is_dir():
        raise FileNotFoundError(video_root)

    frames_dir = Path(args.frames_dir)
    frame_out = frames_dir / "frame"
    wav_out = frames_dir / "wav"
    frame_out.mkdir(parents=True, exist_ok=True)
    wav_out.mkdir(parents=True, exist_ok=True)

    videos = sorted(path for path in video_root.rglob("*") if path.suffix.lower() in VIDEO_EXTS)
    if not videos:
        print(f"No videos found under {video_root}")
        return

    for video_path in tqdm(videos, desc="Extracting raw videos"):
        extract_frames_and_audio(video_path=video_path, video_root=video_root, frame_out=frame_out, wav_out=wav_out, 
                                fps=args.fps, size=args.size, duration=args.duration)


def extract_frames_from_video(video_path):
    import av

    container = av.open(str(video_path))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    frames = [frame.to_rgb().to_ndarray() for frame in container.decode(stream)]
    container.close()
    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")
    return np.stack(frames)


def normalize_rgb_tensor(tensor):
    if tensor.dtype != torch.uint8:
        tensor = tensor.float()
        if tensor.numel() > 0:
            min_value = float(tensor.min())
            max_value = float(tensor.max())
            if min_value >= -1.0 and max_value <= 1.0 and min_value < 0.0:
                tensor = (tensor + 1.0) * 127.5
            elif min_value >= 0.0 and max_value <= 1.0:
                tensor = tensor * 255.0
        tensor = tensor.clamp(0, 255).to(torch.uint8)
    return tensor.contiguous()


def as_rgb_video_tensor(tensor):
    if tensor.ndim != 4:
        return None
    if tensor.shape[0] == 3:
        return tensor
    if tensor.shape[1] == 3:
        return tensor.permute(1, 0, 2, 3).contiguous()
    if tensor.shape[-1] == 3:
        return tensor.permute(3, 0, 1, 2).contiguous()
    return None


def save_modality_chunks(tensor, output_clip_dir, modality, clip_len):
    tensor = normalize_rgb_tensor(tensor)
    save_dir = output_clip_dir / modality
    save_dir.mkdir(parents=True, exist_ok=True)
    _, total_frames, _, _ = tensor.shape
    for start in range(0, total_frames - clip_len + 1, clip_len):
        chunk = tensor[:, start:start + clip_len]
        if chunk.shape[1] < clip_len:
            continue
        torch.save(chunk.detach().clone().contiguous(), save_dir / f"{start:05d}.pt")


def save_rgb_tensor_chunks(tensor_path, output_clip_dir, modality, clip_len):
    tensor = torch.load(tensor_path, map_location="cpu")
    tensor = as_rgb_video_tensor(tensor)
    if tensor is None:
        return False
    save_modality_chunks(tensor, output_clip_dir, modality, clip_len)
    return True


def save_video_chunks(video_path, output_clip_dir, modality, clip_len):
    frames = extract_frames_from_video(video_path)
    tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).contiguous()
    save_modality_chunks(tensor, output_clip_dir, modality, clip_len)


def reshape_attention_chunk(chunk):
    if chunk.ndim == 4:
        return chunk.permute(1, 0, 2, 3).contiguous().to(torch.float16)
    if chunk.ndim != 3:
        raise ValueError(f"Unsupported attention shape: {tuple(chunk.shape)}")

    clip_len, num_tokens, channels = chunk.shape
    side = int(math.isqrt(num_tokens))
    if side * side != num_tokens:
        raise ValueError(f"Attention token count is not square: {num_tokens}")
    chunk = chunk.permute(0, 2, 1).reshape(clip_len, channels, side, side)
    return chunk.permute(1, 0, 2, 3).contiguous().to(torch.float16)


def save_attention_chunks(attn_path, output_clip_dir, clip_len):
    attn = torch.load(attn_path, map_location="cpu")
    save_dir = output_clip_dir / "attn_feat"
    save_dir.mkdir(parents=True, exist_ok=True)

    total_frames = attn.shape[0]
    for start in range(0, total_frames - clip_len + 1, clip_len):
        chunk = attn[start:start + clip_len]
        if chunk.shape[0] < clip_len:
            continue
        torch.save(reshape_attention_chunk(chunk), save_dir / f"{start:05d}.pt")


def find_feature_clip_dirs(feature_root):
    clip_dirs = []
    for root, _, files in os.walk(feature_root):
        names = set(files)
        has_rgb_feature = any(
            f"{modality}.mp4" in names or f"{modality}.pt" in names
            for modality in FEATURE_VIDEOS
        )
        if has_rgb_feature or "attn_feat.pt" in names:
            clip_dirs.append(Path(root))
    return sorted(clip_dirs)


def pack_feature_outputs(args):
    feature_root = Path(args.feature_dir)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    if not feature_root.is_dir():
        raise FileNotFoundError(feature_root)

    failures = []
    clip_dirs = find_feature_clip_dirs(feature_root)
    if not clip_dirs:
        print(f"No feature clip directories found under {feature_root}")
        return

    for clip_dir in tqdm(clip_dirs, desc="Packing feature outputs"):
        rel = clip_dir.relative_to(feature_root)
        output_clip_dir = output_root / rel
        output_clip_dir.mkdir(parents=True, exist_ok=True)
        try:
            for modality in FEATURE_VIDEOS:
                tensor_path = clip_dir / f"{modality}.pt"
                used_tensor = False
                if tensor_path.is_file():
                    used_tensor = save_rgb_tensor_chunks(tensor_path, output_clip_dir, modality, args.clip_len)

                video_path = clip_dir / f"{modality}.mp4"
                if not used_tensor and video_path.is_file():
                    save_video_chunks(video_path, output_clip_dir, modality, args.clip_len)

            attn_path = clip_dir / "attn_feat.pt"
            if attn_path.is_file():
                save_attention_chunks(attn_path, output_clip_dir, args.clip_len)
        except Exception as exc:
            failures.append((str(clip_dir), str(exc)))

    if failures:
        failure_path = output_root / "failed_preprocess.txt"
        with open(failure_path, "w", encoding="utf-8") as handle:
            for clip_dir, error in failures:
                handle.write(f"{clip_dir}\t{error}\n")
        print(f"[Warn] {len(failures)} clips failed. See {failure_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Raw video extraction and feature packing.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    raw = subparsers.add_parser("extract-frames", help="Convert raw videos into frame and wav.")
    raw.add_argument("--video_dir", required=True, help="Input directory containing raw videos. ex) ./MMDF")
    raw.add_argument("--frames_dir", "--frames", required=True, help="Output root. ex) ./MMDF_frames")
    raw.add_argument("--fps", type=float, default=25.0)
    raw.add_argument("--size", type=int, nargs=2, default=(512, 512), metavar=("WIDTH", "HEIGHT"))
    raw.add_argument("--duration", type=float, default=5.0, help="Optional max seconds per video. Use -1 for full videos.")

    pack = subparsers.add_parser("pack-features", help="Split feature outputs into training .pt chunks.")
    pack.add_argument("--feature_dir", required=True, help="Directory produced by hallo/extract_features.py.")
    pack.add_argument("--output_dir", required=True, help="Output root for training tensors.")
    pack.add_argument("--clip_len", type=int, default=16)

    args = parser.parse_args()
    if getattr(args, "duration", None) is not None and args.duration <= 0:
        args.duration = None
    return args


def main():
    args = parse_args()
    if args.command == "extract-frames":
        extract_raw_videos(args)
    elif args.command == "pack-features":
        pack_feature_outputs(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
