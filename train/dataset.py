import os
import glob
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Lambda
from torchvision.transforms.functional import normalize
from typing import List, Dict, Any, Tuple

def _to_float01(x: torch.Tensor) -> torch.Tensor:
    return x.float() / 255.0

class NormalizeVideo:
    """Per-frame channel-wise normalization for a 4D video tensor (C, T, H, W)."""
    def __init__(self, mean: List[float], std: List[float]):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        assert video.dim() == 4, f"Expected (C,T,H,W), got {video.shape}"
        mean = self.mean.to(video.device).view(-1)
        std = self.std.to(video.device).view(-1)
        C, T, H, W = video.shape
        assert mean.numel() == C and std.numel() == C
        out = torch.empty_like(video)
        for t in range(T):
            out[:, t, :, :] = normalize(video[:, t, :, :], mean=mean, std=std)
        return out

class InversionDataset(Dataset):
    """
    Directory layout:
        <root>/<split>/<label>/<model_id>/<clip_id>/
            original/*.pt         (3,T,H,W)
            inverted/*.pt         (3,T,H,W)
            reconstructed/*.pt    (3,T,H,W)
            residual/*.pt         (3,T,H,W)
            attn_feat/*.pt        (C,T,H,W)
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        verbose: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        assert split in ("train", "val", "test"), f"Unsupported split: {split}"

        self.split = split
        self.verbose = verbose
        self.img_channels = 12  # original + inverted + reconstructed + residual, each 3ch

        real_dir = os.path.join(data_dir, split, "real")
        fake_dir = os.path.join(data_dir, split, "fake")

        if not os.path.isdir(real_dir):
            raise FileNotFoundError(f"Real directory not found: {real_dir}")
        if not os.path.isdir(fake_dir):
            raise FileNotFoundError(f"Fake directory not found: {fake_dir}")

        all_samples = self._collect_samples(real_dir, 0) + self._collect_samples(fake_dir, 1)
        self.samples = self._flatten_samples(all_samples)

        self.transform = Compose([
            Lambda(_to_float01),
            NormalizeVideo([0.45] * self.img_channels, [0.225] * self.img_channels),
        ])

        if self.verbose:
            n_real = sum(1 for s in self.samples if s["label"] == 0)
            n_fake = len(self.samples) - n_real
            print(
                f"[InversionDataset] "
                f"root={data_dir} split={split} "
                f"N={len(self.samples)} real={n_real} fake={n_fake}"
            )

    def _collect_samples(self, root: str, label: int) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not os.path.isdir(root):
            return out

        required_modalities = ("original", "inverted", "reconstructed", "residual")

        for model_dir in sorted(glob.glob(os.path.join(root, "*"))):
            model_id = os.path.basename(model_dir)

            for clip_dir in sorted(glob.glob(os.path.join(model_dir, "*"))):
                entry: Dict[str, Any] = {"label": label, "model_id": model_id}

                for k in required_modalities:
                    kdir = os.path.join(clip_dir, k)
                    if os.path.isdir(kdir):
                        files = sorted(glob.glob(os.path.join(kdir, "*.pt")))
                        if files:
                            entry[k] = files

                attn_dir = os.path.join(clip_dir, "attn_feat")
                if os.path.isdir(attn_dir):
                    afiles = sorted(glob.glob(os.path.join(attn_dir, "*.pt")))
                    if afiles:
                        entry["attn_feat"] = afiles

                has_all_rgb = all(k in entry for k in required_modalities)
                has_attn = "attn_feat" in entry

                if has_all_rgb and has_attn:
                    out.append(entry)

        return out

    def _flatten_samples(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        flat: List[Dict[str, Any]] = []
        required_modalities = ("original", "inverted", "reconstructed", "residual")

        for e in entries:
            if any(k not in e for k in required_modalities):
                continue
            if "attn_feat" not in e:
                continue

            counts: List[int] = []
            for k in required_modalities:
                counts.append(len(e[k]))
            counts.append(len(e["attn_feat"]))

            n = min(counts)
            if n == 0:
                continue

            for i in range(n):
                item = {"label": int(e["label"]), "model_id": e["model_id"]}

                for k in required_modalities:
                    item[k] = e[k][i]

                item["attn_feat"] = e["attn_feat"][i]
                flat.append(item)

        return flat

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, str]:
        it = self.samples[idx]

        parts: List[torch.Tensor] = []
        shapes = []

        for k in ("original", "inverted", "reconstructed", "residual"):
            path = it[k]
            vid = torch.load(path)
            assert vid.dim() == 4 and vid.size(0) == 3, (f"{k} must be (3,T,H,W), got {vid.shape} ({path})")
            parts.append(vid)
            shapes.append(vid.shape)

        T, H, W = parts[0].size(1), parts[0].size(2), parts[0].size(3)

        for s in shapes[1:]:
            assert (s[1], s[2], s[3]) == (T, H, W), (f"Mismatched shapes among RGB modalities: {shapes}")

        composite = torch.cat(parts, dim=0).contiguous()  # (12,T,H,W)
        composite = self.transform(composite)

        a_path = it["attn_feat"]
        attn = torch.load(a_path)
        assert attn.dim() == 4, f"attn_feat must be (C,T,H,W), got {attn.shape} ({a_path})"
        attn = attn.float()

        return composite, attn, int(it["label"]), it["model_id"]


if __name__ == "__main__":
    from datasets import load_dataset

    ds = load_dataset("zaqxsw0526/MMDF")
    print(ds["test"][0])
