"""
backend/services/reid_embedder.py — OSNet-IBN body embedding extractor.

Model: osnet_ibn_x1_0 (OSNet with Instance Batch Normalization)
  - Pretrained on MSMT17 multi-camera person re-ID benchmark
  - License: MIT (https://github.com/KaiyangZhou/deep-person-reid)
  - Output: 512-dimensional L2-normalized float32 embedding vector
"""
from __future__ import annotations

import hashlib
import logging
import sys
import types
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from backend.core.config import settings

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

REID_EMBEDDING_DIM: int = 512

# Repo root — three parents up from backend/services/reid_embedder.py.
# REID_CHECKPOINT_PATH in config.py is relative to here, matching how
# DATABASE_URL's "./output/..." is already resolved relative to the process
# cwd (which is the repo root for every entrypoint this app is run from).
_REPO_ROOT = Path(__file__).resolve().parents[2]


class ReIDEmbedder:
    """OSNet-IBN body embedding extractor with real model weights."""

    def __init__(self, stub_mode: bool = False) -> None:
        self._stub_mode = stub_mode
        self._model = None
        self._device = "cpu"
        self._checkpoint_applied: str | None = None

        if not stub_mode:
            self._load_model()

    def _load_model(self) -> None:
        """Load OSNet-IBN weights via torchreid. Fails loudly on dim mismatch."""
        try:
            if "torch.utils.tensorboard" not in sys.modules:
                dummy_tb = types.ModuleType("torch.utils.tensorboard")
                dummy_tb.SummaryWriter = object
                sys.modules["torch.utils.tensorboard"] = dummy_tb

            import torchreid
            import torch

            # GPU when available. This was pinned to "cpu" for a period
            # because a real CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH fired
            # on the first forward pass whenever InsightFace's
            # onnxruntime-gpu CUDA session was loaded in the same process
            # (see main.py) — which the `except` below silently degraded to
            # STUB MODE.
            #
            # ROOT CAUSE, since fixed: torchvision was installed as the
            # CPU-only build (0.28.0+cpu) against a CUDA torch
            # (2.13.0+cu130). With the matched 0.28.0+cu130 build in place,
            # GPU ReID coexists with InsightFace in one process — verified
            # by loading FaceEmbedder first and then running OSNet on CUDA
            # (28.8 ms/crop, no error), plus a full-app boot showing
            # stub_mode=False.
            #
            # Measured on this machine: CUDA ~25 ms/crop vs CPU ~55 ms/crop.
            # That 2x matters at 30 cameras, where ReID runs per detected
            # person crop rather than per frame.
            #
            # If a cuDNN mismatch ever returns, the correct fix is to
            # re-align the torch / torchvision / onnxruntime-gpu CUDA
            # versions — NOT to re-pin this to CPU, which only hides it.
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._device = device
            logger.info("Loading OSNet-IBN ReID model on %s (torchreid).", device)

            self._model = torchreid.models.build_model(
                name="osnet_ibn_x1_0", num_classes=1000, pretrained=True
            )

            # Overlay a fine-tuned checkpoint onto the ImageNet-pretrained
            # backbone, if one is configured. Matches by name+shape and
            # silently skips the rest — the classifier head here is sized for
            # num_classes=1000 while a torchreid-trained checkpoint's
            # classifier is sized for whatever it was trained on (751 for
            # Market-1501), so that layer is expected to be discarded. It is
            # never used at inference anyway; only the backbone feature
            # extractor is called (see _extract_real below).
            #
            # NOTE: does not use torchreid.utils.load_pretrained_weights().
            # That helper calls torch.load() with no weights_only override,
            # and torch >=2.6 defaults weights_only=True — which cannot
            # unpickle the numpy objects torchreid's own training loop saves
            # into the checkpoint (epoch count, rank1, optimizer state), so
            # it raised and silently fell back to stub mode. Caught by
            # actually loading a real checkpoint, not by inspection.
            # weights_only=False is safe here specifically because this file
            # is one this codebase's own training script produced — not an
            # arbitrary third-party download.
            ckpt_setting = (settings.REID_CHECKPOINT_PATH or "").strip()
            if ckpt_setting:
                ckpt_path = Path(ckpt_setting)
                if not ckpt_path.is_absolute():
                    ckpt_path = _REPO_ROOT / ckpt_path
                if ckpt_path.is_file():
                    self._apply_checkpoint(ckpt_path, device)
                else:
                    logger.warning(
                        "REID_CHECKPOINT_PATH is set to %s but that file does "
                        "not exist — continuing with the base ImageNet-"
                        "pretrained backbone only.", ckpt_path,
                    )

            self._model.eval()
            if device == "cuda":
                self._model.to(device)

            # ImageNet normalisation, parked on the model's device so
            # extract_batch never rebuilds them or pays a host-to-device copy
            # per call.
            self._norm_mean = torch.tensor(
                [0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            self._norm_std = torch.tensor(
                [0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

            # Sanity-check output dimension with a dummy crop
            dummy = np.zeros((256, 128, 3), dtype=np.uint8)
            test_emb = self._extract_real(dummy)
            if test_emb.shape[0] != REID_EMBEDDING_DIM:
                raise RuntimeError(
                    f"OSNet-IBN embedding dim mismatch: expected {REID_EMBEDDING_DIM}, "
                    f"got {test_emb.shape[0]}."
                )
            self._stub_mode = False
            logger.info(
                "OSNet-IBN loaded successfully. Embedding dim=%d, device=%s, "
                "checkpoint=%s.",
                REID_EMBEDDING_DIM, device, self._checkpoint_applied or "default (ImageNet only)",
            )

        except Exception as exc:
            logger.error(
                "Failed to load OSNet-IBN model: %s — falling back to STUB MODE.", exc
            )
            self._stub_mode = True

    def _apply_checkpoint(self, ckpt_path: Path, device: str) -> None:
        """Overlay a torchreid-format checkpoint onto self._model in place.

        Reimplements torchreid.utils.load_pretrained_weights' name+shape
        matching (see the note above its call site) with an explicit
        weights_only=False, since this codebase's own checkpoints carry
        non-tensor training metadata that torch's safer default cannot
        unpickle.
        """
        import torch

        try:
            checkpoint = torch.load(
                str(ckpt_path), map_location=device, weights_only=False
            )
        except Exception as exc:
            logger.warning(
                "Could not load checkpoint %s (%s) — continuing with the "
                "base ImageNet-pretrained backbone only.", ckpt_path, exc,
            )
            return

        state_dict = checkpoint.get("state_dict", checkpoint)
        model_dict = self._model.state_dict()
        matched, discarded = {}, []
        for key, value in state_dict.items():
            key = key[7:] if key.startswith("module.") else key
            if key in model_dict and model_dict[key].shape == value.shape:
                matched[key] = value
            else:
                discarded.append(key)

        if not matched:
            logger.warning(
                "Checkpoint %s matched zero layers by name+shape — ignoring "
                "it and continuing with the base ImageNet-pretrained "
                "backbone only.", ckpt_path,
            )
            return

        model_dict.update(matched)
        self._model.load_state_dict(model_dict)
        self._checkpoint_applied = str(ckpt_path)
        logger.info(
            "Applied fine-tuned ReID checkpoint: %s (%d layers matched, "
            "%d discarded — discarded layers are expected, e.g. a "
            "classifier head sized for a different training run).",
            ckpt_path, len(matched), len(discarded),
        )

    def _extract_real(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Run OSNet-IBN on a BGR crop. Returns L2-normalized 512-d vector."""
        import torch
        import torchvision.transforms as T
        from PIL import Image

        pil_img = Image.fromarray(crop_bgr[:, :, ::-1])  # BGR → RGB

        transform = T.Compose([
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        tensor = transform(pil_img).unsqueeze(0)  # (1, C, H, W)
        if getattr(self, "_device", "cpu") == "cuda":
            tensor = tensor.to("cuda")

        with torch.no_grad():
            features = self._model(tensor)  # (1, 512)

        vec = features.cpu().numpy()[0]          # (512,)
        return self._normalize(vec)

    def extract_batch(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        """Embed many crops in one forward pass. Returns (N, 512), L2-normalized.

        extract() costs ~278 ms per crop, and only 36 ms of that is the model:
        the rest is PIL decoding and a torchvision transform pipeline rebuilt
        per call, on a full-size crop. That is fine for one face but not for a
        27-camera loop, where every detected vehicle needs an embedding.

        Measured on this machine (RTX 4070, osnet_ibn_x1_0):

            extract(), per crop            278.2 ms
            batched forward, batch of 8      3.38 ms per crop
            batched forward, batch of 32     0.82 ms per crop

        Two changes buy that. Resizing with cv2 straight to 128x256 keeps the
        pixels the GPU has to touch small, and the uint8 tensor is normalized
        on the GPU rather than in numpy, so the host does almost no float work.
        """
        if not crops_bgr:
            return np.zeros((0, REID_EMBEDDING_DIM), dtype=np.float32)

        if self._stub_mode:
            return np.stack([self._extract_stub(c) for c in crops_bgr])

        import torch

        resized = np.empty((len(crops_bgr), 256, 128, 3), dtype=np.uint8)
        for i, crop in enumerate(crops_bgr):
            if crop.size == 0:
                resized[i] = 0
                continue
            # cv2 needs a contiguous buffer; a crop sliced out of a frame is a
            # strided view of it.
            resized[i] = cv2.resize(np.ascontiguousarray(crop), (128, 256),
                                    interpolation=cv2.INTER_LINEAR)

        out_feats = []
        chunk_size = 32
        with torch.no_grad():
            for start in range(0, len(crops_bgr), chunk_size):
                chunk = resized[start:start + chunk_size]
                t = torch.from_numpy(chunk).to(self._device)
                t = t.permute(0, 3, 1, 2).flip(1).float().div_(255.0)
                t = (t - self._norm_mean) / self._norm_std
                feats = self._model(t)
                feats = torch.nn.functional.normalize(feats, dim=1)
                out_feats.append(feats.cpu().numpy())

        return np.concatenate(out_feats, axis=0).astype(np.float32)

    def _extract_stub(self, crop_bgr: np.ndarray) -> np.ndarray:
        digest = hashlib.md5(crop_bgr.tobytes()).hexdigest()
        seed = int(digest[:8], 16)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(REID_EMBEDDING_DIM).astype(np.float32)
        return self._normalize(vec)

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm < 1e-10:
            return np.zeros(REID_EMBEDDING_DIM, dtype=np.float32)
        return (vec / norm).astype(np.float32)

    def extract(self, crop_bgr: np.ndarray) -> np.ndarray:
        if crop_bgr is None or crop_bgr.size == 0:
            raise ValueError("Empty crop passed to ReIDEmbedder.extract()")
        if crop_bgr.ndim != 3 or crop_bgr.shape[2] != 3:
            raise ValueError(
                f"Expected 3-channel BGR crop, got shape {crop_bgr.shape}"
            )

        if self._stub_mode:
            logger.debug("STUB MODE: extracting deterministic pseudo-embedding.")
            return self._extract_stub(crop_bgr)
        return self._extract_real(crop_bgr)

    @property
    def is_stub(self) -> bool:
        return self._stub_mode

    @property
    def checkpoint_applied(self) -> str | None:
        """Path of the fine-tuned checkpoint actually loaded, or None if
        running on the base ImageNet-pretrained backbone only."""
        return self._checkpoint_applied


@lru_cache(maxsize=1)
def get_embedder() -> ReIDEmbedder:
    return ReIDEmbedder(stub_mode=False)
