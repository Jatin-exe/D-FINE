"""Full D-FINE model using TTNN backend components."""

from __future__ import annotations
from pathlib import Path
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import ttnn

from .ttnn_utils import to_torch_tensor
from .hgnetv2_ttnn_manual import HGNetv2TTNNManual, TTActivation
from .hybrid_encoder_ttnn import HybridEncoderTTNN
from .dfine_decoder_ttnn import DFINETransformerTTNN
from .weight_store import TTNNWeightStore, ModuleKeyRegistry


class DFINE_TTNN(nn.Module):
    """TTNN-accelerated D-FINE model wrapping backbone, encoder, and decoder with preprocessing/postprocessing."""

    def __init__(
        self,
        model_pt: nn.Module,
        device_id: int = 0,
        debug_dump_dir: Optional[Path] = None,
        postprocessor: Optional[nn.Module] = None,
        weight_store_dir: Optional[Path] = None,
        return_ttnn: Optional[bool] = None,
    ):
        super().__init__()
        # Copy input_size from PyTorch model or use default
        self.input_size = getattr(model_pt, 'input_size', (640, 640))
        self.weight_store = None
        self.registry = None
        weight_dir = Path(weight_store_dir) if weight_store_dir is not None else None
        if weight_dir is None:
            env_dir = os.environ.get("TTNN_WEIGHT_DIR")
            if env_dir:
                weight_dir = Path(env_dir)
        if weight_dir is None:
            weight_dir = Path("weight/ttnn_store_single")
        if weight_dir is None:
            raise RuntimeError(
                "TTNN weight store is required. Set TTNN_WEIGHT_DIR or pass weight_store_dir; "
                "export with tools/export/ttnn_export_weights.py."
            )
        manifest = weight_dir / "manifest.json"
        mode = os.environ.get("TTNN_WEIGHT_STORE_MODE")
        if mode is None:
            mode = "load" if manifest.exists() else None
        if mode is None:
            raise RuntimeError(
                "TTNN weight store not found. Run tools/export/ttnn_export_weights.py "
                "or set TTNN_WEIGHT_STORE_MODE=save to create it."
            )
        if mode == "load" and not manifest.exists():
            raise RuntimeError(f"TTNN weight store missing manifest: {manifest}")
        if mode not in ("load", "save"):
            raise RuntimeError(f"Unsupported TTNN weight store mode: {mode}")
        self.weight_store = TTNNWeightStore(weight_dir, mode=mode)
        self.registry = ModuleKeyRegistry(model_pt)

        self.backbone_tt = HGNetv2TTNNManual(
            model_pt.backbone,
            device_id=device_id,
            debug_dump_dir=debug_dump_dir,
            weight_store=self.weight_store,
            registry=self.registry,
        )
        self.encoder_tt = HybridEncoderTTNN(
            model_pt.encoder,
            device=self.backbone_tt.device,
            weight_store=self.weight_store,
            registry=self.registry,
        )
        self.decoder_tt = DFINETransformerTTNN(
            model_pt.decoder,
            device=self.backbone_tt.device,
            weight_store=self.weight_store,
            registry=self.registry,
        )

        # Store postprocessor for end-to-end inference
        self.postprocessor = postprocessor.deploy() if postprocessor is not None else None

        # Enable model cache + warmup to avoid cold-start outliers
        ttnn.CONFIG.enable_model_cache = True
        self.return_ttnn = bool(int(os.environ.get("TTNN_RETURN_TT", "0"))) if return_ttnn is None else bool(return_ttnn)
        if os.environ.get("TTNN_WARMUP", "1") == "1":
            if not getattr(self, "_did_warmup", False):
                self._did_warmup = True
                warmup_size = os.environ.get("TTNN_WARMUP_SIZE")
                if warmup_size:
                    size_val = int(warmup_size)
                    hw = (size_val, size_val)
                else:
                    hw = tuple(self.input_size)
                dummy = torch.zeros((1, 3, *hw), dtype=torch.float32)
                _ = self.forward(dummy)
        if self.weight_store is not None and self.weight_store.mode == "save":
            self.weight_store.flush()

    @staticmethod
    def _env_float(name: str, default: float = 1.0) -> float:
        raw = os.environ.get(name)
        if raw is None:
            return float(default)
        return float(raw)

    def preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load and preprocess image from path."""
        im = Image.open(image_path).convert("RGB")
        w, h = im.size
        orig_size = torch.tensor([[w, h]], dtype=torch.float32)
        tensor = T.Compose([
            T.Resize(self.input_size),
            T.ToTensor()
        ])(im).unsqueeze(0)
        return tensor, orig_size

    def forward(self, x: Union[torch.Tensor, str], score_threshold: Optional[float] = None, return_ttnn: Optional[bool] = None) -> Dict[str, torch.Tensor]:
        """
        End-to-end forward pass through backbone -> encoder -> decoder.

        Args:
            x: Either a tensor [1, 3, H, W] or a path to an image file
            score_threshold: Optional threshold for postprocessing

        Returns:
            If x is a tensor and postprocessor is None: decoder outputs (logits, boxes)
            If x is a string path: postprocessed detections (labels, boxes, scores)
        """
        # Handle image path input
        if isinstance(x, str):
            tensor, orig_size = self.preprocess_image(x)

            # Run model
            bb_feats = self.backbone_tt(tensor)
            enc_feats = self.encoder_tt(bb_feats)
            dec_out = self.decoder_tt(enc_feats)
            if not isinstance(dec_out["pred_logits"], torch.Tensor):
                dec_out = {
                    "pred_logits": to_torch_tensor(dec_out["pred_logits"], expected_shape=dec_out["pred_logits"].shape),
                    "pred_boxes": to_torch_tensor(dec_out["pred_boxes"], expected_shape=dec_out["pred_boxes"].shape),
                }
                logit_scale = self._env_float("TTNN_LOGIT_SCALE", 1.0)
                if logit_scale != 1.0:
                    dec_out["pred_logits"] = dec_out["pred_logits"] * logit_scale

            # Postprocess if available
            if self.postprocessor is not None:
                labels, boxes, scores = self.postprocessor(dec_out, orig_size)

                # Filter by score threshold if provided
                if score_threshold is not None:
                    mask = scores[0] > score_threshold
                    labels = labels[0][mask].unsqueeze(0)
                    boxes = boxes[0][mask].unsqueeze(0)
                    scores = scores[0][mask].unsqueeze(0)

                return {
                    "labels": labels[0],
                    "boxes": boxes[0],
                    "scores": scores[0]
                }
            else:
                return dec_out
        else:
            # Tensor input - just run model
            bb_feats = self.backbone_tt(x)
            enc_feats = self.encoder_tt(bb_feats)
            dec_out = self.decoder_tt(enc_feats)
            return_ttnn_flag = self.return_ttnn if return_ttnn is None else bool(return_ttnn)
            if return_ttnn_flag:
                logit_scale = self._env_float("TTNN_LOGIT_SCALE", 1.0)
                if logit_scale != 1.0:
                    dec_out["pred_logits"] = ttnn.multiply(dec_out["pred_logits"], logit_scale)
                return dec_out
            if not isinstance(dec_out["pred_logits"], torch.Tensor):
                dec_out = {
                    "pred_logits": to_torch_tensor(dec_out["pred_logits"], expected_shape=dec_out["pred_logits"].shape),
                    "pred_boxes": to_torch_tensor(dec_out["pred_boxes"], expected_shape=dec_out["pred_boxes"].shape),
                }
                logit_scale = self._env_float("TTNN_LOGIT_SCALE", 1.0)
                if logit_scale != 1.0:
                    dec_out["pred_logits"] = dec_out["pred_logits"] * logit_scale
            return dec_out

    def forward_ttnn(self, x_act: TTActivation) -> Dict[str, "ttnn.Tensor"]:
        """Run full model on an already-uploaded TTActivation (TTNN-only path)."""
        bb_feats = self.backbone_tt(x_act)
        enc_feats = self.encoder_tt(bb_feats)
        return self.decoder_tt(enc_feats)

    def enable_trace_mode(self, enabled: bool = True, batch: int = 1, strict: bool = True) -> None:
        """Enable trace capture preparation. When strict=False, only precompute decoder trace buffers."""
        if strict:
            if hasattr(self.backbone_tt, "enable_trace_mode"):
                self.backbone_tt.enable_trace_mode(enabled)
            if hasattr(self.encoder_tt, "enable_trace_mode"):
                self.encoder_tt.enable_trace_mode(enabled)
        if hasattr(self.decoder_tt, "enable_trace_mode"):
            self.decoder_tt.enable_trace_mode(enabled, batch=batch, strict=strict)

    def close(self):
        """Release TTNN device resources."""
        self.encoder_tt.close()
        self.decoder_tt.close()
        self.backbone_tt.close()
