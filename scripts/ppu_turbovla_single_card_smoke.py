#!/usr/bin/env python3
"""Run a self-contained TurboVLA single-device train smoke test.

The script creates tiny randomly initialized BERT and DINOv3 backbones so that
the device/training integration can be validated without downloading released
weights.  It exercises the real TurboVLA forward path, loss, backward pass,
optimizer update, and checkpoint serialization.  It is a compatibility test,
not a model-quality benchmark.
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import BertConfig, BertModel, BertTokenizerFast
from transformers.models.dinov3_vit.configuration_dinov3_vit import DINOv3ViTConfig
from transformers.models.dinov3_vit.image_processing_dinov3_vit_fast import (
    DINOv3ViTImageProcessorFast,
)
from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTModel
from turbovla.models.configuration import (
    ActionHeadConfig,
    InteractionConfig,
    TextEncoderConfig,
    TurboVLAConfig,
    VisionEncoderConfig,
)
from turbovla.models.turbovla import TurboVLA

VOCAB = [
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
    ".",
    "?",
    "pick",
    "up",
    "the",
    "red",
    "block",
    "and",
    "place",
    "it",
    "in",
    "box",
    "move",
    "robot",
    "arm",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260803)
    return parser.parse_args()


def ensure_child(path: Path, parent: Path) -> Path:
    path = path.resolve()
    parent = parent.resolve()
    if path != parent and parent not in path.parents:
        raise ValueError(f"refusing to write outside workspace: {path}")
    return path


def prepare_tiny_backbones(model_root: Path) -> tuple[Path, Path]:
    bert_root = model_root / "tiny-random-bert"
    dino_root = model_root / "tiny-random-dinov3-vit"

    if not (bert_root / "model.safetensors").exists():
        bert_root.mkdir(parents=True, exist_ok=True)
        vocab_path = bert_root / "vocab.txt"
        vocab_path.write_text("\n".join(VOCAB) + "\n", encoding="utf-8")
        tokenizer = BertTokenizerFast(vocab_file=str(vocab_path), do_lower_case=True)
        tokenizer.save_pretrained(bert_root)
        config = BertConfig(
            vocab_size=len(VOCAB),
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=128,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            max_position_embeddings=64,
            pad_token_id=VOCAB.index("[PAD]"),
        )
        BertModel(config).save_pretrained(bert_root)

    if not (dino_root / "model.safetensors").exists():
        dino_root.mkdir(parents=True, exist_ok=True)
        config = DINOv3ViTConfig(
            image_size=32,
            patch_size=16,
            num_channels=3,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=128,
            num_register_tokens=4,
            attention_dropout=0.0,
            drop_path_rate=0.0,
            use_gated_mlp=False,
        )
        DINOv3ViTModel(config).save_pretrained(dino_root)
    if not (dino_root / "preprocessor_config.json").exists():
        DINOv3ViTImageProcessorFast(
            do_resize=True,
            size={"height": 32, "width": 32},
            do_center_crop=False,
        ).save_pretrained(dino_root)

    return bert_root, dino_root


def cuda_metric(name: str) -> int | None:
    fn = getattr(torch.cuda, name, None)
    if fn is None:
        return None
    try:
        return int(fn())
    except (AttributeError, RuntimeError):
        return None


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> int:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be positive")

    workspace = args.workspace.resolve()
    if not workspace.is_dir():
        raise FileNotFoundError(workspace)
    artifacts = ensure_child(workspace / "artifacts" / "smoke-models", workspace)
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ensure_child(workspace / "runs" / "ppu-turbovla-single-card" / run_stamp, workspace)
    run_dir.mkdir(parents=True, exist_ok=False)

    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-compatible PPU torch device is unavailable")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("this smoke test requires a CUDA-compatible PPU device")
    torch.cuda.set_device(device)

    bert_root, dino_root = prepare_tiny_backbones(artifacts)
    config = TurboVLAConfig(
        text=TextEncoderConfig(
            model_name_or_path=str(bert_root),
            max_length=16,
            padding_length=16,
            sub_sentence_present=True,
            frozen=True,
            force_eval_when_frozen=True,
            local_files_only=True,
            attention_implementation="eager",
        ),
        vision=VisionEncoderConfig(
            model_name_or_path=str(dino_root),
            image_size=32,
            num_views=3,
            position_embedding="learned_patch",
            encode_views_separately=True,
            frozen=True,
            local_files_only=True,
            attention_implementation="eager",
            compute_precision="bf16_autocast",
            dropout=0.0,
        ),
        interaction=InteractionConfig(
            hidden_dim=64,
            nheads=4,
            num_layers=2,
            dim_feedforward=128,
            enhancer_inner_dim=128,
            text_dropout=0.0,
            fusion_dropout=0.0,
            fusion_droppath=0.0,
            padding_strategy="key_padding_mask",
            residual_style="normalized",
            attention_backend="manual",
            compute_precision="bf16_autocast",
        ),
        action=ActionHeadConfig(
            action_dim=37,
            state_dim=33,
            horizon=4,
            num_state_tokens=2,
            num_layers=2,
            mlp_hidden_dim=128,
            state_hidden_dim=64,
            dropout=0.0,
        ),
    )

    model = TurboVLA(config).to(device)
    model.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=2e-3, weight_decay=0.0)

    pixels = torch.randn(1, 3, 3, 32, 32, device=device)
    state = torch.randn(1, 33, device=device)
    target = torch.tanh(torch.randn(1, 4, 37, device=device))
    instructions = ["pick up the red block and place it in the box ."]

    tracked_name, tracked_parameter = next(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.ndim >= 2
    )
    tracked_before = tracked_parameter.detach().float().cpu().clone()

    peak_memory_reset = True
    try:
        torch.cuda.reset_peak_memory_stats(device)
    except RuntimeError:
        peak_memory_reset = False
    step_metrics: list[dict[str, Any]] = []
    started = time.perf_counter()
    output = None
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        synchronize(device)
        step_started = time.perf_counter()
        output = model(instructions=instructions, samples=pixels, state=state)
        loss = torch.nn.functional.mse_loss(output.float(), target.float())
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=10.0)
        if not torch.isfinite(loss) or not torch.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite training values: loss={loss}, grad_norm={grad_norm}")
        optimizer.step()
        synchronize(device)
        step_metrics.append(
            {
                "step": step + 1,
                "loss": float(loss.detach().cpu()),
                "grad_norm": float(grad_norm.detach().cpu()),
                "elapsed_seconds": time.perf_counter() - step_started,
            }
        )

    total_elapsed = time.perf_counter() - started
    tracked_delta = float((tracked_parameter.detach().float().cpu() - tracked_before).abs().max())
    if tracked_delta <= 0:
        raise RuntimeError(f"optimizer did not update tracked parameter {tracked_name}")
    assert output is not None
    if tuple(output.shape) != (1, 4, 37):
        raise RuntimeError(f"unexpected action output shape: {tuple(output.shape)}")

    properties = torch.cuda.get_device_properties(device)
    report: dict[str, Any] = {
        "status": "passed",
        "test_kind": "synthetic_compatibility_smoke_not_quality_benchmark",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "device": str(device),
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(device),
        "device_total_memory_bytes": int(properties.total_memory),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "frozen_backbones": {"bert": True, "dinov3": True},
        "input": {
            "batch": 1,
            "views": 3,
            "image_shape": [3, 32, 32],
            "state_dim": 33,
            "action_dim": 37,
            "horizon": 4,
        },
        "output_shape": list(output.shape),
        "steps": step_metrics,
        "total_elapsed_seconds": total_elapsed,
        "tracked_parameter": tracked_name,
        "tracked_parameter_max_delta": tracked_delta,
        "max_memory_allocated_bytes": cuda_metric("max_memory_allocated"),
        "max_memory_reserved_bytes": cuda_metric("max_memory_reserved"),
        "peak_memory_reset": peak_memory_reset,
        "config": config.to_dict(),
    }

    checkpoint_path = run_dir / "checkpoint.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "report": report,
        },
        checkpoint_path,
    )
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if set(loaded["model"]) != set(model.state_dict()):
        raise RuntimeError("checkpoint round-trip state keys differ")
    report["checkpoint"] = {
        "path": str(checkpoint_path),
        "size_bytes": checkpoint_path.stat().st_size,
        "round_trip_verified": True,
    }
    report_path = run_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"REPORT_PATH={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
