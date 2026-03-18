#!/usr/bin/env python3
"""
visualize_attention_episode.py

Standalone static visualization runner for a single LIBERO task / episode / set of steps.
It reuses the existing OpenVLA inference path and only reads model-side caches.
"""

import argparse
import json
import math
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parent.parent
OPENVLA_ROOT = REPO_ROOT / "src" / "openvla"
LIBERO_ROOT = OPENVLA_ROOT / "LIBERO"

if str(OPENVLA_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENVLA_ROOT))
if str(LIBERO_ROOT) not in sys.path:
    sys.path.insert(0, str(LIBERO_ROOT))

from libero.libero import benchmark

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)


DEFAULT_STEPS = [10, 12, 13, 15]
TOP_LEVELS = (
    ("top 40%", 0.40, (46, 196, 182, 72)),
    ("top 20%", 0.20, (59, 130, 246, 112)),
    ("top 10%", 0.10, (220, 38, 38, 156)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize OpenVLA/VLA-Pruner attention for one LIBERO episode.")
    parser.add_argument("--pretrained_checkpoint", required=True)
    parser.add_argument("--task_suite_name", required=True)
    parser.add_argument("--task_name", default=None)
    parser.add_argument("--task_index", type=int, default=None)
    parser.add_argument(
        "--mode",
        required=True,
        choices=["baseline", "fastv", "vlapruner_prefill", "vlapruner_nonprefill"],
    )
    parser.add_argument("--episode_idx", type=int, default=0)
    parser.add_argument("--steps", nargs="*", type=int, default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--center_crop", action="store_true", default=True)
    parser.add_argument("--fastv_k", type=int, default=3)
    parser.add_argument("--fastv_r", type=float, default=0.5)
    parser.add_argument("--fastv_image_token_start_index", type=int, default=1)
    parser.add_argument("--fastv_image_token_length", type=int, default=256)
    parser.add_argument("--temporal_w", type=int, default=3)
    parser.add_argument("--temporal_gamma", type=float, default=0.8)
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--render_full_episode", action="store_true", default=False)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--save_frames", action="store_true", default=False)
    parser.add_argument("--save_debug_assets", action="store_true", default=False)
    parser.add_argument("--video_format", choices=["mp4", "gif"], default="mp4")
    parser.add_argument("--fps", type=int, default=5)
    return parser.parse_args()


def build_cfg(args: argparse.Namespace) -> SimpleNamespace:
    mode_flags = {
        "baseline": dict(use_fastv=False, use_temporal=False, use_prefil_attention=False),
        "fastv": dict(use_fastv=True, use_temporal=False, use_prefil_attention=False),
        "vlapruner_prefill": dict(use_fastv=True, use_temporal=True, use_prefil_attention=True),
        "vlapruner_nonprefill": dict(use_fastv=True, use_temporal=True, use_prefil_attention=False),
    }[args.mode]
    return SimpleNamespace(
        model_family="openvla",
        pretrained_checkpoint=args.pretrained_checkpoint,
        load_in_8bit=False,
        load_in_4bit=False,
        center_crop=args.center_crop,
        task_suite_name=args.task_suite_name,
        num_steps_wait=args.num_steps_wait,
        num_trials_per_task=1,
        run_id_note=f"visualize-{args.mode}",
        local_log_dir=str(REPO_ROOT / "experiments" / "logs"),
        use_wandb=False,
        wandb_project="",
        wandb_entity="",
        seed=args.seed,
        use_text_vision_selection=False,
        sparsevlm=False,
        fastv_k=args.fastv_k,
        fastv_r=args.fastv_r,
        fastv_image_token_start_index=args.fastv_image_token_start_index,
        fastv_image_token_length=args.fastv_image_token_length,
        temporal_w=args.temporal_w,
        temporal_gamma=args.temporal_gamma,
        **mode_flags,
    )


def get_max_steps(task_suite_name: str) -> int:
    if task_suite_name == "libero_spatial":
        return 220
    if task_suite_name == "libero_object":
        return 280
    if task_suite_name == "libero_goal":
        return 300
    if task_suite_name == "libero_10":
        return 520
    if task_suite_name == "libero_90":
        return 400
    raise ValueError(f"Unsupported task suite: {task_suite_name}")


def resolve_task(task_suite, task_name: str | None, task_index: int | None):
    if task_index is not None:
        return task_index, task_suite.get_task(task_index)
    if task_name is None:
        raise ValueError("Either --task_index or --task_name must be provided.")
    target = task_name.strip().lower()
    for idx in range(task_suite.n_tasks):
        task = task_suite.get_task(idx)
        if task.language.strip().lower() == target:
            return idx, task
    raise ValueError(f"Task name not found in suite {task_suite.__class__.__name__}: {task_name}")


def infer_grid_size(num_tokens: int) -> int | None:
    side = int(round(math.sqrt(num_tokens)))
    return side if side * side == num_tokens else None


def tensor_to_vector(tensor: torch.Tensor | None, reduce_dims: tuple[int, ...]) -> np.ndarray | None:
    if tensor is None:
        return None
    arr = tensor.float()
    for dim in sorted(reduce_dims, reverse=True):
        arr = arr.mean(dim=dim)
    return arr.detach().cpu().numpy()


def normalize_map(values: np.ndarray | None) -> np.ndarray | None:
    if values is None:
        return None
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return None
    vmin = float(values.min())
    vmax = float(values.max())
    if math.isclose(vmin, vmax):
        return np.zeros_like(values, dtype=np.float32)
    return (values - vmin) / (vmax - vmin)


def make_continuous_overlay(rgb: np.ndarray, values: np.ndarray | None) -> Image.Image | None:
    norm = normalize_map(values)
    if norm is None:
        return None
    side = infer_grid_size(norm.size)
    if side is None:
        return None
    heat = Image.fromarray((norm.reshape(side, side) * 255).astype(np.uint8), mode="L").resize(
        (rgb.shape[1], rgb.shape[0]), resample=Image.Resampling.BILINEAR
    )
    heat_np = np.asarray(heat, dtype=np.float32) / 255.0
    rgb_img = Image.fromarray(rgb.astype(np.uint8)).convert("RGBA")
    overlay = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
    overlay[..., 0] = np.clip(255 * heat_np, 0, 255).astype(np.uint8)
    overlay[..., 1] = np.clip(170 * heat_np, 0, 255).astype(np.uint8)
    overlay[..., 2] = np.clip(40 * heat_np, 0, 255).astype(np.uint8)
    overlay[..., 3] = np.clip(160 * heat_np, 0, 255).astype(np.uint8)
    return Image.alpha_composite(rgb_img, Image.fromarray(overlay, mode="RGBA")).convert("RGB")


def make_patch_overlay(rgb: np.ndarray, values: np.ndarray | None) -> tuple[Image.Image | None, str | None]:
    norm = normalize_map(values)
    if norm is None:
        return None, "unavailable"
    side = infer_grid_size(norm.size)
    if side is None:
        return None, "non-square token grid"
    image = Image.fromarray(rgb.astype(np.uint8)).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    patch_h = image.size[1] / side
    patch_w = image.size[0] / side
    order = np.argsort(norm)[::-1]
    for _, top_ratio, color in TOP_LEVELS:
        top_k = max(1, int(round(norm.size * top_ratio)))
        indices = order[:top_k]
        for idx in indices:
            row, col = divmod(int(idx), side)
            x0 = int(round(col * patch_w))
            y0 = int(round(row * patch_h))
            x1 = int(round((col + 1) * patch_w))
            y1 = int(round((row + 1) * patch_h))
            draw.rectangle([x0, y0, x1, y1], fill=color, outline=(255, 255, 255, 40))
    return Image.alpha_composite(image, overlay).convert("RGB"), None


def make_binary_mask(rgb: np.ndarray, mask_values: np.ndarray | None, color: tuple[int, int, int, int]) -> tuple[Image.Image | None, str | None]:
    if mask_values is None:
        return None, "unavailable"
    mask_values = np.asarray(mask_values, dtype=np.float32)
    side = infer_grid_size(mask_values.size)
    if side is None:
        return None, "non-square token grid"
    image = Image.fromarray(rgb.astype(np.uint8)).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    patch_h = image.size[1] / side
    patch_w = image.size[0] / side
    for idx, value in enumerate(mask_values):
        if value <= 0:
            continue
        row, col = divmod(int(idx), side)
        x0 = int(round(col * patch_w))
        y0 = int(round(row * patch_h))
        x1 = int(round((col + 1) * patch_w))
        y1 = int(round((row + 1) * patch_h))
        draw.rectangle([x0, y0, x1, y1], fill=color, outline=(255, 255, 255, 70))
    return Image.alpha_composite(image, overlay).convert("RGB"), None


def build_keep_prune_masks(pruning_info: dict | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not isinstance(pruning_info, dict):
        return None, None
    original_len = pruning_info.get("original_image_token_length")
    kept_len = pruning_info.get("kept_image_token_length")
    kept_indices = pruning_info.get("kept_indices")
    if original_len is None or kept_indices is None or kept_len is None:
        return None, None
    original_len = int(original_len)
    keep_mask = np.zeros(original_len, dtype=np.float32)
    start = int(pruning_info.get("image_token_start_index", 1))
    if isinstance(kept_indices, torch.Tensor):
        kept_indices = kept_indices.detach().cpu().numpy()
    for idx in kept_indices:
        rel = int(idx) - start
        if 0 <= rel < original_len:
            keep_mask[rel] = 1.0
    prune_mask = 1.0 - keep_mask
    return keep_mask, prune_mask


def draw_centered_multiline(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, fill=(0, 0, 0)) -> None:
    lines = text.split("\n")
    line_height = 16
    total_h = len(lines) * line_height
    y = box[1] + max(0, (box[3] - box[1] - total_h) // 2)
    for line in lines:
        bbox = draw.textbbox((0, 0), line)
        text_w = bbox[2] - bbox[0]
        x = box[0] + max(0, (box[2] - box[0] - text_w) // 2)
        draw.text((x, y), line, fill=fill)
        y += line_height


def wrap_text_block(lines: list[str], width: int) -> str:
    wrapped = []
    for line in lines:
        wrapped.extend(textwrap.wrap(line, width=width) or [""])
    return "\n".join(wrapped)


def make_panel_tile(label: str, image: Image.Image | None, unavailable_reason: str | None, tile_w: int, tile_h: int) -> Image.Image:
    card = Image.new("RGB", (tile_w, tile_h), (252, 252, 252))
    draw = ImageDraw.Draw(card)
    draw.rounded_rectangle([0, 0, tile_w - 1, tile_h - 1], radius=14, outline=(210, 210, 215), width=2, fill=(252, 252, 252))
    draw.rounded_rectangle([10, 10, tile_w - 10, 36], radius=10, fill=(240, 243, 248))
    draw.text((18, 17), label, fill=(20, 20, 20))
    content_box = (12, 48, tile_w - 12, tile_h - 12)
    if image is not None:
        content = image.resize((content_box[2] - content_box[0], content_box[3] - content_box[1]))
        card.paste(content, (content_box[0], content_box[1]))
    else:
        draw.rounded_rectangle(content_box, radius=10, fill=(236, 238, 243))
        draw_centered_multiline(draw, content_box, unavailable_reason or "unavailable", fill=(90, 96, 110))
    return card


def make_legend_tile(tile_w: int, tile_h: int) -> Image.Image:
    card = Image.new("RGB", (tile_w, tile_h), (252, 252, 252))
    draw = ImageDraw.Draw(card)
    draw.rounded_rectangle([0, 0, tile_w - 1, tile_h - 1], radius=14, outline=(210, 210, 215), width=2, fill=(252, 252, 252))
    draw.rounded_rectangle([10, 10, tile_w - 10, 36], radius=10, fill=(240, 243, 248))
    draw.text((18, 17), "Legend", fill=(20, 20, 20))
    y = 64
    for label, _, color in reversed(TOP_LEVELS):
        draw.rounded_rectangle([24, y, 64, y + 24], radius=6, fill=color)
        draw.text((80, y + 4), label, fill=(35, 35, 35))
        y += 40
    draw.text((24, y + 10), "Patch overlay = nested top-k blocks", fill=(70, 75, 85))
    return card


def build_panel(
    title_lines: list[str],
    metadata_lines: list[str],
    tiles: list[tuple[str, Image.Image | None, str | None]],
) -> Image.Image:
    tile_w, tile_h = 300, 300
    cols = 4
    rows = 2
    pad = 18
    header_h = 88
    footer_h = 88
    panel_w = cols * tile_w + (cols + 1) * pad
    panel_h = header_h + rows * tile_h + (rows + 1) * pad + footer_h
    panel = Image.new("RGB", (panel_w, panel_h), (245, 247, 250))
    draw = ImageDraw.Draw(panel)

    draw.rounded_rectangle([pad, pad, panel_w - pad, header_h], radius=18, fill=(255, 255, 255), outline=(220, 224, 230))
    header_text = wrap_text_block(title_lines, width=62)
    draw_centered_multiline(draw, (pad + 12, pad + 8, panel_w - pad - 12, header_h - 8), header_text, fill=(18, 18, 18))

    extended_tiles = tiles + [("Legend", make_legend_tile(tile_w, tile_h), None)]
    while len(extended_tiles) < cols * rows:
        extended_tiles.append(("Reserved", None, "reserved"))

    start_y = header_h + pad
    for idx, (label, image, unavailable_reason) in enumerate(extended_tiles[: cols * rows]):
        row = idx // cols
        col = idx % cols
        x = pad + col * (tile_w + pad)
        y = start_y + row * (tile_h + pad)
        tile = image if label == "Legend" and image is not None else make_panel_tile(label, image, unavailable_reason, tile_w, tile_h)
        panel.paste(tile, (x, y))

    footer_y0 = panel_h - footer_h - pad
    draw.rounded_rectangle([pad, footer_y0, panel_w - pad, panel_h - pad], radius=18, fill=(255, 255, 255), outline=(220, 224, 230))
    midpoint = int(math.ceil(len(metadata_lines) / 2))
    footer_text = wrap_text_block(
        [" | ".join(metadata_lines[:midpoint]), " | ".join(metadata_lines[midpoint:])],
        width=88,
    )
    draw_centered_multiline(draw, (pad + 12, footer_y0 + 8, panel_w - pad - 12, panel_h - pad - 8), footer_text, fill=(45, 50, 60))
    return panel


def write_json(path: Path, payload: dict) -> None:
    serializable = {}
    for key, value in payload.items():
        if isinstance(value, torch.Tensor):
            serializable[key] = value.detach().cpu().tolist()
        elif isinstance(value, np.ndarray):
            serializable[key] = value.tolist()
        else:
            serializable[key] = value
    path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False))


def render_step_artifacts(
    output_dir: Path,
    task_name: str,
    episode_idx: int,
    step: int,
    mode: str,
    rgb: np.ndarray,
    model,
    save_debug_assets: bool,
) -> tuple[Image.Image, dict]:
    telemetry = getattr(model, "last_inference_stats", {}) or {}
    pruning_info = telemetry.get("pruning_info") or {}
    vis_cache = getattr(model, "last_visualization_cache", {}) or {}
    summary = telemetry.get("summary") or {}
    dynamic = telemetry.get("dynamic") or {}

    decode_map = tensor_to_vector(vis_cache.get("action_vision_attentions"), reduce_dims=(1, 2))
    prefill_map = tensor_to_vector(vis_cache.get("prefill_attentions"), reduce_dims=(1, 2))
    current_score = None
    if vis_cache.get("current_selection_score") is not None:
        if isinstance(vis_cache.get("current_selection_score"), torch.Tensor):
            current_score = vis_cache.get("current_selection_score").detach().cpu().numpy().astype(np.float32)
        else:
            current_score = np.asarray(vis_cache.get("current_selection_score"), dtype=np.float32)
    temporal_guide = None
    if vis_cache.get("temporal_guide") is not None:
        if isinstance(vis_cache.get("temporal_guide"), torch.Tensor):
            temporal_guide = vis_cache.get("temporal_guide").detach().cpu().numpy().astype(np.float32)
        else:
            temporal_guide = np.asarray(vis_cache.get("temporal_guide"), dtype=np.float32)

    if decode_map is not None and decode_map.ndim == 2:
        decode_map = decode_map[min(15, decode_map.shape[0] - 1)]
    if prefill_map is not None and prefill_map.ndim == 2:
        prefill_map = prefill_map[min(15, prefill_map.shape[0] - 1)]

    keep_mask, prune_mask = build_keep_prune_masks(pruning_info)
    temporal_reason = None
    if not summary.get("use_temporal"):
        temporal_reason = "temporal disabled"
    elif not dynamic.get("temporal_history_ready"):
        temporal_reason = "history not ready"
    elif temporal_guide is None:
        temporal_reason = "cache missing"

    pref_overlay, pref_reason = make_patch_overlay(rgb, prefill_map)
    decode_overlay, decode_reason = make_patch_overlay(rgb, decode_map)
    temporal_overlay, temporal_overlay_reason = make_patch_overlay(rgb, temporal_guide)
    current_score_overlay, current_score_reason = make_patch_overlay(rgb, current_score)
    keep_mask_overlay, keep_reason = make_binary_mask(rgb, keep_mask, (34, 197, 94, 120))
    prune_mask_overlay, prune_reason = make_binary_mask(rgb, prune_mask, (107, 114, 128, 125))

    if temporal_reason is not None:
        temporal_overlay = None
        temporal_overlay_reason = temporal_reason

    unavailable_reasons = {
        "prefill_overlay": pref_reason,
        "decode_overlay": decode_reason,
        "temporal_guide_overlay": temporal_overlay_reason,
        "current_score_overlay": current_score_reason,
        "keep_mask": keep_reason,
        "prune_mask": prune_reason,
    }
    unavailable = [name for name, reason in unavailable_reasons.items() if reason is not None]

    rgb_image = Image.fromarray(rgb.astype(np.uint8))
    panel = build_panel(
        title_lines=[
            task_name,
            f"episode={episode_idx}   step={step}   mode={mode}",
        ],
        metadata_lines=[
            f"selection_mode={summary.get('selection_mode')}",
            f"use_temporal={summary.get('use_temporal')}",
            f"temporal_history_ready={dynamic.get('temporal_history_ready')}",
            f"temporal_history_len={dynamic.get('temporal_history_len')}",
            f"kept_seq_length={dynamic.get('kept_seq_length')}",
            f"kept_image_token_length={dynamic.get('kept_image_token_length')}",
            f"core_inference_latency_ms={telemetry.get('core_inference_latency_ms')}",
        ],
        tiles=[
            ("RGB", rgb_image, None),
            ("Prefill", pref_overlay, pref_reason),
            ("Decode", decode_overlay, decode_reason),
            ("Temporal Guide", temporal_overlay, temporal_overlay_reason),
            ("Current Score", current_score_overlay, current_score_reason),
            ("Keep Mask", keep_mask_overlay, keep_reason),
            ("Prune Mask", prune_mask_overlay, prune_reason),
        ],
    )

    original_image_token_length = pruning_info.get("original_image_token_length")
    kept_image_token_length = pruning_info.get("kept_image_token_length")
    effective_keep_ratio = None
    if original_image_token_length not in (None, 0) and kept_image_token_length is not None:
        effective_keep_ratio = float(kept_image_token_length) / float(original_image_token_length)

    meta = {
        "task_name": task_name,
        "episode_idx": episode_idx,
        "step": step,
        "mode": mode,
        "selection_mode": (telemetry.get("summary") or {}).get("selection_mode"),
        "use_temporal": (telemetry.get("summary") or {}).get("use_temporal"),
        "temporal_history_ready": dynamic.get("temporal_history_ready"),
        "temporal_history_len": dynamic.get("temporal_history_len"),
        "pruning_layer": (telemetry.get("summary") or {}).get("pruning_layer"),
        "original_seq_length": (telemetry.get("summary") or {}).get("original_seq_length"),
        "kept_seq_length": dynamic.get("kept_seq_length"),
        "original_image_token_length": original_image_token_length,
        "kept_image_token_length": kept_image_token_length,
        "effective_keep_ratio": effective_keep_ratio,
        "core_inference_latency_ms": telemetry.get("core_inference_latency_ms"),
        "temporal_guide_available": temporal_overlay_reason is None,
        "temporal_guide_unavailable_reason": temporal_overlay_reason,
        "unavailable_visualizations": unavailable,
        "unavailable_reasons": unavailable_reasons,
    }
    if save_debug_assets:
        rgb_image.save(output_dir / f"step_{step:04d}_rgb.png")
        if pref_overlay is not None:
            pref_overlay.save(output_dir / f"step_{step:04d}_prefill_overlay.png")
        if decode_overlay is not None:
            decode_overlay.save(output_dir / f"step_{step:04d}_decode_overlay.png")
        if temporal_overlay is not None:
            temporal_overlay.save(output_dir / f"step_{step:04d}_temporal_guide_overlay.png")
        if current_score_overlay is not None:
            current_score_overlay.save(output_dir / f"step_{step:04d}_current_score_overlay.png")
        if keep_mask_overlay is not None:
            keep_mask_overlay.save(output_dir / f"step_{step:04d}_keep_mask.png")
        if prune_mask_overlay is not None:
            prune_mask_overlay.save(output_dir / f"step_{step:04d}_prune_mask.png")
        write_json(output_dir / f"step_{step:04d}_meta.json", meta)
    return panel, meta


def create_video_writer(output_path: Path, fps: int, video_format: str):
    if video_format == "gif":
        return imageio.get_writer(output_path, mode="I", fps=fps)
    return imageio.get_writer(output_path, fps=fps)


def save_panel_frame(panel: Image.Image, frames_dir: Path, step: int) -> Path:
    frame_path = frames_dir / f"step_{step:04d}_panel.png"
    panel.save(frame_path)
    return frame_path


def append_panel_to_video(writer, panel: Image.Image) -> None:
    writer.append_data(np.asarray(panel.convert("RGB")))


def should_capture_step(render_full_episode: bool, step: int, selected_steps: set[int]) -> bool:
    return render_full_episode or step in selected_steps


def get_video_output_path(base_dir: Path, video_format: str) -> Path:
    return base_dir / f"episode_panel.{video_format}"


def main() -> None:
    args = parse_args()
    cfg = build_cfg(args)
    cfg.unnorm_key = cfg.task_suite_name
    steps = sorted(set(args.steps or DEFAULT_STEPS))
    selected_steps = set(steps)
    set_seed_everywhere(cfg.seed)

    model = get_model(cfg)
    if hasattr(model, "enable_visualization_cache"):
        model.enable_visualization_cache = True
    if hasattr(model, "language_model"):
        setattr(model.language_model, "enable_visualization_cache", True)
        if hasattr(model.language_model, "model"):
            setattr(model.language_model.model, "enable_visualization_cache", True)

    processor = get_processor(cfg)
    resize_size = get_image_resize_size(cfg)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    task_index, task = resolve_task(task_suite, args.task_name, args.task_index)
    initial_states = task_suite.get_task_init_states(task_index)
    env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

    output_dir = Path(args.output_dir) / cfg.task_suite_name / args.mode / f"task_{task_index:02d}" / f"episode_{args.episode_idx:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    if args.save_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)

    video_writer = None
    video_output_path = None
    if args.render_full_episode:
        video_output_path = get_video_output_path(output_dir, args.video_format)
        video_writer = create_video_writer(video_output_path, fps=args.fps, video_format=args.video_format)

    env.reset()
    model.reset_av_history()
    obs = env.set_init_state(initial_states[args.episode_idx])
    t = 0
    done = False
    prev_img = None
    last_caches = None
    max_steps = args.max_steps if args.max_steps is not None else get_max_steps(cfg.task_suite_name)
    max_target_step = max_steps if args.render_full_episode else max(steps)

    while t < max_steps + cfg.num_steps_wait and t <= max_target_step and not done:
        if t < cfg.num_steps_wait:
            obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
            t += 1
            continue

        img = get_libero_image(obs, resize_size)
        prev_img = img if prev_img is None else prev_img
        observation = {
            "full_image": img,
            "prev_image": prev_img,
            "state": np.concatenate(
                (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
            ),
        }
        action, last_caches, _ = get_action(
            cfg,
            model,
            observation,
            task_description,
            processor=processor,
            last_caches=last_caches,
        )

        if should_capture_step(args.render_full_episode, t, selected_steps):
            panel, meta = render_step_artifacts(
                output_dir=output_dir,
                task_name=task_description,
                episode_idx=args.episode_idx,
                step=t,
                mode=args.mode,
                rgb=img,
                model=model,
                save_debug_assets=args.save_debug_assets,
            )
            if args.save_frames or not args.render_full_episode:
                save_panel_frame(panel, frames_dir if args.save_frames else output_dir, t)
            if video_writer is not None:
                append_panel_to_video(video_writer, panel)

        action = normalize_gripper_action(action, binarize=True)
        action = invert_gripper_action(action)
        obs, reward, done, info = env.step(action.tolist())
        prev_img = img
        if done and t in steps:
            break
        t += 1

    env.close()
    if video_writer is not None:
        video_writer.close()
        print(f"Saved episode panel video to {video_output_path}")
    print(f"Saved visualization artifacts to {output_dir}")


if __name__ == "__main__":
    main()
