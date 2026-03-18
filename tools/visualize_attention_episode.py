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
from pathlib import Path
from types import SimpleNamespace

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
TOP_RATIOS = (0.125, 0.25, 0.5)
TOP_COLORS = (
    (128, 0, 255, 140),
    (255, 140, 0, 110),
    (255, 220, 0, 80),
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


def make_patch_overlay(rgb: np.ndarray, values: np.ndarray | None) -> Image.Image | None:
    norm = normalize_map(values)
    if norm is None:
        return None
    side = infer_grid_size(norm.size)
    if side is None:
        return None
    image = Image.fromarray(rgb.astype(np.uint8)).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    patch_h = image.size[1] / side
    patch_w = image.size[0] / side
    order = np.argsort(norm)[::-1]
    for top_ratio, color in zip(TOP_RATIOS, TOP_COLORS):
        top_k = max(1, int(round(norm.size * top_ratio)))
        indices = order[:top_k]
        for idx in indices:
            row, col = divmod(int(idx), side)
            x0 = int(round(col * patch_w))
            y0 = int(round(row * patch_h))
            x1 = int(round((col + 1) * patch_w))
            y1 = int(round((row + 1) * patch_h))
            draw.rectangle([x0, y0, x1, y1], fill=color, outline=(255, 255, 255, 40))
    return Image.alpha_composite(image, overlay).convert("RGB")


def make_binary_mask(rgb: np.ndarray, mask_values: np.ndarray | None, color: tuple[int, int, int, int]) -> Image.Image | None:
    if mask_values is None:
        return None
    mask_values = np.asarray(mask_values, dtype=np.float32)
    side = infer_grid_size(mask_values.size)
    if side is None:
        return None
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
    return Image.alpha_composite(image, overlay).convert("RGB")


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


def save_panel(output_path: Path, title: str, images: list[tuple[str, Image.Image | None]]) -> None:
    tile_w, tile_h = 256, 256
    cols = 3
    rows = int(math.ceil(len(images) / cols))
    panel = Image.new("RGB", (cols * tile_w, rows * (tile_h + 24)), (255, 255, 255))
    draw = ImageDraw.Draw(panel)
    draw.text((10, 4), title, fill=(0, 0, 0))
    for idx, (label, image) in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = col * tile_w
        y = row * (tile_h + 24) + 24
        tile = image if image is not None else Image.new("RGB", (tile_w, tile_h), (235, 235, 235))
        tile = tile.resize((tile_w, tile_h))
        panel.paste(tile, (x, y))
        draw.text((x + 8, y - 18), label, fill=(0, 0, 0))
    panel.save(output_path)


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


def capture_step_artifacts(
    output_dir: Path,
    task_name: str,
    episode_idx: int,
    step: int,
    mode: str,
    rgb: np.ndarray,
    model,
) -> None:
    telemetry = getattr(model, "last_inference_stats", {}) or {}
    pruning_info = telemetry.get("pruning_info") or {}
    vis_cache = getattr(model, "last_visualization_cache", {}) or {}

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

    pref_overlay = make_patch_overlay(rgb, prefill_map)
    decode_overlay = make_patch_overlay(rgb, decode_map)
    temporal_overlay = make_patch_overlay(rgb, temporal_guide)
    current_score_overlay = make_patch_overlay(rgb, current_score)
    keep_mask_overlay = make_binary_mask(rgb, keep_mask, (255, 210, 0, 120))
    prune_mask_overlay = make_binary_mask(rgb, prune_mask, (160, 160, 160, 120))

    unavailable = []
    artifact_map = {
        "prefill_overlay": pref_overlay,
        "decode_overlay": decode_overlay,
        "temporal_guide_overlay": temporal_overlay,
        "current_score_overlay": current_score_overlay,
        "keep_mask": keep_mask_overlay,
        "prune_mask": prune_mask_overlay,
    }
    for name, image in artifact_map.items():
        if image is None:
            unavailable.append(name)

    rgb_image = Image.fromarray(rgb.astype(np.uint8))
    rgb_image.save(output_dir / f"step_{step:02d}_rgb.png")
    if pref_overlay is not None:
        pref_overlay.save(output_dir / f"step_{step:02d}_prefill_overlay.png")
    if decode_overlay is not None:
        decode_overlay.save(output_dir / f"step_{step:02d}_decode_overlay.png")
    if temporal_overlay is not None:
        temporal_overlay.save(output_dir / f"step_{step:02d}_temporal_guide_overlay.png")
    if current_score_overlay is not None:
        current_score_overlay.save(output_dir / f"step_{step:02d}_current_score_overlay.png")
    if keep_mask_overlay is not None:
        keep_mask_overlay.save(output_dir / f"step_{step:02d}_keep_mask.png")
    if prune_mask_overlay is not None:
        prune_mask_overlay.save(output_dir / f"step_{step:02d}_prune_mask.png")

    save_panel(
        output_dir / f"step_{step:02d}_panel.png",
        title=f"{task_name} | episode={episode_idx} | step={step} | mode={mode}",
        images=[
            ("rgb", rgb_image),
            ("prefill", pref_overlay),
            ("decode", decode_overlay),
            ("temporal_guide", temporal_overlay),
            ("current_score", current_score_overlay),
            ("keep_mask", keep_mask_overlay),
        ],
    )

    original_image_token_length = pruning_info.get("original_image_token_length")
    kept_image_token_length = pruning_info.get("kept_image_token_length")
    effective_keep_ratio = None
    if original_image_token_length not in (None, 0) and kept_image_token_length is not None:
        effective_keep_ratio = float(kept_image_token_length) / float(original_image_token_length)

    dynamic = telemetry.get("dynamic") or {}
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
        "unavailable_visualizations": unavailable,
    }
    write_json(output_dir / f"step_{step:02d}_meta.json", meta)


def main() -> None:
    args = parse_args()
    cfg = build_cfg(args)
    cfg.unnorm_key = cfg.task_suite_name
    steps = sorted(set(args.steps or DEFAULT_STEPS))
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

    env.reset()
    model.reset_av_history()
    obs = env.set_init_state(initial_states[args.episode_idx])
    t = 0
    done = False
    prev_img = None
    last_caches = None
    max_steps = get_max_steps(cfg.task_suite_name)
    max_target_step = max(steps)

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

        if t in steps:
            capture_step_artifacts(
                output_dir=output_dir,
                task_name=task_description,
                episode_idx=args.episode_idx,
                step=t,
                mode=args.mode,
                rgb=img,
                model=model,
            )

        action = normalize_gripper_action(action, binarize=True)
        action = invert_gripper_action(action)
        obs, reward, done, info = env.step(action.tolist())
        prev_img = img
        if done and t in steps:
            break
        t += 1

    env.close()
    print(f"Saved visualization artifacts to {output_dir}")


if __name__ == "__main__":
    main()
