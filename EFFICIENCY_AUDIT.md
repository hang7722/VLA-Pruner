# Efficiency Metric Code Archaeology (VLA-Pruner/OpenVLA)

This note records where efficiency-related logic actually exists in the public repo, and where it does **not**.

## Key findings

- `run_spatial.sh` and `run_spatial_prefil.sh` only pass flags into `experiments/robot/libero/run_libero_eval.py`; they do not measure latency/FLOPs.
- `run_libero_eval.py` only logs success-related metrics; no latency/FLOPs/speedup logging fields are produced.
- FastV/VLA-Pruner pruning happens inside the patched LLaMA `fastv_forward`, where `num_keep = round(image_token_length * (1 - fastv_r))`.
- Pruning metadata (`original_seq_length`, `kept_indices`, `pruned_indices`, `pruning_layer`) is tracked internally as `pruning_info`.
- `record_flops` exists in config but no execution path in eval uses it to compute FLOPs.

## Primary code path

`run_libero_eval.py` -> `robot_utils.get_action` -> `openvla_utils.get_vla_action` -> `OpenVLAForActionPrediction.predict_action` -> `PrismaticForConditionalGeneration.forward` -> `LlamaForCausalLM.fastv_forward` -> `LlamaModel.fastv_forward` pruning.

## Minimal instrumentation entry points (if needed later)

- `src/openvla/experiments/robot/openvla_utils.py::get_vla_action`: measure per-step `predict_action` wall time (`total`).
- `src/openvla/prismatic/extern/hf/modeling_prismatic.py::predict_action`: split generation stage timing and collect `fastv_config` + temporal state.
- `src/openvla/transformers/src/transformers/models/llama/modeling_llama.py::fastv_forward`: log pre/post sequence length and `num_keep` at prune layer.
- `src/openvla/experiments/robot/libero/run_libero_eval.py`: aggregate per-episode means/p95 and print/log them alongside success rate.
