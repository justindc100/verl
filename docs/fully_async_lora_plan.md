# Fully Async GRPO + vLLM + FSDP: Minimal LoRA plan (merged-only)

## Decision: keep it minimal

To minimize code changes and risk, support **only one LoRA mode** in fully-async:

- `model.lora.rank > 0`
- `model.lora.merge = true`
- **single weight-sync method**: `sync_rollout_weights` (disable checkpoint-engine path for LoRA in fully-async v1)

Everything else is explicitly unsupported in this first implementation:

- `merge=false` (adapter hot-swap path)
- `peft_config` propagation to rollout
- LoRA adapter lifecycle management on vLLM rollout side

## Why this is the smallest viable change

With merged LoRA, rollout only needs normal full-model tensors (`load_weights`) and does not need adapter-specific handling (`TensorLoRARequest`, add/remove adapter, base/adaptor split).

That lets us reuse the existing async detach sync structure with a small trainer-side export change and a strict config gate.

## Minimal code changes

## 1) Add strict config gate for fully-async

At fully-async startup (trainer/rollouter init), enforce:

- If LoRA is enabled, require `model.lora.merge=true`
- If LoRA is enabled, force one sync method (`sync_rollout_weights`) and reject checkpoint-engine LoRA path for now
- Reject unsupported combinations with a clear error

Suggested error message:

> fully_async LoRA v1 only supports merged LoRA (`model.lora.merge=true`) with `sync_rollout_weights`.

## 2) Trainer-side export: merged weights only

In fully-async FSDP detach worker (`experimental/fully_async_policy/fsdp_workers.py`), update `_get_actor_params()`:

- Detect PEFT model + `merge=true`
- Enter merged-LoRA context (same concept as mainline), materialize merged `state_dict`, normalize/convert keys as done today
- Return plain tensors for sync

No adapter metadata is needed.

## 3) ParameterSynchronizer: use a single sync branch

In `experimental/fully_async_policy/param_sync.py`:

- Re-enable sync calls
- For LoRA-enabled fully-async v1, always call:
  - `actor_wg.sync_rollout_weights(...)`
  - `rollout_wg.sync_rollout_weights(...)`
- Skip checkpoint-engine sync for LoRA path in this version

## 4) Rollout side: no LoRA-specific changes required

Because trainer sends merged full weights, rollout remains on standard `inference_model.load_weights(...)` behavior.

## Out of scope (deferred)

- Non-merged LoRA (`merge=false`) for fully-async
- Adapter-only incremental sync
- `peft_config`/`TensorLoRARequest` async integration
- Checkpoint-engine LoRA optimization

## Validation checklist (minimal)

1. Startup guard test:
   - `merge=false` + fully-async fails fast with expected error.
2. Smoke training:
   - short fully-async GRPO run with `merge=true` completes multiple syncs.
3. Rollout sanity:
   - generation works before and after at least one trainer update.
4. Regression:
   - non-LoRA fully-async path remains unchanged.

## Suggested rollout

1. Implement merged-only support.
2. Land with guardrails and docs.
3. Add `merge=false` path only after merged mode is stable.
