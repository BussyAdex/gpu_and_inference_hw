import torch
from torch.profiler import profile as torch_profile, ProfilerActivity, record_function
from transformers import DynamicCache
from utils import (
    build_model,
    get_input_ids,
    slow_loop,
    time_generation,
    MODEL_NAME,
    PROFILE_STEPS,
    RESULTS_DIR,
)


def optimized_loop(model, input_ids, n_steps):
    with torch.inference_mode():
        past_key_values = DynamicCache()
        prompt_len = input_ids.shape[1]

        # Prefill: process the full prompt once, populate the cache
        position_ids = torch.arange(
            prompt_len, device=input_ids.device).unsqueeze(0)
        outputs = model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        next_token_id = torch.argmax(
            outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated = [next_token_id]
        cur_pos = prompt_len

        # Decode: feed one token per step, reuse cache
        for _ in range(n_steps - 1):
            position_ids = torch.tensor([[cur_pos]], device=input_ids.device)
            outputs = model(
                input_ids=next_token_id,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            next_token_id = torch.argmax(
                outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated.append(next_token_id)
            cur_pos += 1

        # ONE sync at the end, not per-step
        return torch.cat(generated, dim=1).squeeze(0).tolist()


def profile(loop_fn, model, input_ids, trace_name: str):
    # wrap loop_fn(model, input_ids, PROFILE_STEPS) with torch.profiler,
    # print the summary table, and export a Chrome trace to RESULTS_DIR / trace_name
    with torch_profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        with record_function("generation_loop"):
            loop_fn(model, input_ids, PROFILE_STEPS)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    trace_path = RESULTS_DIR / trace_name
    prof.export_chrome_trace(str(trace_path))
    print(f"Trace exported to {trace_path}")


def generate_optimized(optimized_trace_name: str) -> float:
    # load the model (consider dtype and other loading options),
    # then call profile() and time_generation() on optimized_loop.
    # Return the elapsed time from time_generation so main() can print a speedup.
    model = build_model(torch.bfloat16)
    input_ids = get_input_ids()

    profile(optimized_loop, model, input_ids, optimized_trace_name)
    elapsed = time_generation(optimized_loop, model, input_ids, "Optimized")
    return elapsed


def main():
    print("=" * 60)
    print("HW2: LLM Inference Optimization")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)

    print("\n--- Part 1: Slow baseline ---")
    model = build_model(torch.float32)
    input_ids = get_input_ids()
    profile(slow_loop, model, input_ids, "v0_slow_trace.json")
    slow_elapsed = time_generation(slow_loop, model, input_ids, "Slow")
    del model
    torch.cuda.empty_cache()

    print("\n--- Part 2: Optimized ---")
    optimized_elapsed = generate_optimized(
        optimized_trace_name="v1_optimized_trace.json")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if optimized_elapsed is None or optimized_elapsed <= 0:
        print("generate_optimized() did not return a positive elapsed time; "
              "cannot compute speedup.")
    else:
        speedup = slow_elapsed / optimized_elapsed
        print(f"  Slow:      {slow_elapsed:6.2f}s")
        print(f"  Optimized: {optimized_elapsed:6.2f}s")
        print(f"  Speedup:   {speedup:6.2f}x  (vs V0 slow baseline)")


if __name__ == "__main__":
    main()


# ============================================================================
# Writeup
# ============================================================================
#
# Changes made and speedup per fix:

# 1. Enabled KV cache (DynamicCache + use_cache=True, feeding one token
# per decode step after prefill):
# V0 21.3s → 7.2s (≈3.0× speedup from baseline).
# Reason: the model no longer recomputes attention over the entire
# prompt at every generation step. After the initial prefill pass,
# previously computed key/value tensors are reused.

# 2. Switched model dtype from fp32 to bfloat16:
# 7.2s → 4.5s (cumulative ≈4.7× speedup).
# Reason: bfloat16 cuts memory bandwidth requirements roughly in half
# while remaining hardware-friendly on modern GPUs. Since decode at
# batch size 1 is largely memory-bandwidth bound, reducing tensor size
# significantly improves throughput.

# 3. Removed per-step .item() GPU synchronization calls:
# 4.5s → 3.9s (cumulative ≈5.5× speedup).
# Reason: calling .item() each iteration forced CPU/GPU synchronization,
# creating bubbles in the execution stream. Keeping tensors on GPU until
# generation completed allowed kernels to execute continuously.
# Perfetto traces confirmed the GPU stream became much denser with
# synchronization gaps removed.

# 4. Wrapped generation loop in torch.inference_mode():
# 3.9s → 3.7s (cumulative ≈5.8× speedup).
# Reason: disables autograd tracking and version counter updates,
# reducing framework overhead during inference. The gain is smaller
# because most runtime was already dominated by model execution.

# Biggest impact and why:
# The KV cache produced the largest improvement. In the baseline version,
# every decode step recomputed attention across the full prompt plus all
# previously generated tokens, causing attention cost to grow repeatedly
# with sequence length. With KV caching enabled, the model computes K/V
# tensors once during prefill and reuses them for subsequent decode steps.

# Conceptually:
# Without cache:
# each step processes ~prompt_len + generated_tokens

# With cache:
# each step processes only the newly generated token while attending to

# cached history.
# This changes generation from repeatedly recomputing large attention
# matrices to incremental decoding, dramatically reducing compute and
# memory traffic per token. For long prompts and autoregressive decoding,
# this optimization dominates all others.
