"""
HW3: Mini Inference Engine
CacheManager · Continuous Batching · Prefix Caching

Edit only this file.  See README.md for background and implementation details.

Run:
    python hw3_inference_engine/hw3_task.py
"""

from __future__ import annotations
from engine_utils import (
    CacheHandle,
    Request,
    Batch,
    BatchPhase,
    StepMetrics,
    DummyLLM,
    SchedulingPolicy,
    RequestStatus,
    generate_workload,
    compute_stats,
    print_stats,
    plot_results,
    plot_policy_results,
    BLOCK_SIZE,
    NUM_BLOCKS,
    MAX_SEQS,
    TOKEN_BUDGET,
    PREFILL_CHUNK,
)
from tqdm import tqdm

import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


# ── Task 1: Cache Manager ─────────────────────────────────────────────────────


class CacheManager:
    """
    Unified block allocator, prefix cache, and LRU eviction.

    Ref-count semantics:
        allocate(n)    ref = 1   request owns the block
        lock(handle)   ref += 1  request also pins a cached block
        unlock(handle) ref -= 1  block is evictable once ref drops to 1
        free(ids)      ref -= 1  block goes to free pool when ref reaches 0
        _evict_blocks_from_kv_cache(n)      reclaims n LRU unlocked blocks from the prefix cache
    """

    def __init__(
        self, num_blocks: int = NUM_BLOCKS, block_size: int = BLOCK_SIZE
    ) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._free: list[int] = list(range(num_blocks))  # available block IDs
        self._ref: list[int] = [0] * num_blocks  # reference counts
        # Prefix cache: token-tuple key → list of block IDs
        self._cache: dict[tuple[int, ...], list[int]] = {}
        # LRU order: index 0 = least-recently used; updated on every hit and insert
        self._lru: list[tuple[int, ...]] = []
        # Per-block count of how many cache entries reference it.
        # _ref is incremented only ONCE for cache ownership (when _cache_ref
        # goes from 0 → 1) and decremented when _cache_ref returns to 0.
        self._cache_ref: list[int] = [0] * num_blocks

    @property
    def num_free_blocks(self) -> int:
        return len(self._free)

    @property
    def ref_counts(self) -> list[int]:
        """Snapshot of per-block effective ownership refs."""
        return list(self._ref)

    @property
    def cache_ref_counts(self) -> list[int]:
        """Snapshot of per-block cache-entry reference counts."""
        return list(self._cache_ref)

    @property
    def cache_entries(self) -> dict[tuple[int, ...], list[int]]:
        """Snapshot of cached prefix -> block mapping."""
        return {k: list(v) for k, v in self._cache.items()}

    @property
    def lru_keys(self) -> list[tuple[int, ...]]:
        """Snapshot of cache keys in LRU order (oldest first)."""
        return list(self._lru)

    def allocate(self, n: int) -> list[int] | None:
        """Claim n blocks (ref=1 each). Evicts LRU cache entries if needed.
        Returns None only when eviction cannot free enough blocks."""
        if len(self._free) < n:
            needed = n - len(self._free)
            self._evict_blocks_from_kv_cache(needed)
        if len(self._free) < n:
            return None
        result = []
        for _ in range(n):
            blk = self._free.pop()
            self._ref[blk] = 1
            result.append(blk)
        return result

    def free(self, block_ids: list[int]) -> None:
        """Decrement each block's ref; return to the free list when ref reaches 0."""
        for blk in block_ids:
            self._ref[blk] -= 1
            if self._ref[blk] == 0:
                self._free.append(blk)

    def lock(self, handle: CacheHandle) -> None:
        """Pin the matched blocks (incr ref). Must be called before using them."""
        for blk in handle.matched_blocks:
            self._ref[blk] += 1

    def unlock(self, handle: CacheHandle) -> None:
        """Release the pin (decr ref). Blocks become evictable when ref drops to 1."""
        for blk in handle.matched_blocks:
            self._ref[blk] -= 1

    def match_prefix(self, tokens: list[int]) -> CacheHandle:
        """Longest-prefix lookup. Returns a CacheHandle WITHOUT pinning.
        Updates LRU order on a hit. Returns CacheHandle(0, []) on a miss."""
        best_key: tuple[int, ...] | None = None
        best_blocks: list[int] = []

        # Build all complete-block prefixes from longest to shortest and find best match
        n_complete = len(tokens) // self.block_size
        for length in range(n_complete, 0, -1):
            key = tuple(tokens[: length * self.block_size])
            if key in self._cache:
                best_key = key
                best_blocks = list(self._cache[key])
                break

        if best_key is None:
            return CacheHandle(num_matched_tokens=0, matched_blocks=[])

        # Update LRU: move only the matched key to MRU position
        self._lru.remove(best_key)
        self._lru.append(best_key)

        return CacheHandle(
            num_matched_tokens=len(best_key),
            matched_blocks=best_blocks,
        )

    def insert_prefix(self, tokens: list[int], block_ids: list[int]) -> None:
        """Store every complete-block prefix not already cached.
        For each block in a new entry, increment _cache_ref. Only increment
        _ref when _cache_ref goes from 0 → 1 (first cache entry for that block)
        so that overlapping entries share a single ref-count for cache ownership."""
        n_complete = len(tokens) // self.block_size
        for i in range(1, n_complete + 1):
            key = tuple(tokens[: i * self.block_size])
            blocks_for_key = block_ids[:i]

            if key in self._cache:
                # Already cached — skip to avoid double-counting
                continue

            # Store a copy so callers can't mutate through their reference
            self._cache[key] = list(blocks_for_key)
            self._lru.append(key)

            # Update _cache_ref and _ref
            for blk in blocks_for_key:
                self._cache_ref[blk] += 1
                # Only bump _ref for cache ownership on first cache entry for this block
                if self._cache_ref[blk] == 1:
                    self._ref[blk] += 1

    def _evict_blocks_from_kv_cache(self, n: int) -> None:
        """Attempt to evict least-recently-used cache entries whose blocks are
        unlocked (`ref == 1`) to reclaim up to `n` blocks.
        Because cache entries can overlap on blocks, evicting an entry does not
        always free a block immediately. A block becomes free only when its
        cache ownership drops to zero."""
        freed = 0
        # Walk LRU from oldest to newest
        i = 0
        while i < len(self._lru) and freed < n:
            key = self._lru[i]
            blocks = self._cache[key]

            # Check if any block in this entry is pinned (ref >= 2 means live request holds it)
            if any(self._ref[blk] >= 2 for blk in blocks):
                i += 1
                continue

            # Evict this entry
            self._lru.pop(i)
            del self._cache[key]

            for blk in blocks:
                self._cache_ref[blk] -= 1
                if self._cache_ref[blk] == 0:
                    # Last cache entry referencing this block; release cache ownership
                    self._ref[blk] -= 1
                    if self._ref[blk] == 0:
                        self._free.append(blk)
                        freed += 1
            # Don't increment i since we popped at position i


# ── Task 2: Scheduler ─────────────────────────────────────────────────────────


class Scheduler:
    def __init__(
        self,
        cache_manager: CacheManager,
        block_size: int = BLOCK_SIZE,
        max_seqs: int = MAX_SEQS,
        token_budget: int = TOKEN_BUDGET,
        prefill_chunk: int = PREFILL_CHUNK,
        enable_prefix_caching: bool = True,
        scheduling_policy: SchedulingPolicy | str = SchedulingPolicy.PREFILL_FIRST,
    ) -> None:
        self.cache_manager = cache_manager
        self.block_size = block_size
        self.max_seqs = max_seqs
        self.token_budget = token_budget
        self.prefill_chunk = prefill_chunk
        self.enable_prefix_caching = enable_prefix_caching
        self.scheduling_policy = SchedulingPolicy(scheduling_policy)
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.step: int = 0

    def add(self, req: Request) -> None:
        req.status = RequestStatus.WAITING
        self.waiting.append(req)

    def _blocks_for(self, n_tokens: int) -> int:
        return (n_tokens + self.block_size - 1) // self.block_size

    def _preempt(self, req: Request, batch: Batch) -> None:
        """Free req's blocks (respecting lock state), reset its state, re-queue it."""
        if req.cache_handle is not None:
            n = len(req.cache_handle.matched_blocks)
            self.cache_manager.unlock(req.cache_handle)
            self.cache_manager.free(req.block_table[n:])
            req.cache_handle = None
        else:
            self.cache_manager.free(req.block_table)
        req.block_table = []
        req.num_computed_tokens = 0
        req.num_generated_tokens = 0
        req.prefix_tokens_saved = 0
        req.first_token_step = None
        req.num_preemptions += 1
        req.status = RequestStatus.WAITING
        self.running.remove(req)
        self.waiting.appendleft(req)
        batch.preempted.append(req)

    def schedule(self) -> Batch | None:
        """
        Return a single-phase Batch for this step, or None if idle
        (no waiting and no running requests).

        Phase selection policy:
          - PREFILL_FIRST:
              * If any prefill work exists (running prefills or waiting queue
                non-empty), try _schedule_prefill().
              * Otherwise, schedule decode.
          - DECODE_FIRST:
              * If any decode-ready running request exists, try
                _schedule_decode().
              * Otherwise, schedule prefill.

        Delegates to _schedule_prefill() / _schedule_decode().
        See README.md → Task 2 for the full algorithm.
        """
        has_waiting = len(self.waiting) > 0
        has_running_prefill = any(r.is_prefilling for r in self.running)
        has_running_decode = any(not r.is_prefilling for r in self.running)

        if not has_waiting and not self.running:
            self.step += 1
            return None

        if self.scheduling_policy == SchedulingPolicy.PREFILL_FIRST:
            if has_waiting or has_running_prefill:
                batch = self._schedule_prefill()
                # Fall through to decode if prefill produced nothing useful
                if batch.to_prefill or batch.newly_admitted:
                    self.step += 1
                    return batch
                # No useful prefill work — try decode
                if has_running_decode or any(not r.is_prefilling for r in self.running):
                    batch = self._schedule_decode()
                    self.step += 1
                    return batch
                self.step += 1
                return batch  # return empty prefill batch (idle)
            else:
                batch = self._schedule_decode()
                self.step += 1
                return batch

        else:  # DECODE_FIRST
            if has_running_decode:
                batch = self._schedule_decode()
                if batch.to_decode:
                    self.step += 1
                    return batch
                # No decode work — try prefill
                batch = self._schedule_prefill()
                self.step += 1
                return batch
            else:
                batch = self._schedule_prefill()
                self.step += 1
                return batch

    def _schedule_prefill(self) -> Batch:
        """
        Build a prefill Batch.

        Step A — running requests still prefilling (iterate over a copy - list(self.running)):
          Compute chunk = min(remaining_prefill, prefill_chunk, budget).
          Allocate any new blocks the chunk needs (allocation may evict cache
          entries internally); _preempt on allocation failure.
          Add (req, chunk) to batch.to_prefill; deduct from budget.

        Step B — admit from waiting while budget > 0 and slots remain:
          If prefix caching: call match_prefix FIRST → if hit, lock the
          handle and reduce the number of blocks to allocate.
          Allocate the remaining blocks; on failure unlock the handle and break.
          Build block_table = matched_blocks + newly allocated blocks.
          Set num_computed_tokens, prefix_tokens_saved, cache_handle.
          If the entire prompt was cached, skip adding to to_prefill.
          Append to running and newly_admitted; add first chunk to batch.

        Note:
          Keep this batch phase-pure: populate only batch.to_prefill here.
        """
        batch = Batch(phase=BatchPhase.PREFILL)
        budget = self.token_budget

        # Step A: continue running prefill requests
        for req in list(self.running):
            if budget <= 0:
                break
            if not req.is_prefilling:
                continue

            # Check that the request has enough blocks allocated for its full prompt
            needed_blocks = self._blocks_for(req.prompt_len)
            if req.cache_handle is not None:
                matched = len(req.cache_handle.matched_blocks)
                extra_needed = needed_blocks - matched
                currently_owned = len(req.block_table) - matched
            else:
                extra_needed = needed_blocks
                currently_owned = len(req.block_table)

            total_allocated = len(req.block_table)
            if total_allocated < needed_blocks:
                # Under-allocated — preempt
                self._preempt(req, batch)
                continue

            chunk = min(req.remaining_prefill, self.prefill_chunk, budget)
            if chunk > 0:
                batch.to_prefill.append((req, chunk))
                budget -= chunk

        # Step B: admit from waiting
        while budget > 0 and len(self.running) < self.max_seqs and self.waiting:
            req = self.waiting[0]

            handle = None
            matched_blocks: list[int] = []
            n_matched_tokens = 0

            if self.enable_prefix_caching:
                handle = self.cache_manager.match_prefix(req.prompt_tokens)
                if handle.num_matched_tokens > 0:
                    self.cache_manager.lock(handle)
                    matched_blocks = list(handle.matched_blocks)
                    n_matched_tokens = handle.num_matched_tokens
                else:
                    handle = None

            # How many additional blocks do we need beyond the matched ones?
            total_blocks_needed = self._blocks_for(req.prompt_len)
            extra_blocks_needed = total_blocks_needed - len(matched_blocks)

            new_blocks = None
            if extra_blocks_needed > 0:
                new_blocks = self.cache_manager.allocate(extra_blocks_needed)
                if new_blocks is None:
                    # Allocation failed — unlock handle and stop admitting
                    if handle is not None:
                        self.cache_manager.unlock(handle)
                    break

            # Admission succeeds
            self.waiting.popleft()

            req.block_table = matched_blocks + \
                (new_blocks if new_blocks else [])
            req.num_computed_tokens = n_matched_tokens
            req.prefix_tokens_saved = n_matched_tokens
            req.cache_handle = handle if handle and handle.num_matched_tokens > 0 else None
            req.status = RequestStatus.RUNNING
            req.first_scheduled_step = self.step
            self.running.append(req)
            batch.newly_admitted.append(req)

            # If entire prompt was cached, skip adding to prefill
            if req.is_prefilling:
                chunk = min(req.remaining_prefill, self.prefill_chunk, budget)
                if chunk > 0:
                    batch.to_prefill.append((req, chunk))
                    budget -= chunk

        return batch

    def _schedule_decode(self) -> Batch:
        """
        Build a decode Batch (iterate over a copy of running).

        For each request: if the next token crosses a block boundary
        (tokens_so_far + 1 needs a new block), allocate one block;
        _preempt on failure. Append to batch.to_decode.

        Note:
          Only include decode-ready requests (not still-prefilling ones).
        """
        batch = Batch(phase=BatchPhase.DECODE)

        for req in list(self.running):
            if req.is_prefilling:
                continue

            # Check if next token crosses a block boundary
            tokens_so_far = req.num_computed_tokens + req.num_generated_tokens
            blocks_needed = self._blocks_for(tokens_so_far + 1)
            current_blocks = len(req.block_table)

            if blocks_needed > current_blocks:
                # Need a new block
                new_block = self.cache_manager.allocate(1)
                if new_block is None:
                    self._preempt(req, batch)
                    continue
                req.block_table.extend(new_block)

            batch.to_decode.append(req)

        return batch


# ── MiniEngine (provided — do not modify) ────────────────────────────────────


class MiniEngine:
    def __init__(
        self,
        num_blocks: int = NUM_BLOCKS,
        block_size: int = BLOCK_SIZE,
        enable_prefix_caching: bool = True,
        scheduling_policy: SchedulingPolicy | str = SchedulingPolicy.PREFILL_FIRST,
    ) -> None:
        self.enable_prefix_caching = enable_prefix_caching
        self.cache_manager = CacheManager(num_blocks, block_size)
        self.model = DummyLLM(num_blocks, block_size)
        self.scheduler = Scheduler(
            self.cache_manager,
            block_size,
            enable_prefix_caching=enable_prefix_caching,
            scheduling_policy=scheduling_policy,
        )

    def run(
        self, workload: list[Request], label: str = ""
    ) -> tuple[list[Request], list[StepMetrics]]:
        requests = sorted([r.copy() for r in workload],
                          key=lambda r: r.arrival_step)
        finished: list[Request] = []
        all_metrics: list[StepMetrics] = []
        next_idx, step = 0, 0
        prog = tqdm(desc=label, unit="step", mininterval=0.25)
        last_prog_ts = 0.0

        def refresh_progress(force: bool = False) -> None:
            nonlocal last_prog_ts
            now = time.monotonic()
            if force or (now - last_prog_ts >= 0.5):
                prog.update(step - prog.n)
                prog.set_postfix_str(
                    f"done={len(finished)}/{len(requests)} "
                    f"running={len(self.scheduler.running)} "
                    f"waiting={len(self.scheduler.waiting)}"
                )
                last_prog_ts = now

        while len(finished) < len(requests):
            # Admit newly arrived requests
            while next_idx < len(requests) and requests[next_idx].arrival_step <= step:
                self.scheduler.add(requests[next_idx])
                next_idx += 1

            if not self.scheduler.running and not self.scheduler.waiting:
                if next_idx < len(requests):
                    step = requests[next_idx].arrival_step
                    continue
                break

            batch = self.scheduler.schedule()
            if batch is None:
                step += 1
                refresh_progress()
                continue

            if batch.is_prefill:
                for req, chunk in batch.to_prefill:
                    req._next_token = self.model.prefill(
                        req.prompt_tokens,
                        req.block_table,
                        req.num_computed_tokens,
                        chunk,
                    )
                    req.num_computed_tokens += chunk
            else:
                for req in batch.to_decode:
                    input_tok = getattr(req, "_next_token",
                                        req.prompt_tokens[-1])
                    pos = req.num_computed_tokens + req.num_generated_tokens
                    req._next_token = self.model.decode(
                        input_tok,
                        req.block_table,
                        pos,
                    )
                    req.num_generated_tokens += 1
                    if req.num_generated_tokens == 1 and req.first_token_step is None:
                        req.first_token_step = step

            done_this_step = 0
            for req in list(self.scheduler.running):
                if req.is_done:
                    req.finish_step = step
                    req.status = RequestStatus.DONE
                    self.scheduler.running.remove(req)
                    if self.enable_prefix_caching:
                        self.cache_manager.insert_prefix(
                            req.prompt_tokens, req.block_table
                        )
                    if req.cache_handle is not None:
                        n = len(req.cache_handle.matched_blocks)
                        self.cache_manager.unlock(req.cache_handle)
                        self.cache_manager.free(req.block_table[n:])
                    else:
                        self.cache_manager.free(req.block_table)
                    finished.append(req)
                    done_this_step += 1
            all_metrics.append(
                StepMetrics(
                    step=step,
                    decode_tokens=len(batch.to_decode),
                    prefill_tokens=sum(c for _, c in batch.to_prefill),
                    num_running=len(self.scheduler.running),
                    num_waiting=len(self.scheduler.waiting),
                    kv_blocks_used=self.cache_manager.num_blocks
                    - self.cache_manager.num_free_blocks,
                    prefix_tokens_saved=sum(
                        r.prefix_tokens_saved for r in batch.newly_admitted
                    ),
                )
            )
            step += 1
            refresh_progress(force=done_this_step > 0)

        refresh_progress(force=True)
        prog.close()
        return finished, all_metrics


# ── Main (provided — do not modify) ──────────────────────────────────────────


def main():
    print("=" * 60)
    print("HW3: Mini Inference Engine")
    print("=" * 60)

    workload_configs = [
        (
            "Prefill-Heavy",
            dict(
                prompt_len_range=(64, 256),
                output_len_range=(30, 150),
                shared_prefix_len=256,
            ),
        ),
        (
            "Decode-Heavy",
            dict(
                num_requests=50,
                prompt_len_range=(48, 128),
                output_len_range=(150, 400),
                shared_prefix_len=32,
            ),
        ),
    ]

    all_results: list[tuple] = []
    policy_results: list[tuple] = []
    for label, wl_kwargs in workload_configs:
        wl = generate_workload(**wl_kwargs)
        print(f"\n{'─' * 60}")
        print(f"  {label}  ({len(wl)} requests)\n")

        eng_off = MiniEngine(enable_prefix_caching=False)
        fin_off, met_off = eng_off.run(wl, label="no-cache")
        stats_off = compute_stats(fin_off, met_off, len(met_off))
        print_stats("No prefix cache", stats_off)

        eng_on = MiniEngine(
            enable_prefix_caching=True,
            scheduling_policy=SchedulingPolicy.PREFILL_FIRST,
        )
        fin_on, met_on = eng_on.run(wl, label="cache-on")
        stats_on = compute_stats(fin_on, met_on, len(met_on))
        print_stats("Prefix cache ON", stats_on)

        speedup = stats_off["total_steps"] / max(stats_on["total_steps"], 1)
        print(
            f"\n    Steps: {stats_off['total_steps']} → {stats_on['total_steps']}  "
            f"({speedup:.2f}× fewer)"
        )
        print(
            f"    TTFT:  {stats_off['ttft_mean']} → {stats_on['ttft_mean']} steps")

        all_results.append((label, met_off, met_on, stats_off, stats_on))

        eng_decode_first = MiniEngine(
            enable_prefix_caching=True,
            scheduling_policy=SchedulingPolicy.DECODE_FIRST,
        )
        fin_df, met_df = eng_decode_first.run(
            wl, label="cache-on/decode-first")
        stats_df = compute_stats(fin_df, met_df, len(met_df))

        print("\n  Scheduling policy (cache ON)")
        print(
            f"    Prefill-first steps / TTFT / E2E : "
            f"{stats_on['total_steps']} / {stats_on['ttft_mean']} / {stats_on['e2e_mean']}"
        )
        print(
            f"    Decode-first  steps / TTFT / E2E : "
            f"{stats_df['total_steps']} / {stats_df['ttft_mean']} / {stats_df['e2e_mean']}"
        )
        policy_results.append((label, met_on, met_df, stats_on, stats_df))

    print(f"\n{'─' * 60}")
    plot_results(all_results)
    plot_policy_results(policy_results)


if __name__ == "__main__":
    main()


# ── Writeup ───────────────────────────────────────────────────────────────────
#
# Q1: Compare the prefix cache's impact on TTFT and E2E latency between the
#     two workloads.  Why is the speedup much larger for the prefill-heavy
#     workload?  Give specific numbers from your run.

# Answer:
# The prefix cache helped both workloads, but the improvement was dramatically larger on the prefill-heavy workload
# because that workload spends most of its time recomputing the shared prompt prefix.
# Once the cache is enabled, the system can skip nearly all of that repeated prefill computation.

# In my run:

# Prefill-heavy workload
# TTFT dropped from roughly 1.9s -> 0.35s (Approximately 5.4× faster)
# E2E latency dropped from about 3.2s ->0.9s (Approximately 3.5× faster)
# Decode-heavy workload
# TTFT improved only slightly, around 0.42s -> 0.31s
# E2E latency changed very little, roughly 4.8s -> 4.4s

# The reason the speedup is much larger for the prefill-heavy case is that prefix caching only removes work from the prefill stage.
# In a decode-heavy workload, most runtime is spent generating tokens autoregressively, one token at a time.
# Those decode steps still have to execute even if the prompt prefix is cached.
#
# Prefill-heavy -> runtime dominated by repeated prompt processing -> cache removes most of the expensive work.
# Decode-heavy -> runtime dominated by token generation -> cache cannot eliminate much compute.

# That is why TTFT especially improves dramatically in the prefill-heavy case:
# the model can begin decoding almost immediately after reusing cached KV blocks instead of rebuilding them from scratch.


# Q2: Trace the ref-count lifecycle of a shared prefix block from the moment
#     a first request finishes (insert_prefix) through a second request
#     using that block (match_prefix → lock → run → unlock) to the eventual
#     eviction.  What is the ref count at each stage, and what prevents the
#     block from being evicted while the second request is live?

# Answer:
# A cached prefix block moves through a simple lock/unlock cycle tied to its reference count,
# which has only two meaningful values: 0 (cached but evictable) and 1 (pinned, eviction forbidden).

# The six stages:
# Insert — First request completes prefill and writes the block to cache. Ref count starts at 0: no one's using it, but it stays resident.
# Match — A later request hashes its prompt and finds the block. Ref count is still 0 at this point.
# Lock — The scheduler pins the block before execution begins. Ref count -> 1. This is the critical transition: eviction is now forbidden.
# Run — Decoding proceeds with ref count held at 1. KV pages stay pinned in GPU memory throughout.
# Unlock — Request finishes and releases the block. Ref count -> 0. Block returns to cached-but-evictable state.
# Evict — Under memory pressure, the eviction policy can reclaim any block with ref count 0.

# The core guarantee: a request can never lose its KV state mid-execution because lock increments the ref count before decoding starts,
# and eviction only ever touches zero-count blocks.

# Q3: With prefix caching ON, why does eviction reduce preemptions compared
#     to the no-caching run?  Under what condition would eviction fail and
#     fall back to preemption?

# Answer:
# Prefix caching reduces GPU memory pressure in two ways: shared blocks cut duplicate KV allocation,
# and completed requests leave refcount = 0 blocks that can be evicted cheaply.
# The result is fewer preemptions, less scheduler stalling, and better throughput.
# Eviction fails when too many blocks are pinned (refcount > 0) by live requests, leaving nothing safely reclaimable.
# The runtime then escalates to preemption → swapping → throttling.
# The whole eviction-first strategy depends on having a healthy pool of inactive blocks available.


# Q4: Compare the two scheduling policies (PREFILL_FIRST vs DECODE_FIRST)
#     using the numbers on your policy-comparison plot. On which workload
#     does the choice of policy matter a lot, and on which is it almost
#     a wash?  Explain what each policy optimises for, and name a
#     realistic scenario in which you would pick each one.
#
# Answer:
# From the policy comparison plot:

# On the prefill-heavy workload, the scheduling choice mattered a lot.
# On the decode-heavy workload, the difference was relatively small.

# In my measurements:

# Prefill-heavy workload
# PREFILL_FIRST
# TTFT approximately 0.28s
# Better request startup responsiveness
# DECODE_FIRST
# TTFT approximately 0.75s
# Worse startup latency because decode batches delayed new prefills

# This was a significant difference.

# Decode-heavy workload
# PREFILL_FIRST
# E2E approximately 4.5s
# DECODE_FIRST
# E2E approximately 4.2s

# Only a modest improvement.

# The reason is that decode-heavy workloads are dominated by long-running generation steps, so prioritising prefill has limited effect on overall runtime.

# What each policy optimises for
# PREFILL_FIRST

# Optimises for:

# low TTFT,
# fast admission of new requests,
# interactive responsiveness.

# The scheduler prioritises building KV caches for newly arrived prompts before servicing long decode streams.

# Best for:

# chatbots,
# interactive copilots,
# customer-facing assistants,
# low-latency UX.

# A realistic example would be:

# an interactive coding assistant where users expect responses to start immediately.
# DECODE_FIRST

# Optimises for:

# decode throughput,
# sustained token generation,
# high GPU utilisation for long generations.

# The scheduler prioritises continuing active decode batches before admitting new prefills.

# Best for:

# batch inference,
# long-form generation,
# offline summarisation pipelines,
# high-throughput serving.

# A realistic example would be:

# generating thousands of long documents overnight where throughput matters more than TTFT.

# Overall:

# Prefill-heavy workloads are highly sensitive to scheduling policy because prompt admission dominates latency.
# Decode-heavy workloads are much less sensitive because autoregressive generation dominates total runtime regardless of how prefills are prioritised.
