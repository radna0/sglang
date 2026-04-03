from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.moe.routed_experts_capturer import get_global_experts_capturer
from sglang.srt.managers.io_struct import (
    AbortReq,
    BatchEmbeddingOutput,
    BatchTokenIDOutput,
)
from sglang.srt.managers.schedule_batch import (
    BaseFinishReason,
    Req,
    RequestStage,
    ScheduleBatch,
)
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.server_args import get_global_server_args
from sglang.srt.speculative.dflash_utils import resolve_dflash_overlap_token_ids
from sglang.srt.tracing.trace import trace_slice, trace_slice_batch, trace_slice_end

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import (
        EmbeddingBatchResult,
        GenerationBatchResult,
        ScheduleBatch,
        Scheduler,
    )

logger = logging.getLogger(__name__)

DEFAULT_FORCE_STREAM_INTERVAL = 50


def _parse_debug_output_len_range() -> tuple[int, int] | None:
    raw = (os.environ.get("SGLANG_DEBUG_OUTPUT_ID_LEN_RANGE") or "").strip()
    if not raw:
        return None
    try:
        lo_s, hi_s = raw.split(":", 1)
        lo = int(lo_s.strip())
        hi = int(hi_s.strip())
    except Exception:
        return None
    if hi < lo:
        lo, hi = hi, lo
    return lo, hi


def _parse_debug_logits_topk() -> int:
    raw = (os.environ.get("SGLANG_DEBUG_LOGITS_TOPK") or "").strip()
    if not raw:
        return 0
    try:
        k = int(raw)
    except Exception:
        return 0
    return max(0, k)


def _fa3_trace_output_ids_enabled() -> bool:
    return os.environ.get("SGLANG_FA3_TRACE_OUTPUT_IDS", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _overlap_debug_enabled() -> bool:
    return (os.environ.get("SGLANG_DFLASH_DEBUG_OVERLAP_RELEASE") or "").strip().lower() not in (
        "",
        "0",
        "false",
        "off",
        "no",
    )


class SchedulerOutputProcessorMixin:
    """
    This class implements the output processing logic for Scheduler.
    We put them into a separate file to make the `scheduler.py` shorter.
    """

    def _release_kv_cache_and_draft(
        self: Scheduler, req: Req, *, is_insert: bool = True
    ):
        if req.req_pool_idx is None:
            if _overlap_debug_enabled():
                logger.warning(
                    "Skip generic KV release for already-detached DFlash req: rid=%s completed=%s committed_len=%s allocated_len=%s",
                    getattr(req, "rid", None),
                    getattr(req, "finished", lambda: False)() if hasattr(req, "finished") else None,
                    getattr(req, "kv_committed_len", None),
                    getattr(req, "kv_allocated_len", None),
                )
            draft_worker = getattr(self, "draft_worker", None)
            hook = getattr(draft_worker, "on_req_finished", None) if draft_worker else None
            if hook is not None:
                hook(req)
            return
        release_kv_cache(req, self.tree_cache, is_insert=is_insert)
        draft_worker = getattr(self, "draft_worker", None)
        hook = getattr(draft_worker, "on_req_finished", None) if draft_worker else None
        if hook is not None:
            hook(req)

    def _get_storage_backend_type(self) -> str:
        """Get storage backend type from tree_cache."""
        storage_backend_type = "none"
        cache_controller = getattr(self.tree_cache, "cache_controller", None)
        if cache_controller and hasattr(cache_controller, "storage_backend"):
            storage_backend = cache_controller.storage_backend
            if storage_backend is not None:
                storage_backend_type = type(storage_backend).__name__
        return storage_backend_type

    def _get_cached_tokens_details(self, req: Req) -> Optional[dict]:
        """Get detailed cache breakdown for a request, if available.

        Returns:
            - None if HiCache is not enabled
            - {"device": X, "host": Y} if HiCache enabled but L3 storage is not
            - {"device": X, "host": Y, "storage": Z, "storage_backend": "..."} if L3 enabled
        """
        # Only show details if HiCache is enabled
        if not getattr(self, "enable_hierarchical_cache", False):
            return None

        # Only show if there are any cached tokens
        if (
            req.cached_tokens_device > 0
            or req.cached_tokens_host > 0
            or req.cached_tokens_storage > 0
        ):
            details = {
                "device": req.cached_tokens_device,
                "host": req.cached_tokens_host,
            }
            # Only include storage fields if L3 storage is enabled
            if getattr(self, "enable_hicache_storage", False):
                details["storage"] = req.cached_tokens_storage
                details["storage_backend"] = self._get_storage_backend_type()
            return details
        return None

    def process_batch_result_prebuilt(self: Scheduler, batch: ScheduleBatch):
        assert self.disaggregation_mode == DisaggregationMode.DECODE
        for req in batch.reqs:
            req.check_finished()
            if req.finished():
                req.time_stats.forward_entry_time = req.time_stats.completion_time = (
                    time.perf_counter()
                )
                trace_slice_end(
                    RequestStage.DECODE_QUICK_FINISH,
                    req.rid,
                    thread_finish_flag=True,
                )
                self._release_kv_cache_and_draft(req)

        # Note: Logprobs should be handled on the prefill engine.
        trace_slice_batch(RequestStage.DECODE_FAKE_OUTPUT, batch.reqs)
        self.stream_output(batch.reqs, batch.return_logprob)

    def maybe_collect_routed_experts(self: Scheduler, req: Req):
        """Collect routed experts for a finished request."""
        req.routed_experts = get_global_experts_capturer().get_routed_experts(
            req_pool_idx=req.req_pool_idx,
            seqlen=req.seqlen,
            req_to_token_pool=self.req_to_token_pool,
        )

    def maybe_collect_customized_info(
        self: Scheduler, i: int, req: Req, logits_output: LogitsProcessorOutput
    ):
        if logits_output is not None and logits_output.customized_info is not None:
            if req.customized_info is None:
                req.customized_info = {}
            for k, v in logits_output.customized_info.items():
                if k not in req.customized_info:
                    req.customized_info[k] = []
                req.customized_info[k].append(v[i])

    def _handle_dflash_overlap_preprocessed_req(
        self: Scheduler,
        req: Req,
        i: int,
        logits_output: LogitsProcessorOutput,
    ) -> None:
        """Handle a DFlash req whose output_ids/finish state were committed in verify.

        In this mode the output processor should only:
        - release KV exactly once for newly finished requests
        - collect customized_info
        - avoid replaying token commit logic
        """
        if not req.is_retracted and req.req_pool_idx is not None:
            # DFlash overlap commits output_ids inside verify. Tree-cache refresh and
            # final release paths consume fill_ids/cache_protected_len, so rebuild the
            # request-visible sequence before any cache action runs.
            req.fill_ids = req.origin_input_ids + req.output_ids

        if self.enable_overlap and (req.finished() or req.is_retracted):
            if _overlap_debug_enabled():
                logger.warning(
                    "DFLASH overlap pre-processed req: rid=%s finished=%s retracted=%s req_pool_idx=%s kv_committed_len=%s kv_allocated_len=%s cache_protected_len=%s prefix_indices_len=%s committed_freed=%s overalloc_freed=%s",
                    getattr(req, "rid", None),
                    req.finished(),
                    req.is_retracted,
                    getattr(req, "req_pool_idx", None),
                    getattr(req, "kv_committed_len", None),
                    getattr(req, "kv_allocated_len", None),
                    getattr(req, "cache_protected_len", None),
                    (
                        None
                        if getattr(req, "prefix_indices", None) is None
                        else int(len(getattr(req, "prefix_indices")))
                    ),
                    getattr(req, "kv_committed_freed", None),
                    getattr(req, "kv_overallocated_freed", None),
            )
            if req.finished() and req.req_pool_idx is not None:
                if not req.kv_overallocated_freed:
                    # Keep the overallocated tail intact here. In overlap mode the
                    # committed prefix is refreshed into the tree cache first, then
                    # `release_kv_cache()` frees the remaining speculative tail via
                    # `pop_overallocated_kv_cache()`. Collapsing the allocator length
                    # early would strand borrowed verify slots and trigger an idle leak.
                    self.maybe_collect_routed_experts(req)
                    committed_token_len = int(
                        getattr(req, "kv_committed_len", 0) or 0
                    )
                    if committed_token_len <= 0 or committed_token_len > len(req.fill_ids):
                        committed_token_len = len(req.fill_ids)
                    prefix_indices = getattr(req, "prefix_indices", None)
                    prefix_indices_len = (
                        0 if prefix_indices is None else int(len(prefix_indices))
                    )
                    cache_protected_len = int(
                        getattr(req, "cache_protected_len", 0) or 0
                    )
                    needs_cache_refresh = (
                        cache_protected_len < committed_token_len
                        or prefix_indices_len < committed_token_len
                    )
                    refreshed_prefix = None
                    req_to_token_pool = getattr(self, "req_to_token_pool", None)
                    if needs_cache_refresh:
                        current_prefix = getattr(req, "prefix_indices", None)
                        if (
                            current_prefix is not None
                            and int(len(current_prefix)) >= committed_token_len
                        ):
                            refreshed_prefix = current_prefix[:committed_token_len].to(
                                torch.int64, copy=True
                            )
                        elif req_to_token_pool is not None and req.req_pool_idx is not None:
                            refreshed_prefix = req_to_token_pool.req_to_token[
                                req.req_pool_idx, :committed_token_len
                            ].to(torch.int64, copy=True)
                        if refreshed_prefix is not None:
                            req.prefix_indices = refreshed_prefix
                            if (
                                req_to_token_pool is not None
                                and req.req_pool_idx is not None
                            ):
                                req_to_token_pool.write(
                                    (req.req_pool_idx, slice(0, committed_token_len)),
                                    refreshed_prefix,
                                )
                    if self.server_args.disaggregation_decode_enable_offload_kvcache:
                        if not self.decode_offload_manager.offload_kv_cache(req):
                            if _overlap_debug_enabled():
                                req_row = (
                                    req_to_token_pool.req_to_token[
                                        req.req_pool_idx, : len(req.fill_ids)
                                    ]
                                    if req_to_token_pool is not None
                                    and req.req_pool_idx is not None
                                    else None
                                )
                                prefix_dump = (
                                    None
                                    if refreshed_prefix is None
                                    else (
                                        refreshed_prefix.tolist()
                                        if refreshed_prefix.numel() <= 16
                                        else refreshed_prefix[-8:].tolist()
                                    )
                                )
                                req_row_dump = (
                                    req_row.tolist()
                                    if req_row.numel() <= 16
                                    else req_row[-8:].tolist()
                                )
                                logger.warning(
                                    "DFLASH overlap refreshed pre-processed req bookkeeping: rid=%s committed_token_len=%s cache_protected_len=%s prefix_indices_len=%s prefix_dump=%s req_row_dump=%s",
                                    getattr(req, "rid", None),
                                    committed_token_len,
                                    getattr(req, "cache_protected_len", None),
                                    (
                                        None
                                        if refreshed_prefix is None
                                        else int(len(refreshed_prefix))
                                    ),
                                    prefix_dump,
                                    req_row_dump,
                                )
                            if req.req_pool_idx is not None:
                                self._release_kv_cache_and_draft(req, is_insert=False)
                            else:
                                if _overlap_debug_enabled():
                                    logger.warning(
                                        "DFLASH overlap finished req already released before final cleanup; skipping generic release: "
                                        "rid=%s kv_committed_len=%s kv_allocated_len=%s committed_freed=%s overalloc_freed=%s",
                                        getattr(req, "rid", None),
                                        getattr(req, "kv_committed_len", None),
                                        getattr(req, "kv_allocated_len", None),
                                        getattr(req, "kv_committed_freed", None),
                                        getattr(req, "kv_overallocated_freed", None),
                                    )
                                draft_worker = getattr(self, "draft_worker", None)
                                hook = (
                                    getattr(draft_worker, "on_req_finished", None)
                                    if draft_worker
                                    else None
                                )
                                if hook is not None:
                                    hook(req)
                    else:
                        if _overlap_debug_enabled():
                            print(
                                "DFLASH overlap final release debug "
                                f"rid={getattr(req, 'rid', None)} "
                                f"kv_committed_len={getattr(req, 'kv_committed_len', None)} "
                                f"kv_allocated_len={getattr(req, 'kv_allocated_len', None)} "
                                f"fill_ids_len={len(req.fill_ids)} "
                                f"output_ids_len={len(req.output_ids)} "
                                f"prefix_indices_len={0 if getattr(req, 'prefix_indices', None) is None else len(getattr(req, 'prefix_indices'))} "
                                f"cache_protected_len={getattr(req, 'cache_protected_len', None)}",
                                flush=True,
                            )
                        if _overlap_debug_enabled() and needs_cache_refresh:
                            req_row = (
                                req_to_token_pool.req_to_token[
                                    req.req_pool_idx, : len(req.fill_ids)
                                ]
                                if req_to_token_pool is not None
                                and req.req_pool_idx is not None
                                else None
                            )
                            prefix_dump = (
                                None
                                if refreshed_prefix is None
                                else (
                                    refreshed_prefix.tolist()
                                    if refreshed_prefix.numel() <= 16
                                    else refreshed_prefix[-8:].tolist()
                                )
                            )
                            req_row_dump = (
                                req_row.tolist()
                                if req_row.numel() <= 16
                                else req_row[-8:].tolist()
                            )
                            logger.warning(
                                "DFLASH overlap refreshed pre-processed req bookkeeping: rid=%s committed_token_len=%s cache_protected_len=%s prefix_indices_len=%s prefix_dump=%s req_row_dump=%s",
                                getattr(req, "rid", None),
                                committed_token_len,
                                getattr(req, "cache_protected_len", None),
                                (
                                    None
                                    if refreshed_prefix is None
                                    else int(len(refreshed_prefix))
                                ),
                                prefix_dump,
                                req_row_dump,
                            )
                        if req.req_pool_idx is not None:
                            self._release_kv_cache_and_draft(req, is_insert=False)
                        else:
                            if _overlap_debug_enabled():
                                logger.warning(
                                    "DFLASH overlap finished req already released before final cleanup; skipping generic release: "
                                    "rid=%s kv_committed_len=%s kv_allocated_len=%s committed_freed=%s overalloc_freed=%s",
                                    getattr(req, "rid", None),
                                    getattr(req, "kv_committed_len", None),
                                    getattr(req, "kv_allocated_len", None),
                                    getattr(req, "kv_committed_freed", None),
                                    getattr(req, "kv_overallocated_freed", None),
                                )
                            draft_worker = getattr(self, "draft_worker", None)
                            hook = (
                                getattr(draft_worker, "on_req_finished", None)
                                if draft_worker
                                else None
                            )
                            if hook is not None:
                                hook(req)

                    req.time_stats.completion_time = time.perf_counter()
                    if _overlap_debug_enabled():
                        logger.warning(
                            "DFLASH overlap released pre-processed req: rid=%s req_pool_idx=%s committed_freed=%s overalloc_freed=%s",
                            getattr(req, "rid", None),
                            getattr(req, "req_pool_idx", None),
                            getattr(req, "kv_committed_freed", None),
                            getattr(req, "kv_overallocated_freed", None),
                        )
        elif not req.is_retracted and req.req_pool_idx is not None:
            # DFlash overlap already committed the accepted prefix and KV mapping in
            # verify. With the overlap output-processing barrier in place, this
            # callback now runs before the next scheduling step can reuse the live
            # running batch. At this point we must refresh the unfinished request in
            # radix/SWA so queued prefills cannot evict the just-committed prefix.
            self.tree_cache.cache_unfinished_req(req)

        self.maybe_collect_customized_info(i, req, logits_output)

    def process_batch_result_prefill(
        self: Scheduler,
        batch: ScheduleBatch,
        result: Union[GenerationBatchResult, EmbeddingBatchResult],
    ):
        skip_stream_req = None

        if self.is_generation:
            if result.copy_done is not None:
                result.copy_done.synchronize()

            (
                logits_output,
                next_token_ids,
                extend_input_len_per_req,
                extend_logprob_start_len_per_req,
            ) = (
                result.logits_output,
                result.next_token_ids,
                result.extend_input_len_per_req,
                result.extend_logprob_start_len_per_req,
            )

            # Move next_token_ids and logprobs to cpu
            next_token_ids = next_token_ids.tolist()
            if batch.return_logprob:
                if logits_output.next_token_logprobs is not None:
                    logits_output.next_token_logprobs = (
                        logits_output.next_token_logprobs.tolist()
                    )
                if logits_output.input_token_logprobs is not None:
                    logits_output.input_token_logprobs = tuple(
                        logits_output.input_token_logprobs.tolist()
                    )

            hidden_state_offset = 0

            # Check finish conditions
            logprob_pt = 0

            for i, (req, next_token_id) in enumerate(zip(batch.reqs, next_token_ids)):
                if req.finished() or req.is_retracted:
                    # decode req in mixed batch or retracted req
                    continue

                if req.is_chunked <= 0:
                    if req.time_stats.prefill_finished_ts == 0.0:
                        req.time_stats.prefill_finished_ts = time.time()

                    # req output_ids are set here
                    req.output_ids.append(next_token_id)
                    req.check_finished()

                    if req.finished():
                        self.maybe_collect_routed_experts(req)
                        self._release_kv_cache_and_draft(req)
                        req.time_stats.completion_time = time.perf_counter()
                    elif not batch.decoding_reqs or req not in batch.decoding_reqs:
                        # This updates radix so others can match
                        self.tree_cache.cache_unfinished_req(req)

                    self.maybe_collect_customized_info(i, req, logits_output)

                    if batch.return_logprob:
                        assert extend_logprob_start_len_per_req is not None
                        assert extend_input_len_per_req is not None
                        extend_logprob_start_len = extend_logprob_start_len_per_req[i]
                        extend_input_len = extend_input_len_per_req[i]

                        num_input_logprobs = self._calculate_num_input_logprobs(
                            req, extend_input_len, extend_logprob_start_len
                        )

                        if req.return_logprob:
                            self.add_logprob_return_values(
                                i,
                                req,
                                logprob_pt,
                                next_token_ids,
                                num_input_logprobs,
                                logits_output,
                            )
                        logprob_pt += num_input_logprobs

                    if (
                        req.return_hidden_states
                        and logits_output.hidden_states is not None
                    ):
                        req.hidden_states.append(
                            logits_output.hidden_states[
                                hidden_state_offset : (
                                    hidden_state_offset := hidden_state_offset
                                    + len(req.origin_input_ids)
                                )
                            ]
                            .cpu()
                            .clone()
                            .tolist()
                        )

                    if req.grammar is not None:
                        # FIXME: this try-except block is for handling unexpected xgrammar issue.
                        try:
                            req.grammar.accept_token(next_token_id)
                        except ValueError as e:
                            # Grammar accept_token can raise ValueError if the token is not in the grammar.
                            # This can happen if the grammar is not set correctly or the token is invalid.
                            logger.error(
                                f"Grammar accept_token failed for req {req.rid} with token {next_token_id}: {e}"
                            )
                            self.abort_request(AbortReq(rid=req.rid))
                        req.grammar.finished = req.finished()

                    trace_slice(
                        RequestStage.PREFILL_FORWARD,
                        req.rid,
                        auto_next_anon=not req.finished(),
                        thread_finish_flag=req.finished(),
                    )

                else:
                    # being chunked reqs' prefill is not finished
                    req.is_chunked -= 1
                    # There is only at most one request being currently chunked.
                    # Because this request does not finish prefill,
                    # we don't want to stream the request currently being chunked.
                    skip_stream_req = req

                    # Incrementally update input logprobs.
                    if batch.return_logprob:
                        extend_logprob_start_len = extend_logprob_start_len_per_req[i]
                        extend_input_len = extend_input_len_per_req[i]
                        if extend_logprob_start_len < extend_input_len:
                            # Update input logprobs.
                            num_input_logprobs = self._calculate_num_input_logprobs(
                                req, extend_input_len, extend_logprob_start_len
                            )
                            if req.return_logprob:
                                self.add_input_logprob_return_values(
                                    i,
                                    req,
                                    logits_output,
                                    logprob_pt,
                                    num_input_logprobs,
                                    last_prefill_chunk=False,
                                )
                            logprob_pt += num_input_logprobs

                    trace_slice(
                        RequestStage.PREFILL_CHUNKED_FORWARD,
                        req.rid,
                        auto_next_anon=True,
                    )

        else:  # embedding or reward model
            if result.copy_done is not None:
                result.copy_done.synchronize()

            is_sparse = envs.SGLANG_EMBEDDINGS_SPARSE_HEAD.is_set()

            embeddings = result.embeddings

            if is_sparse:
                batch_ids, token_ids = embeddings.indices()
                values = embeddings.values()

                embeddings = [{} for _ in range(embeddings.size(0))]
                for i in range(batch_ids.shape[0]):
                    embeddings[batch_ids[i].item()][token_ids[i].item()] = values[
                        i
                    ].item()
            else:
                if isinstance(embeddings, torch.Tensor):
                    embeddings = embeddings.tolist()
                else:
                    embeddings = [tensor.tolist() for tensor in embeddings]

            # Check finish conditions
            for i, req in enumerate(batch.reqs):
                if req.is_retracted:
                    continue

                req.embedding = embeddings[i]
                if req.is_chunked <= 0:
                    # Dummy output token for embedding models
                    req.output_ids.append(0)
                    req.check_finished()

                    if req.finished():
                        self._release_kv_cache_and_draft(req)
                    else:
                        self.tree_cache.cache_unfinished_req(req)
                else:
                    # being chunked reqs' prefill is not finished
                    req.is_chunked -= 1

                trace_slice(
                    RequestStage.PREFILL_FORWARD,
                    req.rid,
                    auto_next_anon=not req.finished(),
                    thread_finish_flag=req.finished(),
                )

        self.stream_output(batch.reqs, batch.return_logprob, skip_stream_req)

        if self.current_scheduler_metrics_enabled:
            can_run_cuda_graph = getattr(result, "can_run_cuda_graph", False)
            self.log_prefill_stats(
                prefill_stats=batch.prefill_stats,
                can_run_cuda_graph=can_run_cuda_graph,
                dp_cooperation_info=batch.dp_cooperation_info,
            )

    def _resolve_spec_overlap_token_ids(
        self: Scheduler, result: GenerationBatchResult, batch: ScheduleBatch
    ) -> List[List[int]]:
        """Resolve the padding next token ids for speculative decoding with overlap."""
        assert result.next_token_ids.is_cpu
        assert result.accept_lens.is_cpu

        accept_lens = result.accept_lens.tolist()
        result.num_accepted_tokens = sum(accept_lens) - len(batch.reqs)
        if result.accept_length_per_req_cpu is None:
            result.accept_length_per_req_cpu = [x - 1 for x in accept_lens]

        if batch.spec_algorithm.is_dflash_family():
            predict_tokens = resolve_dflash_overlap_token_ids(
                flat_token_ids=result.next_token_ids,
                accept_lens=result.accept_lens,
            )
            return predict_tokens

        next_token_ids = result.next_token_ids.tolist()
        stride = self.draft_worker.speculative_num_draft_tokens
        predict_tokens = [
            next_token_ids[i * stride : i * stride + accept_lens[i]]
            for i, _req in enumerate(batch.reqs)
        ]

        for i, req in enumerate(batch.reqs):
            req.kv_committed_len += accept_lens[i]
            req.spec_verify_ct += 1

            accepted_draft_tokens = result.accept_length_per_req_cpu[i]
            req.spec_accepted_tokens += accepted_draft_tokens
            req.update_spec_acceptance_histogram(accepted_draft_tokens)

        return predict_tokens

    def _accumulate_dflash_ssd_metrics(
        self: Scheduler, result: GenerationBatchResult, batch: ScheduleBatch
    ) -> None:
        hit_cts = getattr(result, "spec_ssd_hit_ct", None)
        prepare_cts = getattr(result, "spec_ssd_prepare_ct", None)
        prepare_failure_cts = getattr(result, "spec_ssd_prepare_failure_ct", None)
        cache_pending = getattr(result, "spec_ssd_cache_pending", None)
        overlap_launch_cts = getattr(result, "spec_ssd_overlap_launch_ct", None)
        overlap_wait_cts = getattr(result, "spec_ssd_overlap_wait_ct", None)
        difficulty_gate_skip_cts = getattr(
            result, "spec_ssd_difficulty_gate_skip_ct", None
        )
        fanout_gate_skip_cts = getattr(result, "spec_ssd_fanout_gate_skip_ct", None)
        fanout_escalation_cts = getattr(
            result, "spec_ssd_fanout_escalation_ct", None
        )
        fanout_alt_budget = getattr(result, "spec_ssd_fanout_alt_budget", None)

        if (
            hit_cts is None
            and prepare_cts is None
            and prepare_failure_cts is None
            and cache_pending is None
            and overlap_launch_cts is None
            and overlap_wait_cts is None
            and difficulty_gate_skip_cts is None
            and fanout_gate_skip_cts is None
            and fanout_escalation_cts is None
            and fanout_alt_budget is None
        ):
            return

        for i, req in enumerate(batch.reqs):
            if hit_cts is not None and i < len(hit_cts):
                req.spec_ssd_hit_ct = int(hit_cts[i])
            if prepare_cts is not None and i < len(prepare_cts):
                req.spec_ssd_prepare_ct = int(prepare_cts[i])
            if prepare_failure_cts is not None and i < len(prepare_failure_cts):
                req.spec_ssd_prepare_failure_ct = int(prepare_failure_cts[i])
            if cache_pending is not None and i < len(cache_pending):
                req.spec_ssd_cache_pending = int(cache_pending[i])
            if overlap_launch_cts is not None and i < len(overlap_launch_cts):
                req.spec_ssd_overlap_launch_ct = int(overlap_launch_cts[i])
            if overlap_wait_cts is not None and i < len(overlap_wait_cts):
                req.spec_ssd_overlap_wait_ct = int(overlap_wait_cts[i])
            if (
                difficulty_gate_skip_cts is not None
                and i < len(difficulty_gate_skip_cts)
            ):
                req.spec_ssd_difficulty_gate_skip_ct = int(
                    difficulty_gate_skip_cts[i]
                )
            if fanout_gate_skip_cts is not None and i < len(fanout_gate_skip_cts):
                req.spec_ssd_fanout_gate_skip_ct = int(fanout_gate_skip_cts[i])
            if fanout_escalation_cts is not None and i < len(fanout_escalation_cts):
                req.spec_ssd_fanout_escalation_ct = int(fanout_escalation_cts[i])
            if fanout_alt_budget is not None and i < len(fanout_alt_budget):
                req.spec_ssd_fanout_alt_budget = int(fanout_alt_budget[i])

    def process_batch_result_idle(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ):
        if result.copy_done is not None:
            result.copy_done.synchronize()

        self.stream_output_generation(
            batch.reqs, batch.return_logprob, is_idle_batch=True
        )

    def process_batch_result_dllm(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ):
        if result.copy_done is not None:
            result.copy_done.synchronize()

        self.token_to_kv_pool_allocator.free_group_begin()

        for idx in range(batch.batch_size()):
            # If no new tokens generated, meaning the prefilling stage
            if not result.next_token_ids:
                break

            req = batch.reqs[idx]
            next_token_ids = result.next_token_ids[idx].tolist()
            self.num_generated_tokens += len(next_token_ids)

            for _token_idx, next_token_id in enumerate(next_token_ids):
                req.output_ids.append(next_token_id)
                req.check_finished()
                if req.finished():
                    self._release_kv_cache_and_draft(req)
                    req.time_stats.completion_time = time.perf_counter()
                    break

                self.tree_cache.cache_unfinished_req(req)

        self.stream_output(batch.reqs, batch.return_logprob)
        self.token_to_kv_pool_allocator.free_group_end()

        if self.current_scheduler_metrics_enabled:
            can_run_cuda_graph = getattr(result, "can_run_cuda_graph", False)
            self.log_prefill_stats(
                prefill_stats=batch.prefill_stats,
                can_run_cuda_graph=can_run_cuda_graph,
                dp_cooperation_info=batch.dp_cooperation_info,
            )

    def process_batch_result_decode(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ):
        if result.copy_done is not None:
            result.copy_done.synchronize()

        logits_output = result.logits_output
        can_run_cuda_graph = result.can_run_cuda_graph
        next_token_ids = result.next_token_ids
        force_plain_decode_output_processing = bool(
            getattr(result, "force_plain_decode_output_processing", False)
        )

        dflash_overlap_preprocessed = (
            self.enable_overlap
            and batch.is_spec_v2
            and batch.spec_algorithm.is_dflash_family()
            and bool(getattr(result, "dflash_overlap_preprocessed", False))
        )

        if batch.spec_algorithm.is_none() or force_plain_decode_output_processing:
            next_token_ids = next_token_ids.tolist()
            if batch.return_logprob:
                next_token_logprobs = logits_output.next_token_logprobs.tolist()
        elif batch.is_spec_v2 and not dflash_overlap_preprocessed:
            next_token_ids = self._resolve_spec_overlap_token_ids(result, batch)
        elif dflash_overlap_preprocessed:
            next_token_ids = [None] * len(batch.reqs)

        self.num_generated_tokens += len(batch.reqs)
        if not batch.spec_algorithm.is_none() and not force_plain_decode_output_processing:
            self.update_spec_metrics(batch.batch_size(), result.num_accepted_tokens)
            self._accumulate_dflash_ssd_metrics(result, batch)
        if self.enable_metrics:
            self.metrics_collector.increment_cuda_graph_pass(value=can_run_cuda_graph)

        self.token_to_kv_pool_allocator.free_group_begin()

        # NOTE: in any case, we should check finish here
        # if finished, also clean up committed kv cache and over-allocated kv cache here

        # Check finish condition
        debug_output_len_range = _parse_debug_output_len_range()
        for i, (req, next_token_id) in enumerate(zip(batch.reqs, next_token_ids)):
            req: Req

            if dflash_overlap_preprocessed and (
                self.enable_overlap and (req.finished() or req.is_retracted)
            ):
                self._handle_dflash_overlap_preprocessed_req(req, i, logits_output)
                continue

            if dflash_overlap_preprocessed:
                self._handle_dflash_overlap_preprocessed_req(req, i, logits_output)
                continue

            new_accepted_len = 1
            len_before = len(req.output_ids)
            if batch.spec_algorithm.is_none() or force_plain_decode_output_processing:
                req.output_ids.append(next_token_id)
            elif batch.is_spec_v2:
                # Only spec v2's output_ids are updated here.
                req.output_ids.extend(next_token_id)
                new_accepted_len = len(next_token_id)
            if (
                debug_output_len_range is not None
                and debug_output_len_range[0] <= len_before <= debug_output_len_range[1]
            ):
                logger.info(
                    "decode output trace: spec=%s len_before=%d next_token=%s len_after=%d seq_len=%s",
                    (
                        "NONE"
                        if force_plain_decode_output_processing
                        else getattr(batch.spec_algorithm, "name", str(batch.spec_algorithm))
                    ),
                    int(len_before),
                    next_token_id,
                    int(len(req.output_ids)),
                    int(batch.seq_lens[i].item()) if batch.seq_lens is not None else None,
                )
                debug_logits_topk = _parse_debug_logits_topk()
                if (
                    debug_logits_topk > 0
                    and logits_output is not None
                    and logits_output.next_token_logits is not None
                ):
                    row_logits = logits_output.next_token_logits[i]
                    top_k = min(debug_logits_topk, int(row_logits.shape[-1]))
                    if top_k > 0:
                        top_vals, top_ids = torch.topk(row_logits, k=top_k, dim=-1)
                        logger.info(
                            "decode logits trace: len_before=%d top_ids=%s top_vals=%s",
                            int(len_before),
                            top_ids.detach().to("cpu", non_blocking=False).tolist(),
                            [
                                round(float(x), 6)
                                for x in top_vals.detach()
                                .to("cpu", non_blocking=False)
                                .tolist()
                            ],
                        )

            # Update Mamba last track seqlen
            self._mamba_prefix_cache_update(req, batch, result, i)

            req.check_finished(new_accepted_len)

            if req.finished():
                self.maybe_collect_routed_experts(req)

                if self.server_args.disaggregation_decode_enable_offload_kvcache:
                    # Asynchronously offload KV cache; release_kv_cache will be called after Device->Host transfer completes
                    if not self.decode_offload_manager.offload_kv_cache(req):
                        self._release_kv_cache_and_draft(req)
                else:
                    self._release_kv_cache_and_draft(req)

                req.time_stats.completion_time = time.perf_counter()

            self.maybe_collect_customized_info(i, req, logits_output)

            if req.return_logprob and batch.spec_algorithm.is_none():
                # speculative worker handles logprob in speculative decoding
                req.output_token_logprobs_val.append(next_token_logprobs[i])
                req.output_token_logprobs_idx.append(next_token_id)
                if req.top_logprobs_num > 0:
                    req.output_top_logprobs_val.append(
                        logits_output.next_token_top_logprobs_val[i]
                    )
                    req.output_top_logprobs_idx.append(
                        logits_output.next_token_top_logprobs_idx[i]
                    )
                if req.token_ids_logprob is not None:
                    req.output_token_ids_logprobs_val.append(
                        logits_output.next_token_token_ids_logprobs_val[i]
                    )
                    req.output_token_ids_logprobs_idx.append(
                        logits_output.next_token_token_ids_logprobs_idx[i]
                    )

            if req.return_hidden_states and logits_output.hidden_states is not None:
                req.hidden_states.append(
                    logits_output.hidden_states[i].cpu().clone().tolist()
                )

            if req.grammar is not None:
                # FIXME: this try-except block is for handling unexpected xgrammar issue.
                try:
                    if batch.spec_algorithm.is_none():
                        # Normal decode: single token
                        req.grammar.accept_token(next_token_id)
                    elif batch.is_spec_v2:
                        # Speculative decode: next_token_id is a list of accepted tokens
                        for token_id in next_token_id:
                            req.grammar.accept_token(token_id)
                except ValueError as e:
                    # Grammar accept_token can raise ValueError if the token is not in the grammar.
                    # This can happen if the grammar is not set correctly or the token is invalid.
                    logger.error(
                        f"Grammar accept_token failed for req {req.rid} with token {next_token_id}: {e}"
                    )
                    self.abort_request(AbortReq(rid=req.rid))
                req.grammar.finished = req.finished()

        self.stream_output(batch.reqs, batch.return_logprob)
        self.token_to_kv_pool_allocator.free_group_end()

        self.forward_ct_decode = (self.forward_ct_decode + 1) % (1 << 30)
        if self.current_scheduler_metrics_enabled:
            if self.forward_ct_decode % self.server_args.decode_log_interval == 0:
                self.log_decode_stats(can_run_cuda_graph, running_batch=batch)
            self.log_decode_stats_every_iteration(
                batch, num_accepted_tokens=result.num_accepted_tokens
            )

    def _mamba_prefix_cache_update(
        self, req: Req, batch: ScheduleBatch, result: GenerationBatchResult, i: int
    ) -> None:
        seq_len = len(req.origin_input_ids) + len(req.output_ids) - 1
        if req.mamba_ping_pong_track_buffer is not None:
            mamba_track_interval = get_global_server_args().mamba_track_interval
            if batch.spec_algorithm.is_none() and seq_len % mamba_track_interval == 0:
                # for non-spec decode, we update mamba_last_track_seqlen at the end of each track interval
                req.mamba_next_track_idx = 1 - req.mamba_next_track_idx
                req.mamba_last_track_seqlen = seq_len
            elif (
                not batch.spec_algorithm.is_none()
                and result.accept_length_per_req_cpu is not None
            ):
                # for spec decode, update mamba_last_track_seqlen if this iteration crosses a track interval
                actual_seq_len = req.seqlen - 1
                if (
                    actual_seq_len // mamba_track_interval
                    != (actual_seq_len - result.accept_length_per_req_cpu[i])
                    // mamba_track_interval
                ):
                    req.mamba_last_track_seqlen = (
                        actual_seq_len // mamba_track_interval * mamba_track_interval
                    )

    def _process_input_token_logprobs(
        self, req: Req, input_token_logprobs: List
    ) -> None:
        """Process input token logprobs values and indices."""
        is_multi_item_scoring = self._is_multi_item_scoring(req)

        # Process logprob values - handle multi-item scoring vs regular requests
        if is_multi_item_scoring:
            # Multi-item scoring: use all logprobs as-is
            req.input_token_logprobs_val = input_token_logprobs
        else:
            # Regular request: add None at start, remove last (sampling token)
            req.input_token_logprobs_val = [None] + input_token_logprobs[:-1]

        # Process logprob indices based on scoring type
        if is_multi_item_scoring:
            # Multi-item scoring: only include delimiter token positions
            relevant_tokens = req.origin_input_ids[req.logprob_start_len :]
            input_token_logprobs_idx = [
                token_id
                for token_id in relevant_tokens
                if token_id == self.server_args.multi_item_scoring_delimiter
            ]
        else:
            # Regular request: include all tokens from logprob_start_len onwards
            input_token_logprobs_idx = req.origin_input_ids[req.logprob_start_len :]

        # Clip padded hash values from image tokens to prevent detokenization errors
        req.input_token_logprobs_idx = [
            x if x < self.model_config.vocab_size - 1 else 0
            for x in input_token_logprobs_idx
        ]

    def _process_input_top_logprobs(self, req: Req) -> None:
        """Process input top logprobs."""
        if req.top_logprobs_num <= 0:
            return

        is_multi_item_scoring = self._is_multi_item_scoring(req)

        # Initialize arrays - multi-item scoring starts empty, others start with None
        req.input_top_logprobs_val = [] if is_multi_item_scoring else [None]
        req.input_top_logprobs_idx = [] if is_multi_item_scoring else [None]

        # Extend arrays with temp values
        for val, idx in zip(
            req.temp_input_top_logprobs_val,
            req.temp_input_top_logprobs_idx,
            strict=True,
        ):
            req.input_top_logprobs_val.extend(val)
            req.input_top_logprobs_idx.extend(idx)

        # Remove last token (sampling token) for non multi-item scoring requests
        if not is_multi_item_scoring:
            req.input_top_logprobs_val.pop()
            req.input_top_logprobs_idx.pop()

        # Clean up temp storage
        req.temp_input_top_logprobs_idx = None
        req.temp_input_top_logprobs_val = None

    def _process_input_token_ids_logprobs(self, req: Req) -> None:
        """Process input token IDs logprobs."""
        if req.token_ids_logprob is None:
            return

        is_multi_item_scoring = self._is_multi_item_scoring(req)

        # Initialize arrays - multi-item scoring starts empty, others start with None
        req.input_token_ids_logprobs_val = [] if is_multi_item_scoring else [None]
        req.input_token_ids_logprobs_idx = [] if is_multi_item_scoring else [None]

        # Process temp values - convert tensors to lists and extend arrays
        for val, idx in zip(
            req.temp_input_token_ids_logprobs_val,
            req.temp_input_token_ids_logprobs_idx,
            strict=True,
        ):
            val_list = val.tolist() if isinstance(val, torch.Tensor) else val
            req.input_token_ids_logprobs_val.extend(
                val_list if isinstance(val_list, list) else [val_list]
            )
            req.input_token_ids_logprobs_idx.extend(idx)

        # Remove last token (sampling token) for non multi-item scoring requests
        if not is_multi_item_scoring:
            req.input_token_ids_logprobs_val.pop()
            req.input_token_ids_logprobs_idx.pop()

        # Clean up temp storage
        req.temp_input_token_ids_logprobs_idx = None
        req.temp_input_token_ids_logprobs_val = None

    def _calculate_relevant_tokens_len(self, req: Req) -> int:
        """Calculate the expected length of logprob arrays based on whether multi-item scoring is enabled.

        For multi-item scoring, only delimiter positions have logprobs.
        For regular requests, all positions from logprob_start_len onwards have logprobs.
        """
        is_multi_item_scoring = self._is_multi_item_scoring(req)
        relevant_tokens = req.origin_input_ids[req.logprob_start_len :]

        if is_multi_item_scoring:
            # Multi-item scoring: count delimiter tokens from logprob_start_len onwards
            return sum(
                1
                for token_id in relevant_tokens
                if token_id == self.server_args.multi_item_scoring_delimiter
            )
        else:
            # Regular request: all tokens from logprob_start_len onwards
            return len(relevant_tokens)

    def _calculate_num_input_logprobs(
        self, req: Req, extend_input_len: int, extend_logprob_start_len: int
    ) -> int:
        """Calculate the number of input logprobs based on whether multi-item scoring is enabled.

        For multi-item scoring, only delimiter positions have logprobs.
        For regular requests, all positions in the range have logprobs.
        """
        is_multi_item_scoring = self._is_multi_item_scoring(req)

        if is_multi_item_scoring:
            # Multi-item scoring: count delimiter tokens in the relevant portion
            relevant_tokens = req.origin_input_ids[
                extend_logprob_start_len:extend_input_len
            ]
            return sum(
                1
                for token_id in relevant_tokens
                if token_id == self.server_args.multi_item_scoring_delimiter
            )
        else:
            # Regular request: all tokens in the range
            return extend_input_len - extend_logprob_start_len

    def _is_multi_item_scoring(self, req: Req) -> bool:
        """Check if request uses multi-item scoring.

        Multi-item scoring applies to prefill-only requests when a delimiter
        token is configured. In this mode, only positions containing the
        delimiter token receive logprobs.
        """
        return req.is_prefill_only and self.server_args.multi_item_scoring_delimiter

    def add_input_logprob_return_values(
        self: Scheduler,
        i: int,
        req: Req,
        output: LogitsProcessorOutput,
        logprob_pt: int,
        num_input_logprobs: int,
        last_prefill_chunk: bool,  # If True, it means prefill is finished.
    ):
        """Incrementally add input logprobs to `req`.

        Args:
            i: The request index in a batch.
            req: The request. Input logprobs inside req are modified as a
                consequence of the API
            fill_ids: The prefill ids processed.
            output: Logit processor output that's used to compute input logprobs
            last_prefill_chunk: True if it is the last prefill (when chunked).
                Some of input logprob operation should only happen at the last
                prefill (e.g., computing input token logprobs).
        """
        assert output.input_token_logprobs is not None
        if req.input_token_logprobs is None:
            req.input_token_logprobs = []
        if req.temp_input_top_logprobs_val is None:
            req.temp_input_top_logprobs_val = []
        if req.temp_input_top_logprobs_idx is None:
            req.temp_input_top_logprobs_idx = []
        if req.temp_input_token_ids_logprobs_val is None:
            req.temp_input_token_ids_logprobs_val = []
        if req.temp_input_token_ids_logprobs_idx is None:
            req.temp_input_token_ids_logprobs_idx = []

        if req.input_token_logprobs_val is not None:
            # The input logprob has been already computed. It only happens
            # upon retract.
            if req.top_logprobs_num > 0:
                assert req.input_token_logprobs_val is not None
            return

        # Important for the performance.
        assert isinstance(output.input_token_logprobs, tuple)
        input_token_logprobs: Tuple[int] = output.input_token_logprobs
        input_token_logprobs = input_token_logprobs[
            logprob_pt : logprob_pt + num_input_logprobs
        ]
        req.input_token_logprobs.extend(input_token_logprobs)

        if req.top_logprobs_num > 0:
            req.temp_input_top_logprobs_val.append(output.input_top_logprobs_val[i])
            req.temp_input_top_logprobs_idx.append(output.input_top_logprobs_idx[i])

        if req.token_ids_logprob is not None:
            req.temp_input_token_ids_logprobs_val.append(
                output.input_token_ids_logprobs_val[i]
            )
            req.temp_input_token_ids_logprobs_idx.append(
                output.input_token_ids_logprobs_idx[i]
            )

        if last_prefill_chunk:
            input_token_logprobs = req.input_token_logprobs
            req.input_token_logprobs = None
            assert req.input_token_logprobs_val is None
            assert req.input_token_logprobs_idx is None
            assert req.input_top_logprobs_val is None
            assert req.input_top_logprobs_idx is None

            # Process all input logprob types using helper functions
            self._process_input_token_logprobs(req, input_token_logprobs)
            self._process_input_top_logprobs(req)

            self._process_input_token_ids_logprobs(req)

            if req.return_logprob:
                relevant_tokens_len = self._calculate_relevant_tokens_len(req)
                assert len(req.input_token_logprobs_val) == relevant_tokens_len
                assert len(req.input_token_logprobs_idx) == relevant_tokens_len
                if req.top_logprobs_num > 0:
                    assert len(req.input_top_logprobs_val) == relevant_tokens_len
                    assert len(req.input_top_logprobs_idx) == relevant_tokens_len
                if req.token_ids_logprob is not None:
                    assert len(req.input_token_ids_logprobs_val) == relevant_tokens_len
                    assert len(req.input_token_ids_logprobs_idx) == relevant_tokens_len

    def add_logprob_return_values(
        self: Scheduler,
        i: int,
        req: Req,
        pt: int,
        next_token_ids: List[int],
        num_input_logprobs: int,
        output: LogitsProcessorOutput,
    ):
        """Attach logprobs to the return values."""
        if output.next_token_logprobs is not None:
            req.output_token_logprobs_val.append(output.next_token_logprobs[i])
            req.output_token_logprobs_idx.append(next_token_ids[i])

        # Only add input logprobs if there are input tokens to process
        # Note: For prefill-only requests with default logprob_start_len, this will be 0,
        # meaning we only compute output logprobs (which is the intended behavior)
        if num_input_logprobs > 0:
            self.add_input_logprob_return_values(
                i, req, output, pt, num_input_logprobs, last_prefill_chunk=True
            )
        else:
            self._initialize_empty_logprob_containers(req)

        if req.top_logprobs_num > 0:
            req.output_top_logprobs_val.append(output.next_token_top_logprobs_val[i])
            req.output_top_logprobs_idx.append(output.next_token_top_logprobs_idx[i])

        if (
            req.token_ids_logprob is not None
            and output.next_token_token_ids_logprobs_val is not None
        ):
            # Convert GPU tensor to list if needed
            logprobs_val = output.next_token_token_ids_logprobs_val[i]
            if isinstance(logprobs_val, torch.Tensor):
                logprobs_val = logprobs_val.tolist()
            req.output_token_ids_logprobs_val.append(logprobs_val)
            req.output_token_ids_logprobs_idx.append(
                output.next_token_token_ids_logprobs_idx[i]
            )

        return num_input_logprobs

    def _initialize_empty_logprob_containers(self, req: Req) -> None:
        """
        Initialize logprob fields to empty lists if unset.

        This is needed for prefill-only requests where the normal initialization
        flow might be bypassed, but downstream code expects these fields to be lists.
        """
        if req.input_token_logprobs_val is None:
            req.input_token_logprobs_val = []
        if req.input_token_logprobs_idx is None:
            req.input_token_logprobs_idx = []
        if req.input_top_logprobs_val is None:
            req.input_top_logprobs_val = []
        if req.input_top_logprobs_idx is None:
            req.input_top_logprobs_idx = []
        if req.input_token_ids_logprobs_val is None:
            req.input_token_ids_logprobs_val = []
        if req.input_token_ids_logprobs_idx is None:
            req.input_token_ids_logprobs_idx = []

    def stream_output(
        self: Scheduler,
        reqs: List[Req],
        return_logprob: bool,
        skip_req: Optional[Req] = None,
    ):
        """Stream the output to detokenizer."""
        if self.is_generation:
            self.stream_output_generation(reqs, return_logprob, skip_req)
        else:  # embedding or reward model
            self.stream_output_embedding(reqs)

        if envs.SGLANG_TEST_CRASH_AFTER_STREAM_OUTPUTS.get() > 0:
            self._trigger_crash_for_tests(
                envs.SGLANG_TEST_CRASH_AFTER_STREAM_OUTPUTS.get()
            )

    def _trigger_crash_for_tests(self, crash_threshold: int):
        # Crash trigger: crash after stream_output is called N times
        # This is used for testing purposes.
        if not hasattr(self, "_test_stream_output_count"):
            self._test_stream_output_count = 0
        self._test_stream_output_count += 1
        if self._test_stream_output_count >= crash_threshold:
            raise RuntimeError(
                f"Test crash after stream_output called {self._test_stream_output_count} times"
            )

    def stream_output_generation(
        self: Scheduler,
        reqs: List[Req],
        return_logprob: bool,
        skip_req: Optional[Req] = None,
        is_idle_batch: bool = False,
    ):
        rids = []
        http_worker_ipcs = []
        finished_reasons: List[BaseFinishReason] = []

        decoded_texts = []
        decode_ids_list = []
        read_offsets = []
        output_ids = []

        skip_special_tokens = []
        spaces_between_special_tokens = []
        no_stop_trim = []
        prompt_tokens = []
        completion_tokens = []
        cached_tokens = []
        cached_tokens_details = []  # Detailed breakdown by cache source
        spec_verify_ct = []
        spec_accepted_tokens = []
        spec_acceptance_histogram = []
        retraction_counts = []
        output_hidden_states = None
        load = self.get_load()
        routed_experts = None
        customized_info = {}

        queue_times = []
        forward_entry_times = []
        prefill_launch_delays = []
        prefill_launch_latencies = []
        prefill_finished_timestamps = []

        if return_logprob:
            input_token_logprobs_val = []
            input_token_logprobs_idx = []
            output_token_logprobs_val = []
            output_token_logprobs_idx = []
            input_top_logprobs_val = []
            input_top_logprobs_idx = []
            output_top_logprobs_val = []
            output_top_logprobs_idx = []
            input_token_ids_logprobs_val = []
            input_token_ids_logprobs_idx = []
            output_token_ids_logprobs_val = []
            output_token_ids_logprobs_idx = []
        else:
            input_token_logprobs_val = input_token_logprobs_idx = (
                output_token_logprobs_val
            ) = output_token_logprobs_idx = input_top_logprobs_val = (
                input_top_logprobs_idx
            ) = output_top_logprobs_val = output_top_logprobs_idx = (
                input_token_ids_logprobs_val
            ) = input_token_ids_logprobs_idx = output_token_ids_logprobs_val = (
                output_token_ids_logprobs_idx
            ) = None

        for req in reqs:
            if req is skip_req:
                continue

            # Multimodal partial stream chunks break the detokenizer, so drop aborted requests here.
            if self.model_config.is_multimodal_gen and req.to_finish:
                continue

            if req.finished():
                if req.finished_output:
                    # With the overlap schedule, a request will try to output twice and hit this line twice
                    # because of the one additional delayed token. This "continue" prevented the dummy output.
                    continue
                req.finished_output = True
                if req.finished_len is None:
                    req.finished_len = len(req.output_ids)
                should_output = True
            else:
                if req.stream:
                    stream_interval = (
                        req.sampling_params.stream_interval or self.stream_interval
                    )

                    # origin stream_interval logic
                    should_output = (
                        len(req.output_ids) % stream_interval == 1
                        if not self.model_config.is_multimodal_gen
                        and stream_interval > 1
                        else len(req.output_ids) % stream_interval == 0
                    )

                    if should_output:
                        # check_match_stop_str_prefix if  tail_str's suffix match stop_str prefix
                        should_output &= not req.check_match_stop_str_prefix()
                else:
                    should_output = (
                        len(req.output_ids) % DEFAULT_FORCE_STREAM_INTERVAL == 0
                        if not self.model_config.is_multimodal_gen
                        else False
                    )

            if should_output:
                send_token_offset = req.send_token_offset
                send_output_token_logprobs_offset = (
                    req.send_output_token_logprobs_offset
                )
                rids.append(req.rid)
                http_worker_ipcs.append(req.http_worker_ipc)
                finished_reasons.append(
                    req.finished_reason.to_json() if req.finished_reason else None
                )
                decoded_texts.append(req.decoded_text)
                decode_ids, read_offset = req.init_incremental_detokenize()

                if self.model_config.is_multimodal_gen:
                    decode_ids_list.append(decode_ids)
                else:
                    decode_ids_list.append(decode_ids[req.send_decode_id_offset :])

                # Exclude the tokens after stop condition
                output_ids_ = req.output_ids_through_stop

                req.send_decode_id_offset = len(decode_ids)
                read_offsets.append(read_offset)
                output_ids.append(output_ids_[send_token_offset:])
                req.send_token_offset = len(output_ids_)
                skip_special_tokens.append(req.sampling_params.skip_special_tokens)
                spaces_between_special_tokens.append(
                    req.sampling_params.spaces_between_special_tokens
                )
                no_stop_trim.append(req.sampling_params.no_stop_trim)
                prompt_tokens.append(len(req.origin_input_ids))
                completion_tokens.append(len(output_ids_))
                cached_tokens.append(req.cached_tokens)

                # Collect detailed cache breakdown if available
                cached_tokens_details.append(self._get_cached_tokens_details(req))

                retraction_counts.append(req.retraction_count)

                queue_times.append(req.time_stats.get_queueing_time())
                forward_entry_times.append(req.time_stats.forward_entry_time)

                prefill_launch_delays.append(req.time_stats.get_prefill_launch_delay())
                prefill_launch_latencies.append(
                    req.time_stats.get_prefill_launch_latency()
                )
                prefill_finished_timestamps.append(
                    req.time_stats.get_prefill_finished_ts()
                )

                if not self.spec_algorithm.is_none():
                    spec_verify_ct.append(req.spec_verify_ct)
                    spec_accepted_tokens.append(req.spec_accepted_tokens)
                    spec_acceptance_histogram.append(req.spec_acceptance_histogram)
                    customized_info.setdefault("spec_ssd_hit_ct", []).append(
                        int(getattr(req, "spec_ssd_hit_ct", 0))
                    )
                    customized_info.setdefault("spec_ssd_prepare_ct", []).append(
                        int(getattr(req, "spec_ssd_prepare_ct", 0))
                    )
                    customized_info.setdefault("spec_ssd_prepare_failure_ct", []).append(
                        int(getattr(req, "spec_ssd_prepare_failure_ct", 0))
                    )
                    customized_info.setdefault("spec_ssd_cache_pending", []).append(
                        int(
                            getattr(req, "spec_ssd_cache_pending", 0)
                            or len(getattr(req, "dflash_ssd_cache", {}) or {})
                            or (
                                1
                                if getattr(req, "dflash_ssd_cached_proposal", None)
                                is not None
                                else 0
                            )
                        )
                    )
                    customized_info.setdefault("spec_ssd_overlap_launch_ct", []).append(
                        int(getattr(req, "spec_ssd_overlap_launch_ct", 0))
                    )
                    customized_info.setdefault("spec_ssd_overlap_wait_ct", []).append(
                        int(getattr(req, "spec_ssd_overlap_wait_ct", 0))
                    )
                    customized_info.setdefault(
                        "spec_ssd_difficulty_gate_skip_ct", []
                    ).append(int(getattr(req, "spec_ssd_difficulty_gate_skip_ct", 0)))
                    customized_info.setdefault(
                        "spec_ssd_fanout_gate_skip_ct", []
                    ).append(int(getattr(req, "spec_ssd_fanout_gate_skip_ct", 0)))
                    customized_info.setdefault(
                        "spec_ssd_fanout_escalation_ct", []
                    ).append(int(getattr(req, "spec_ssd_fanout_escalation_ct", 0)))
                    customized_info.setdefault(
                        "spec_ssd_fanout_alt_budget", []
                    ).append(int(getattr(req, "spec_ssd_fanout_alt_budget", 0)))
                    customized_info.setdefault(
                        "spec_accept_length_step_min", []
                    ).append(
                        (
                            None
                            if getattr(req, "spec_accept_length_step_min", None)
                            is None
                            else int(getattr(req, "spec_accept_length_step_min", 0))
                        )
                    )
                    customized_info.setdefault(
                        "spec_accept_length_step_max", []
                    ).append(
                        (
                            None
                            if getattr(req, "spec_accept_length_step_max", None)
                            is None
                            else int(getattr(req, "spec_accept_length_step_max", 0))
                        )
                    )
                    customized_info.setdefault("spec_dflash_verify_mode_last", []).append(
                        getattr(req, "spec_dflash_verify_mode_last", None)
                    )
                    customized_info.setdefault("spec_dflash_debug_stat_ct", []).append(
                        int(getattr(req, "spec_dflash_debug_stat_ct", 0))
                    )
                    customized_info.setdefault(
                        "spec_dflash_verify_append_path_last", []
                    ).append(getattr(req, "spec_dflash_verify_append_path_last", None))
                    for key in (
                        "spec_dflash_verify_append_path_fused_ct",
                        "spec_dflash_verify_append_path_direct_ct",
                        "spec_dflash_verify_append_path_staged_ct",
                    ):
                        customized_info.setdefault(key, []).append(
                            int(getattr(req, key, 0))
                        )
                    customized_info.setdefault("spec_dflash_max_steps_last", []).append(
                        (
                            None
                            if getattr(req, "spec_dflash_max_steps_last", None) is None
                            else int(getattr(req, "spec_dflash_max_steps_last", 0))
                        )
                    )
                    customized_info.setdefault("spec_dflash_max_steps_min", []).append(
                        (
                            None
                            if getattr(req, "spec_dflash_max_steps_min", None) is None
                            else int(getattr(req, "spec_dflash_max_steps_min", 0))
                        )
                    )
                    customized_info.setdefault("spec_dflash_max_steps_max", []).append(
                        (
                            None
                            if getattr(req, "spec_dflash_max_steps_max", None) is None
                            else int(getattr(req, "spec_dflash_max_steps_max", 0))
                        )
                    )
                    customized_info.setdefault("spec_dflash_max_steps_mean", []).append(
                        getattr(req, "spec_dflash_max_steps_mean", None)
                    )
                    for key in (
                        "spec_dflash_effective_draft_token_num_last",
                        "spec_dflash_effective_draft_token_num_min",
                        "spec_dflash_effective_draft_token_num_max",
                        "spec_dflash_effective_draft_token_num_mean",
                        "spec_dflash_effective_step_count_last",
                        "spec_dflash_effective_step_count_min",
                        "spec_dflash_effective_step_count_max",
                        "spec_dflash_effective_step_count_mean",
                        "spec_dflash_total_draft_token_num",
                    ):
                        customized_info.setdefault(key, []).append(
                            getattr(req, key, None)
                        )
                    for key in (
                        "spec_dflash_accept_ratio_mean",
                        "spec_dflash_tv_mean",
                        "spec_dflash_p_entropy_mean",
                        "spec_dflash_q_entropy_mean",
                        "spec_dflash_p_max_mean",
                        "spec_dflash_q_max_mean",
                        "spec_dflash_q_max_mean_first",
                        "spec_dflash_q_max_min_first",
                        "spec_dflash_q_ent_mean_first",
                        "spec_dflash_adaptive_temp_mul",
                    ):
                        customized_info.setdefault(key, []).append(
                            getattr(req, key, None)
                        )
                    customized_info.setdefault(
                        "spec_dflash_pq_disabled_rounds_left", []
                    ).append(
                        int(getattr(req, "spec_dflash_pq_disabled_rounds_left", 0))
                    )

                if return_logprob:
                    if (
                        req.return_logprob
                        and not req.input_logprob_sent
                        # Decode server does not send input logprobs
                        and self.disaggregation_mode != DisaggregationMode.DECODE
                    ):
                        input_token_logprobs_val.append(req.input_token_logprobs_val)
                        input_token_logprobs_idx.append(req.input_token_logprobs_idx)
                        input_top_logprobs_val.append(req.input_top_logprobs_val)
                        input_top_logprobs_idx.append(req.input_top_logprobs_idx)
                        input_token_ids_logprobs_val.append(
                            req.input_token_ids_logprobs_val
                        )
                        input_token_ids_logprobs_idx.append(
                            req.input_token_ids_logprobs_idx
                        )
                        req.input_logprob_sent = True
                    else:
                        input_token_logprobs_val.append([])
                        input_token_logprobs_idx.append([])
                        input_top_logprobs_val.append([])
                        input_top_logprobs_idx.append([])
                        input_token_ids_logprobs_val.append([])
                        input_token_ids_logprobs_idx.append([])

                    if req.return_logprob:
                        output_token_logprobs_val.append(
                            req.output_token_logprobs_val[
                                send_output_token_logprobs_offset:
                            ]
                        )
                        output_token_logprobs_idx.append(
                            req.output_token_logprobs_idx[
                                send_output_token_logprobs_offset:
                            ]
                        )
                        output_top_logprobs_val.append(
                            req.output_top_logprobs_val[
                                send_output_token_logprobs_offset:
                            ]
                        )
                        output_top_logprobs_idx.append(
                            req.output_top_logprobs_idx[
                                send_output_token_logprobs_offset:
                            ]
                        )
                        output_token_ids_logprobs_val.append(
                            req.output_token_ids_logprobs_val[
                                send_output_token_logprobs_offset:
                            ]
                        )
                        output_token_ids_logprobs_idx.append(
                            req.output_token_ids_logprobs_idx[
                                send_output_token_logprobs_offset:
                            ]
                        )
                        req.send_output_token_logprobs_offset = len(
                            req.output_token_logprobs_val
                        )
                    else:
                        output_token_logprobs_val.append([])
                        output_token_logprobs_idx.append([])
                        output_top_logprobs_val.append([])
                        output_top_logprobs_idx.append([])
                        output_token_ids_logprobs_val.append([])
                        output_token_ids_logprobs_idx.append([])

                if req.return_hidden_states:
                    if output_hidden_states is None:
                        output_hidden_states = []
                    output_hidden_states.append(req.hidden_states)
                if req.return_routed_experts:
                    if routed_experts is None:
                        routed_experts = []
                    routed_experts.append(req.routed_experts)

                if req.customized_info is not None:
                    for k, v in req.customized_info.items():
                        if k not in customized_info:
                            customized_info[k] = []
                        customized_info[k].append(v[send_token_offset:])

            if (
                req.finished()
                and self.attn_tp_rank == 0
                and self.server_args.enable_request_time_stats_logging
            ):
                req.log_time_stats()

        # Send to detokenizer
        if reqs or is_idle_batch:
            if self.model_config.is_multimodal_gen:
                return
            if _fa3_trace_output_ids_enabled() and output_ids:
                logger.info(
                    "[FA3OutputPath][scheduler] rids=%s output_ids=%s decoded_texts=%s",
                    rids[:2],
                    output_ids[:2],
                    decoded_texts[:2],
                )
            self.send_to_detokenizer.send_output(
                BatchTokenIDOutput(
                    rids=rids,
                    http_worker_ipcs=http_worker_ipcs,
                    spec_verify_ct=spec_verify_ct,
                    spec_accepted_tokens=spec_accepted_tokens,
                    spec_acceptance_histogram=spec_acceptance_histogram,
                    queue_time=queue_times,
                    forward_entry_time=forward_entry_times,
                    prefill_launch_delay=prefill_launch_delays,
                    prefill_launch_latency=prefill_launch_latencies,
                    prefill_finished_ts=prefill_finished_timestamps,
                    finished_reasons=finished_reasons,
                    decoded_texts=decoded_texts,
                    decode_ids=decode_ids_list,
                    read_offsets=read_offsets,
                    output_ids=output_ids,
                    skip_special_tokens=skip_special_tokens,
                    spaces_between_special_tokens=spaces_between_special_tokens,
                    no_stop_trim=no_stop_trim,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cached_tokens=cached_tokens,
                    cached_tokens_details=cached_tokens_details,
                    input_token_logprobs_val=input_token_logprobs_val,
                    input_token_logprobs_idx=input_token_logprobs_idx,
                    output_token_logprobs_val=output_token_logprobs_val,
                    output_token_logprobs_idx=output_token_logprobs_idx,
                    input_top_logprobs_val=input_top_logprobs_val,
                    input_top_logprobs_idx=input_top_logprobs_idx,
                    output_top_logprobs_val=output_top_logprobs_val,
                    output_top_logprobs_idx=output_top_logprobs_idx,
                    input_token_ids_logprobs_val=input_token_ids_logprobs_val,
                    input_token_ids_logprobs_idx=input_token_ids_logprobs_idx,
                    output_token_ids_logprobs_val=output_token_ids_logprobs_val,
                    output_token_ids_logprobs_idx=output_token_ids_logprobs_idx,
                    output_token_entropy_val=None,
                    output_hidden_states=output_hidden_states,
                    routed_experts=routed_experts,
                    customized_info=customized_info,
                    placeholder_tokens_idx=None,
                    placeholder_tokens_val=None,
                    retraction_counts=retraction_counts,
                    load=load,
                )
            )

    def stream_output_embedding(self: Scheduler, reqs: List[Req]):
        rids = []
        http_worker_ipcs = []
        finished_reasons: List[BaseFinishReason] = []

        embeddings = []
        prompt_tokens = []
        cached_tokens = []
        cached_tokens_details = []  # Detailed breakdown by cache source
        queue_times = []
        forward_entry_times = []
        prefill_launch_delays = []
        prefill_launch_latencies = []
        prefill_finished_timestamps = []
        retraction_counts = []
        for req in reqs:
            if req.finished():
                rids.append(req.rid)
                http_worker_ipcs.append(req.http_worker_ipc)
                finished_reasons.append(req.finished_reason.to_json())
                embeddings.append(req.embedding)
                prompt_tokens.append(len(req.origin_input_ids))
                cached_tokens.append(req.cached_tokens)

                # Collect detailed cache breakdown if available
                cached_tokens_details.append(self._get_cached_tokens_details(req))

                queue_times.append(req.time_stats.get_queueing_time())
                forward_entry_times.append(req.time_stats.forward_entry_time)

                prefill_launch_delays.append(req.time_stats.get_prefill_launch_delay())
                prefill_launch_latencies.append(
                    req.time_stats.get_prefill_launch_latency()
                )
                prefill_finished_timestamps.append(
                    req.time_stats.get_prefill_finished_ts()
                )
                retraction_counts.append(req.retraction_count)
        self.send_to_detokenizer.send_output(
            BatchEmbeddingOutput(
                rids=rids,
                http_worker_ipcs=http_worker_ipcs,
                queue_time=queue_times,
                forward_entry_time=forward_entry_times,
                prefill_launch_delay=prefill_launch_delays,
                prefill_launch_latency=prefill_launch_latencies,
                prefill_finished_ts=prefill_finished_timestamps,
                finished_reasons=finished_reasons,
                embeddings=embeddings,
                prompt_tokens=prompt_tokens,
                cached_tokens=cached_tokens,
                cached_tokens_details=cached_tokens_details,
                placeholder_tokens_idx=None,
                placeholder_tokens_val=None,
                retraction_counts=retraction_counts,
            )
        )
