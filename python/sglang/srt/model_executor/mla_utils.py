# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Small helpers shared across GPT-OSS / MLA runtime pieces."""

from __future__ import annotations


def get_flashmla_mla_kv_cache_dim(
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    mla_rope_num_kv_heads: int = 1,
    attention_backend: str | None = None,
) -> int:
    """Return the KV-cache width FlashMLA should see for MLA models.

    The shared-latent GPT-OSS MLA export keeps a one-head rope tail and pads the
    cache width to the legacy 576 floor for FlashMLA compatibility. When the
    checkpoint carries a native rope-head geometry (mla_rope_num_kv_heads > 1),
    return the exact latent+rope width so the runtime stays aligned with the
    converted checkpoint.
    """

    kv_cache_dim = int(kv_lora_rank) + int(qk_rope_head_dim) * int(
        mla_rope_num_kv_heads
    )
    if int(mla_rope_num_kv_heads) <= 1 and kv_cache_dim < 576:
        return 576
    return kv_cache_dim
