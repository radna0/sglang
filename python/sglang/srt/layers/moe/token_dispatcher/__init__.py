from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    BaseDispatcherConfig,
    CombineInput,
    CombineInputChecker,
    CombineInputFormat,
    DispatchOutput,
    DispatchOutputChecker,
    DispatchOutputFormat,
)
from sglang.srt.layers.moe.token_dispatcher.deepep import (
    DeepEPConfig,
    DeepEPDispatcher,
    DeepEPLLCombineInput,
    DeepEPLLDispatchOutput,
    DeepEPNormalCombineInput,
    DeepEPNormalDispatchOutput,
)
try:
    # FlashInfer can be absent or partially broken (e.g. missing cuda-python).
    # Keep import-time robust; raise only if user selects the FlashInfer backend.
    from sglang.srt.layers.moe.token_dispatcher.flashinfer import (  # type: ignore
        FlashinferDispatcher,
        FlashinferDispatchOutput,
    )
except Exception:  # pragma: no cover

    class _FlashinferUnavailable:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Flashinfer token dispatcher is unavailable. "
                "Install a working flashinfer + cuda-python stack, or select a non-flashinfer backend."
            )

    FlashinferDispatcher = _FlashinferUnavailable  # type: ignore
    FlashinferDispatchOutput = _FlashinferUnavailable  # type: ignore
from sglang.srt.layers.moe.token_dispatcher.fuseep import NpuFuseEPDispatcher
from sglang.srt.layers.moe.token_dispatcher.mooncake import (
    MooncakeCombineInput,
    MooncakeDispatchOutput,
    MooncakeEPDispatcher,
)
from sglang.srt.layers.moe.token_dispatcher.moriep import (
    MoriEPDispatcher,
    MoriEPNormalCombineInput,
    MoriEPNormalDispatchOutput,
)
from sglang.srt.layers.moe.token_dispatcher.standard import (
    StandardCombineInput,
    StandardDispatcher,
    StandardDispatchOutput,
)

__all__ = [
    "BaseDispatcher",
    "BaseDispatcherConfig",
    "CombineInput",
    "CombineInputChecker",
    "CombineInputFormat",
    "DispatchOutput",
    "DispatchOutputFormat",
    "DispatchOutputChecker",
    "FlashinferDispatchOutput",
    "FlashinferDispatcher",
    "MooncakeCombineInput",
    "MooncakeDispatchOutput",
    "MooncakeEPDispatcher",
    "MoriEPNormalDispatchOutput",
    "MoriEPNormalCombineInput",
    "MoriEPDispatcher",
    "StandardDispatcher",
    "StandardDispatchOutput",
    "StandardCombineInput",
    "DeepEPConfig",
    "DeepEPDispatcher",
    "DeepEPNormalDispatchOutput",
    "DeepEPLLDispatchOutput",
    "DeepEPLLCombineInput",
    "DeepEPNormalCombineInput",
    "NpuFuseEPDispatcher",
]
