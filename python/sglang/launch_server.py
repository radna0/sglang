"""Launch the inference server."""

import asyncio
import faulthandler
import os
import sys
import time
import traceback

from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.common import suppress_noisy_warnings

suppress_noisy_warnings()
faulthandler.enable(all_threads=True)


def run_server(server_args):
    """Run the server based on server_args.grpc_mode and server_args.encoder_only."""
    if server_args.encoder_only:
        if server_args.grpc_mode:
            from sglang.srt.disaggregation.encode_grpc_server import (
                serve_grpc_encoder,
            )

            asyncio.run(serve_grpc_encoder(server_args))
        else:
            from sglang.srt.disaggregation.encode_server import launch_server

            launch_server(server_args)
    elif server_args.grpc_mode:
        from sglang.srt.entrypoints.grpc_server import serve_grpc

        asyncio.run(serve_grpc(server_args))
    elif getattr(server_args, "use_ray", False):
        try:
            from sglang.srt.ray.http_server import launch_server
        except ImportError:
            raise ImportError(
                "Ray is required for --use-ray mode. "
                "Install it with: pip install 'sglang[ray]'"
            )

        launch_server(server_args)
    else:
        # Default mode: HTTP mode.
        from sglang.srt.entrypoints.http_server import launch_server

        launch_server(server_args)


if __name__ == "__main__":
    import warnings

    warnings.warn(
        "'python -m sglang.launch_server' is still supported, but "
        "'sglang serve' is the recommended entrypoint.\n"
        "  Example: sglang serve --model-path <model> [options]",
        UserWarning,
        stacklevel=1,
    )

    server_args = prepare_server_args(sys.argv[1:])

    try:
        run_server(server_args)
    except BaseException:
        traceback.print_exc()
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            time.sleep(0.2)
        except Exception:
            pass
        raise
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            time.sleep(0.2)
        except Exception:
            pass
        kill_process_tree(os.getpid(), include_parent=False)
