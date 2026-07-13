import multiprocessing
import time
import traceback
from collections.abc import Generator
from contextlib import contextmanager
from multiprocessing import Queue
from pathlib import Path
from queue import Empty
from typing import Callable, cast

from click.testing import Result
from typer.testing import CliRunner

from lclstreamer.cmd.lclstreamer import app

runner: CliRunner = CliRunner()

configuration: str = """
source_identifier: "none"
skip_incomplete_events: false

data_sources:
    random:
        type: GenericRandomNumpyArray
        array_shape: 200,20
        array_dtype: float32

event_source:
    type: InternalEventSource
    number_of_events_to_generate: 1009

processing_pipeline:
    type: BatchProcessingPipeline
    batch_size: 10

data_serializer:
    type: HDF5BinarySerializer
    compression_level: 3
    compression: zfp
    fields:
        random: /data/random

data_handlers:
  - type: BinaryDataStreamingDataHandler
    urls:
        - "tcp://127.0.0.1:50101"
    role: client
    library: zmq
    socket_type: push
"""


@contextmanager
def child_process(  # pyright: ignore[reportUnknownParameterType]
    target_func: Callable[  # pyright: ignore[reportUnknownParameterType]
        [Queue, str],  # pyright: ignore[reportMissingTypeArgument]
        None,
    ],
    *args: str,
) -> Generator[
    tuple[Queue, multiprocessing.Process],  # pyright: ignore[reportMissingTypeArgument]
    None,
]:
    """
    A context manager that sets up a queue
    and starts a new multiprocessing.Process to run a target function.

    It ensures the process is terminated after a timeout upon exit.
    """
    p = None

    try:
        # Use spawn method for better process isolation
        ctx = multiprocessing.get_context("spawn")
        queue: Queue = ctx.Queue()  # pyright: ignore[reportMissingTypeArgument]
        p = ctx.Process(
            target=target_func,  # pyright: ignore[reportUnknownArgumentType]
            args=(queue,) + tuple(args),  # pyright: ignore[reportUnknownArgumentType]
        )
        p.start()
        yield queue, p  # pyright: ignore[reportReturnType]

    finally:
        if p is not None:
            p.join(timeout=15.0)
            if p.is_alive():
                p.terminate()
                p.join()


def run_pull(
    queue: Queue,  # pyright: ignore[reportMissingTypeArgument, reportUnknownParameterType]
    uri: str,
) -> None:
    import traceback

    from zmq import PULL, RCVTIMEO, Again, Context, ZMQError

    count: int = 0
    expected_count: int = 101

    try:
        with Context() as context:
            with context.socket(PULL) as socket:
                socket.setsockopt(RCVTIMEO, 10000)
                socket.bind("tcp://*:50101")
                queue.put({"ready": True})  # pyright: ignore[reportUnknownMemberType]
                while True:
                    try:
                        _: bytes = socket.recv()
                        count += 1
                        if count >= expected_count:
                            break
                        time.sleep(0.001)
                    except Again:
                        break

        if count < expected_count:
            queue.put(  # pyright: ignore[reportUnknownMemberType]
                {
                    "success": False,
                    "message": f"Timeout: received {count} messages, expected {expected_count}",
                },
                timeout=5.0,
            )
        else:
            queue.put(  # pyright: ignore[reportUnknownMemberType]
                {"success": True, "message:": f"Only received {count} messages"},
                timeout=5.0,
            )
    except ZMQError as exception:
        queue.put(  # pyright: ignore[reportUnknownMemberType]
            {
                "success": False,
                "message": f"{str(exception)}\n{traceback.format_exc()}",
            },
            timeout=5.0,
        )


def test_app() -> None:
    with child_process(run_pull, "tcp://127.0.0.1:50101") as (
        queue,  # pyright: ignore[reportUnknownVariableType]
        _,
    ):
        # Wait for the server to be ready before starting the client
        try:
            ready_msg: dict[str, bool | str] = cast(
                dict[str, bool | str], queue.get(timeout=15.0)
            )
            if not ready_msg.get("ready"):
                assert False, "Server did not report ready within 15 seconds"
        except Empty:
            assert False, "Server did not start within 15 seconds"

        # Add extra delay to ensure server socket is fully ready to receive
        time.sleep(1.0)

        with runner.isolated_filesystem():
            current_directory: Path = Path.cwd()
            configuration_file_name: Path = current_directory / "lclstreamer.yaml"
            configuration_file_name.write_text(configuration, "utf-8")
            absolute_config_path: Path = configuration_file_name.resolve()
            result: Result = runner.invoke(app, ["--config", str(absolute_config_path)])
            print("--- Output")
            print(result.output)
            if result.exception is not None and result.exc_info is not None:
                print("--- Exceptions")
                print(result.exception)
                # print(result.exc_info)
                traceback.print_tb(result.exc_info[2])

            assert result.exit_code == 0, (
                f"Client failed with exit code {result.exit_code}"
            )

            try:
                server_result: dict[str, bool | str] = cast(
                    dict[str, bool | str], queue.get(timeout=10.0)
                )
                if server_result.get("success"):
                    print("Succeeded! Expected 101 messages, got 101")
                else:
                    print("Failed! Subprocess error:")
                    print(server_result["message"])
            except Empty:
                assert False, "Server did not report results within 15 seconds"
