#!/sdf/group/lcls/ds/ana/sw/conda2/manage/bin/psconda.sh

from collections.abc import Iterator
from pathlib import Path
from typing import (
    Annotated,
    Any,
)

import typer
from ..utils.logging import log
import logging
import socket
from mpi4py import MPI
from stream.core import Source, stream
from stream.ops import map, take, tap  # pyright: ignore[reportUnknownVariableType]

from ..data_handlers.setup import initialize_data_handlers
from ..data_serializers.setup import initialize_data_serializer
from ..event_data_sources.setup import initialize_event_source
from ..models.parameters import Parameters
from ..processing_pipelines.setup import initialize_processing_pipeline
from ..utils.parameters import load_configuration_parameters
from ..utils.protocols import (
    DataHandlerProtocol,
    DataSerializerProtocol,
    EventSourceProtocol,
    ProcessingPipelineProtocol,
)
from ..utils.stream import (
    clock,
)
from ..utils.typing import StrFloatIntNDArray

app = typer.Typer()


@stream
def _filter_incomplete_events(
    events: Iterator[dict[str, StrFloatIntNDArray | None]], max_consecutive: int = 100
) -> Iterator[dict[str, StrFloatIntNDArray | None]]:
    """
    Drops events that are incomplete

    Incomplete events are events where the retrieval of one or more data items
    failed

    Arguments:

        events: An event iterator
        max_consecutive (int): maximum number of consecutive frames containing any "missing" value before terminating early

    Returns:

        events: An event iterator
    """
    consecutive: int = 0
    ev_num: int = 0
    num_dropped: int = 0
    nfailed: dict[str, int] = {}  # number from each detector
    for ev_num, event in enumerate(events):
        if all(v is not None for v in event.values()):
            yield event
            consecutive = 0
            continue
        for name, v in event.items():
            if v is None:
                nfailed[name] = nfailed.get(name, 0) + 1
        consecutive += 1
        num_dropped += 1
        if consecutive >= max_consecutive:
            break
    if consecutive >= max_consecutive:
        print(f"Stopping early after {consecutive} errors.")
    if num_dropped > 0:
        print(f"Failed detector counts: {nfailed}.")
    print(f"Processed {ev_num + 1} events with {num_dropped} dropped.")


def _data_counter(data: bytes) -> int:
    """
    Computes the size of the input data

    Arguments:

        data: A byte object storing the data

    Returns:

        size: The size of the data in bytes
    """
    return len(data)


@app.command()
def main(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="configuration file (default: monitor.yaml file in the current "
            "working directory",
        ),
    ] = Path("lclstreamer.yaml"),
    num_events: Annotated[
        int,
        typer.Option(
            "--num-events", "-n", help="number of data events to read before stopping"
        ),
    ] = 0,
    debug: Annotated[
        bool,
        typer.Option(
            "--debug", "-d", help="enable showing debug info",
            is_flag=True,
        ),
    ] = False,
) -> None:
    """
    An application that retrieves data from an event source, processes it, serializes
    it, and passes it to a series of data handlers that forwards it to external
    applications. The event source, data processing, serialization strategy, and
    further data handling are defined by the content of a configuration file
    """

    if debug:
        log.setLevel(logging.DEBUG)

    # 1. Read and recover configuration parameters
    mpi_size: int = MPI.COMM_WORLD.Get_size()
    mpi_rank: int = MPI.COMM_WORLD.Get_rank()

    parameters: Parameters = load_configuration_parameters(filename=config)

    log.debug(f"[Rank {mpi_rank} {socket.gethostname()}] Initializing event source....")

    source: EventSourceProtocol = initialize_event_source(
        parameters=parameters,
        worker_pool_size=mpi_size,
        worker_rank=mpi_rank,
    )

    log.debug(f"[Rank {mpi_rank} {socket.gethostname()}] Initializing event source: Done!")

    log.debug(f"[Rank {mpi_rank} {socket.gethostname()}] Initializing processing pipeline....")
    processing_pipeline: ProcessingPipelineProtocol = initialize_processing_pipeline(
        parameters
    )
    log.debug(f"[Rank {mpi_rank} {socket.gethostname()}] Initializing processing pipeline: Done!")

    log.debug(f"[Rank {mpi_rank} {socket.gethostname()}] Initializing data serializer....")
    data_serializer: DataSerializerProtocol = initialize_data_serializer(parameters)
    log.debug(f"[Rank {mpi_rank} {socket.gethostname()}] Initializing data serializer: Done!")

    log.debug(f"[Rank {mpi_rank} {socket.gethostname()}] Initializing data handlers....")
    data_handlers: list[DataHandlerProtocol] = initialize_data_handlers(parameters)
    log.debug(f"[Rank {mpi_rank} {socket.gethostname()}] Initializing data handlers: Done!")

    workflow: Any = source.get_events()

    if num_events > 0:
        workflow >>= take(num_events)

    if parameters.skip_incomplete_events is True:
        workflow >>= _filter_incomplete_events(max_consecutive=1)

    workflow >>= processing_pipeline

    workflow = Source(workflow)
    workflow >>= data_serializer

    workflow = Source(workflow)
    data_handler: DataHandlerProtocol
    for data_handler in data_handlers:
        workflow >>= tap(data_handler)

    workflow >>= map(_data_counter)

    for stat in workflow >> clock():
        log.debug(f"[Rank {mpi_rank}] {stat}]")

    print(f"[Rank {mpi_rank}] Hello, I'm done now.  Have a most excellent day!")
