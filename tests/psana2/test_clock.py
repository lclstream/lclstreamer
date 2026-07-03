from typing import Any

from mpi4py import MPI
from stream.core import Source, Stream

from lclstreamer.utils.stream import clock

mpi_rank: int = MPI.COMM_WORLD.rank


def test_clock():
    src: Source[int] = Source(range(1, 100, 3))

    stat: Stream[Any, Any]
    for stat in src >> clock():  # pyright: ignore[reportUnknownVariableType]
        print(f"[Rank {mpi_rank}] {stat}]", flush=True)
