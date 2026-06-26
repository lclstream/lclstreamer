import os
from collections.abc import Generator
from typing import Any, cast

from mpi4py import MPI
from psana import DataSource  # type: ignore
from stream.core import source

from ...models.parameters import DataSourceParameters, Psana2EventSourceParameters
from ...utils.logging import log_error_and_exit, log_info
from ...utils.protocols import (
    DataSourceProtocol,
    EventSourceProtocol,
)
from ...utils.typing import StrFloatIntNDArray
from ..generic.data_sources import GenericRandomNumpyArray as GenericRandomNumpyArray
from .data_sources import (
    Psana2DetectorInterface as Psana2DetectorInterface,
)
from .data_sources import (
    Psana2RunInfo as Psana2RunInfo,
)
from .data_sources import (
    Psana2Timestamp as Psana2Timestamp,
)


def _parse_source_identifier(source_identifier: str) -> dict[str, str | int]:
    # Parses a source identifier string into a keyword-argument dictionary
    # The source identifier is a comma-separated string of key=value pairs

    rank: int = MPI.COMM_WORLD.Get_rank()

    source_dict: dict[str, str | int] = {}
    source_items: list[str] = source_identifier.split(",")
    item: str
    for item in source_items:
        print(item)
        if item.startswith("shmem="):
            source_dict["shmem"] = item.split("shmem=")[1].strip().lstrip()
        elif item.startswith("exp="):
            source_dict["exp"] = item.split("exp=")[1].strip().lstrip()
        elif item.startswith("run="):
            source_dict["run"] = int(item.split("run=")[1].strip().lstrip())
        elif item.startswith("files="):
            source_dict["files"] = item.split("files=")[1].strip().lstrip()
        elif item.startswith("drp="):
            source_dict["drp"] = item.split("drp=")[1].strip().lstrip()
        elif item.startswith("max_events="):
            source_dict["max_events"] = int(
                item.split("max_events=")[1].strip().lstrip()
            )
        elif item.strip().lstrip() == "live":
            source_dict["live"] = True
            log_info(f"Rank: {rank} running in live mode")
        elif item.strip().lstrip() == "ffb":
            os.environ["SIT_PSDM_DATA"] = "/sdf/data/lcls/drpsrcf/ffb"
            log_info(f"Rank: {rank} reading data from the FFB filesystem")
        else:
            log_error_and_exit(
                "Part of the source string for psana2 cannot be error=:\n{item}"
            )
    return source_dict


class Psana2EventSource(EventSourceProtocol):
    """
    See documentation of the `__init__` function.
    """

    def __init__(
        self,
        parameters: Psana2EventSourceParameters,
        data_source_parameters: dict[str, DataSourceParameters],
        source_identifier: str,
        worker_pool_size: int,
        worker_rank: int,
    ) -> None:
        """
        Initializes a Psana2 Event Source

        Arguments:

            parameters: The event source configuration parameters

            worker_pool_size: The size of the worker pool

            worker_rank: The rank of the worker calling the function
        """
        del worker_pool_size
        del worker_rank

        if parameters.type != "Psana2EventSource":
            log_error_and_exit("Event source parameters do not match the expected type")

        if "shmem" in source_identifier:
            log_error_and_exit(
                "Shared memory mode is not currently available for the psana2 data "
                "event source"
            )
        else:
            data_source_arguments: dict[str, str | int] = _parse_source_identifier(
                source_identifier
            )
            psana_data_source: Any = (  # pyright: ignore[reportUnknownVariableType]
                DataSource(**data_source_arguments)
            )
            self._psana_run: Any = next(
                psana_data_source.runs()  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            )
            self._event_source: Any = cast(
                Generator[Any],
                self._psana_run.events(),  # pyright: ignore[reportUnknownMemberType]
            )

        # self._event_source = DataSource(parameters.source_identifier).events()

        self._data_sources: dict[str, DataSourceProtocol] = {}
        data_source_name: str
        for data_source_name in data_source_parameters:
            try:
                data_source_class: type[DataSourceProtocol] = globals()[
                    data_source_parameters[data_source_name].type
                ]

                self._data_sources[data_source_name] = data_source_class(
                    name=data_source_name,
                    parameters=data_source_parameters[data_source_name],
                    additional_info={
                        "run": self._psana_run,  # pyright: ignore[reportUnknownMemberType]
                        "source_identifier": source_identifier,
                    },
                )
            except NameError as e:
                log_error_and_exit(
                    f"Data source {data_source_parameters[data_source_name].type} "
                    f"is not available for backend Psana2EventSource: {e}"
                )

    @source
    def get_events(
        self,
    ) -> Generator[dict[str, StrFloatIntNDArray | None]]:
        """
        Retrieves an event from the data source

        Returns:

            data: A dictionary storing data for an event
        """
        psana_event: Any
        for psana_event in self._event_source:
            data: dict[str, Any | None] = {}

            data_source_name: str
            for data_source_name in self._data_sources:
                try:
                    data[data_source_name] = self._data_sources[
                        data_source_name
                    ].get_data(event=psana_event)
                except (TypeError, AttributeError) as e:
                    log_info(e)
                    data[data_source_name] = None
            yield data
