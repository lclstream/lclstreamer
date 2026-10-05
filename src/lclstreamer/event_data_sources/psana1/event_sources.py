from collections.abc import Generator
from typing import Any, cast

from psana import DataSource, MPIDataSource  # type: ignore
from stream.core import source

from ...models.parameters import DataSourceParameters, Psana1EventSourceParameters
from ...utils.logging import log_info, log_error_and_exit
from ...utils.protocols import DataSourceProtocol, EventSourceProtocol
from ...utils.typing import StrFloatIntNDArray
from ..generic.data_sources import GenericRandomNumpyArray as GenericRandomNumpyArray
from .data_sources import (
    Psana1DetectorInterface as Psana1DetectorInterface,
)
from .data_sources import (
    Psana1Timestamp as Psana1Timestamp,
)


class Psana1EventSource(EventSourceProtocol):
    """
    See documentation of the `__init__` function
    """

    def __init__(
        self,
        parameters: Psana1EventSourceParameters,
        data_source_parameters: dict[str, DataSourceParameters],
        source_identifier: str,
        worker_pool_size: int,
        worker_rank: int,
    ) -> None:
        """
        Initializes a Psana1 Event Source

        Arguments:

            parameters: The event source configuration parameters

            worker_pool_size: The size of the worker pool

            worker_rank: The rank of the worker calling the function
        """
        del worker_pool_size
        del worker_rank

        if parameters.type != "Psana1EventSource":
            log_error_and_exit("Event source parameters do not match the expected type")

        if "shmem" in source_identifier:
            self._event_source: Generator[Any] = cast(
                Generator[Any],
                DataSource(  # pyright: ignore[reportUnknownMemberType]
                    source_identifier
                ).events(),
            )
        else:
            psana_source_string: str = source_identifier
            if not psana_source_string.endswith(":smd"):
                psana_source_string = f"{psana_source_string}:smd"
            self._event_source = cast(
                Generator[Any],
                MPIDataSource(  # pyright: ignore[reportUnknownMemberType]
                    psana_source_string
                ).events(),
            )

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
                    additional_info={"source_identifier": source_identifier},
                )
            except NameError:
                log_error_and_exit(
                    f"Data source {data_source_parameters[data_source_name].type} "
                    "is not available for backend Psana1EventSource"
                )

    @source
    def get_events(
        self,
    ) -> Generator[dict[str, StrFloatIntNDArray | None]]:
        """
        Retrieves events from the data source

        Yields:

            data: A dictionary storing data for an event
        """
        psana_event: Any
        for psana_event in self._event_source:
            data: StrFloatIntNDArray | None
            data_dict: dict[str, StrFloatIntNDArray | None] = {}
            data_source_name: str

            for data_source_name in self._data_sources:
                name: str
                try:
                    name, data = self._data_sources[
                        data_source_name
                    ].get_data(event=psana_event)
                except (TypeError, AttributeError) as e:
                    log_info(e)
                    name, data = "Invalid", None
                data_dict[name] = data
            yield data_dict
