from collections.abc import Generator

from stream.core import source

from ...models.parameters import (
    DataSourceParameters,
    InternalEventSourceParameters,
)
from ...utils.logging import log_error_and_exit
from ...utils.protocols import DataSourceProtocol, EventSourceProtocol
from ...utils.typing import (
    StrFloatIntNDArray,
)
from .data_sources import (
    FloatValue as FloatValue,
)
from .data_sources import (
    GenericRandomNumpyArray as GenericRandomNumpyArray,
)
from .data_sources import (
    IntValue as IntValue,
)
from .data_sources import (
    MpiRank as MpiRank,
)


class InternalEventSource(EventSourceProtocol):
    """
    See documentation of the `__init__` function
    """

    def __init__(
        self,
        parameters: InternalEventSourceParameters,
        data_source_parameters: dict[str, DataSourceParameters],
        source_identifier: str,
        worker_pool_size: int,
        worker_rank: int,
    ) -> None:
        """
        Initializes an Internal Event Source

        This Event Source does not rely on any external framework to generate events
        and is only compatible with data sources that don't use any external
        framework to generate data. It is intended mainly for testing

        Arguments:

            parameters: The event source configuration parameters

            worker_pool_size: The size of the worker pool

            worker_rank: The rank of the worker calling the function
        """
        del worker_pool_size
        del worker_rank

        if parameters.type != "InternalEventSource":
            log_error_and_exit("Event source parameters do not match the expected type")

        self.number_of_events_to_generate: int = parameters.number_of_events_to_generate

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
                    "is not available for backend InternalEventSource"
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
        for i in range(self.number_of_events_to_generate):
            data: dict[str, StrFloatIntNDArray | None] = {}
            data_source_name: str

            for data_source_name in self._data_sources:
                try:
                    data[data_source_name] = self._data_sources[
                        data_source_name
                    ].get_data(event=i)
                except (TypeError, AttributeError):
                    data[data_source_name] = None
            yield data
