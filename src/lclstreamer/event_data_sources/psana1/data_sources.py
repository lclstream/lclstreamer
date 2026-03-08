from typing import Any

import numpy
from numpy.typing import NDArray
from psana import Detector, EventId  # type: ignore

from ...models.parameters import DataSourceParameters
from ...utils.logging import log_error_and_exit
from ...utils.protocols import DataSourceProtocol
from ..generic.data_sources import BaseDetectorInterface

class Psana1Timestamp(DataSourceProtocol):
    """
    See documentation of the `__init__` function
    """

    def __init__(
        self,
        name: str,
        parameters: DataSourceParameters,
        additional_info: dict[str, Any],
    ):
        """
        Initializes a Psana1 Timestamp Data Source

        Arguments:

            name: An identifier for the data source

            parameters: The data source configuration parameters
        """
        del name
        del parameters
        del additional_info

    def get_data(self, event: Any) -> NDArray[numpy.uint64]:
        """
        Retrieves timestamp information from a psana1 event

        Encodes the timestamp as a uint64 matching psana2's native format:
        upper 32 bits = seconds, lower 32 bits = nanoseconds.

        Arguments:

            event: A psana1 event

        Returns:

            timestamp: a numpy scalar (uint64) with seconds in upper 32 bits
            and nanoseconds in lower 32 bits
        """
        psana_event_id: Any = event.get(
            EventId  # pyright: ignore[reportAttributeAccessIssue]
        )
        timestamp_epoch_format: Any = psana_event_id.time()
        seconds = int(timestamp_epoch_format[0])
        nanoseconds = int(timestamp_epoch_format[1])
        return numpy.array(
            (seconds << 32) | nanoseconds,
            dtype=numpy.uint64,
        )


class Psana1DetectorInterface(BaseDetectorInterface):
    """
    See documentation of the `__init__` function
    """

    def __init__(
        self,
        name: str,
        parameters: DataSourceParameters,
        additional_info: dict[str, Any],
    ):
        """
        Initializes Psana1 Detector Interface Data Source

        Arguments:

            name: An identifier for the data source

            parameters: The data source configuration parameters

            additional_info: Contains run information etc. (not used for psana1)
        """

        super().__init__(name, parameters, additional_info)
        del additional_info

    def _create_detector(self):
        return Detector(self._detector_name)

    def _setup_special_fields(self, psana_fields, data_caller):
        if psana_fields == ["eventCodes"]:
            data_caller = self._get_evr_codes
            self.dtype = numpy.int64
        return data_caller

    def _get_evr_codes(self, name, base, event):
        evr_codes: list[numpy.int64] = base(event)
        data: list[numpy.int64] = numpy.pad(evr_codes,
            pad_width=(0, 256 - len(evr_codes)),
            mode="constant",
            constant_values=(0, 0),
        )
        return (name, numpy.array(data, dtype=self.dtype))
