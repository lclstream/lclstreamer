from typing import Any

import numpy
from numpy.typing import NDArray
from psana import Detector, EventId  # type: ignore

from ...models.parameters import DataSourceParameters
from ...utils.logging import log_error_and_exit
from ...utils.protocols import DataSourceProtocol


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


class Psana1DetectorInterface(DataSourceProtocol):
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
        """
        del additional_info
        extra_parameters: dict[str, Any] | None = parameters.__pydantic_extra__

        self._name: str = name
        if extra_parameters is None:
            log_error_and_exit(
                f"Entries needed by the {name} data source are not defined"
            )
            return  # For the type checker
        if "psana_name" not in extra_parameters:
            log_error_and_exit(
                f"Entry 'psana_name' is not defined for data source {name}"
            )
        if "psana_fields" not in extra_parameters:
            if ":" in extra_parameters["psana_name"]:
                self._is_pv: bool = True
            else:
                log_error_and_exit(
                    f"Entry 'psana_fields' is not defined for data source {name}"
                )
        else:
            fields: list[str] | str = extra_parameters["psana_fields"]
            self._det_params: list[str] = (
                [fields] if isinstance(fields, str) else fields
            )

        self.dtype: type
        if "dtype" not in extra_parameters:
            self.dtype = numpy.float64
        else:
            self.dtype = extra_parameters["dtype"]

        self._detector_interface: Any = Detector(extra_parameters["psana_name"])

    def get_data(self, event: Any) -> NDArray[Any]:
        """
        Retrieves data via the Detector Interface from a psana1 event

        Arguments:

            event: A psana1 event

         Returns:

            value: The retrieved data in the format of a numpy array
        """

        data: list[Any] = []

        base: Any
        if getattr(self, "_is_pv", False):
            base = self._detector_interface
            data.append(base(event))
        else:
            for param in self._det_params:

                base = self._detector_interface

                subfields: list[str] = param.split(".")
                for field in subfields:
                    if hasattr(base, field):
                        base = getattr(base, field)
                    else:
                        log_error_and_exit(f"Detector {base} has no parameter {field}")
                if callable(base):
                    try:
                        base = base(event)
                    except TypeError:
                        base = base()
                    data.append(base)
                else:
                    data.append(base)

        if len(data) == 1:
            data = data[0]
            if isinstance(data, dict):
                log_error_and_exit(
                    f"Data for the psana2 data source {self._name} has "
                    "the format of a dictionary!"
                )
            if param == "eventCodes":
                # special case for event codes
                return numpy.pad(data,
                        pad_width=(0, 256 - len(data)),
                        mode="constant",
                        constant_values=(0, 0),
                    )
        return numpy.array(data, dtype=self.dtype)
