from typing import Any

import numpy
from numpy.typing import NDArray

from ...models.parameters import DataSourceParameters
from ...utils.logging import log_error_and_exit
from ...utils.protocols import DataSourceProtocol
from ..generic.data_sources import BaseDetectorInterface

class Psana2Timestamp(DataSourceProtocol):
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
        Initializes a psana2 Timestamp Data Source

        Arguments:

            name: An identifier for the data source

            parameters: The data source configuration parameters
        """
        pass

    def get_data(self, event: Any) -> NDArray[numpy.uint64]:
        """
        Retrieves timestamp information from a psana2 event.

        The psana2 event.timestamp is natively a uint64 with seconds in the
        upper 32 bits and nanoseconds in the lower 32 bits.

        Arguments:

            event: A psana2 event

        Returns:

            timestamp: a numpy scalar (uint64) with seconds in upper 32 bits
            and nanoseconds in lower 32 bits
        """
        return numpy.array(event.timestamp, dtype=numpy.uint64)


class Psana2DetectorInterface(BaseDetectorInterface):
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
        Initializes a psana2 Detector values data source

        Arguments:

            name: An identifier for the data source

            parameters: The data source configuration parameters

            additional_info: Contains run information etc.
        """
        self._additional_info = additional_info
        super().__init__(name, parameters, additional_info)

    def _create_detector(self):
        return self._additional_info["run"].Detector(
            self._detector_name
        )

class Psana2RunInfo(DataSourceProtocol):
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
        Initializes a Psana2 Run Info Data Source

        Arguments:

            name: An identifier for the data source

            parameters: The data source configuration parameters
        """
        run: Any = additional_info["run"]
        self._run_data: dict[str, NDArray[numpy.str_]] = {  # pyright: ignore[reportUnknownMemberType]
            "experiment": numpy.array(run.expt, dtype=numpy.str_),  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
            "run_timestamp": numpy.array(str(run.timestamp), dtype=numpy.str_),  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownArgumentType]
            "run_number": numpy.array(str(run.runnum), dtype=numpy.str_),  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownArgumentType]
            "source_identifier": numpy.array(additional_info["source_identifier"], dtype=numpy.str_),
        }

    def get_data(self, event: Any) -> dict[str, NDArray[numpy.str_]]:
        """
        Retrieves the detector info from a psana2 event

        Arguments:

            event: A psana2 event

         Returns:

            value: The retrieved data in the format of a numpy array
        """

        return self._run_data
