from typing import Any

import numpy
from numpy.typing import NDArray

from ...models.parameters import DataSourceParameters
from ...utils.logging import log_error_and_exit
from ...utils.protocols import DataSourceProtocol


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

    def get_data(self, event: Any) -> NDArray[numpy.float64]:
        """
        Retrieves timestamp information from a psana2 event

        Arguments:

            event: A psana2 event

        Returns:

            timestamp: a 1D numpy array (of type float64) containing the timestamp
            information
        """
        return numpy.array(event.timestamp, dtype=numpy.float64)


class Psana2DetectorInterface(DataSourceProtocol):
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
        """
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
            self._det_fields: list[str] = (
                [fields] if isinstance(fields, str) else fields
            )

        # dtype: None means preserve original dtype (e.g., uint16 for raw detector data)
        self.dtype: type | None
        if "dtype" not in extra_parameters:
            self.dtype = None  # Preserve original dtype
        else:
            self.dtype = extra_parameters["dtype"]

        self._detector_interface: Any = additional_info["run"].Detector(
            extra_parameters["psana_name"]
        )

    def get_data(self, event: Any) -> NDArray[Any]:
        """
        Retrieves Detector values from a psana2 event

        Arguments:

            event: A psana2 event

         Returns:

            value: The retrieved data in the format of a numpy array
        """

        data: list[Any] = []

        base: Any
        if getattr(self, "_is_pv", False):
            base = self._detector_interface
            data.append(base(event))
        else:
            for psana_field in self._det_fields:
                base = self._detector_interface
                subfields: list[str] = psana_field.split(".")
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

        # TODO: Streamline logic
        if len(data) == 1:
            data = data[0]
            if isinstance(data, dict):
                log_error_and_exit(
                    f"Data for the psana2 data source {self._name} has "
                    "the format of a dictionary!"
                )
            else:
                # Preserve original dtype if self.dtype is None
                if self.dtype is None:
                    return numpy.asarray(data)
                return numpy.array(data, dtype=self.dtype)
        # Preserve original dtype if self.dtype is None
        if self.dtype is None:
            return numpy.asarray(data)
        return numpy.array(data, dtype=self.dtype)


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
        self._run_data: list[str] = [  # pyright: ignore[reportUnknownMemberType]
            run.expt,  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
            str(run.timestamp),  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownArgumentType]
            str(run.runnum),  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownArgumentType]
            additional_info["source_identifier"],
        ]

    def get_data(self, event: Any) -> NDArray[numpy.str_]:
        """
        Retrieves the detector info from a psana2 event

        Arguments:

            event: A psana2 event

         Returns:

            value: The retrieved data in the format of a numpy array
        """

        return numpy.array(self._run_data, dtype=numpy.str_)
