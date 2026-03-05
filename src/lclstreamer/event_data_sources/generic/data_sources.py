from typing import Any

import numpy
from numpy.typing import NDArray

from ...models.parameters import DataSourceParameters
from ...utils.logging import log_error_and_exit
from ...utils.protocols import DataSourceProtocol


class FloatValue(DataSourceProtocol):
    """
    See documentation of the `__init__` function.
    """

    def __init__(
        self,
        name: str,
        parameters: DataSourceParameters,
        additional_info: dict[str, Any],
    ):
        """
        Initializes a Float Value Data Source

        Arguments:

            name: An identifier for the data source

            parameters: The configuration parameters
        """
        del additional_info
        extra_parameters: dict[str, Any] | None = parameters.__pydantic_extra__
        if extra_parameters is None:
            log_error_and_exit(
                f"Entries needed by the {name} data source are not defined"
            )
        if "value" not in extra_parameters:
            log_error_and_exit(
                f"Entry 'array_shape' is not defined for data source {name}"
            )
        try:
            self._value: float = float(extra_parameters["value"])
        except ValueError:
            log_error_and_exit(
                f"Entry 'value' is not a valid float for data source {name}"
            )

    def get_data(self, event: Any) -> NDArray[numpy.float64]:
        """
        Retrieves the float value defined in the configuration file as an 1d array

        Arguments:

            event: A psana1 event

        Returns:

            random: an 1d array storing the value defined by the data source
            configuration parameters.
        """
        return numpy.array(self._value, dtype=numpy.float64)


class IntValue(DataSourceProtocol):
    """
    See documentation of the `__init__` function.
    """

    def __init__(
        self,
        name: str,
        parameters: DataSourceParameters,
        additional_info: dict[str, Any],
    ):
        """
        Initializes a Int Value Data Source

        Arguments:

            name: An identifier for the data source

            parameters: The configuration parameters
        """
        del additional_info
        extra_parameters: dict[str, Any] | None = parameters.__pydantic_extra__
        if extra_parameters is None:
            log_error_and_exit(
                f"Entries needed by the {name} data source are not defined"
            )
        if "value" not in extra_parameters:
            log_error_and_exit(
                f"Entry 'array_shape' is not defined for data source {name}"
            )
        try:
            self._value: int = int(extra_parameters["value"])
        except ValueError:
            log_error_and_exit(
                f"Entry 'value' is not a valid int for data source {name}"
            )

    def get_data(self, event: Any) -> NDArray[numpy.int_]:
        """
        Retrieves the int value defined in the configuration file as an 1d array

        Arguments:

            event: A psana1 event

        Returns:

            random: an 1d array storing the value defined by the data source
            configuration parameters.
        """
        return numpy.array(self._value, dtype=numpy.int_)


class GenericRandomNumpyArray(DataSourceProtocol):
    """
    See documentation of the `__init__` function.
    """

    def __init__(
        self,
        name: str,
        parameters: DataSourceParameters,
        additional_info: dict[str, Any],
    ):
        """
        Initializes a Generic Random Numpy Array Data Source.

        Arguments:

            name: An identifier for the data source

            parameters: The configuration parameters
        """
        del additional_info
        extra_parameters: dict[str, Any] | None = parameters.__pydantic_extra__
        if extra_parameters is None:
            log_error_and_exit(
                f"Entries needed by the {name} data source are not defined"
            )
        if "array_shape" not in extra_parameters:
            log_error_and_exit(
                f"Entry 'array_shape' is not defined for data source {name}"
            )
        if "array_dtype" not in extra_parameters:
            log_error_and_exit(
                f"Entry 'array_dtype' is not defined for data source {name}"
            )

        try:
            self._array_shape: tuple[int, ...] = tuple(
                int(x) for x in extra_parameters["array_shape"].split(",")
            )
        except ValueError:
            log_error_and_exit(
                f"Parameter 'array_dtype' for data source {name} is malformed"
            )
        try:
            self._array_dtype: numpy.dtype[numpy.int_ | numpy.float64] = numpy.dtype(
                extra_parameters["array_dtype"]
            )
        except TypeError:
            log_error_and_exit(
                f"Dtype {extra_parameters['array_dtype']} is not available in numpy"
            )

    def get_data(self, event: Any) -> NDArray[numpy.float64 | numpy.int_]:
        """
        Retrieves an array of int of float random numbers

        Arguments:

            event: A psana1 event

        Returns:

            random: an array of the type and size requested by the user, containing
            random data (either of integer or floating type)
        """
        del event
        if numpy.issubdtype(self._array_dtype, numpy.integer):
            return numpy.random.randint(low=0, high=255, size=self._array_shape).astype(
                self._array_dtype
            )
        elif numpy.issubdtype(self._array_dtype, numpy.floating):
            return numpy.random.random(self._array_shape).astype(self._array_dtype)
        else:
            log_error_and_exit(
                "Only random arrays of integer of floating types are currently "
                "supported"
            )


class GenericRandomTimestamp(DataSourceProtocol):
    """
    See documentation of the `__init__` function.
    """

    def __init__(
        self,
        name: str,
        parameters: DataSourceParameters,
        additional_info: dict[str, Any],
    ):
        """
        Initializes a Generic Random Timestamp Data Source

        Arguments:

            name: An identifier for the data source

            parameters: The configuration parameters

            additional_info: Additional information (unused)
        """
        del name
        del parameters
        del additional_info

    def get_data(self, event: Any) -> NDArray[numpy.float64]:
        """
        Retrieves a random timestamp value (Unix epoch time in seconds)

        Arguments:

            event: An event object (unused)

        Returns:

            timestamp: A 0-dimensional numpy float64 array containing a random
                timestamp in the range ~2020-2023
        """
        del event
        random_timestamp: float = numpy.random.uniform(1600000000, 1700000000)
        return numpy.array(random_timestamp, dtype=numpy.float64)


class GenericRandomWavelength(DataSourceProtocol):
    """
    See documentation of the `__init__` function.
    """

    def __init__(
        self,
        name: str,
        parameters: DataSourceParameters,
        additional_info: dict[str, Any],
    ):
        """
        Initializes a Generic Random Wavelength Data Source

        Arguments:

            name: An identifier for the data source

            parameters: The configuration parameters

            additional_info: Additional information (unused)
        """
        del name
        del parameters
        del additional_info

    def get_data(self, event: Any) -> NDArray[numpy.float64]:
        """
        Retrieves a random wavelength value in the typical X-ray range

        Arguments:

            event: An event object (unused)

        Returns:

            wavelength: A 0-dimensional numpy float64 array containing a random
                wavelength value between 0.1 and 10.0 Angstroms
        """
        del event
        random_wavelength: float = numpy.random.uniform(0.1, 10.0)
        return numpy.array(random_wavelength, dtype=numpy.float64)


class SourceIdentifier(DataSourceProtocol):
    """
    See documentation of the `__init__` function.
    """

    def __init__(
        self,
        name: str,
        parameters: DataSourceParameters,
        additional_info: dict[str, Any],
    ):
        """
        Initializes a Source Identifier Data Source.

        Arguments:

            name: An identifier for the data source

            parameters: The configuration parameters

            additional_info: A dictionary of additional information, expected to
                contain a ``source_identifier`` key whose value is stored and
                returned by `get_data`
        """
        del name
        del parameters
        self._source_identifier: NDArray[numpy.str_] = numpy.array(
            additional_info["source_identifier"]
        )

    def get_data(self, event: Any) -> NDArray[numpy.str_]:
        """
        Retrieves the source identifier as a numpy string array.

        Arguments:

            event: An event object (unused)

        Returns:

            source_identifier: A 0-dimensional numpy string array containing the
                source identifier defined at initialization
        """
        return self._source_identifier

class BaseDetectorInterface(DataSourceProtocol):
    def __init__(
        self,
        name: str,
        parameters: DataSourceParameters,
        additional_info: dict[str, Any],
    ):
        """
        Initializes the DetectorInterface base class

        Arguments:

            name: An identifier for the data source

            parameters: The data source configuration parameters
        """
        self._name: str = name
        extra_parameters: dict[str, Any] | None = parameters.__pydantic_extra__
        self._call_get_data: list[tuple[str, Any, Any]] = []

        if extra_parameters is None:
            log_error_and_exit(
                f"Entries needed by the {name} data source are not defined"
            )
            return  # For the type checker
        if "psana_name" not in extra_parameters:
            log_error_and_exit(
                f"Entry 'psana_name' is not defined for data source {name}"
            )

        self._detector_name: str = extra_parameters["psana_name"]
        detector_interface: Any = self._create_detector()

        self.dtype: type
        if "dtype" not in extra_parameters:
            self.dtype = numpy.float64
        else:
            self.dtype = extra_parameters["dtype"]

        if "psana_fields" not in extra_parameters:
            if ":" in self._detector_name:
                # it is a PV
                self._call_get_data.append((self._detector_name, detector_interface, self._get_callable_with_event))
            else:
                log_error_and_exit(
                    f"Entry 'psana_fields' is not defined for data source {name}"
                )
        else:
            fields: list[str] | str = extra_parameters["psana_fields"]
            det_fields: list[str] = ([fields] if isinstance(fields, str) else fields)
            det_fields = [f.split(".") for f in det_fields]

            for psana_fields in det_fields:
                data_caller: Any = None
                base = detector_interface
                psana_field: str = ".".join([self._detector_name, *psana_fields])

                for field in psana_fields:
                    # Find the full name of the function we will call
                    if hasattr(base, field):
                        base = getattr(base, field)
                    else:
                        log_error_and_exit(f"Detector {base} has no parameter {field}")

                if callable(base):
                    # Check if bound method or not plus the number of args
                    arg_number = base.__code__.co_argcount - (1 if hasattr(base, "__self__") else 0)
                    if arg_number > 0:
                        data_caller = self._get_callable_with_event
                    else:
                        data_caller = self._get_callable_with_noevent
                else:
                    data_caller = self._get_noncallable

                data_caller = self._setup_special_fields(psana_fields, data_caller)

                self._call_get_data.append((psana_field, base, data_caller))

    def _setup_special_fields(self, psana_fields, data_caller):
        return data_caller

    def _get_callable_with_event(self, name, base, event):
        return (name, numpy.array(base(event), dtype=self.dtype))

    def _get_callable_with_noevent(self, name, base, event):
        return (name, numpy.array(base(), dtype=self.dtype))

    def _get_noncallable(self, name, base, event):
        return (name, numpy.array(base, dtype=self.dtype))

    def _create_detector(self, *args, **kwargs):
        raise NotImplementedError("Derived classes have to implement their _create_detector")

    def get_data(self, event: Any) -> NDArray[Any]:
        """
        Retrieves Detector values from a psana event

        Arguments:

            event: A psana1 or 2 event

         Returns:

            value: The retrieved data in the format of a numpy array
        """
        data_dict: dict[str, Any] = {}
        name: str
        base: Any
        data_caller: Any
        data = Any

        for name, base, data_caller in self._call_get_data:
            name, data = data_caller(name, base, event)
            if isinstance(data, dict):
                log_error_and_exit(
                    f"Data for the psana data source {self._name} has "
                    "the format of a dictionary! HSD detectors are not supported yet."
                )
            data_dict[name] = data

        return data_dict
