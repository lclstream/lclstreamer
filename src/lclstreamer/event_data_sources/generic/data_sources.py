from typing import Any

import numpy
from numpy.typing import NDArray

from ...models.parameters import DataSourceParameters
from ...utils.logging import log_error_and_exit
from ...utils.protocols import DataSourceProtocol


class ConstValue(DataSourceProtocol):
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
        Initializes a Const Value Data Source

        Arguments:

            name: An identifier for the data source

            parameters: The configuration parameters
        """
        del additional_info
        extra_parameters: dict[str, Any] | None = parameters.model_dump()
        if extra_parameters is None:
            log_error_and_exit(
                f"Entries needed by the {name} data source are not defined"
            )
        self._dtype = extra_parameters["dtype"]
        try:
            numpy.dtype(self._dtype)
        except (TypeError, ValueError):
            log_error_and_exit(
                f"Entry 'dtype' is not defined for data source {name}"
            )
        raw_value: numpy.number = extra_parameters["value"]
        cast_value: numpy.NDArray[numpy.number] = numpy.array(raw_value, dtype=self._dtype)
        if not numpy.array_equal(cast_value, numpy.array(raw_value)):
                 log_error_and_exit(
                    f"Value '{raw_value}' is not dtype '{self._dtype}' "
                    f"for data source {name}."
                )
        self._data_dict: dict[str, NDArray[numpy.number]] = {name: cast_value}

    def get_data(self, event: Any) -> dict[str, NDArray[numpy.number]]:
        """
        Retrieves the constant float or int value defined in the configuration file as an 1d array

        Arguments:

            event: An internal, psana1 or psana2 event

        Returns:

            An 1d array storing the value defined by the data source
            configuration parameters.
        """
        print(self._data_dict)
        return self._data_dict

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
        extra_parameters: dict[str, Any] | None = parameters.model_dump()
        if extra_parameters is None:
            log_error_and_exit(
                f"Entries needed by the {name} data source are not defined"
            )
        try:
            self._array_shape: tuple[int, ...] = tuple(extra_parameters["array_shape"])
        except ValueError:
            log_error_and_exit(
                f"Parameter 'array_shape' for data source {name} is malformed"
            )
        try:
            self._array_dtype: numpy.dtype[numpy.number] = numpy.dtype(
                extra_parameters["array_dtype"]
            )
        except TypeError:
            log_error_and_exit(
                f"Dtype {extra_parameters['array_dtype']} is not available in numpy"
            )
        self._always_random = extra_parameters["always_random"] # Check in models whether it is bool not here
        self._name = name

        if not self._always_random:
            # Pre-generate the array and re-use it to save computing time
            self._array = self._gen_data(self._array_dtype, self._array_shape)

    def _gen_data(self, dtype, shape) -> NDArray[numpy.number]:
        """
        Generates an array of int of float random numbers

        Arguments:

            dtype: Type of numbers

            shape: Shape of array

        Returns: an array of the type and size requested by the user, containing
            random data (either of integer or floating type)
        """
        if numpy.issubdtype(self._array_dtype, numpy.integer):
            return numpy.random.randint(
                low=0, high=255, size=self._array_shape).astype(self._array_dtype
            )
        elif numpy.issubdtype(self._array_dtype, numpy.floating):
            return numpy.random.random(self._array_shape).astype(self._array_dtype)
        else:
            log_error_and_exit(
                "Only random arrays of integer of floating types are currently "
                "supported"
            )


    def get_data(self, event: Any) -> dict[str, NDArray[numpy.number]]:
        """
        Retrieves an array of int of float random numbers

        Arguments:

            event: A psana1 or psana2 event (doesn't matter)

        Returns:

            A dictionary of random numbers as requested by the user.
        """
        del event
        data_dict: dict[str, Any] = {}
        if self._always_random:
            data_dict[self._name] = self._gen_data(self._array_dtype, self._array_shape)
        else:
            data_dict[self._name] = self._array
        return data_dict


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
        extra_parameters: dict[str, Any] | None = parameters.model_dump()
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

        if extra_parameters["psana_fields"] is None:
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
        return (name, numpy.asarray(base(event), dtype=self.dtype))

    def _get_callable_with_noevent(self, name, base, event):
        return (name, numpy.asarray(base(), dtype=self.dtype))

    def _get_noncallable(self, name, base, event):
        return (name, numpy.asarray(base, dtype=self.dtype))

    def _create_detector(self, *args, **kwargs):
        raise NotImplementedError("Derived classes have to implement their _create_detector")

    def get_data(self, event: Any) -> dict[str, NDArray[numpy.number]]:
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
