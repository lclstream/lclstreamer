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


class MpiRank(DataSourceProtocol):
    """
    Data source that returns the MPI rank of the current process.

    Useful for tracking which producer sent each event.
    """

    def __init__(
        self,
        name: str,
        parameters: DataSourceParameters,
        additional_info: dict[str, Any],
    ):
        """
        Initializes an MPI Rank data source.

        Attempts to get rank from:
        1. MPI library (if available and initialized)
        2. Environment variables (OMPI_COMM_WORLD_RANK, PMI_RANK, SLURM_PROCID)
        3. Falls back to 0 if not in MPI context

        Arguments:
            name: An identifier for the data source
            parameters: The configuration parameters
            additional_info: Additional runtime info
        """
        del name
        del parameters
        del additional_info

        self._rank: int = self._get_mpi_rank()

    def _get_mpi_rank(self) -> int:
        """Get MPI rank from various sources."""
        import os

        # Try MPI library first
        try:
            from mpi4py import MPI
            if MPI.COMM_WORLD.Get_size() > 1:
                return MPI.COMM_WORLD.Get_rank()
        except ImportError:
            pass
        except Exception:
            pass

        # Try environment variables set by MPI launchers
        for env_var in ['OMPI_COMM_WORLD_RANK', 'PMI_RANK', 'SLURM_PROCID',
                        'MV2_COMM_WORLD_RANK', 'MPICH_RANK']:
            rank_str = os.environ.get(env_var)
            if rank_str is not None:
                try:
                    return int(rank_str)
                except ValueError:
                    pass

        # Not in MPI context
        return 0

    def get_data(self, event: Any) -> NDArray[numpy.int32]:
        """
        Returns the MPI rank as a numpy array.

        Arguments:
            event: Event object (unused)

        Returns:
            1D numpy array containing the MPI rank
        """
        del event
        return numpy.array(self._rank, dtype=numpy.int32)
