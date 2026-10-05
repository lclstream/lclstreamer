from dataclasses import dataclass, field
from typing import Any

import numpy
from numpy.typing import DTypeLike

from ...utils.logging import log_error_and_exit
from ...utils.typing import StrFloatIntNDArray


@dataclass
class DataContainer:
    """
    Dataclass used to store accumulated numpy arrays

    Attributes:

        data: A list of numpy arrays accumulated so far for this data source

        dtype: The numpy dtype of the arrays, inferred from the first array added

        shape: The shape of each individual array, inferred from the first array added
    """

    data: list[StrFloatIntNDArray] = field(default_factory=list)
    #data: dict[str, StrFloatIntNDArray | dict [str, StrFloatIntNDArray | None] | None] = field(default_factory=dict)
    dtype: DTypeLike | None = None
    shape: tuple[int, ...] | None = None


class DataStorage:
    """
    See documentation of the `__init__` function
    """

    def __init__(self) -> None:
        """
        Initializes a Data Storage object

        Data Storage objects are containers that can store numpy arrays and allow
        bulk retrieval of the stored data
        """

        self._data_containers: dict[str, DataContainer | dict[str, DataContainer]] = {}
        self._count: int = 0

    def __len__(self) -> int:
        """
        Returns the number of data entries currently stored

        Returns:

            count: The number of times `add_data` has been called since the last
                reset
        """
        return self._count

    def add_data(self, data: dict[str, StrFloatIntNDArray | None]) -> None:
        """
        Adds data to the Data Storage object

        The function takes a dictionary storing numpy arrays, each identified
        by a dictionary key label. When called for the first time, it uses
        the incoming data to determine labels and dtypes of the numpy arrays to
        accumulate. All subsequent calls of the function will only accept data arrays
        with the same labels and dtypes as the initial call, or data whose value is
        None. If the data value is None, this function will the fill the missing data
        with appropriate null values (numpy.NaN for float data, the number -999 for int
        data, and the string "None" for str data)

        Arguments:

            data: a dictionary storing numpy arrays
        """
        if len(self._data_containers) == 0:
            data_source_name: str
            for data_source_name, data_value in data.items():
                if data_value is None:
                    log_error_and_exit(
                        f"Data entry {data_source_name} was none in the first "
                        "event. Impossible to determine data size"
                    )
                else:
                    data_container = DataContainer(
                        data=[data_value],
                        dtype=data_value.dtype,
                        shape=data_value.shape,
                    )
                    self._data_containers[data_source_name] = data_container
        else:
            for data_source_name, data_value in data.items():
                if data_source_name not in self._data_containers:
                    log_error_and_exit(
                        "The data labels in the current event are not in the labels "
                        "used to initialize the Data Storage container"
                    )
                data_container = self._data_containers[data_source_name]

                if data_value is None:
                    if data_container.shape is not None:
                        if numpy.issubdtype(
                            data_container.dtype, numpy.signedinteger[Any]
                        ):
                            data_container.subdata.append(
                                numpy.full(data_container.shape, "-999")
                            )
                            continue
                        elif numpy.issubdtype(
                            data_container.dtype, numpy.floating[Any]
                        ):
                            data_container.subdata.append(
                                numpy.full(
                                    data_container.shape,
                                    numpy.float64("nan"),
                                    dtype=data_container.dtype,
                                )
                            )
                            continue
                        else:
                            data_container.subdata.append(
                                numpy.full(
                                    data_container.shape, "None", dtype=numpy.str_
                                )
                            )
                            continue
                else:
                    if data_value.dtype != data_container.dtype:
                        log_error_and_exit(
                            f"The dtype of the data entry {data_source_name} in the "
                            "current event does not match the dtype of the data "
                            "with which this label was originally initialized"
                        )
                    if data_value.shape != data_container.shape:
                        log_error_and_exit(
                            f"The shape of the data entry {data_source_name} in the "
                            "current event does not match the shape of the data "
                            "with which this label was originally initialized"
                        )
                    data_container.data.append(data_value)
        self._count += 1

    def retrieve_stored_data(self) -> dict[str, StrFloatIntNDArray | None]:
        """
        Retuns the data stored in the Data Storage container object

        The data is returned as dictionary of numpy arrays. The keys of the
        dictionary match the labels of the stored data. The array associated
        with each label stores the accumulated data, with the fist axis
        representing each subsequent data item added, and the rest of the axes
        representing the accumulated data

        Returns:

            stored_data: A dictionary containing the data accumulated by the
                Data Storage container
        """

        stored_data: dict[str, StrFloatIntNDArray | None] = {}

        data_source_name: str
        for data_source_name in self._data_containers:
            stored_data[data_source_name] = numpy.stack(
                self._data_containers[data_source_name].data,
            )

        return stored_data

    def reset_data_storage(self) -> None:
        """
        Resets the Data Storage container

        Clears all accumulated arrays from every data container and resets the
        internal event counter to zero. The container labels and dtypes inferred
        from the first event are preserved so that the storage can be reused for
        a new batch without re-initialization
        """
        data_source_name: str
        for data_source_name in self._data_containers:
            self._data_containers[data_source_name].data = []
        self._count = 0
