from dataclasses import dataclass, field
from typing import Any

import numpy
from numpy.typing import DTypeLike

from ...utils.logging import log_error, log_error_and_exit
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
    # data: dict[str, StrFloatIntNDArray | dict [str, StrFloatIntNDArray | None] | None] = field(default_factory=dict)
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
        # Maps each input source name to the flattened sub-keys it produced on the
        # first event, so when a whole source is missing (None) on a later event we
        # know which sub-containers to back-fill instead of dropping the event.
        self._source_subkeys: dict[str, list[str]] = {}
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
            for data_source_name in data:
                data_value: Any | None = data[data_source_name]
                if data_value is None:
                    # A source missing on the very first event cannot be sized yet, but
                    # the rank must not die over it. Skip it now and initialize its
                    # container lazily on the first later event that carries it; until
                    # then its absence is simply "missing this event" downstream.
                    log_error(
                        f"Data entry {data_source_name} was None on the first event; "
                        "deferring initialization until a later event provides it."
                    )
                    continue
                elif isinstance(data_value, dict):
                    data_container: dict[str, DataContainer]
                    for sub_data_name, sub_data in data_value.items():
                        if sub_data is None:
                            # A sub-field missing on the first event cannot be sized
                            # yet; defer it and initialize its container on a later
                            # event that carries it.
                            continue
                        subdata_container = DataContainer(
                            data=[sub_data],
                            dtype=sub_data.dtype,
                            shape=sub_data.shape,
                        )
                        self._data_containers[sub_data_name] = subdata_container
                    self._source_subkeys[data_source_name] = [
                        sub_key
                        for sub_key, sub_data in data_value.items()
                        if sub_data is not None
                    ]
                else:
                    data_container = DataContainer(
                        data=[data_value],
                        dtype=data_value.dtype,
                        shape=data_value.shape,
                    )
                    self._data_containers[data_source_name] = data_container
                    self._source_subkeys[data_source_name] = [data_source_name]
        else:
            for data_name, subdata in data.items():
                if isinstance(subdata, dict):
                    dataitems = list(subdata.items())
                elif subdata is None:
                    # The whole source is missing this event. Back-fill every
                    # sub-container it owns (recorded on the first event) so the event
                    # is still emitted -- a missing optional source never drops a frame.
                    dataitems = [
                        (sub_key, None)
                        for sub_key in self._source_subkeys.get(data_name, [data_name])
                    ]
                else:
                    dataitems = [(data_name, subdata)]
                for data_source_name, data_value in dataitems:
                    if data_source_name not in self._data_containers:
                        if data_value is None:
                            # Still missing and never initialized; nothing to size a
                            # container from yet. Skip it for this event.
                            continue
                        # First appearance of a label that was deferred on earlier
                        # events (missing on the first event). Initialize its container
                        # now and back-fill the events already accumulated in this batch
                        # with null placeholders so the stacked arrays stay aligned.
                        new_container = DataContainer(
                            data=[],
                            dtype=data_value.dtype,
                            shape=data_value.shape,
                        )
                        for _ in range(self._count):
                            new_container.data.append(self._null_value(new_container))
                        self._data_containers[data_source_name] = new_container
                        subkeys = self._source_subkeys.setdefault(data_name, [])
                        if data_source_name not in subkeys:
                            subkeys.append(data_source_name)
                    data_container = self._data_containers[data_source_name]

                    if data_value is None:
                        if data_container.shape is not None:
                            if numpy.issubdtype(
                                data_container.dtype, numpy.signedinteger
                            ):
                                data_container.data.append(
                                    numpy.full(
                                        data_container.shape,
                                        -999,
                                        dtype=data_container.dtype,
                                    )
                                )
                                continue
                            elif numpy.issubdtype(data_container.dtype, numpy.floating):
                                data_container.data.append(
                                    numpy.full(
                                        data_container.shape,
                                        numpy.float64("nan"),
                                        dtype=data_container.dtype,
                                    )
                                )
                                continue
                            else:
                                data_container.data.append(
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

    def _null_value(self, container: DataContainer) -> StrFloatIntNDArray:
        """Null placeholder matching a container's dtype and shape: NaN for floating
        data, -999 for signed integers, and the string "None" for everything else."""
        if numpy.issubdtype(container.dtype, numpy.signedinteger):
            return numpy.full(container.shape, -999, dtype=container.dtype)
        elif numpy.issubdtype(container.dtype, numpy.floating):
            return numpy.full(
                container.shape, numpy.float64("nan"), dtype=container.dtype
            )
        return numpy.full(container.shape, "None", dtype=numpy.str_)

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
