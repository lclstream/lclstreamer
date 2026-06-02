# Data Sources



## Psana1 Data Sources

The following Data Source classes are compatible with the `Psana1EventSource` event
source.



### Psana1Timestamp

This Data Source class retrieves timestamp information from a psana1 data event.

* The timestamp information is returned as a numpy array of type `uint64`. where seconds are stored in the upper 32 bits and nanoseconds in the lower 32 bits.

#### *Configuration Parameters for Psana1Timestamp*

This Data Source does not require any configuration parameters beyond the mandatory
`type` entry.



### Psana1DetectorInterface

This Data Source class provides a generic interface for retrieving data from any
detector or EPICS process variable stored in a psana1 data event. It accesses the
detector or variable via the psana1 `Detector` interface and supports using attribute
names and callable methods to retrieve the desired data.

* The retrieved data is returned as a numpy array. The default type is `float64` unless
  overridden via the `dtype` configuration parameter.

* If the `psana_name` contains a colon (`:`) and `psana_fields` is not specified, the
  data source is treated as an EPICS process variable (PV) and its current value is
  retrieved directly.

* If the `psana_name` refers to a detector whose `psana_fields` value is
  `eventCodes`, the returned array is always a 1D integer array of size 256. The EVR
  codes associated with the event are stored at the beginning of the array; any
  remaining space is filled with the integer value `0`.

#### *Configuration Parameters for Psana1DetectorInterface*

* `psana_name` (str): The name or alias of the detector or EPICS variable in the psana1
  framework. Example: `Jungfrau1M`

* `psana_fields` (str or list of str): This parameter is optional when `psana_name`
  contains a colon (PV access). Otherwise it is required. It specifies the dot-separated
  chain of psana1 detector attributes and/or callable methods used to retrieve the data.
  If a list is provided, each element is retrieved independently and the results are
  stacked into a single array. Example: `calib` or `["calib", "raw"]`

* `dtype` (str): This parameter is optional. It specifies the numpy dtype of the
  returned array. The default value is `float64`. Example: `int32`




## Psana2 Data Sources

The following Data Source classes are compatible with the `Psana2EventSource` event
source.



### Psana2Timestamp

This Data Source class retrieves timestamp information from a psana2 data event.

* The timestamp information is returned as a 1D numpy array of type `float64`.

#### *Configuration Parameters for Psana2Timestamp*

This Data Source does not require any configuration parameters beyond the mandatory
`type` entry.



### Psana2DetectorInterface

This Data Source class provides a generic interface for retrieving data from any
detector or EPICS process variable stored in a psana2 data event. It accesses the
detector or variable via the psana2 `run.Detector` interface and supports using
attribute names and callable methods to retrieve the desired data.

* The retrieved data is returned as a numpy array. The default type is `float64` unless
  overridden via the `dtype` configuration parameter.

* If the `psana_name` contains a colon (`:`) and `psana_fields` is not specified, the
  data source is treated as a process variable and its current value is retrieved
  directly.

#### *Configuration Parameters for Psana2DetectorInterface*

* `psana_name` (str): The name or alias of the detector or process variable in the
  psana2 framework. Example: `epicsinfo`

* `psana_fields` (str or list of str): This parameter is optional when `psana_name`
  contains a colon (PV access). Otherwise it is required. It specifies the dot-separated
  chain of psana2 detector attributes and/or callable methods used to retrieve the data.
  If a list is provided, each element is retrieved independently and the results are
  stacked into a single array. Example: `raw.calib`

* `dtype` (str): This parameter is optional. It specifies the numpy dtype of the
  returned array. The default value is `float64`. Example: `float32`




### Psana2RunInfo

This Data Source class retrieves run-level metadata from a psana2 data run. It returns
information that is constant across all events in a run,.

* The metadata is returned as a 1D numpy array of type `str` containing four elements,
  in order: the experiment name, the run start timestamp, the run number, and the
  source identifier string.

#### *Configuration Parameters for Psana2RunInfo*

This Data Source does not require any configuration parameters beyond the mandatory
`type` entry.



## Generic Data Sources

The following Data Source classes are not tied to any specific data-acquisition
framework and can be used with any compatible Event Source.



### GenericRandomNumpyArray

This Data Source class generates a random numerical array of a size and type chosen by
the user. It is primarily intended for testing and development.

* The generated array has a shape and dtype defined by the configuration parameters.
  Integer arrays are filled with random integers uniformly sampled from `[0, 255]`.
  Floating-point arrays are filled with random values uniformly sampled from `[0, 1)`.

#### *Configuration Parameters for GenericRandomNumpyArray*

* `array_shape` (str): The shape of the generated array, expressed as a
  comma-separated list of integers following the same convention as numpy's `shape`
  attribute. Example: `512, 512`, or `10,` for a column vector, `1,` for a single value.

* `array_dtype` (str): The numerical type of the generated array, in the same format
  as numpy dtype strings. Only integer and floating-point types are supported.
  Example: `float32`
  
* `always_random` (bool): If set to `True`, it will re-generate the array at every event.
  Otherwise it will generate one array and re-use it for every event (e.g. saves CPU time
  for network throughput testing).


### ConstValue

This Data Source class returns a fixed floating-point or integer scalar value defined in the
configuration file.

#### *Configuration Parameters for FloatValue*

* `value` (float): The scalar floating-point value to be returned by this data source
  on every event. Example: `3.14` or `[6, 3, 1]`.

* `dtype` (str): The numerical type of the generated constant. In the same format
  as numpy dtype strings. Only integer and floating-point types are supported.
  Example: `float32` or `uint16`.


---


### SourceIdentifier

This Data Source class returns the `source_identifier` string that was provided in the
top-level LCLStreamer configuration. It is useful for embedding provenance information
into the serialized data.

* The source identifier is returned as a 0-dimensional numpy array of type `str`.

#### *Configuration Parameters for SourceIdentifier*

This Data Source does not require any configuration parameters beyond the mandatory
`type` entry.
