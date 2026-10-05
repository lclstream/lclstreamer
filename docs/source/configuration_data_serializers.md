# Data Serializers



## HDF5BinarySerializer

This Data Serializer class turns the data into a binary blob with the internal structure
of an HDF5 file.

### *Configuration Parameters for HDF5BinarySerializer*

* `compression` (str): This parameter is optional. If present, the HDF5 Data Serializer
  compresses the data with the specified algorithm during serialization. Possible values
  are: `gzip`, `gzip_with_shuffle` (byteshuffle is performed before gzip compression is
  applied), `bitshuffle_with_lz4`, `bitshuffle_with_zstd` (bitshuffle is performed
  before LZ4 or Zstd compression is applied), and `zfp`. If this parameter is not
  specified, or is set to `null`, no compression is applied during serialization. The
  default value of this parameter is `null`. Example: `gzip`

* `compression_level` (int): This parameter is optional. If the compression algorithm
  specified by the `compression` configuration parameter supports compression levels,
  this parameter specifies the compression level applied during serialization. The
  default value of this parameter is `3`. Example: `5`

* `fields` (dict of str): This entry is a dictionary that specifies where each data
  source will be stored in the internal structure of the HDF5 file. Each key in the
  dictionary is the name of a data source, and each value is the internal HDF5 path
  where the data source is stored. If a data source is not present in the dictionary,
  it is excluded from the serialization process and does not appear in the binary blob
  containing the serialized data. Example:

  ```yaml
  fields:
    timestamp: /data/timestamp
    detector_data: /data/data
  ```


## SimplonBinarySerializer

This Data Serializer class turns the data accumulated by LCLStreamer into a binary blob
with the internal structure of a Simplon message. It follows version 1.8 of the Simplon
specification published by Dectris.

Each call to the serializer produces a Simplon image message containing the detector
frame for the latest event in the batch. When the last LCLStream worker processes the
first batch, it additionally emits a Simplon start message with run and detector
metadata. At the end of the stream, the last worker emits a Simplon end
message. Each message is a CBOR-encoded dictionary whose `type` entry is `start`,
`image` or `end`, and which identifies the run with `run_id`.

The detector frame and spectrum are sent uncompressed.

* The following data sources must be present in the `data_sources` section of the
  configuration file when using this serializer: `timestamp`, `detector_data`,
  `detector_geometry`, and `run_info`.

* The serializer reads the detector ID and geometry from the `jungfrau._detid` and
  `jungfrau.raw._det_geotxt_default` data keys, and the beam data from
  `ebeamh.raw.ebeamUndAngX`, `ebeamh.raw.ebeamUndAngY`, `ebeamh.raw.ebeamUndPosX`,
  `ebeamh.raw.ebeamUndPosY` and `ebeamh.raw.ebeamPhotonEnergy`. Use `->` aliases in
  `psana_fields` to give the fields these names (see
  `examples/lclstreamer-psana2-simplon.yml`). The beam data is optional.

### *Configuration Parameters for SimplonBinarySerializer*

* `data_source_to_serialize` (str): The name of the data source whose array is 
  embedded in each Simplon image message. This name must correspond
  to a key defined in the `data_sources` section of the configuration file.
  Example: `detector_data`

* `polarization_fraction` (float): The fraction of linear polarization of the X-ray
  beam, as a value between 0 and 1. This value is included in the Simplon start
  message. Example: `0.99`

* `polarization_axis` (list of float): A three-element list representing the
  polarization axis direction vector. This value is included in the Simplon start
  message. Example: `[0.0, 1.0, 0.0]`

* `data_collection_rate` (str): A human-readable string describing the nominal data
  collection rate of the detector. This value is included in the Simplon start message.
  Example: `120 Hz`

* `detector_name` (str): A human-readable name identifying the main detector that
  generates the data encoded in the Simplon image messages. This value is included
  in the Simplon start message. Example: `Jungfrau 1M`

* `detector_type` (str): A string identifying the model or type of the main detector
  that generates the data encoded in the Simplon image messages. This value is
  included in the Simplon start message. Example: `Jungfrau 1M`

* `photon_wavelength_source` (str): This parameter is optional. The data key of the
  photon wavelength PV, reported in nm. Its value is sent in Angstrom as
  `photon_wavelength` in each image message, or 0 when the reading is missing. When the
  parameter is not set, `photon_wavelength` is 0 and is sent only with the beam data.
  Example: `photon_wavelength`

* `spectrometer_source` (str): This parameter is optional. The data key of a spectrometer
  array, sent in each image message as `spectrometer_data`, with `spectrometer_dtype`
  and `spectrometer_shape`.
  It is left out of an event whose reading is missing. Example: `spectrometer`
