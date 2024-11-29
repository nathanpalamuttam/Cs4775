### Basset
###### Deep convolutional neural networks for DNA sequence analysis.
--------------------------------------------------------------------------------
## Preprocess

<a name="preprocess_features.py"/>
#### preprocess_features.py

Merge a set of feature BED files for training into a single [BED](../docs/file_specs.md#bed) and [activity table](../docs/file_specs.md#table).

| Arguments | Type | Description |
| --- | --- | --- |
| target_beds_file | table listing labels and [BED](../docs/file_specs.md#bed) | One line per sample- label then BED path |

| Options | Variable | Description |
| --- | --- | --- |
| -a | db_act_file | Existing database [activity table](../docs/file_specs.md#table) |
| -b | db_bed | Existing database [BED](../docs/file_specs.md#bed) |
| -c | chrom_lengths_file | Table of chromosome lengths |
| -m | merge_overlap | Overlap length (after extension to feature_size) above which to merge features [Default: 200] |
| -n | no_db_activity | Do not pass along the activities of the database sequences [Default: False] |
| -o | out_prefix | Output file prefix [Default: features] |
| -s | feature_size | Extend features to this size [Default: 600] |
| -y | ignore_y | Ignore Y chromsosome features [Default: False] |


--------------------------------------------------------------------------------
<a name="seq_hdf5.py"/>
#### seq_hdf5.py

Construct an HDF5 file, dividng the data into training, validation, and test subsets.

| Arguments | Type | Description |
| --- | --- | --- |
| fasta_file | FASTA | FASTA file of sequences. |
| targets_file | [Table](../docs/file_specs.md#table) | Targets activity table. |
| out_file | [HDF5](../docs/file_specs.md#hdf5) | Output HDF5 file. |

| Options | Variable | Description |
| --- | --- | --- |
| -b | batch_size | Align sizes with batch size |
| -c | counts | Validation and training percentages are given as raw counts [Default: False] |
| -r | permute | Permute sequences [Default: False] |
| -s | random_seed | numpy.random seed [Default: 1] |
| -t | test_pct | Test % [Default: 0] |
| -v | valid_pct | Validation % [Default: 0] |


--------------------------------------------------------------------------------
<a name="basset_sample.py"/>
#### basset_sample.py

Sample sequences from an existing database.

| Arguments | Type | Description |
| --- | --- | --- |
| db_bed | [BED](../docs/file_specs.md#bed) | Existing database BED. |
| db_act_file | [Table](../docs/file_specs.md#table) | Existing database activity table. |
| sample_seqs | int | Number of sequences to sample. |
| output_prefix | str | Filename prefix for output BED and activity table files. |

| Options | Variable | Description |
| --- | --- | --- |
| -s | seed | Random number generator seed [Default: 1] |
