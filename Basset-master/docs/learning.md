### Basset
###### Deep convolutional neural networks for DNA sequence analysis.
--------------------------------------------------------------------------------
### Learning

<a name="train"/>
###### basset_train.lua

Train a convolutional neural network on the given data.

| Argument | Type | Description |
| --- | --- | --- |
| data_file | [HDF5](../docs/file_specs.md#hdf5) | Input training and validation data |

| Option | Description |
| --- | --- |
| -cuda | Run on GPGPU |
| -job | Table of job hyper-parameters |
| -max_epochs | Maximum training epochs to perform |
| -restart | Restart an interrupted training run using the given model file |
| -result | Write the loss value to this file (useful for Bayes Opt) |
| -save | Prefix for saved models [Default: dnacnn] |
| -seed | Seed the model with the parameters of another in the given model file |
| -rand | Random number generator seed |
| -stagnant_t | Allowed epochs with stagnant validation loss [Default: 10] |


--------------------------------------------------------------------------------
<a name="test"/>
###### basset_test.lua

Report model performance on the given test data, producing files with AUC and points along the ROC curves for each sample.

| Argument | Type | Description |
| --- | --- | --- |
| model_file | [Model](../docs/file_specs.md#model) | Saved model to use |
| data_file | [HDF5](../docs/file_specs.md#hdf5) | Input training and validation data |
| out_dir | | Output directory |

| Option | Description |
| --- | --- |
| -cuda | Run on GPGPU |

--------------------------------------------------------------------------------
<a name="predict"/>
###### basset_predict.lua

Predict activity for a new set of sequences.

| Arguments | Type | Description |
| --- | --- | --- |
| model_file | [Model](../docs/file_specs.md#model) | Saved model to use |
| data_file | [HDF5](../docs/file_specs.md#hdf5) | Input training and validation data |
| out_file | | Output file |

| Option | Description |
| --- | --- |
| -cuda | Run on GPU [Default: False] |
| -norm | Normalize all targets to a 0.05 frequency |