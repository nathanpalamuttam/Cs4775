### Basset
###### Deep convolutional neural networks for DNA sequence analysis.
--------------------------------------------------------------------------------
### Visualization

<a name="motifs"/>
###### basset_motifs.py

Collect statistics and make plots to explore the first convolution layer of the given model using the given sequences.

| Argument | Type | Description |
| --- | --- | --- |
| model_file | [Model](../docs/file_specs.md#model) | Trained model |
| test_hdf5_file | [HDF5](../docs/file_specs.md#hdf5) | Test data |

| Options | Variable | Description |
| --- | --- | --- |
| -d | model_hdf5_file | Pre-computed model output as HDF5 |
| -o | out_dir | Output directory |
| -m | meme_db | MEME database used to annotate motifs |
| -s | sample | Sample sequences from the test set |
| -t | trim_filters | Trim uninformative positions off the filter ends |

--------------------------------------------------------------------------------
<a name="infl"/>
###### basset_motifs_infl.py

Collect statistics and make plots to explore the first convolution layer of the given model using the given sequences.

| Argument | Type | Description |
| --- | --- | --- |
| model_file | [Model](../docs/file_specs.md#model) | Trained model |
| test_hdf5_file | [HDF5](../docs/file_specs.md#hdf5) | Test data |

| Options | Variable | Description |
| --- | --- | --- |
| -b | batch_size | Batch size (affects memory usage) [Default: 1000] |
| -d | model_hdf5_file | Pre-computed model output as HDF5 |
| -i | informative_only | Plot informative filters only |
| -m | motifs_file | Motifs table file output by [basset_motifs.py](visualization.md#motifs) |
| -n | norm_targets | Use the norm of the target influences as the primary influence measure |
| -o | out_dir | Output directory |
| --subset | subset_file | Subset targets to those in this file |
| -s | sample | Sample sequences from the test set |
| -t | targets_file | File specifying target indexes and labels |
| --width | heat_width | Heatmaps width [Default: 10] |
| --height | heat_height | Heatmaps height [Default: 10] |
| --font | heat_font | Heatmaps font size [Default: 0.4] |

--------------------------------------------------------------------------------
<a name="sat"/>
###### basset_sat.py

Perform an in silico saturated mutagenesis of the given test sequences using the given model.

| Argument | Type | Description |
| --- | --- | --- |
| model_file | [Model](../docs/file_specs.md#model) | Trained model |
| input_file | FASTA or [HDF5](../docs/file_specs.md#hdf5) | Test data |

| Options | Variable | Description |
| --- | --- | --- |
| -a | input_activity_file | Optional activitiy table matching an input FASTA file |
| -d | model_hdf5_file | Pre-computed model output as HDF5 |
| -m | min_limit | Minimum heat map limit [Default: 0.1] |
| -n | center_nt | Center nt to mutate and plot in the heat map [Default: 200] |
| -o | out_dir | Output directory |
| -s | sample | Sample sequences from the test set |
| -t | targets | Comma-separated list of target indexes to plot (or -1 for all) |

--------------------------------------------------------------------------------
<a name="sat_vcf"/>
###### basset_sat_vcf.py

Perform an in silico saturated mutagenesis of the regions surrounding a list of SNPs given in VCF format using the given model.

| Argument | Type | Description |
| --- | --- | --- |
| model_file | [Model](../docs/file_specs.md#model) | Trained model |
| vcf_file | [VCF](https://samtools.github.io/hts-specs/VCFv4.2.pdf) | SNPs |

| Options | Variable | Description |
| --- | --- | --- |
| -d | model_hdf5_file | Pre-computed model output as HDF5 |
| -f | genome_fasta | Genome FASTA from which sequences will be drawn |
| -l | seq_len | Sequence length provided to the model |
| -m | min_limit | Minimum heat map limit [Default: 0.1] |
| -n | center_nt | Nt around the SNP to mutate and plot in the heat map [Default: 200] |
| -o | out_dir | Output directory |
| -t | targets | Comma-separated list of target indexes to plot (or -1 for all) |

--------------------------------------------------------------------------------
<a name="sad"/>
###### basset_sad.py

Compute SNP Accessibility Difference scores for SNPs in a VCF file using the given model.

| Argument | Type | Description |
| --- | --- | --- |
| model_file | [Model](../docs/file_specs.md#model) | Trained model |
| vcf_file | [VCF](https://samtools.github.io/hts-specs/VCFv4.2.pdf) | SNPs |

| Options | Variable | Description |
| --- | --- | --- |
| -d | model_hdf5_file | Pre-computed model output as HDF5 |
| -f | genome_fasta | Genome FASTA from which sequences will be drawn |
| -i | index_snp | SNPs are labeled with their index SNP in column 6 |
| -l | seq_len | Sequence length provided to the model |
| -m | min_limit | Minimum heat map limit [Default: 0.1] |
| -n | center_nt | Nt around the SNP to mutate and plot in the heat map [Default: 200] |
| -o | out_dir | Output directory |
| -s | score | SNPs are labeld with scores as column 7 |
