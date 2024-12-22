# CS 4775 Project

## Group Members
- Ayaan
- Gavin
- Kevin
- Nathan

---

## Acquiring Dataset

1. Visit the [ChIP-Atlas website](https://chip-atlas.org/).
2. Navigate to the **Peak Browser**.
3. Choose the following parameters:
   - **Chip**: TFs and Others
   - **Cell Type**: Neural
   - **Threshold**: 100
4. Download the resulting **BED file**.
5. Obtain the **hg38 FASTA file** as a reference genome.
6. Extract the BED file into a sequence file using the following command:
   ```bash
   bedtools getfasta -fi hg38.fasta -bed regions.bed -s -fo extracted_sequences_strand.fasta
7. To convert the dataset into the required file types:
   - Download 7Zip
   - Extract .gz
   - Extract .tar

## Running TBiNet

1. Update the `testmat` variable in the script to point to the location of your extracted FASTA file:
   ```python
   testmat = "path/to/extracted_sequences_strand.fasta"
2. Run the script to test the model.
3. The AUROC scores will be generated automatically as output.

## Running DeepBind
1. Set up your python virtual environment by installing cPicke, deepbind_util, tensorflow.
2. Convert the untared file from step 7 above back into the .gz file.
3. Using the scripts provided in the DeepBind folder, change the data directory path to be the data of the sequence.gz file you converted above.
4. Run the training process by calling python deepbind_train.py top. This would run training on the top 500 examples. You can also pass all as the argument for the first 1000.
5. Testing can also be done by repeating the steps above for the test file. 
