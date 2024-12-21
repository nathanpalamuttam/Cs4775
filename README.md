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
