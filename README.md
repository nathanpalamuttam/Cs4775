CS 4775 Project
Group Members: Ayaan, Gavin, Kevin, Nathan

Acquiring Dataset:
- Go to https://chip-atlas.org/
- Click on Peak Browser
- Choose the following parameters:
1. Chip: TFs and Others
2. Neural
3. 100
- Download the BED file
- Acquire the hg38 fasta file as a reference
- Extract the BED file into a sequence file using this command: bedtools getfasta -fi hg38.fasta -bed regions.bed -s -fo extracted_sequences_strand.fasta

TBiNet
- change the testmat variable into the file where your fasta file is located
- test the model, and the AUROC scores will automatically generate

DeepBind
//TODO
  

 
