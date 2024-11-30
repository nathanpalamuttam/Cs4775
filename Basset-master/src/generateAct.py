def generate_act_file(fasta_file, output_txt, default_value=1):
    with open(fasta_file, 'r') as fasta, open(output_txt, 'w') as act_file:
        act_file.write("SampleName\n")  # Write header
        for line in fasta:
            if line.startswith(">"):
                header = line[1:].strip()  # Remove ">"
                act_file.write(f"{header}\t{default_value}\n")  # Assign default target

# Usage
generate_act_file("learn_cd4.fa", "learn_cd4_act.txt")
print("DONE")
