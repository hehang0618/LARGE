import os
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import pandas as pd

def process_fasta(input_file, output_dir):
    temp_dir = os.path.join(output_dir, 'temp/')
    os.makedirs(temp_dir, exist_ok=True)
    input_file_name = os.path.basename(input_file)
    records = list(SeqIO.parse(input_file, "fasta"))
    
    def replace_nonstandard_amino_acids(sequence, replacements={'J': 'L', '*': ''}):
        return ''.join(replacements.get(aa, aa) for aa in sequence)
    
    modified_records = []
    for record in records:
        modified_sequence = replace_nonstandard_amino_acids(str(record.seq))
        # Store original description and unmodified sequence for later sorting
        modified_records.append({
            'description': record.description,
            'seq': modified_sequence,
            'length': len(modified_sequence)
        })
    
    # Sort records by sequence length (descending)
    modified_records.sort(key=lambda x: x['length'], reverse=True)
    
    # Create SeqRecord objects with sequential IDs based on sorted order
    sorted_seqrecords = []
    for idx, record_dict in enumerate(modified_records, start=1):
        seq_record = SeqRecord(
            Seq(record_dict['seq']),
            id=str(idx),
            description=record_dict['description']
        )
        sorted_seqrecords.append(seq_record)
    
    # Write sorted sequences to FASTA
    output_fasta_file_rep = os.path.join(temp_dir, f"{input_file_name}_rep.fasta")
    SeqIO.write(sorted_seqrecords, output_fasta_file_rep, "fasta")
    
    # Create CSV from sorted records
    cate_dir = os.path.join(temp_dir, 'Cate')
    os.makedirs(cate_dir, exist_ok=True)
    
    def create_csv_from_records(seq_records, output_dir, input_file_name):
        data = {'Seq ID': [], 'Seq Name': []}
        for record in seq_records:
            data['Seq ID'].append(record.id)
            data['Seq Name'].append(record.description)
        df = pd.DataFrame(data)
        cate_csv = os.path.join(output_dir, f'{input_file_name}.csv')
        df.to_csv(cate_csv, index=False)
    
    create_csv_from_records(sorted_seqrecords, cate_dir, input_file_name)