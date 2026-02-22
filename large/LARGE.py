#!/usr/bin/env python3
import os
import subprocess
import sys
import argparse
import shutil
import pyrodigal
import Bio.SeqIO
import importlib.util
import large

from large.readfasta import process_fasta
from large.plm_embed_esm1b_LARGE import run_esm_embedding
from large.batch_data_LARGE import batch_data
from large.glm_embed import glm_embed
from large.LARGE_predict import *


def parse_arguments():
    spec = importlib.util.find_spec('large')
    script_dir = list(large.__path__)[0]
    default_output = "./output/"
    default_model_dir = script_dir + "/model/"
    
    parser = argparse.ArgumentParser(description="Process sequences and run LARGE prediction.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-p", "--protein", metavar="filename", help="Input protein sequence file in FASTA format")
    group.add_argument("-n", "--nucleotide", metavar="filename", help="Input nucleotide sequence file in FASTA format (will be translated to protein)")
    parser.add_argument("-c", "--cpu", type=int, default=10, help="Number of cpus for sequence loading (default: %(default)s)")
    parser.add_argument("-t", "--topn", type=int, default=1, help="Extract the top n prediction results (default: %(default)s)")
    parser.add_argument("-g", "--gpu", type=int, default=1, help="Which gpu for LARGE (default: %(default)s)")
    parser.add_argument("-l", "--length", type=int, default=5000, help="Cut sequence longer than how many amino acids. (default: %(default)s)")
    parser.add_argument("-o", "--output_dir", default=default_output, help="Output dir (default: %(default)s)")
    parser.add_argument("-m", "--model_dir", default=default_model_dir, help="Downloaded models dir. If you don't set the download path manually, Don't set it.(default: %(default)s)")
    parser.add_argument("--probability", default=0.5, help="Set the cutoff probability(default: %(default)s)")
    parser.add_argument("--clean", action="store_true", help="If delete the intermediate file")
    parser.add_argument("--cate", help="Training model")
    parser.add_argument("--model", default="fast" ,help="Fast or slow model")    
    args = parser.parse_args()
    return args
    
def mark_step_finished(step_num, output_dir):
    finish_file = os.path.join(output_dir, f"finish_step{step_num}")
    open(finish_file, "w").close()

def step_finished(step_num, output_dir):
    finish_file = os.path.join(output_dir, f"finish_step{step_num}")
    return os.path.exists(finish_file)

def create_directory_if_not_exists(path):
    os.makedirs(path, exist_ok=True)

def translate_nucleotide_to_protein(input_file, outfaa):
    records = list(Bio.SeqIO.parse(input_file, "fasta"))
    orf_finder = pyrodigal.GeneFinder(meta=True)
    with open(outfaa, 'w') as out_handle:
        for record in records:
            for i, pred in enumerate(orf_finder.find_genes(bytes(record.seq))):
                out_handle.write(f">{record.id}_{i+1}\n")
                out_handle.write(f"{pred.translate()}\n")
    return outfaa

def main():
    args = parse_arguments()
    create_directory_if_not_exists(args.output_dir)
    
    tempdir = os.path.join(args.output_dir, "temp/")
    os.makedirs(tempdir, exist_ok=True)
    # Step 0: Process fasta file
    if args.protein:
        input_file = args.protein
        input_file_name = os.path.basename(input_file)
    else:
        input_file = args.nucleotide
        outfaa = os.path.join(tempdir, os.path.basename(input_file) + ".protein.faa")
        input_file_name = os.path.basename(outfaa)
    if not step_finished(0, args.output_dir):
        print("--------------------------------------------------")
        print("Step 0: Processing file",input_file)
        if args.nucleotide:
            print("Translating nucleotide sequences to protein sequences...")
            try:
                input_file = translate_nucleotide_to_protein(input_file, outfaa)
            except Exception as e:
                print(f"Error occurred while translating nucleotides: {e}")
                sys.exit(1)

        try:
            process_fasta(input_file, args.output_dir)
            mark_step_finished(0, args.output_dir)
        except subprocess.CalledProcessError as e:
            print(f"Error occurred in Step 0: {e}")
            sys.exit(1)
    else:
        print("Step 0 has been done.")
    
    # Step 1: Embed sequences using ESM-1b model and process sequence metadata
    rep_file = os.path.join(tempdir, input_file_name + "_rep.fasta")
    pklfile = os.path.join(tempdir, f"LARGE_seq_cate_fasta_{input_file_name}.esm.embs.pkl")
    fastatsvfile = os.path.join(tempdir, f"LARGE_seq_cate_fasta_{input_file_name}.tsv")
    batch_dir = os.path.join(tempdir, f"batched_data_{input_file_name}")
    os.makedirs(batch_dir, exist_ok=True)
    if not step_finished(1, args.output_dir):
        print("--------------------------------------------------")
        print("Step 1: Embedding sequences...")
        try:
            regression_path=os.path.join(args.model_dir, "esm2_t33_650M_UR50D-contact-regression.pt")
            model_path=os.path.join(args.model_dir, "esm2_t33_650M_UR50D.pt")
            sequence_representations = run_esm_embedding(
                fasta_file=rep_file, output_path=pklfile, num_workers=args.cpu, num_gpus=args.gpu,
                regression_path=regression_path, model_path=model_path, max_length=5000,model=args.model)
            batch_data(sequence_representations,  batch_dir)
            mark_step_finished(1, args.output_dir)
        except subprocess.CalledProcessError as e:
            print(f"Error occurred in Step 1: {e}")
            sys.exit(1)
    else:
        print("Step 1 has been done.")
    
    # Step 2: Using the GLM model
    if not step_finished(2, args.output_dir):
        print("--------------------------------------------------")
        print("Step 2: Run GLM-META...")
        try:
            model = os.path.join(args.model_dir, "pytorch_model.bin")
            glm_embed(model_path=model, data_path=batch_dir, output_path=tempdir, num_gpus=args.gpu,num_workers=args.cpu)
            mark_step_finished(2, args.output_dir)
        except subprocess.CalledProcessError as e:
            print(f"Error occurred in Step 2: {e}")
            sys.exit(1)
    else:
        print("Step 2 has been done.")
    
    # Step 3: Run prediction using LARGE model
    print("--------------------------------------------------")
    print("Step 3: Generating final results...")
    try:
        predict(results_folder=tempdir, file_name=input_file_name, num_gpus=args.gpu, 
        model_path=args.model_dir, topn=args.topn,probability=args.probability)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred in Step 3: {e}")
    if args.clean:
        shutil.rmtree(tempdir)

if __name__ == "__main__":
    main()
