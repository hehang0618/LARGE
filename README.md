# **LARGE**

A large language model based tool for antibiotic resistance gene identify and estimates.

## **Getting Started**

### **Prerequisites**

- Python 3.10.13
- fair-esm=1.0.2
- biopython=1.82
- pytorch=2.1.2
- transformers=4.22.2
- scikit-learn=1.1.2
- h5py=3.7.0
- prodigal
- huggingface_hub
- bio=1.6.0
- fairscale=0.4.10
- All required packages listed in **`requirements.txt`** and **`environment.yml`**

### **Installation**

LARGE is under Python 3.10*, therefore, it is recommended to run it via virtual environment. 
You must have the **`GPUs`** to install and use this software. We recommend that you use a gpu with at least 10GB of memory

1. Clone the repository:
    
    ```bash
    git clone https://github.com/hehang0618/LARGE
    ```
    
2. Create a conda envirment by using the environment.yml

    ```bash
    cd LARGE
    conda env create -f environment.yml
    ```
   Or you can use mamba to create the environment(recommended)
    ```bash
    cd LARGE
    mamba env create -f environment.yml
    ```

3. activate the environment

    ```bash
    conda activate large
    ```

4. install some other packages by pip

    ```bash
    pip install .
    ```

5. Model Download

   Our models are uploaded to Hugging Face at [https://huggingface.co/westlakehang/large](https://huggingface.co/westlakehang/large). You can either:
   - Download all files from the website and place them in the **`model`** folder in the conda envs folder(etc. ~/miniconda3/envs/large/lib/python3.10/site-packages/large_v1/model/).
   - Run the **`modeldownload.py`** script to automatically download the models.

## **Usage**

### Basic Usage

Run the main script with optional arguments:

```bash
LARGE -p [protein_file_name] -o [output_folder_name] 
```

### Processing MetaBAT Assembled Genomes

If you have metagenome-assembled genomes (MAGs) from MetaBAT, you can use the following script to process multiple bins:

```bash
#!/bin/bash

# Initialize conda environment
source /data/hehang/miniconda3/etc/profile.d/conda.sh
conda activate large

# Set base directory
BASE_DIR="~/metabat"

# Process bins from a list of IDs
INPUT_FILE="${BASE_DIR}/id_list1-100.txt"

while read -r id; do
    mkdir -p "${BASE_DIR}/1.pipline/output/${id}"
    
    # Process each bin file
    for bin_file in "${BASE_DIR}/0.data/${id}/bins/bin."*.fa; do
        bin_filename=$(basename "$bin_file")
        bin_number=$(echo "$bin_filename" | grep -oP 'bin\.\K\d+')
        output_dir="${BASE_DIR}/output/${id}/bin.${bin_number}"
        
        # Skip if already processed
        if [ -f "${output_dir}/finish_step2" ]; then
            continue
        fi
        
        # Create output directory and run LARGE
        mkdir -p "$output_dir"
        LARGE -n "$bin_file" -o "$output_dir" -g 0 --clean
        
        # Check execution status
        if [ $? -eq 0 ]; then
            echo "LARGE success:$id $bin_filename"
        else
            echo "error: LARGE fail:$id  $bin_filename"
        fi
    done
    echo "----------------------------"
done < "$INPUT_FILE"
```

This script will:
1. Process multiple bins from MetaBAT output
2. Create separate output directories for each bin
3. Skip already processed bins
4. Clean up intermediate files automatically
5. Provide success/failure status for each bin

### Input Options
- `-p`, `--protein`: Specify a protein sequence input file, Must be a fasta file.
- `-n`, `--nucleotide`: Specify a nucleotide sequence input file, Must be a fasta file.

### Performance Configuration
- `-w`, `--workers`: Number of CPUs for sequence loading (default: 10)
- `-g`, `--gpus`: Number of GPUs to use (default: 1)
- `-t`, `--topn`: Extract top N prediction results (default: 1)

### Output and Model Management
- `-o`, `--output_dir`: Output directory (default: `./output/`)
- `-m`, `--model_dir`: Path to downloaded models directory
- `--clean`: Delete intermediate files after processing
- `--training`: Enable model training mode

## **About**

If you use LARGE in published research, please cite:

[Not now, maybe soon? I don't know.]

## **License**

LARGE is under the MIT licence. However, please take a look at te comercial restrictions of the databases used during the mining process (CARD, ARDB, and UniProt). 

## **Contact**

If need any asistance please contact: [hehang@westlake.edu.cn](hehang@westlake.edu.cn)
