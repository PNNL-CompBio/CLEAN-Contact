# CLEAN-Contact

This repository contains the code and data for the paper "CLEAN-Contact: Contrastive Learning-enabled Enzyme Functional Annotation Prediction with Structural Inference".

## Introduction
CLEAN-Contact requires protein sequences and structures as input. The sequence inputs can be CSV or FASTA files. The 
structure inputs must be PDB files. 

Please note that if the proteins in the input CSV or FASTA files already have structures in the [Alphafold database](https://alphafold.ebi.ac.uk/), 
CLEAN-Contact will pull those PDB files for the user. In that case the user will only need 
to include their protein sequences as input. Otherwise, if the protein structures are not already in the Alphafold 
database, the user should obtain their PDB files from another method before running CLEAN-Contact.

## Installation and Setup
### Requirements
Python == 3.10.13, PyTorch == 2.1.1, torchvision == 0.16.1;
fair-esm == 2.0.0, pytorch-cuda == 12.1

### Installation
1. Clone the code and start setting up the conda environment
    ```bash
    git clone https://github.com/PNNL-CompBio/CLEAN-Contact.git
    cd CLEAN-Contact
    conda create -n clean-contact python=3.10 -y
    conda activate clean-contact
    conda install -c conda-forge biopython biotite matplotlib numpy pandas pyyaml scikit-learn scipy tensorboardx tqdm
    ```
2. Install PyTorch and torchvision with CUDA
   * Find your operating system's installation method here: https://pytorch.org/get-started/locally/
3. Install fair-esm
    ```
    python -m pip install fair-esm==2.0.0
    python build.py install
    git clone https://github.com/facebookresearch/esm.git
    ```
### Setup
1. Create required folders:

    ```
   python
   >>> from src.CLEAN.utils import ensure_dirs
   >>> ensure_dirs()
    ```
2. Download the precomputed embeddings and distance map for both training and test data from 
[here](https://drive.google.com/drive/folders/1yw0P8kjiqCUPyYAZdI-GIpSGEnSOScVZ?usp=sharing) and put both `esm_data` 
and `distance_map` folders under the `data` folder.


## Pre-inference

Before running the inference step, extract the sequence representations and structure representations for your own data 
and then merge them. 

Sequence inputs can be in a CSV format or FASTA format and should be placed in the `data` folder. CSVs must have the 
columns: "Entry", "EC number", and "Sequence", where only "EC number" should be empty. 

Structure inputs must be in PDB format. CLEAN-Contact will grab the PDBs from the Alphafold database if the structure 
is available, otherwise use your own pre-generated PDB files as input. In either case create your PDB folder, such 
as <pdb-dir>, in the top level directory of CLEAN-Contact where extract_structure_representation.py is.

### Extract sequence and structure representations
#### Data in CSV format

For example, your `<csv-file>` is `data/split100_reduced.csv`. Then run the following commands: 

```bash
python extract_structure_representation.py \
    --input data/split100_reduced.csv \
    --pdb-dir <pdb-dir> 
```

```
python
>>> from src.CLEAN.utils import csv_to_fasta, retrieve_esm2_embedding
>>> csv_to_fasta('data/split100_reduced.csv', 'data/split100_reduced.fasta') # fasta file will be 'data/split100_reduced.fasta'
>>> retrieve_esm2_embedding('split100_reduced')
```

#### Data in FASTA format

For example, your `<fasta-file>` is `data/split100_reduced.fasta`. Then run the following commands:

```
python
>>> from src.CLEAN.utils import fasta_to_csv, retrieve_esm2_embedding
>>> fasta_to_csv('split100_reduced') # output will be 'data/split100_reduced.csv'
>>> retrieve_esm2_embedding('split100_reduced')
```

```bash
python extract_structure_representation.py \
    --input data/split100_reduced.csv \
    --pdb-dir <pdb-dir> 
```

### Merge representations and compute distance map

Run the following commands to merge the sequence and structure representations:

```
python
>>> from src.CLEAN.utils import merge_sequence_structure_emb
>>> merge_sequence_structure_emb(<csv-file>)
```

If your data will be used as training data, run the following commands to compute distance map:

```
python
>>> from src.CLEAN.utils import compute_esm_distance
>>> compute_esm_distance(<csv-file>)
```

## Inference

If your dataset is in `csv` format, you can use the following command to inference the model:

```bash
python inference.py \
    --train-data split100_reduced \
    --test-data <test-data> \
    --gmm <gmm> \
    --method <method>
```

Replace `<test-data>` with your test data name, `<gmm>` with the list of fitted Gaussian Mixture Models (GMMs) and `<method>` with the `maxsep` or `pvalue`.

If you provide `<gmm>`, the model will use the fitted GMMs to compute prediction confidence. 

Run `python extract_confidence_result.py` and `python print_prediction_confidence_results.py` to extract and print the prediction confidence results, respectively, to reproduce results in Fig. S4-6.

We provide the fitted GMMs based on `maxsep` at `gmm_test/gmm_lst.pkl`. 

If your dataset is in `fasta` format, you can use the following command to inference the model:

```bash
python inference_fasta.py \
    --train-data split100_reduced \
    --fasta <fasta-file> \
    --gmm <gmm> \
    --method <method>
```

Performance metrics measured by Precision, Recall, F-1, and AUROC will be printed out. Per sample predictions will be saved in `results` folder.

## Training

Sequences whose EC number has only one sequence are required to mutated to generate positive samples. We provide the mutated sequences in `data/split100_reduced_single_seq_ECs.csv`. To get your own mutated sequences, run the following command:

```
python
>>> from src.CLEAN.utils import mutate_single_seq_ECs
>>> mutate_single_seq_ECs('split100_reduced')
```

```bash
python mutate_conmap_for_single_EC.py \
    --fasta data/split100_reduced_single_seq_ECs.fasta 
```

```
python
>>> from src.CLEAN.utils import fasta_to_csv, merge_sequence_structure_emb
>>> fasta_to_csv('split100_reduced_single_seq_ECs')
>>> merge_sequence_structure_emb('split100_reduced_single_seq_ECs')
```

To train the model mentioned in the main text (`addition` model), modify arguments in `train-split100-reduced-resnet50-esm2-2560-addition-triplet.sh` and run the following command:

```bash
./train-split100-reduced-resnet50-esm2-2560-addition-triplet.sh
```

To train models with the other combinations (`contact_1` and `contact_2`), modify arguments in `train-split100-reduced-resnet50-esm2-2560-contact_1-triplet.sh` and `train-split100-reduced-resnet50-esm2-2560-contact_2-triplet.sh`, respectively, and run the following command:

```bash
./train-split100-reduced-resnet50-esm2-2560-contact_1-triplet.sh
./train-split100-reduced-resnet50-esm2-2560-contact_2-triplet.sh
```
