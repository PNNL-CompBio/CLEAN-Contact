import argparse
import os
from pathlib import Path
from src.CLEAN.utils import *
from src.CLEAN.infer import infer_maxsep, infer_pvalue

def eval_parse():
    # only argument passed is the fasta file name to infer
    # located in ./data/[args.fasta_data].fasta
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data', type=str, default='split100', help='Training data name')
    parser.add_argument('--fasta', type=Path, required=True, help='Fasta file to infer')
    parser.add_argument('--model-name', type=str, default='split100_reduced_resnet50_esm2_2560_addition_triplet', help='Trained model file name')
    parser.add_argument('--gmm', type=Path, default=None, help='File name for list of GMM models')
    parser.add_argument('--method', type=str, default='maxsep', help='Inference method')
    args = parser.parse_args()
    return args


def main():
    args = eval_parse()
    train_data = 'split100'
    test_data = args.fasta_data 
    # converting fasta to dummy csv file, will delete after inference
    # esm embedding are taken care of
    prepare_infer_fasta(test_data) 
    # inferred results is in
    # results/[args.fasta_data].csv
    infer_maxsep(train_data, test_data, report_metrics=False, pretrained=True, gmm = './data/pretrained/gmm_ensumble.pkl')
    if args.method == 'maxsep':
        infer_maxsep(train_data, test_data, report_metrics=True, pretrained=False, model_name=args.model_name, gmm=args.gmm)
    elif args.method == 'pvalue':
        infer_pvalue(train_data, test_data, report_metrics=True, pretrained=False, model_name=args.model_name, gmm=args.gmm)
    else:
        raise ValueError(f'Invalid method: {args.method}')
    # removing dummy csv file
    os.remove("data/"+ test_data +'.csv')
    

if __name__ == '__main__':
    main()
