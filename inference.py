from pathlib import Path
from src.CLEAN.infer import infer_maxsep, infer_pvalue

def main(args):

    print(args)

    if args.method == 'maxsep':
        infer_maxsep(args.train_data, args.test_data, report_metrics=True, pretrained=False, model_name=args.model_name, gmm=args.gmm)
    elif args.method == 'pvalue':
        infer_pvalue(args.train_data, args.test_data, report_metrics=True, pretrained=False, model_name=args.model_name, gmm=args.gmm)
    else:
        raise ValueError(f'Invalid method: {args.method}')

def parse_args():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data', type=str, default='split100_reduced', help='Training data name')
    parser.add_argument('--test-data', type=str, default='new', help='Test data name')
    parser.add_argument('--model-name', type=str, default='split100_reduced_resnet50_esm2_2560_addition_triplet', help='Trained model file name')
    parser.add_argument('--gmm', type=Path, default=None, help='File name for list of GMM models')
    parser.add_argument('--method', type=str, default='maxsep', help='Inference method')

    return parser.parse_args()