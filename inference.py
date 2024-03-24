from pathlib import Path
from src.CLEAN.infer import infer_maxsep, infer_pvalue

def main(args):

    print(args)

    if 'maxsep' in args.method:
        infer_maxsep(args.train_data, args.test_data, report_metrics=True, pretrained=False, model_name=args.model_name, gmm=args.gmm)
    if 'pvalue' in args.method:
        infer_pvalue(args.train_data, args.test_data, report_metrics=True, pretrained=False, model_name=args.model_name, gmm=args.gmm)
    elif 'maxsep' not in args.method:
        raise ValueError(f'Invalid method: {args.method}')

def parse_args():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data', type=str, default='split100_reduced', help='Training data name')
    parser.add_argument('--test-data', type=str, default='new', help='Test data name')
    parser.add_argument('--model-name', type=str, default='split100_reduced_resnet50_esm2_2560_addition_triplet', help='Trained model file name')
    parser.add_argument('--gmm', type=Path, default=None, help='File name for list of GMM models')
    parser.add_argument('--method', nargs='+', default='maxsep', help='Inference method')

    return parser.parse_args()
