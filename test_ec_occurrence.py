from src.CLEAN.infer import infer_pvalue, infer_maxsep

if __name__ == '__main__':
    train = 'split100_reduced'
    tests = ['merge_{}']
    occurrences = ['0_5', '5_10', '10_50', '50_100', '100']
    model_names = ['split100_reduced_resnet50_esm2_2560_addition_o256_triplet_7000', 'split100_reduced_resnet50_esm2_2560_addition_o256_triplet_best']

    for test in tests:
        for occur in occurrences:
            for model_name in model_names:
                test_ = test.format(occur)
                print(test_, model_name)
                infer_pvalue(train, test_, report_metrics=True, pretrained=False, model_name=model_name)
                print('\n')
                infer_maxsep(train, test_, report_metrics=True, pretrained=False, model_name=model_name)
                print('*' * 40)
