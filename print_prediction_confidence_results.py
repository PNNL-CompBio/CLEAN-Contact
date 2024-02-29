from src.CLEAN.evaluate import get_pred_labels, get_pred_probs, get_true_labels, get_eval_metrics
import os
import numpy as np

# true_label, all_label = get_true_labels('data/merge_new_price')

for conf in np.linspace(0.1, 1, 10):
# for conf in [0.9, 1.0]:
    print(f'{conf:.1}')
    fname = 'results/merge_new_price_maxsep_confidence_0_{:.1f}'.format(conf)
    true_label, all_label = get_true_labels('data/merge_new_price_maxsep_confidence_0_{:.1f}'.format(conf))
    pred_label = get_pred_labels(fname, '')
    pred_probs = get_pred_probs(fname, '')
    pre, rec, f1, roc, acc, _, _, _ = get_eval_metrics(
            pred_label, pred_probs, true_label, all_label)
    roc = float(roc)
    print('#' * 80)
    print('>>> confidence: {:.1f}'.format(conf))
    print(f'>>> precision: {pre:.3} | recall: {rec:.3}'
            f'| F1: {f1:.3} | AUC: {roc:.3} ')
    print('-' * 80)

print()
print('*' * 100)
print('*' * 100)
print()

for conf in ['0_0.5', '0.5_1.0']:
    fname = 'results/merge_new_price_maxsep_lv34_confidence_{:s}'.format(conf)
    true_label, all_label = get_true_labels('data/merge_new_price_maxsep_lv34_confidence_{:s}'.format(conf))
    pred_label = get_pred_labels(fname, '')
    pred_probs = get_pred_probs(fname, '')
    pre, rec, f1, roc, acc, _, _, _ = get_eval_metrics(
    pred_label, pred_probs, true_label, all_label)
    roc = float(roc)
    print('#' * 80)
    print('>>> confidence: {:s}'.format(conf))
    print(f'>>> precision: {pre:.3} | recall: {rec:.3}'
            f'| F1: {f1:.3} | AUC: {roc:.3} | ACC: {acc:.3}')
    print('-' * 80)