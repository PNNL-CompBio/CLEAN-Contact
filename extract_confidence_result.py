import csv
import numpy as np
import pandas as pd

with open('results/merge_new_price_maxsep.csv', 'r') as f:
    reader = csv.reader(f)
    pred = list(reader)

with open('results/merge_new_price_maxsep_confidence.csv', 'r') as f:
    reader = csv.reader(f)
    conf = list(reader)

pred_conf_dict = dict()
for line in pred:
    preds_confs = {i.split('/')[0]: [i.split('/')[1]] for i in line[1:]}
    pred_conf_dict[line[0]] = preds_confs

for line in conf:
    for ec_conf in line[1:]:
        pred_conf_dict[line[0]][ec_conf.split('/')[0]].append(ec_conf.split('/')[1].split('_')[0])

results = dict()
data_conf = dict()
for i in np.linspace(0.1, 1, 10):
    results['confidence_0_{:.1f}'.format(i)] = dict()
    data_conf['confidence_0_{:.1f}'.format(i)] = []

data = pd.read_csv('data/merge_new_price.csv', sep='\t')

results_lvl_3_4 = {
    'confidence_0_0.5': dict(),
    'confidence_0.5_1.0': dict(),
}
data_conf_lvl_3_4 = {
    'confidence_0_0.5': [],
    'confidence_0.5_1.0': [],
}

for ent in pred_conf_dict:
    for k in results:
        results[k][ent] = list()
    results_lvl_3_4['confidence_0.5_1.0'][ent] = list()
    results_lvl_3_4['confidence_0_0.5'][ent] = list()
    for ec in pred_conf_dict[ent]:
        if float(pred_conf_dict[ent][ec][1]) > 0.9:
            results['confidence_0_1.0'][ent].append(f'{ec}/{pred_conf_dict[ent][ec][0]}')
            data_conf['confidence_0_1.0'].append(data.loc[data['Entry'] == ent])
        # else:
        #     results['confidence_0_0.9'][ent].append(f'{ec}/{pred_conf_dict[ent][ec][0]}')
        #     results['confidence_0_1.0'][ent].append(f'{ec}/{pred_conf_dict[ent][ec][0]}')
        elif float(pred_conf_dict[ent][ec][1]) > 0.8:
            results['confidence_0_0.9'][ent].append(f'{ec}/{pred_conf_dict[ent][ec][0]}')
            results['confidence_0_1.0'][ent].append(f'{ec}/{pred_conf_dict[ent][ec][0]}')
            data_conf['confidence_0_0.9'].append(data.loc[data['Entry'] == ent])
            data_conf['confidence_0_1.0'].append(data.loc[data['Entry'] == ent])
        elif float(pred_conf_dict[ent][ec][1]) > 0.7:
            results['confidence_0_0.8'][ent].append(f'{ec}/{pred_conf_dict[ent][ec][0]}')
            results['confidence_0_0.9'][ent].append(f'{ec}/{pred_conf_dict[ent][ec][0]}')
            results['confidence_0_1.0'][ent].append(f'{ec}/{pred_conf_dict[ent][ec][0]}')
            data_conf['confidence_0_0.8'].append(data.loc[data['Entry'] == ent])
            data_conf['confidence_0_0.9'].append(data.loc[data['Entry'] == ent])
            data_conf['confidence_0_1.0'].append(data.loc[data['Entry'] == ent])
        elif float(pred_conf_dict[ent][ec][1]) > 0.6:
            results['confidence_0_0.7'][ent].append(f'{ec}/{pred_conf_dict[ent][ec][0]}')
            results['confidence_0_0.8'][ent].append(f'{ec}/{pred_conf_dict[ent][ec][0]}')
            results['confidence_0_0.9'][ent].append(f'{ec}/{pred_conf_dict[ent][ec][0]}')
            results['confidence_0_1.0'][ent].append(f'{ec}/{pred_conf_dict[ent][ec][0]}')
            data_conf['confidence_0_0.7'].append(data.loc[data['Entry'] == ent])
            data_conf['confidence_0_0.8'].append(data.loc[data['Entry'] == ent])
            data_conf['confidence_0_0.9'].append(data.loc[data['Entry'] == ent])
            data_conf['confidence_0_1.0'].append(data.loc[data['Entry'] == ent])
        elif float(pred_conf_dict[ent][ec][1]) > 0.5:
            results['confidence_0_0.6'][ent].append(f'{ec}/{pred_conf_dict[ent][ec][0]}')
            results['confidence_0_0.7'][ent].append(f'{ec}/{pred_conf_dict[ent][ec][0]}')
            results['confidence_0_0.8'][ent].append(f'{ec}/{pred_conf_dict[ent][ec][0]}')
            results['confidence_0_0.9'][ent].append(f'{ec}/{pred_conf_dict[ent][ec][0]}')
            results['confidence_0_1.0'][ent].append(f'{ec}/{pred_conf_dict[ent][ec][0]}')
            data_conf['confidence_0_0.6'].append(data.loc[data['Entry'] == ent])
            data_conf['confidence_0_0.7'].append(data.loc[data['Entry'] == ent])
            data_conf['confidence_0_0.8'].append(data.loc[data['Entry'] == ent])
            data_conf['confidence_0_0.9'].append(data.loc[data['Entry'] == ent])
            data_conf['confidence_0_1.0'].append(data.loc[data['Entry'] == ent])
        elif float(pred_conf_dict[ent][ec][1]) > 0.4:
            results['confidence_0_0.5'][ent].append(f'{ec}/{pred_conf_dict[ent][ec][0]}')
            results['confidence_0_0.6'][ent].append(f'{ec}/{pred_conf_dict[ent][ec][0]}')
            results['confidence_0_0.7'][ent].append(f'{ec}/{pred_conf_dict[ent][ec][0]}')
            results['confidence_0_0.8'][ent].append(f'{ec}/{pred_conf_dict[ent][ec][0]}')
            results['confidence_0_0.9'][ent].append(f'{ec}/{pred_conf_dict[ent][ec][0]}')
            results['confidence_0_1.0'][ent].append(f'{ec}/{pred_conf_dict[ent][ec][0]}')
            data_conf['confidence_0_0.5'].append(data.loc[data['Entry'] == ent])
            data_conf['confidence_0_0.6'].append(data.loc[data['Entry'] == ent])
            data_conf['confidence_0_0.7'].append(data.loc[data['Entry'] == ent])
            data_conf['confidence_0_0.8'].append(data.loc[data['Entry'] == ent])
            data_conf['confidence_0_0.9'].append(data.loc[data['Entry'] == ent])
            data_conf['confidence_0_1.0'].append(data.loc[data['Entry'] == ent])
        elif float(pred_conf_dict[ent][ec][1]) > 0.3:
            results['confidence_0_0.4'][ent].append(f'{ec}/{pred_conf_dict[ent][ec][0]}')
            results['confidence_0_0.5'][ent].append(f'{ec}/{pred_conf_dict[ent][ec][0]}')
            results['confidence_0_0.6'][ent].append(f'{ec}/{pred_conf_dict[ent][ec][0]}')
            results['confidence_0_0.7'][ent].append(f'{ec}/{pred_conf_dict[ent][ec][0]}')
            results['confidence_0_0.8'][ent].append(f'{ec}/{pred_conf_dict[ent][ec][0]}')
            results['confidence_0_0.9'][ent].append(f'{ec}/{pred_conf_dict[ent][ec][0]}')
            results['confidence_0_1.0'][ent].append(f'{ec}/{pred_conf_dict[ent][ec][0]}')
            data_conf['confidence_0_0.4'].append(data.loc[data['Entry'] == ent])
            data_conf['confidence_0_0.5'].append(data.loc[data['Entry'] == ent])
            data_conf['confidence_0_0.6'].append(data.loc[data['Entry'] == ent])
            data_conf['confidence_0_0.7'].append(data.loc[data['Entry'] == ent])
            data_conf['confidence_0_0.8'].append(data.loc[data['Entry'] == ent])
            data_conf['confidence_0_0.9'].append(data.loc[data['Entry'] == ent])
            data_conf['confidence_0_1.0'].append(data.loc[data['Entry'] == ent])
        elif float(pred_conf_dict[ent][ec][1]) > 0.2:
            results['confidence_0_0.3'][ent].append(f'{ec}/{pred_conf_dict[ent][ec][0]}')
            results['confidence_0_0.4'][ent].append(f'{ec}/{pred_conf_dict[ent][ec][0]}')
            results['confidence_0_0.5'][ent].append(f'{ec}/{pred_conf_dict[ent][ec][0]}')
            results['confidence_0_0.6'][ent].append(f'{ec}/{pred_conf_dict[ent][ec][0]}')
            results['confidence_0_0.7'][ent].append(f'{ec}/{pred_conf_dict[ent][ec][0]}')
            results['confidence_0_0.8'][ent].append(f'{ec}/{pred_conf_dict[ent][ec][0]}')
            results['confidence_0_0.9'][ent].append(f'{ec}/{pred_conf_dict[ent][ec][0]}')
            results['confidence_0_1.0'][ent].append(f'{ec}/{pred_conf_dict[ent][ec][0]}')
            data_conf['confidence_0_0.3'].append(data.loc[data['Entry'] == ent])
            data_conf['confidence_0_0.4'].append(data.loc[data['Entry'] == ent])
            data_conf['confidence_0_0.5'].append(data.loc[data['Entry'] == ent])
            data_conf['confidence_0_0.6'].append(data.loc[data['Entry'] == ent])
            data_conf['confidence_0_0.7'].append(data.loc[data['Entry'] == ent])
            data_conf['confidence_0_0.8'].append(data.loc[data['Entry'] == ent])
            data_conf['confidence_0_0.9'].append(data.loc[data['Entry'] == ent])
            data_conf['confidence_0_1.0'].append(data.loc[data['Entry'] == ent])
        elif float(pred_conf_dict[ent][ec][1]) > 0.1:
            results['confidence_0_0.2'][ent].append(f'{ec}/{pred_conf_dict[ent][ec][0]}')
            results['confidence_0_0.3'][ent].append(f'{ec}/{pred_conf_dict[ent][ec][0]}')
            results['confidence_0_0.4'][ent].append(f'{ec}/{pred_conf_dict[ent][ec][0]}')
            results['confidence_0_0.5'][ent].append(f'{ec}/{pred_conf_dict[ent][ec][0]}')
            results['confidence_0_0.6'][ent].append(f'{ec}/{pred_conf_dict[ent][ec][0]}')
            results['confidence_0_0.7'][ent].append(f'{ec}/{pred_conf_dict[ent][ec][0]}')
            results['confidence_0_0.8'][ent].append(f'{ec}/{pred_conf_dict[ent][ec][0]}')
            results['confidence_0_0.9'][ent].append(f'{ec}/{pred_conf_dict[ent][ec][0]}')
            results['confidence_0_1.0'][ent].append(f'{ec}/{pred_conf_dict[ent][ec][0]}')
            data_conf['confidence_0_0.2'].append(data.loc[data['Entry'] == ent])
            data_conf['confidence_0_0.3'].append(data.loc[data['Entry'] == ent])
            data_conf['confidence_0_0.4'].append(data.loc[data['Entry'] == ent])
            data_conf['confidence_0_0.5'].append(data.loc[data['Entry'] == ent])
            data_conf['confidence_0_0.6'].append(data.loc[data['Entry'] == ent])
            data_conf['confidence_0_0.7'].append(data.loc[data['Entry'] == ent])
            data_conf['confidence_0_0.8'].append(data.loc[data['Entry'] == ent])
            data_conf['confidence_0_0.9'].append(data.loc[data['Entry'] == ent])
            data_conf['confidence_0_1.0'].append(data.loc[data['Entry'] == ent])
        else:
            results['confidence_0_0.1'][ent].append(f'{ec}/{pred_conf_dict[ent][ec][0]}')
            results['confidence_0_0.2'][ent].append(f'{ec}/{pred_conf_dict[ent][ec][0]}')
            results['confidence_0_0.3'][ent].append(f'{ec}/{pred_conf_dict[ent][ec][0]}')
            results['confidence_0_0.4'][ent].append(f'{ec}/{pred_conf_dict[ent][ec][0]}')
            results['confidence_0_0.5'][ent].append(f'{ec}/{pred_conf_dict[ent][ec][0]}')
            results['confidence_0_0.6'][ent].append(f'{ec}/{pred_conf_dict[ent][ec][0]}')
            results['confidence_0_0.7'][ent].append(f'{ec}/{pred_conf_dict[ent][ec][0]}')
            results['confidence_0_0.8'][ent].append(f'{ec}/{pred_conf_dict[ent][ec][0]}')
            results['confidence_0_0.9'][ent].append(f'{ec}/{pred_conf_dict[ent][ec][0]}')
            results['confidence_0_1.0'][ent].append(f'{ec}/{pred_conf_dict[ent][ec][0]}')
            data_conf['confidence_0_0.1'].append(data.loc[data['Entry'] == ent])
            data_conf['confidence_0_0.2'].append(data.loc[data['Entry'] == ent])
            data_conf['confidence_0_0.3'].append(data.loc[data['Entry'] == ent])
            data_conf['confidence_0_0.4'].append(data.loc[data['Entry'] == ent])
            data_conf['confidence_0_0.5'].append(data.loc[data['Entry'] == ent])
            data_conf['confidence_0_0.6'].append(data.loc[data['Entry'] == ent])
            data_conf['confidence_0_0.7'].append(data.loc[data['Entry'] == ent])
            data_conf['confidence_0_0.8'].append(data.loc[data['Entry'] == ent])
            data_conf['confidence_0_0.9'].append(data.loc[data['Entry'] == ent])
            data_conf['confidence_0_1.0'].append(data.loc[data['Entry'] == ent])

        if float(pred_conf_dict[ent][ec][1]) > 0.5:
            results_lvl_3_4['confidence_0.5_1.0'][ent].append(f'{ec}/{pred_conf_dict[ent][ec][0]}')
            data_conf_lvl_3_4['confidence_0.5_1.0'].append(data.loc[data['Entry'] == ent])
        else:
            ec_lvs = ec.split('.')
            ec_lvs[-1] = '-'
            new_ec = '.'.join(ec_lvs)
            pred_ec = f'{new_ec}/{pred_conf_dict[ent][ec][0]}'
            # pred_ec = pred_ec[:-1]  + '-'
            results_lvl_3_4['confidence_0_0.5'][ent].append(pred_ec)
            # results_lvl_3_4['confidence_0.5_1.0'][ent].append(f'{ec}/{pred_conf_dict[ent][ec][0]}')
            # true_ec = data.loc[data['Entry'] == ent]

            data_conf_lvl_3_4['confidence_0_0.5'].append(data.loc[data['Entry'] == ent])
            # data_conf_lvl_3_4['confidence_0.5_1.0'].append(data.loc[data['Entry'] == ent])

write_results = {k: [] for k in results}
# print(write_results)
for k in results:
    for ent in results[k]:
        if len(results[k][ent]) > 0:
            # print(k)
            # print(ent)
            # print(results[k][ent])
            write_results[k].append([ent] + results[k][ent])
# print(write_results[k])
# print(results['confidence_0_1.0'])
for key in write_results:
    with open('results/merge_new_price_maxsep_' + key + '.csv', 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(write_results[key])

for key in data_conf:
    data_conf[key] = pd.concat(data_conf[key])
    data_conf[key] = data_conf[key].drop_duplicates()
    data_conf[key].to_csv('data/merge_new_price_maxsep_' + key + '.csv', sep='\t', index=False)

write_results_lvl_3_4 = {k: [] for k in results_lvl_3_4}
for k in results_lvl_3_4:
    for ent in results_lvl_3_4[k]:
        if len(results_lvl_3_4[k][ent]) > 0:
            write_results_lvl_3_4[k].append([ent] + results_lvl_3_4[k][ent])

for key in write_results_lvl_3_4:
    with open('results/merge_new_price_maxsep_lv34_' + key + '.csv', 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(write_results_lvl_3_4[key])

def f(x):
    ecs = x.split(';')
    new_ecs = []
    for ec in ecs:
        ec_lvs = ec.split('.')
        ec_lvs[-1] = '-'
        new_ec = '.'.join(ec_lvs)
        new_ecs.append(new_ec)
    return ';'.join(new_ecs)

for key in data_conf_lvl_3_4:
    data_conf_lvl_3_4[key] = pd.concat(data_conf_lvl_3_4[key])
    data_conf_lvl_3_4[key] = data_conf_lvl_3_4[key].drop_duplicates()
    if '0_0.5' in key:
        data_conf_lvl_3_4[key]['EC number'] = data_conf_lvl_3_4[key]['EC number'].apply(f)
    data_conf_lvl_3_4[key].to_csv('data/merge_new_price_maxsep_lv34_' + key + '.csv', sep='\t', index=False)