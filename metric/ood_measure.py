import numpy as np

from metric.utils.measures import get_measures


def get_and_print_results(out_score, in_score):
    aurocs, auprs, fprs = [], [], []

    measures = get_measures(out_score, in_score)
    aurocs.append(measures[0]);
    auprs.append(measures[1]);
    fprs.append(measures[2])

    auroc = np.mean(aurocs);
    aupr = np.mean(auprs);
    fpr = np.mean(fprs)

    return auroc, aupr, fpr


def eval_ood_measure(conf, seg_label, train_id_in=1, train_id_out=0):
    in_scores = conf[seg_label == train_id_in]
    out_scores = conf[seg_label == train_id_out]

    if (len(out_scores) != 0) and (len(in_scores) != 0):
        auroc, aupr, fpr = get_and_print_results(out_scores, in_scores)
        return auroc, aupr, fpr
    else:
        return None
