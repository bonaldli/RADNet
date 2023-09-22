import numpy as np
from tqdm import tqdm

__all__ = ['BCubed_measure']


def BCubed_measure(gt_labels, pd_labels, return_detail=False, verbose=False):
    """
    Measure clustering performance by Bcubed method.

    Args:
        gt_labels: ground truth labels (int).
        pd_labels: predicted labels (int).

    Returns:
        average precision
        average recall
        average f1-score: 2.0 * precision * recall / (precision + recall)
        average expansion
    """
    
    def _indexing(labels, ignore_labels=[-1]):
        label_to_index = {}
        for idx, lab in enumerate(labels):
            if lab in ignore_labels:
                continue
            if lab in label_to_index:
                label_to_index[lab].append(idx)
            else:
                label_to_index[lab] = [idx]
        return label_to_index

    gt_dict = _indexing(gt_labels)
    pd_dict = _indexing(pd_labels)

    num_gt_ids = len(gt_dict.keys())
    precision = np.zeros((num_gt_ids, 1), dtype=np.float)
    recall    = np.zeros((num_gt_ids, 1), dtype=np.float)
    expansion = np.zeros((num_gt_ids, 1), dtype=np.float)
    gt_amount    = np.zeros((num_gt_ids, 1), dtype=np.float)

    i = 0
    for gt_k, gt_v in tqdm(gt_dict.items(), desc='Testing', disable=not verbose):
        pd_keys = np.unique(pd_labels[gt_v])
        expansion[i] = len(pd_keys)
        gt_amount[i] = len(gt_v)
        for pd_k in pd_keys:
            num_intersect = np.intersect1d(gt_v, pd_dict[pd_k]).shape[0]
            precision[i] += num_intersect**2 / len(pd_dict[pd_k])
            recall[i]    += num_intersect**2 / gt_amount[i]
        i += 1


    gt_num = len((gt_labels >= 0).nonzero()[0])
    ave_pre = np.sum(precision) / gt_num
    ave_rec = np.sum(recall) / gt_num
    fscore = 2.0 * ave_pre * ave_rec / (ave_pre + ave_rec)
    ave_exp = np.average(expansion)

    if return_detail:
      detail_dict = {'precision': precision, 'recall': recall, 'expansion': expansion}
      return ave_pre, ave_rec, fscore, ave_exp, detail_dict

    return ave_pre, ave_rec, fscore, ave_exp

