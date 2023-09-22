__all__ = ['test', 'clustering_statistics', 'clustering_size_distribution']

import sys, os
import numpy as np

from .BCubed_measure import BCubed_measure
from .pairwise_measure import pairwise_measure

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())
from base_utils import *
# from database import ClusterDB

def test(gt_labels, pred_labels, verbose=False):
  num_clusters, sz_ge3, sz2, sz1, cap_left, max_cluster_size = clustering_statistics(pred_labels)
  print('{:<10}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
    'Statistics', num_clusters, sz_ge3, sz2, sz1, cap_left, max_cluster_size))

  p, r, f, e = BCubed_measure(gt_labels, pred_labels, return_detail=False, verbose=verbose)
  print('{:<10}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}'.format(
    'Bcubed', p, r, f, e))

  return '{:<10}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}'.format(
    'Bcubed', p, r, f, e)

def clustering_size_distribution(pred_labels, dis=[3, 5]):
  unique_labels, label_counts = np.unique(pred_labels, return_counts=True)
  num_dis = []
  for idx in range(len(dis) + 1):
    if idx == 0:
      num_dis.append(np.where(label_counts < dis[idx])[0].shape[0])
    if idx == len(dis):
      num_dis.append(np.where(label_counts >= dis[idx - 1])[0].shape[0])
    if idx !=0 and idx != len(dis):
      num_dis.append(np.where((label_counts >= dis[idx - 1]) & (label_counts < dis[idx]))[0].shape[0])
  max_cluster_size = np.max(label_counts)
  return max_cluster_size, num_dis


def clustering_statistics(pred_labels):
  unique_labels, label_counts = np.unique(pred_labels, return_counts=True)
  num_clusters = len(unique_labels)
  sz_ge3 = len((label_counts >= 3).nonzero()[0])
  sz2    = len((label_counts == 2).nonzero()[0])
  sz1    = len((label_counts == 1).nonzero()[0])
  cap_left = sz2 * 2 + sz1
  max_cluster_size = np.max(label_counts)
  return num_clusters, sz_ge3, sz2, sz1, cap_left, max_cluster_size

def doc_stastics(incremental_clustering, all_gt_labels, label_num):
    all_clusters = ClusterDB(min_size=1)
    all_clusters += incremental_clustering.cluster_db
    all_clusters += incremental_clustering.document_db
    print('calculate unique')
    gt_labels = parse_gt_labels(all_clusters.all_capture_ids, all_gt_labels)
    labels = np.unique(gt_labels)
    pd_labels = parse_predict_labels(all_clusters.all_capture_ids)
    print('done')

    print('calculate  Doc Statistics')
    times2num = {}
    pd_cus = set()
    for label in labels:
        if label == -1:
            continue
        indexs = np.where(gt_labels == label)
        pd_alls = pd_labels[indexs]
        pd_alls_unqiue = np.unique(pd_alls)
        pd_cus.update(pd_alls_unqiue.tolist())
        num = pd_alls_unqiue.shape[0]
        if num in times2num:
            times2num[num] += 1
        else:
            times2num[num] = 1
    pd_cus = list(pd_cus)
    print(len(pd_cus))
    print('done')

    times = [times2num[i] if i in times2num else 0 for i in range(1, 6)]
    times.append(labels.shape[0] - 1 - sum(times))

    print('{:<10}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
        'Doc Statistics', incremental_clustering.document_db.num_document,
        labels.shape[0] - 1, times[0], times[1], times[2], times[3], times[4], times[5]))
    # p = float(sum(each_doc_contain_num_IDS)) / float(len(each_doc_contain_num_IDS))
    p = 0
    r = float(labels.shape[0] - 1) / float(label_num)
    e = float(sum([times * times2num[times] if times <= 100 else 0 for times in times2num])) / float(
        labels.shape[0] - 1)

    print('{:<10}\t{:.6f}\t{:.6f}\t{:.6f}'.format('Doc evaluation', p, r, e))

    del all_clusters
    del gt_labels
    del pd_labels
    del pd_cus
