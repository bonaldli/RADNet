from __future__ import division

import argparse
import numpy as np
from scipy import sparse as sp

__all__ = ['pairwise_measure']



def check_clusterings(labels_true, labels_pred):
  """Check that the two clusterings matching 1D integer arrays."""
  labels_true = np.asarray(labels_true)
  labels_pred = np.asarray(labels_pred)

  # input checks
  if labels_true.ndim != 1:
    raise ValueError(
      "labels_true must be 1D: shape is %r" % (labels_true.shape,))
  if labels_pred.ndim != 1:
    raise ValueError(
      "labels_pred must be 1D: shape is %r" % (labels_pred.shape,))
  if labels_true.shape != labels_pred.shape:
    raise ValueError(
      "labels_true and labels_pred must have same size, got %d and %d"
      % (labels_true.shape[0], labels_pred.shape[0]))
  return labels_true, labels_pred



def contingency_matrix(labels_true, labels_pred, eps=None, sparse=False):
  if eps is not None and sparse:
    raise ValueError("Cannot set 'eps' when sparse=True")

  classes, class_idx = np.unique(labels_true, return_inverse=True)
  clusters, cluster_idx = np.unique(labels_pred, return_inverse=True)
  n_classes = classes.shape[0]
  n_clusters = clusters.shape[0]
  # Using coo_matrix to accelerate simple histogram calculation,
  # i.e. bins are consecutive integers
  # Currently, coo_matrix is faster than histogram2d for simple cases
  contingency = sp.coo_matrix((np.ones(class_idx.shape[0]),
                 (class_idx, cluster_idx)),
                shape=(n_classes, n_clusters),
                dtype=np.int)
  if sparse:
    contingency = contingency.tocsr()
    contingency.sum_duplicates()
  else:
    contingency = contingency.toarray()
    if eps is not None:
      # don't use += as contingency is integer
      contingency = contingency + eps
  return contingency


def fowlkes_mallows_score(labels_true, labels_pred, sparse=True):
  labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
  n_samples, = labels_true.shape

  c = contingency_matrix(labels_true, labels_pred, sparse=sparse)
  tk = np.dot(c.data, c.data) - n_samples
  pk = np.sum(np.asarray(c.sum(axis=0)).ravel() ** 2) - n_samples
  qk = np.sum(np.asarray(c.sum(axis=1)).ravel() ** 2) - n_samples
  avg_pre = tk / pk
  avg_rec = tk / qk
  fscore = 2. * avg_pre * avg_rec / (avg_pre + avg_rec)
  return avg_pre, avg_rec, fscore, -1, None


def pairwise_measure(gt_labels, pred_labels, sparse=True):
  num_gt_idxs = len(gt_labels)
  pred_labels = pred_labels[:num_gt_idxs]
  return fowlkes_mallows_score(gt_labels, pred_labels, sparse)


