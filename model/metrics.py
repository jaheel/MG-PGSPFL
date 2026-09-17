"""
Shared evaluation module for all federated baselines.
=====================================================
Provides standard evaluation functions so that all methods are
compared with identical metrics and identical computation logic.

Metrics:
  - f1_macro(preds, labels)        : mean per-class F1 (macro-F1)
  - recall_at_k(preds, labels, k)  : top-k coverage of true labels
  - recall_for(preds, labels, idxs): recall over a class subset
  - evaluate(model, ...)           : full evaluation, returns a metrics dict

Usage:
  from metrics import evaluate
  metrics = evaluate(model, test_patients, centers, N_ICD, BATCH_SIZE, DEVICE,
                     dataset_cls, collate_fn)
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from collections import defaultdict

def f1_macro(preds, labels, threshold=0.5):
    """Mean per-class F1 (macro-F1)."""
    p = (preds >= threshold).astype(int)
    f1s = []
    for c in range(labels.shape[1]):
        tp = (p[:, c] * labels[:, c]).sum()
        fp = ((p[:, c] == 1) & (labels[:, c] == 0)).sum()
        fn = ((p[:, c] == 0) & (labels[:, c] == 1)).sum()
        if tp + fp + fn == 0:
            continue
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        rec  = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0)
    return np.mean(f1s) if f1s else 0.0


def recall_at_k(preds, labels, k):
    """Fraction of true labels covered by the top-k predictions."""
    total = 0.0
    for i in range(len(preds)):
        topk = set(np.argsort(preds[i])[-k:])
        true = set(np.where(labels[i] >= 0.5)[0])
        total += len(topk & true) / max(1, len(true))
    return total / len(preds)


def recall_at_k_for_classes(preds, labels, k, class_indices):
    """Fraction of true labels (within the given class subset) covered by top-k predictions."""
    if len(class_indices) == 0:
        return 0.0
    idx_set = set(class_indices)
    total = 0.0
    for i in range(len(preds)):
        topk = set(np.argsort(preds[i])[-k:])
        true = set(np.where(labels[i] >= 0.5)[0])
        true_in = true & idx_set  # only count target classes
        hit_in = topk & true_in
        total += len(hit_in) / max(1, len(true_in))
    return total / len(preds) if total > 0 else 0.0


def recall_for(preds, labels, indices):
    """Recall over a given class subset (used for head/tail separately)."""
    if len(indices) == 0:
        return 0.0
    p = (preds >= 0.5).astype(int)
    recalls = []
    for c in indices:
        tp = (p[:, c] * labels[:, c]).sum()
        fn = ((p[:, c] == 0) & (labels[:, c] == 1)).sum()
        recalls.append(tp / (tp + fn) if tp + fn > 0 else 0.0)
    return np.mean(recalls)


@torch.no_grad()
def evaluate(model, test_patients, centers, N_ICD, batch_size, device,
             dataset_cls, collate_fn):
    """
    Full evaluation.

    Args:
      model         : model (must implement model(x, lengths) -> preds)
      test_patients : {center: [pid, ...]}
      centers       : [center_name, ...]
      N_ICD         : number of ICD classes
      batch_size    : batch size
      device        : torch device
      dataset_cls   : SeqDataset class (only takes pids)
      collate_fn    : collate function

    Returns:
      dict with the evaluation metrics
    """
    model.eval()
    all_preds_global, all_labels_global = [], []
    per_client_recall = []     # client-level recall@10
    per_client_head_recall = {'10': [], '20': []}
    per_client_tail_recall = {'10': [], '20': []}
    per_client_valid_head = []  # count clients with valid head samples
    per_client_valid_tail = []

    for center in centers:
        t_pids = test_patients.get(center, [])
        if len(t_pids) == 0:
            continue
        ds = dataset_cls(t_pids)
        dl = DataLoader(ds, batch_size, collate_fn=collate_fn)
        preds_c, labels_c = [], []
        for x, lens, y in dl:
            preds_c.append(model(x.to(device), lens).cpu().numpy())
            labels_c.append(y.numpy())
        if not preds_c:
            continue
        preds_c = np.vstack(preds_c)
        labels_c = np.vstack(labels_c)
        all_preds_global.append(preds_c)
        all_labels_global.append(labels_c)

        # Per-client: local head/tail split (20/80)
        freq_c = labels_c.sum(axis=0)
        sorted_idx = np.argsort(freq_c)[::-1]
        n_head = max(1, int(N_ICD * 0.2))
        head_idx = sorted_idx[:n_head]
        tail_idx = sorted_idx[n_head:]

        # Client-level Recall@10, Recall@20
        for k in [10, 20]:
            per_client_recall.append((k, recall_at_k(preds_c, labels_c, k)))

        # Per-client Recall-head@k, Recall-tail@k
        for k in [10, 20]:
            # Head: skip samples without head labels
            has_head = labels_c[:, head_idx].sum(axis=1) > 0
            if has_head.sum() > 0:
                rh = recall_at_k_for_classes(preds_c[has_head], labels_c[has_head], k, head_idx)
                per_client_head_recall[str(k)].append(rh)
                per_client_valid_head.append(center)

            # Tail: skip samples without tail labels
            has_tail = labels_c[:, tail_idx].sum(axis=1) > 0
            if has_tail.sum() > 0:
                rt = recall_at_k_for_classes(preds_c[has_tail], labels_c[has_tail], k, tail_idx)
                per_client_tail_recall[str(k)].append(rt)
                per_client_valid_tail.append(center)

    # Global metrics
    ap = np.vstack(all_preds_global); al = np.vstack(all_labels_global)

    result = {}
    for k in [10, 20]:
        result[f'recall_at_{k}'] = recall_at_k(ap, al, k)

    # Head/tail: mean over valid clients
    for k in [10, 20]:
        rh_list = per_client_head_recall[str(k)]
        rt_list = per_client_tail_recall[str(k)]
        result[f'recall_head_at_{k}'] = np.mean(rh_list) if rh_list else 0.0
        result[f'recall_tail_at_{k}'] = np.mean(rt_list) if rt_list else 0.0

    # Client-Recall: mean/std of per-client recall@10
    r10_list = [v for k, v in per_client_recall if k == 10]
    r20_list = [v for k, v in per_client_recall if k == 20]
    result['client_recall_at_10'] = np.mean(r10_list) if r10_list else 0.0
    result['client_recall_std_at_10'] = np.std(r10_list) if r10_list else 0.0
    result['client_recall_at_20'] = np.mean(r20_list) if r20_list else 0.0
    result['client_recall_std_at_20'] = np.std(r20_list) if r20_list else 0.0

    result['n_valid_head_clients'] = len(set(per_client_valid_head))
    result['n_valid_tail_clients'] = len(set(per_client_valid_tail))

    return result
