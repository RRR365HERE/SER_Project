"""
Classification metrics: accuracy, precision, recall, F1, confusion matrix,
and ROC/AUC (one-vs-rest for multiclass).
"""

import numpy as np


def accuracy_score(y_true, y_pred):
    """Fraction of predictions that match the ground truth."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true, y_pred, labels=None):
    """Compute confusion matrix C where C[i, j] = number of samples
    truly in class i that were predicted as class j.

    Diagonal entries are correct predictions; off-diagonal entries
    are the model's mistakes.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))

    n = len(labels)
    label_to_idx = {label: i for i, label in enumerate(labels)}

    cm = np.zeros((n, n), dtype=int)
    for true, pred in zip(y_true, y_pred):
        if true in label_to_idx and pred in label_to_idx:
            cm[label_to_idx[true], label_to_idx[pred]] += 1
    return cm


def precision_recall_f1(y_true, y_pred, labels=None, average=None):
    """Compute per-class precision, recall, and F1.

    Per-class definitions (one-vs-rest):
      precision = TP / (TP + FP)
      recall    = TP / (TP + FN)
      f1        = 2 * P * R / (P + R)

    Parameters
    ----------
    average : None, 'macro', or 'weighted'
        None     -> per-class arrays
        'macro'  -> simple mean across classes
        'weighted' -> mean weighted by class support (count in y_true)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp   # column sum minus diagonal
    fn = cm.sum(axis=1) - tp   # row sum minus diagonal
    support = cm.sum(axis=1)   # actual count per class

    # Safe division: return 0 if denominator is 0
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
        recall    = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)
        denom = precision + recall
        f1 = np.where(denom > 0, 2 * precision * recall / denom, 0.0)

    if average is None:
        return precision, recall, f1
    if average == "macro":
        return precision.mean(), recall.mean(), f1.mean()
    if average == "weighted":
        w = support / support.sum()
        return (precision @ w, recall @ w, f1 @ w)
    raise ValueError(f"Unknown average: {average}")


def precision_score(y_true, y_pred, labels=None, average=None):
    return precision_recall_f1(y_true, y_pred, labels=labels, average=average)[0]


def recall_score(y_true, y_pred, labels=None, average=None):
    return precision_recall_f1(y_true, y_pred, labels=labels, average=average)[1]


def f1_score(y_true, y_pred, labels=None, average=None):
    return precision_recall_f1(y_true, y_pred, labels=labels, average=average)[2]


def classification_report(y_true, y_pred, labels=None, label_names=None):
    """Return a string with a per-class precision/recall/F1/support table,
    plus macro avg, weighted avg, and overall accuracy."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    if label_names is None:
        label_names = [str(l) for l in labels]

    p, r, f = precision_recall_f1(y_true, y_pred, labels=labels, average=None)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    support = cm.sum(axis=1)

    rows = [f"{'class':<15} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>10}",
            "-" * 60]
    for i, name in enumerate(label_names):
        rows.append(f"{name:<15} {p[i]:>10.3f} {r[i]:>10.3f} {f[i]:>10.3f} {support[i]:>10d}")
    rows.append("-" * 60)

    pm, rm, fm = precision_recall_f1(y_true, y_pred, labels=labels, average="macro")
    pw, rw, fw = precision_recall_f1(y_true, y_pred, labels=labels, average="weighted")
    rows.append(f"{'macro avg':<15} {pm:>10.3f} {rm:>10.3f} {fm:>10.3f} {support.sum():>10d}")
    rows.append(f"{'weighted avg':<15} {pw:>10.3f} {rw:>10.3f} {fw:>10.3f} {support.sum():>10d}")
    rows.append(f"{'accuracy':<15} {accuracy_score(y_true, y_pred):>10.3f}")
    return "\n".join(rows)


def _binary_roc_auc(y_true_binary, y_score):
    """Compute area under the ROC curve via trapezoidal integration.

    Sweeps the decision threshold from highest score to lowest, building
    the ROC curve as (FPR, TPR) pairs, then integrates.
    """
    y_true_binary = np.asarray(y_true_binary)
    y_score = np.asarray(y_score)

    n_pos = int(y_true_binary.sum())
    n_neg = len(y_true_binary) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")  # AUC undefined if one class is absent

    # Sort by score descending; walk through thresholds from high to low
    order = np.argsort(-y_score)
    y_sorted = y_true_binary[order]

    tp_cum = np.cumsum(y_sorted)
    fp_cum = np.cumsum(1 - y_sorted)

    tpr = np.concatenate([[0.0], tp_cum / n_pos])
    fpr = np.concatenate([[0.0], fp_cum / n_neg])

    return float(np.sum(np.diff(fpr) * (tpr[:-1] + tpr[1:]) / 2))



def roc_auc_score(y_true, y_score, average="macro"):
    """ROC AUC. Binary or multiclass (one-vs-rest).

    Parameters
    ----------
    y_true : (n_samples,)
        True integer class labels.
    y_score : (n_samples,) for binary, or (n_samples, n_classes) for multiclass
        Predicted probabilities or decision scores.
    average : 'macro' or None
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    classes = np.unique(y_true)

    if y_score.ndim == 1:  # Binary
        return _binary_roc_auc(y_true, y_score)

    # Multiclass: one-vs-rest, then average
    aucs = []
    for i, cls in enumerate(classes):
        y_bin = (y_true == cls).astype(int)
        aucs.append(_binary_roc_auc(y_bin, y_score[:, i]))
    aucs = np.array(aucs)
    return float(aucs.mean()) if average == "macro" else aucs