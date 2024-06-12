from __future__ import division
import numpy as np
from sklearn.metrics.cluster import contingency_matrix, normalized_mutual_info_score
from sklearn.metrics import precision_score, recall_score
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Define all the available metrics
__all__ = ['pairwise', 'bcubed', 'nmi', 'precision', 'recall', 'accuracy', 'new_metric']

def _validate_labels(gt_labels, pred_labels):
    """
    Validate that the ground truth and predicted labels are 1D arrays of the same shape.

    Args:
        gt_labels (numpy.ndarray): Ground truth labels.
        pred_labels (numpy.ndarray): Predicted labels.
    
    Raises:
        ValueError: If input conditions are not met.
    """
    if gt_labels.ndim != 1:
        raise ValueError(f"gt_labels must be 1D: shape is {gt_labels.shape}")
    if pred_labels.ndim != 1:
        raise ValueError(f"pred_labels must be 1D: shape is {pred_labels.shape}")
    if gt_labels.shape != pred_labels.shape:
        raise ValueError(f"gt_labels and pred_labels must have the same size, got {gt_labels.shape[0]} and {pred_labels.shape[0]}")
    return gt_labels, pred_labels

def _get_label_to_indices(labels):
    """
    Map each label to the list of indices where it occurs.

    Args:
        labels (numpy.ndarray): Array of labels.
    
    Returns:
        dict: Mapping from label to list of indices.
    """
    label_to_indices = {}
    for index, label in enumerate(labels):
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(index)
    return label_to_indices

def _compute_fscore(precision, recall):
    """
    Compute the F1-score from precision and recall.

    Args:
        precision (float): Precision value.
        recall (float): Recall value.
    
    Returns:
        float: Computed F1-score.
    """
    return 2.0 * precision * recall / (precision + recall)

def fowlkes_mallows_score(gt_labels, pred_labels, sparse=True):
    """
    Compute the Fowlkes-Mallows index.

    Args:
        gt_labels (numpy.ndarray): Ground truth labels.
        pred_labels (numpy.ndarray): Predicted labels.
        sparse (bool): Whether to use a sparse contingency matrix.
    
    Returns:
        tuple: Precision, recall, and F1-score.
    """
    n_samples = gt_labels.shape[0]

    c_matrix = contingency_matrix(gt_labels, pred_labels, sparse=sparse)
    tk = np.dot(c_matrix.data, c_matrix.data) - n_samples
    pk = np.sum(np.asarray(c_matrix.sum(axis=0)).ravel()**2) - n_samples
    qk = np.sum(np.asarray(c_matrix.sum(axis=1)).ravel()**2) - n_samples
    avg_pre = tk / pk
    avg_rec = tk / qk
    fscore = _compute_fscore(avg_pre, avg_rec)
    return avg_pre, avg_rec, fscore

def pairwise(gt_labels, pred_labels, sparse=True):
    """
    Evaluate using pairwise Fowlkes-Mallows index.

    Args:
        gt_labels (numpy.ndarray): Ground truth labels.
        pred_labels (numpy.ndarray): Predicted labels.
        sparse (bool): Whether to use a sparse contingency matrix.
    
    Returns:
        tuple: Precision, recall, and F1-score.
    """
    _validate_labels(gt_labels, pred_labels)
    avg_pre, avg_rec, fscore = fowlkes_mallows_score(gt_labels, pred_labels, sparse)
    print(f'#pairwise: avg_pre:{avg_pre:.4f}, avg_rec:{avg_rec:.4f}, fscore:{fscore:.4f}')
    return avg_pre, avg_rec, fscore

def bcubed(gt_labels, pred_labels):
    """
    Evaluate using BCubed metrics.

    Args:
        gt_labels (numpy.ndarray): Ground truth labels.
        pred_labels (numpy.ndarray): Predicted labels.
    
    Returns:
        tuple: Precision, recall, and F1-score.
    """
    _validate_labels(gt_labels, pred_labels)

    gt_label_to_indices = _get_label_to_indices(gt_labels)
    pred_label_to_indices = _get_label_to_indices(pred_labels)

    total_precision = 0.0
    total_recall = 0.0
    total_gt_items = sum(len(indices) for indices in gt_label_to_indices.values())

    for gt_indices in gt_label_to_indices.values():
        unique_pred_labels = np.unique(pred_labels[gt_indices])
        for pred_label in unique_pred_labels:
            pred_indices = pred_label_to_indices[pred_label]
            common_elements = np.intersect1d(gt_indices, pred_indices).size
            total_precision += (common_elements ** 2) / len(pred_indices)
            total_recall += (common_elements ** 2) / len(gt_indices)

    avg_precision = total_precision / total_gt_items
    avg_recall = total_recall / total_gt_items
    fscore = _compute_fscore(avg_precision, avg_recall)
    print(f'#bcubed: avg_pre:{avg_precision:.4f}, avg_rec:{avg_recall:.4f}, fscore:{fscore:.4f}')
    return avg_precision, avg_recall, fscore

def nmi(gt_labels, pred_labels):
    """
    Compute the Normalized Mutual Information (NMI) score.

    Args:
        gt_labels (numpy.ndarray): Ground truth labels.
        pred_labels (numpy.ndarray): Predicted labels.
    
    Returns:
        float: NMI score.
    """
    return normalized_mutual_info_score(gt_labels, pred_labels)

def precision(gt_labels, pred_labels):
    """
    Compute the precision score.

    Args:
        gt_labels (numpy.ndarray): Ground truth labels.
        pred_labels (numpy.ndarray): Predicted labels.
    
    Returns:
        float: Precision score.
    """
    return precision_score(gt_labels, pred_labels)

def recall(gt_labels, pred_labels):
    """
    Compute the recall score.

    Args:
        gt_labels (numpy.ndarray): Ground truth labels.
        pred_labels (numpy.ndarray): Predicted labels.
    
    Returns:
        float: Recall score.
    """
    return recall_score(gt_labels, pred_labels)

def accuracy(gt_labels, pred_labels):
    """
    Compute the accuracy score.

    Args:
        gt_labels (numpy.ndarray): Ground truth labels.
        pred_labels (numpy.ndarray): Predicted labels.
    
    Returns:
        float: Accuracy score.
    """
    return np.mean(gt_labels == pred_labels)

def new_metric(gt_labels, pred_labels, sparse=True):
    """
    Compute a custom metric based on precision and recall thresholds.

    Args:
        gt_labels (numpy.ndarray): Ground truth labels.
        pred_labels (numpy.ndarray): Predicted labels.
        sparse (bool): Whether to use a sparse contingency matrix.
    
    Returns:
        None
    """
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

    c_matrix = contingency_matrix(gt_labels, pred_labels, sparse=sparse)
    true_class_counts = np.asarray(c_matrix.sum(axis=1)).ravel()
    pred_cluster_counts = np.asarray(c_matrix.sum(axis=0)).ravel()
    true_class_count = true_class_counts.shape[0]
    pred_cluster_count = pred_cluster_counts.shape[0]

    correct_positive_counts = np.zeros(len(thresholds))

    for j in range(c_matrix.shape[1]):
        col = c_matrix[:, j]
        precisions = col.data / pred_cluster_counts[j]
        recalls = col.toarray().ravel() / true_class_counts
        recalls = recalls[recalls > 0.0]
        for i, threshold in enumerate(thresholds):
            correct_positive_counts[i] += np.sum((precisions > threshold) & (recalls > threshold))

    false_positive_counts = pred_cluster_count - correct_positive_counts
    false_negative_counts = true_class_count - correct_positive_counts
    avg_precision = correct_positive_counts / pred_cluster_count
    avg_recall = correct_positive_counts / true_class_count
    identity_f1_scores = 2 * correct_positive_counts / (2 * correct_positive_counts + false_positive_counts + false_negative_counts)

    for i, threshold in enumerate(thresholds):
        print(f'#theta:{threshold}, avg_pre:{avg_precision[i]:.4f}, avg_rec:{avg_recall[i]:.4f}, Class F-score:{identity_f1_scores[i]:.4f}')