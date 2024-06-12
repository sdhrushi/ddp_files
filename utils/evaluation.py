import numpy as np
from utils import metrics
from configs import config


def evaluate_performance(metrics_to_evaluate):
    """
    Evaluate the performance of clustering results using specified metrics.
    
    Args:
        metrics_to_evaluate (list): List of metric names to evaluate.
    """
    # Load predicted labels from the results file
    predicted_labels = np.load(config.result_path)

    # Load ground truth labels from the label file
    with open(config.label_path, 'r') as file:
        ground_truth_labels = np.array([line.strip() for line in file.readlines()], dtype=np.uint32)

    # Display the number of unique classes in ground truth and predicted labels
    print(f'# of classes: {len(np.unique(ground_truth_labels))}, # of predicted classes: {len(np.unique(predicted_labels))}')

    # Evaluate each specified metric
    for metric_name in metrics_to_evaluate:
        metric_function = getattr(metrics, metric_name)
        metric_function(ground_truth_labels, predicted_labels)