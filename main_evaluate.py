from utils.evaluation import evaluate_performance

def main():
    """
    Main function to evaluate clustering performance using specified metrics.
    """
    evaluation_metrics = ['pairwise', 'bcubed', 'nmi', 'class_f_score']
    evaluate_performance(evaluation_metrics)

if __name__ == '__main__':
    main()