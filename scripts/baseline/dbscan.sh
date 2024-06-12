#!/bin/bash

# Set up the paths and parameters
DATA_DIR=./data
DATA_NAME=part_test

OUTPUT_DIR=$DATA_DIR/baseline_results
GROUND_TRUTH_LABELS=$DATA_DIR/labels/$DATA_NAME.meta

FEATURE_DIM=256

export PYTHONPATH=.

# Define the clustering method and parameters
CLUSTER_METHOD=aro
SIMILARITY_THRESHOLD=0.0
KNN_NEIGHBORS=80
NUM_PROCESSES=16
PREDICTED_LABELS=$OUTPUT_DIR/${DATA_NAME}_${CLUSTER_METHOD}_k_${KNN_NEIGHBORS}_th_${SIMILARITY_THRESHOLD}/pred_labels.txt

# Run the baseline clustering
python tools/baseline_cluster.py \
    --prefix $DATA_DIR \
    --oprefix $OUTPUT_DIR \
    --name $DATA_NAME \
    --dim $FEATURE_DIM \
    --method $CLUSTER_METHOD \
    --knn $KNN_NEIGHBORS \
    --th_sim $SIMILARITY_THRESHOLD \
    --num_process $NUM_PROCESSES

# Define and run evaluation metrics
EVALUATION_METRICS=("pairwise" "bcubed")

for METRIC in "${EVALUATION_METRICS[@]}"
do
    python evaluation/evaluate.py \
        --metric $METRIC \
        --gt_labels $GROUND_TRUTH_LABELS \
        --pred_labels $PREDICTED_LABELS
done