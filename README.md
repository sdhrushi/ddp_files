# SCD & Community Detection | Clustering
Pytorch code for performing clustering on images using features from Vision-Language models. 

Dataset paths are stored in dataset_catalog.json, which need to be modified to local paths. The imagenet dataset follows the standard folder structure. For other datasets, please refer to the scripts from VISSL to download and prepare. CLIP's labels and prompt templates are stored in classes.json and templates.json.

# Pre-Training 
"""
python -m torch.distributed.run --nproc_per_node=16 train.py --dataset [name_of_dataset] --clip_model ViT-B/16 
"""
