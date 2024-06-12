import numpy as np
import faiss
from tqdm import tqdm
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')


def normalize_l2(vec):
    """
    Perform L2 normalization on the input vectors.
    
    Args:
        vec (numpy.ndarray): Input vectors to normalize.
    
    Returns:
        numpy.ndarray: L2 normalized vectors.
    """
    return vec / np.linalg.norm(vec, axis=1, keepdims=True)


def load_features(feature_path, feature_dim=256):
    """
    Load features from a file.

    Args:
        feature_path (str): Path to the feature file.
        feature_dim (int): Dimension of the features.

    Returns:
        numpy.ndarray: Loaded features.
    """
    if feature_path.endswith('.npy'):
        features = np.load(feature_path).astype(np.float32)
    else:
        features = np.fromfile(feature_path, dtype=np.float32).reshape(-1, feature_dim)
    return features


def compute_faiss_knn(feature_path, knn_output_path, feature_dim, k=256):
    """
    Compute the k-nearest neighbors using FAISS and save the results.

    Args:
        feature_path (str): Path to the feature file.
        knn_output_path (str): Path to save the KNN results.
        feature_dim (int): Dimension of the features.
        k (int): Number of neighbors to compute.

    Returns:
        tuple: Neighbor indices and similarities.
    """
    features = load_features(feature_path, feature_dim)
    print(f'Feature shape: {features.shape}')
    features = normalize_l2(features)

    index = faiss.IndexFlatIP(feature_dim)

    # Use a single GPU
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, index)

    index.add(features)
    batch_size = 200000
    num_batches = int(np.ceil(features.shape[0] / batch_size))

    all_sims = np.empty((0, k + 1), dtype=np.float32)
    all_nbrs = np.empty((0, k + 1), dtype=np.uint32)

    for i in tqdm(range(num_batches)):
        start = i * batch_size
        end = (i + 1) * batch_size
        query_batch = features[start:end]
        sims_batch, nbrs_batch = index.search(query_batch, k + 1)
        all_sims = np.vstack((all_sims, sims_batch))
        all_nbrs = np.vstack((all_nbrs, nbrs_batch))

    # Adjust results by removing self-matches
    for i in range(all_nbrs.shape[0]):
        if i != all_nbrs[i, 0]:
            for j, nbr in enumerate(all_nbrs[i, 1:], start=1):
                if i == nbr:
                    all_nbrs[i, 1:j + 1] = all_nbrs[i, :j]
                    all_sims[i, 1:j + 1] = all_sims[i, :j]
                    break

    all_sims = all_sims[:, 1:]
    all_nbrs = all_nbrs[:, 1:]

    knn_data = [(nbr.astype(np.uint32), sim.astype(np.float32)) for nbr, sim in zip(all_nbrs, all_sims)]
    np.savez_compressed(knn_output_path, data=np.array(knn_data))
    
    return all_nbrs, all_sims