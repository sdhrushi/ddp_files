import os
import numpy as np
from time import time
from tqdm import tqdm
import infomap
from utils.faiss_knn import faiss_knn
from configs import config
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')


def detect_outliers(delta_p, window_size):
    """
    Detect outliers in the transition probabilities.
    
    Args:
        delta_p (numpy.ndarray): Difference in probabilities.
        window_size (int): Size of the window to consider for outlier detection.
    
    Returns:
        numpy.ndarray: Indices of detected outliers.
    """
    omega = window_size
    z = np.zeros_like(delta_p, dtype=np.float32)
    for j in tqdm(range(delta_p.shape[1] - omega, -1, -1)):
        mu_test = np.mean(delta_p[:, j:j + omega], axis=1)
        mu_ref = np.mean(delta_p[:, j:], axis=1)
        sigma_ref = np.std(delta_p[:, j:], axis=1)
        q = j + (omega + 1) // 2
        z[:, q] = np.abs(mu_test - mu_ref) / sigma_ref
    q_star = np.argmax(z, axis=1)
    return q_star


class InfoMapClustering:
    def __init__(self):
        self.omega = config.window_size
        self.topK = config.topK
        self.knn_path = config.knn_path
        self.label_path = config.label_path
        self.feat_path = config.feat_path
        self.feat_dim = config.feat_dim
        self.result_path = config.result_path
        
        # Create necessary directories
        os.makedirs(os.path.dirname(self.knn_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.result_path), exist_ok=True)
        
        # Load KNN data
        self._load_knn()
        self.start_time = time()

    def _load_knn(self):
        """
        Load K-Nearest Neighbors (KNN) data.
        """
        start_time = time()
        if os.path.exists(self.knn_path):
            knn_data = np.load(self.knn_path)['data']
            if isinstance(knn_data, list):
                knn_data = np.array(knn_data)
            self.nbrs = knn_data[:, 0, :self.topK].astype(np.int32)
            self.sims = knn_data[:, 1, :self.topK].astype(np.float32)
        else:
            self.nbrs, self.sims = faiss_knn(self.feat_path, self.knn_path, self.feat_dim, self.topK)
        print(f'Time to load KNN: {time() - start_time:.2f}s')

    def compute_transition_probabilities(self, threshold=0.62):
        """
        Compute transition probabilities based on a similarity threshold.
        
        Args:
            threshold (float): Similarity threshold for considering links.
        """
        single_nodes, links, weights = [], [], []
        for i in tqdm(range(self.nbrs.shape[0])):
            for j, nbr in enumerate(self.nbrs[i]):
                if self.sims[i, j] >= threshold:
                    links.append((i, nbr))
                    weights.append(self.sims[i, j])
                else:
                    break
            if not links:
                single_nodes.append(i)
        self.links = np.array(links, dtype=np.uint32)
        self.weights = np.array(weights, dtype=np.float32)
        self.single_nodes = np.array(single_nodes, dtype=np.uint32)

    def adjust_probabilities(self):
        """
        Adjust transition probabilities using outlier detection.
        """
        probabilities = self.sims / np.sum(self.sims, axis=1, keepdims=True)
        delta_p = probabilities[:, :-1] - probabilities[:, 1:]
        outlier_indices = detect_outliers(delta_p, self.omega)
        print(f'Time for outlier detection: {time() - self.start_time:.2f}s')
        
        single_nodes, links, weights = [], [], []
        for i, cutoff in enumerate(outlier_indices):
            for j in self.nbrs[i, :cutoff + 1]:
                if i != j:
                    links.append((i, j))
                    weights.append(probabilities[i, j])
            if not links:
                single_nodes.append(i)
        self.links = np.array(links, dtype=np.uint32)
        self.weights = np.array(weights, dtype=np.float32)
        self.single_nodes = np.array(single_nodes, dtype=np.uint32)

    def perform_clustering(self):
        """
        Perform clustering using Infomap algorithm.
        """
        infomap_instance = infomap.Infomap("--two-level", flow_model='undirected')
        for (i, j), sim in tqdm(zip(self.links, self.weights)):
            infomap_instance.addLink(i, j, sim)
        
        # Free memory
        del self.links
        del self.weights

        infomap_instance.run(seed=100)
        label_to_indices = {}
        self.index_to_label = {}

        for node in infomap_instance.iterTree():
            if node.moduleIndex() not in label_to_indices:
                label_to_indices[node.moduleIndex()] = []
            label_to_indices[node.moduleIndex()].append(node.physicalId)

        for label, indices in label_to_indices.items():
            start_index = 2 if label == 0 else 1
            for idx in indices[start_index:]:
                self.index_to_label[idx] = label

        if len(self.single_nodes) > 0:
            new_label = len(label_to_indices)
            for node in self.single_nodes:
                if node not in self.index_to_label:
                    self.index_to_label[node] = new_label
                    label_to_indices[new_label] = [node]
                    new_label += 1

        print(f'Time for clustering: {time() - self.start_time:.2f}s')
        predicted_labels = np.full(len(self.index_to_label), -1)
        for idx, label in self.index_to_label.items():
            predicted_labels[idx] = label

        np.save(self.result_path, predicted_labels)