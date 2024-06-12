from model.InfoMapClustering import InfoMapClustering


if __name__ == '__main__':

    _info = InfoMapClustering()
    _info.adjust_probabilities()
    _info.perform_clustering()