
window_size = 20
topK = 256

feat_dim = 256
dataset = 'eurosat'
test_name = 'part_test'

knn_path = './data/{}/knns/{}_faiss_top{}.npz'.format(dataset, test_name, topK)
feat_path = './data/{}/features/{}.bin'.format(dataset, test_name)
label_path = './data/{}/labels/{}.meta'.format(dataset, test_name)
result_path = './result/{}/part1_test_top{}_winds{}.npy'.format(dataset, topK, window_size)
