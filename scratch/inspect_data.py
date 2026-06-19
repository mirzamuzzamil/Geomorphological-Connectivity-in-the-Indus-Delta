import numpy as np

data = np.load('data/indus_dataset_v2.npz')
print("Keys:", list(data.keys()))
for k in data.keys():
    print(k, data[k].shape)
