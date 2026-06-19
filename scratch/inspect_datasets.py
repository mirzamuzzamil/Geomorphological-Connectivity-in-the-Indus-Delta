import numpy as np
for name in ["data/indus_dataset.npz", "data/indus_dataset_v2.npz"]:
    print(f"\nInspecting {name}...")
    try:
        data = np.load(name)
        for k in data.files:
            arr = data[k]
            print(f"  Key: {k}, Shape: {arr.shape}, Dtype: {arr.dtype}")
    except Exception as e:
        print(f"  Error loading: {e}")
