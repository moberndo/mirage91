import numpy as np


def load_dataset(config: dict):
    X = np.load(config["data"]["dataset_path"], allow_pickle=True)

    labels = (np.array(X[:, 0])).astype(np.float32)
    X = np.array(X[:, 1].tolist())
    Label = labels

    X = X[:, :, 200 * 2:200 * 6]
    X = X[:, :, 0:-1:3]

    combined_indices = np.where(np.isin(Label, config["data"]["selected_classes"]))[0]
    # Use the combined indices to index X and Label
    X = X[combined_indices, :, :]
    Label = Label[combined_indices]

    c = np.random.permutation(X.shape[0])
    X = X[c, :, :]
    Label = Label[c]

    # Get unique values and create a mapping
    unique_values = np.unique(Label)
    mapping = {val: idx for idx, val in enumerate(unique_values)}

    # Apply the mapping to the Label array
    Label = np.array([mapping[val] for val in Label])

    return X, Label