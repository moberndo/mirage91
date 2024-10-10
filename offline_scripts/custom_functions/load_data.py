import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


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

def load_dataloader(x, y, device, config: dict, shuffle: bool = False) -> torch.utils.data.DataLoader:
    X_np = x

    X_mean = np.mean(X_np, axis=(0, 1, 2), keepdims=True)
    X_std = np.std(X_np, axis=(0, 1, 2), keepdims=True)
    X_normalized = (X_np - X_mean) / X_std

    x = torch.tensor(X_normalized, dtype=torch.float32)

    data = np.expand_dims(x, axis=1)
    img = (torch.from_numpy(data).to(device)).type(torch.float32)
    label = (torch.from_numpy(y).to(device)).type(torch.long)
    dataset = torch.utils.data.TensorDataset(img, label)

    class_counts = torch.bincount(label)
    class_weights = 1. / class_counts.float()
    sample_weights = class_weights[label]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    return torch.utils.data.DataLoader(dataset=dataset, sampler=sampler, batch_size=config["training"]["batch_size"])
