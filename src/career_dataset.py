import torch
from torch.utils.data import Dataset

class CareerDataset(Dataset):

    def __init__(self, X, y):

        # xử lý cả DataFrame và numpy
        if hasattr(X, "values"):
            X = X.values

        if hasattr(y, "values"):
            y = y.values

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]