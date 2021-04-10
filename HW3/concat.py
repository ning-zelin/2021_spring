import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
x_1 = torch.randn(12, 100) ; y_1 = torch.randn(12)
x_2 = torch.randn(12, 100) ; y_2 = torch.randn(12)
class Data_1(Dataset):
    def __init__(self):
        self.features = x_1
        self.targets = y_1

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

    def __len__(self):
        return len(self.targets)
class Data_2(Dataset):
    def __init__(self):
        self.features = x_2
        self.targets = y_2

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

    def __len__(self):
        return len(self.targets)
data_1 = Data_1()
data_2 = Data_2()

Contact_data = ConcatDataset([data_1, data_2])
loader = DataLoader(Contact_data, batch_size = 1, shuffle = False)
for x, y in loader:
    print(x.shape, y.shape)
    print(len(loader))
    break
