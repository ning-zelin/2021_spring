# Import necessary packages.
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset
from torchvision.datasets import DatasetFolder
from tqdm.auto import tqdm

train_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    transforms.Resize((128, 128)),

    transforms.ToTensor(),
    transforms.Normalize(0, 1)
])

test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(0, 1)
])

batch_size = 1

train_set = DatasetFolder(r"D:\DeepLearning\Data\food-11\food-11\training\labeled", loader = lambda x: Image.open(x),
                          extensions = "jpg", transform = train_tfm)
valid_set = DatasetFolder(r"D:\DeepLearning\Data\food-11\food-11\training\labeled", loader = lambda x: Image.open(x),
                          extensions = "jpg", transform = test_tfm)
unlabeled_set = DatasetFolder(r"D:\DeepLearning\Data\food-11\food-11\training\unlabeled\0_1",
                              loader = lambda x: Image.open(x), extensions = "jpg", transform = train_tfm)
test_set = DatasetFolder(r"D:\DeepLearning\Data\food-11\food-11\testing", loader = lambda x: Image.open(x),
                         extensions = "jpg", transform = test_tfm)

train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True, )
valid_loader = DataLoader(valid_set, batch_size = batch_size, shuffle = True, )
test_loader = DataLoader(test_set, batch_size = batch_size, shuffle = False)


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # The arguments for commonly used modules:
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)

        # input image size: [3, 128, 128]
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(4, 4, 0),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 11)
        )

    def forward(self, x):
        x = self.cnn_layers(x)

        x = x.flatten(1)

        x = self.fc_layers(x)
        return x


class contact_set(torch.utils.data.Dataset):
    def __init__(self, old_set, new_set):
        self.old_set = old_set
        self.new_set = new_set

    def __len__(self):
        return len(self.old_set) + len(self.new_set)

    def __getitem__(self, idx):
        if idx < len(self.old_set):
            return self.old_set[idx]
        else:
            return self.new_set[idx - len(self.old_set)]


def get_pseudo_label(old_set, unlabeled_set, model, batch_size = 350, threshold = 0.8):
    data_loader = DataLoader(unlabeled_set, batch_size)
    model.eval()
    softmax = nn.Softmax(-1)
    for img_batch, _ in tqdm(data_loader):
        with torch.no_grad():
            logits = model(img_batch.cuda()).cpu()
        probs = softmax(logits)
        probs_max_bs_1, pos_max_bs_1 = probs.max(-1)
        bool_index = probs_max_bs_1 >= threshold
        new_data = img_batch[bool_index]
        new_label = pos_max_bs_1[bool_index]
        new_set = TensorDataset(new_data, new_label)
        old_set = contact_set(old_set, new_set)
    return new_set


device = "cuda" if torch.cuda.is_available() else "cpu"

model = Classifier().to(device)
model.device = device

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr = 0.0003, weight_decay = 1e-5)

n_epochs = 100

for epoch in range(n_epochs):

    if epoch >= 0:
        new_set = get_pseudo_label(train_loader, unlabeled_set, model)
        concat_dataset = ConcatDataset([train_set, new_set])

        train_loader = DataLoader(concat_dataset, batch_size = 10)

    model.train()

    train_loss = []
    train_accs = []

    for batch in tqdm(train_loader):
        imgs, labels = batch

        logits = model(imgs.to(device))

        loss = criterion(logits, labels.to(device))

        optimizer.zero_grad()

        loss.backward()

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm = 10)

        optimizer.step()

        acc = (logits.argmax(dim = -1) == labels.to(device)).float().mean()

        train_loss.append(loss.item())
        train_accs.append(acc)

    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)

    print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

    model.eval()

    valid_loss = []
    valid_accs = []

    for batch in tqdm(valid_loader):
        imgs, labels = batch

        with torch.no_grad():
            logits = model(imgs.to(device))

        loss = criterion(logits, labels.to(device))

        acc = (logits.argmax(dim = -1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        valid_loss.append(loss.item())
        valid_accs.append(acc)

    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)

    print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

model.eval()

predictions = []

for batch in tqdm(test_loader):
    imgs, labels = batch

    with torch.no_grad():
        logits = model(imgs.to(device))

    predictions.extend(logits.argmax(dim = -1).cpu().numpy().tolist())

with open("predict.csv", "w") as f:
    f.write("Id,Category\n")

    for i, pred in enumerate(predictions):
        f.write(f"{i},{pred}\n")
