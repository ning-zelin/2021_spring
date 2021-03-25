import numpy as np
import torch
import random
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# setup seed
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(20)

# prepare data
train = np.load(r'D:\DeepLearning\Data\Lee\hw2\train_11.npy')
train_label = np.load(r'D:\DeepLearning\Data\Lee\hw2\train_label_11.npy')
test = np.load(r'D:\DeepLearning\Data\Lee\hw2\test_11.npy')

class TIMITDataset(Dataset):
    def __init__(self,X, y = None):
        self.data = torch.from_numpy(X).float()
        if y is not None:
            y = y.astype(np.int)
            self.label = torch.LongTensor(y)
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is None:
            return self.data[idx]
        else:
            return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)

VAL_RATIO = 0.2

percent = int(train.shape[0] * (1 - VAL_RATIO))
train_x, train_y , val_x, val_y = train[:percent], train_label[:percent],train[percent:], \
    train_label[percent:]

BATCH_SIZE = 64
train_set = TIMITDataset(train_x, train_y)
val_set = TIMITDataset(val_x, val_y)
train_loader = DataLoader(train_set, BATCH_SIZE, shuffle = True)
val_loader = DataLoader(val_set, BATCH_SIZE)

# design model
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.net1 = nn.Sequential(
            nn.Linear(429, 1024),
            nn.SiLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 2048),
            nn.SiLU(),
            nn.Dropout(0.4),
            nn.Linear(2048, 4096),
            nn.SiLU(),
            nn.Dropout(0.4),
            nn.Linear(4096, 1024),
            nn.SiLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.SiLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.SiLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 39)
        )

    def forward(self, x):
        return self.net1(x)



num_epoch = 20
learning_rate = 1e-4

model_path = r'./model.ckpt'
model = Classifier().cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

device = 'cuda:0'
best_acc = 0.0
for epoch in range(num_epoch):
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0
    # training
    model.train()  # set the model to training mode
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        batch_loss = criterion(outputs, labels)
        _, train_pred = torch.max(outputs, 1)  # get the index of the class with the highest probability
        batch_loss.backward()
        optimizer.step()
        train_acc += (train_pred.cpu() == labels.cpu()).sum().item()
        train_loss += batch_loss.item()
    # validation
    if len(val_set) > 0:
        model.eval()  # set the model to evaluation mode
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                batch_loss = criterion(outputs, labels)
                _, val_pred = torch.max(outputs, 1)
                val_acc += (
                        val_pred.cpu() == labels.cpu()).sum().item()  # get the index of the class with the highest probability
                val_loss += batch_loss.item()
            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                epoch + 1, num_epoch, train_acc / len(train_set), train_loss / len(train_loader),
                val_acc / len(val_set), val_loss / len(val_loader)
            ))
            # if the model improves, save a checkpoint at this epoch
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), model_path)
                print('saving model with acc {:.3f}'.format(best_acc / len(val_set)))
    else:
        print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
            epoch + 1, num_epoch, train_acc / len(train_set), train_loss / len(train_loader)
        ))
# if not validating, save the last epoch
if len(val_set) == 0:
    torch.save(model.state_dict(), model_path)
    print('saving model at last epoch')
