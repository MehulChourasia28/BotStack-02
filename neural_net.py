import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

df = pd.read_csv(r"/kaggle/input/dataset/battleship_dataset.csv")
arr = df.values.astype('float32')
boards = arr[:, :100].reshape(-1, 10, 10)
ships  = arr[:, 100:].reshape(-1, 10, 10)

tensor_boards = torch.from_numpy(boards).unsqueeze(1)
tensor_ships  = torch.from_numpy(ships).unsqueeze(1)

dataset = TensorDataset(tensor_boards, tensor_ships)
train_set, test_set = torch.utils.data.random_split(dataset, [45000, 5000])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_set,  batch_size=32, shuffle=False)

class CNN(nn.Module):
    def __init__(self, in_channels=1, base_filters=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, base_filters,    kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(base_filters, base_filters*2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(base_filters*2, base_filters, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(base_filters, 1,              kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return self.conv4(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = CNN().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 10
threshold = 0.5               # decisions above → ship

for epoch in range(1, num_epochs+1):
    # ---- TRAINING ----
    model.train()
    running_loss = 0.0
    train_correct = 0
    train_total   = 0

    for i, (inputs, labels) in enumerate(train_loader, 1):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # compute batch accuracy
        preds = (outputs > threshold).float()
        train_correct += (preds == labels).sum().item()
        train_total   += labels.numel()

        if i % 100 == 0:
            avg_loss = running_loss / 100
            running_loss = 0.0
            print(f"[Epoch {epoch} Batch {i:4d}]  loss: {avg_loss:.4f}")

    train_acc = train_correct / train_total
    # ---- VALIDATION / TEST ----
    model.eval()
    test_correct = 0
    test_total   = 0
    test_loss    = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            test_loss += criterion(outputs, labels).item() * inputs.size(0)

            preds = (outputs > threshold).float()
            test_correct += (preds == labels).sum().item()
            test_total   += labels.numel()

    test_loss /= len(test_loader.dataset)
    test_acc  = test_correct / test_total

    print(f"Epoch {epoch:2d} ▶ train_acc: {train_acc*100:5.2f}%  "
          f"test_acc: {test_acc*100:5.2f}%  test_loss: {test_loss:.4f}")

print("Finished Training")

torch.save(model.state_dict(),"model.pt")


# redefine test_loader for single-sample batches
test_loader_one = DataLoader(test_set, batch_size=1, shuffle=True)

model.eval()
with torch.no_grad():
    # get exactly one (input, label) pair
    sample_input, sample_label = next(iter(test_loader_one))
    sample_input  = sample_input.to(device)   # shape [1,1,10,10]
    sample_label  = sample_label.to(device)

    # forward + threshold
    raw_output = model(sample_input)          # [1,1,10,10]
    pred_map   = (raw_output > threshold).float()

# convert to numpy
inp_np   = sample_input.cpu().squeeze().numpy()   # [10,10]
label_np = sample_label.cpu().squeeze().numpy()   # [10,10]
pred_np  = pred_map.cpu().squeeze().numpy()       # [10,10]

# print arrays
print("Input board:")
print(inp_np.astype(int))
print("\nActual ships:")
print(label_np.astype(int))
print("\nPredicted ships:")
print(pred_np.astype(int))