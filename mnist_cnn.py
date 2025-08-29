# mnist_cnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path

# ==== CNN Modeli ====
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))          # [B,32,26,26]
        x = F.relu(self.conv2(x))          # [B,64,24,24]
        x = F.max_pool2d(x, 2)             # [B,64,12,12]
        x = self.dropout1(x)
        x = torch.flatten(x, 1)            # [B,9216]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def get_loaders(batch_train=64, batch_test=1000):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = datasets.MNIST("data", train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST("data", train=False, download=True, transform=transform)
    return (
        DataLoader(train_ds, batch_size=batch_train, shuffle=True),
        DataLoader(test_ds,  batch_size=batch_test)
    )

def train_and_evaluate(epochs=3, lr=1e-3, device=None, save_path="models/mnist_cnn.pt"):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_loader, test_loader = get_loaders()

    def _train(epoch):
        model.train()
        for idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = F.nll_loss(out, target)
            loss.backward()
            optimizer.step()
            if idx % 200 == 0:
                print(f"Train Epoch: {epoch} [{idx*len(data)}/{len(train_loader.dataset)}]  Loss: {loss.item():.6f}")

    def _test():
        model.eval()
        loss_sum = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                out = model(data)
                loss_sum += F.nll_loss(out, target, reduction="sum").item()
                pred = out.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        loss = loss_sum / len(test_loader.dataset)
        acc = 100.0 * correct / len(test_loader.dataset)
        print(f"\nTest set: Avg loss: {loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({acc:.2f}%)\n")

    for ep in range(1, epochs + 1):
        _train(ep)
        _test()

    Path("models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model kaydedildi: {save_path}")

if __name__ == "__main__":
    # Sadece bu dosyayı doğrudan çalıştırırsan eğitim yapılır.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_and_evaluate(device=device)
