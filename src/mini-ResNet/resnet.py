import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import argparse

torch.set_default_dtype(torch.float64)


class MiniResNet(nn.Module):
    def __init__(self, num_classes=10, act=nn.ReLU):
        super(MiniResNet, self).__init__()
        # initial Convolution Layer
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.act1 = act()
        self.pool1 = nn.AvgPool2d(2, 2)  # Reduce 32 x 32 -> 16 x 16

        # first residual block
        self.res1_conv1 = nn.Conv2d(16, 16, kernel_size=3, padding=1)  
        self.res1_act1 = act()
        self.res1_conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.res1_act2 = act()  # after skip connection

        # second residual block
        self.res2_conv1 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.res2_act1 = act()
        self.res2_conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.res2_act2 = act()  # after skip connection
        self.pool2 = nn.AvgPool2d(2, 2)  # Reduce 16 x 16 -> 8 x 8

        # fully connected layers
        self.fc1 = nn.Linear(16 * 8 * 8, 128)
        self.act2 = act()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # initial conv
        out = self.pool1(self.act1(self.conv1(x)))

        # first residual block
        res = out  # skip connection
        out = self.res1_act1(self.res1_conv1(out))
        out = self.res1_conv2(out)
        out += res  # skip connection
        out = self.res1_act2(out)

        # second residual block
        res = out
        out = self.res2_act1(self.res2_conv1(out))
        out = self.res2_conv2(out)
        out += res
        out = self.res2_act2(out)
        out = self.pool2(out)

        # classifier
        out = out.view(out.size(0), -1)  # flatten
        out = self.act2(self.fc1(out))
        out = self.fc2(out)
        return out


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}"
                f" ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
            )


def test(model, device, test_loader):
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    acc = 100.0 * correct / len(test_loader.dataset)
    print(
        f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)}"
        f" ({acc:.0f}%)\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mini ResNet Training")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    torch.manual_seed(args.seed)
    transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010),
            ),
        ]
    )
    train_data = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transforms
    )
    test_data = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transforms
    )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=use_cuda,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=use_cuda,
    )
    model = MiniResNet(num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    param_dict = {
        name: p.detach().cpu().numpy().tolist() for name, p in model.named_parameters()
    }
    out_name = f"mini_resnet_model_params.json"
    with open(out_name, "w") as f:
        json.dump(param_dict, f, indent=2)