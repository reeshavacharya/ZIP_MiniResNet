import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

torch.set_default_dtype(torch.float64)


class MiniResNet(nn.Module):
    def __init__(self, num_classes=10, act=nn.ReLU):
        super(MiniResNet, self).__init__()
        # initial Convolution Layer
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.act1 = act()
        self.pool1 = nn.MaxPool2d(2, 2)  # Reduce 32 x 32 -> 16 x 16

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
        self.pool2 = nn.MaxPool2d(2, 2)  # Reduce 16 x 16 -> 8 x 8

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


def main():
    p = argparse.ArgumentParser(
        description="Inference + layer-input dump for mini-ResNet on CIFAR10 (fp64)"
    )
    args = p.parse_args()
    params_path = f"mini_resnet_model_params.json"
    out_path = f"mini_resnet_inference_layer_inputs.json"
    try:
        with open(params_path, "r") as f:
            params = json.load(f)
    except FileNotFoundError:
        print(f"Parameter file {params_path} not found.")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MiniResNet(act=nn.ReLU).to(device).double()
    state = model.state_dict()
    for name in list(state.keys()):
        if name not in params:
            raise KeyError(f"Missing key in params JSON: {name}")
        arr=np.array(params[name], dtype=np.float64)
        state[name] = torch.from_numpy(arr).to(device)
    model.load_state_dict(state)
    print("Model parameters loaded successfully.")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_dataset = datasets.CIFAR10("./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)
    image, label = next(iter(test_loader))
    image = image.to(device).double()
    activations = {}

    def pre_hook(name):
        def _h(module, inp):
            activations[name] = inp[0].detach().cpu().numpy().tolist()
        return _h
    
    model.conv1.register_forward_pre_hook(pre_hook("conv1_input"))
    model.act1.register_forward_pre_hook(pre_hook("act1_input"))
    model.pool1.register_forward_pre_hook(pre_hook("pool1_input"))

    model.res1_conv1.register_forward_pre_hook(pre_hook("res1_conv1_input"))
    model.res1_act1.register_forward_pre_hook(pre_hook("res1_act1_input"))
    model.res1_conv2.register_forward_pre_hook(pre_hook("res1_conv2_input"))
    model.res1_act2.register_forward_pre_hook(pre_hook("res1_act2_input"))

    model.res2_conv1.register_forward_pre_hook(pre_hook("res2_conv1_input"))
    model.res2_act1.register_forward_pre_hook(pre_hook("res2_act1_input"))
    model.res2_conv2.register_forward_pre_hook(pre_hook("res2_conv2_input"))
    model.res2_act2.register_forward_pre_hook(pre_hook("res2_act2_input"))
    model.pool2.register_forward_pre_hook(pre_hook("pool2_input"))
    
    model.fc1.register_forward_pre_hook(pre_hook("fc1_input"))
    model.act2.register_forward_pre_hook(pre_hook("act2_input"))
    model.fc2.register_forward_pre_hook(pre_hook("fc2_input"))

    model.eval()
    with torch.no_grad():
        out = model(image)
    pred = out.argmax(dim=1).item()
    meta = {
        "_meta": {
            "activation": 'relu',
            "dtype": "float64",
            "pred": int(pred),
            "label": int(label.item())
        }
    }
    meta.update(activations)
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved layer-by-layer inputs to '{out_path}'.")

if __name__ == "__main__":
    main()
