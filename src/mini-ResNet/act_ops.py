# act1_y_yprime_dump.py
import argparse
import os
import json
import numpy as np
import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)


def float64_to_hex(x: float) -> str:
    u = np.float64(x).view(np.uint64)
    return f"0x{int(u):016x}"


def main():
    script_dir = os.path.abspath(os.path.dirname(__file__))
    parser = argparse.ArgumentParser(
        description="activation for act*_input, dump y,y' in hex"
    )
    parser.add_argument("--key", required=True, help="Layer key, e.g. act1_input")
    TARGET_ROOT = os.path.abspath(
        os.path.join(
            script_dir,
            "..",
            "..",
            "proof_generation",
            "ZIP_proof_generation",
            "ZIP_lookup",
            "examples",
        )
    )
    args = parser.parse_args()
    layer_key = args.key
    acts_path = os.path.join(script_dir, "mini_resnet_inference_layer_inputs.json")
    layer_num = layer_key.replace("_input", "").replace("act_", "")
    out_dir = os.path.join(TARGET_ROOT, f"y_yprime_examples_mini_resnet_act_{layer_num}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"relu_y_yprime.txt")
    with open(acts_path) as f:
        acts = json.load(f)
    if layer_key not in acts:
        raise KeyError(
            f"Expected key '{layer_key}' in mini_resnet_inference_layer_inputs.json"
        )

    x_np = np.array(acts[layer_key], dtype=np.float64)
    x = torch.from_numpy(x_np).to(torch.float64)

    bound = 30.0
    y_prime = torch.clamp(x, min=0.0, max=bound)
    y = F.relu(x)

    def to_hex_pairs(y_np, yp_np):
        for yi, ypi in zip(y_np.reshape(-1), yp_np.reshape(-1)):
            u_y = np.float64(yi).view(np.uint64)
            u_yp = np.float64(ypi).view(np.uint64)
            yield f"0x{int(u_y):016x}, 0x{int(u_yp):016x}\n"

    y_np = y.cpu().numpy()
    yp_np = y_prime.cpu().numpy()
    with open(out_path, "w") as txt:
        txt.writelines(to_hex_pairs(y_np, yp_np))
    print(f"[{layer_key}] act='relu'  shape={tuple(x_np.shape)}  elements={x_np.size}")


if __name__ == "__main__":
    main()