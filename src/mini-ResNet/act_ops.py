# act1_y_yprime_dump.py
import argparse
import os
import json
import math
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
            "..",  # src/
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
    # For GELU we log y = gelu(x) and y' = clamp(x, [-5, 5])
    # to match the NFGen approximation domain used for the tables.
    out_path = os.path.join(out_dir, f"gelu_y_yprime.txt")
    with open(acts_path) as f:
        acts = json.load(f)
    if layer_key not in acts:
        raise KeyError(
            f"Expected key '{layer_key}' in mini_resnet_inference_layer_inputs.json"
        )

    x_np = np.array(acts[layer_key], dtype=np.float64)
    x = torch.from_numpy(x_np).to(torch.float64)

    # For GELU, use the clipped input as y' and GELU(y') as y.
    # Clamp to (-5, 5) so all y' lie in the approximation range
    # that the NFGen tables were generated for.
    bound = 5.0
    y_prime = torch.clamp(x, min=-bound, max=bound)
    # Evaluate GELU on the clamped input using PyTorch's
    # tanh-based approximation so that the proof circuit
    # compares against the same function family.
    y = F.gelu(y_prime, approximate="tanh")

    def to_hex_pairs(y_np, yp_np):
        for yi, ypi in zip(y_np.reshape(-1), yp_np.reshape(-1)):
            u_y = np.float64(yi).view(np.uint64)
            u_yp = np.float64(ypi).view(np.uint64)
            yield f"0x{int(u_y):016x}, 0x{int(u_yp):016x}\n"

    y_np = y.cpu().numpy()
    yp_np = y_prime.cpu().numpy()
    with open(out_path, "w") as txt:
        txt.writelines(to_hex_pairs(y_np, yp_np))
    print(f"[{layer_key}] act='gelu'  shape={tuple(x_np.shape)}  elements={x_np.size}")


if __name__ == "__main__":
    main()