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

    TARGET_ROOT = os.path.abspath(
        os.path.join(
            script_dir,
            "..", "..",
            "proof_generation", "ZIP_proof_generation", "ZIP_lookup", "examples"
        )
    )
    acts_path = os.path.join(script_dir, "lenet_inference_layer_inputs.json")