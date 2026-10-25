"""
Efficiency numbers for the paper: parameters, FLOPs, latency and peak memory per crop.

    python src/profile_model.py --config configs/hybrid.yaml            # GPU if available
    python src/profile_model.py --config configs/hybrid.yaml --device cpu

Latency = median of --runs forward passes after --warmup passes, batch 1, one crop
of data.crop_size. Report the device name printed here next to the numbers.
FLOPs need `pip install fvcore` (reported as fvcore counts multiply-adds).
"""
import argparse
import platform
import time

import numpy as np
import torch
import yaml

from model_factory import build_model


def main(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    roi = tuple(cfg["data"]["crop_size"])
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(cfg["model"], roi).to(device).eval()
    x = torch.randn(1, 4, *roi, device=device)

    params = sum(p.numel() for p in model.parameters())
    dev_name = torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor() or "CPU"
    print(f"model={cfg['model']['name']} crop={roi} device={dev_name}")
    print(f"parameters: {params / 1e6:.2f} M")

    try:
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(model, x)
        flops.unsupported_ops_warnings(False)
        print(f"GMACs (fvcore): {flops.total() / 1e9:.1f}")
    except ImportError:
        print("GMACs: install fvcore to count")

    with torch.no_grad():
        for _ in range(args.warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(args.runs):
            t0 = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    print(f"latency per crop: median {np.median(times):.1f} ms (IQR {np.percentile(times, 25):.1f}-{np.percentile(times, 75):.1f})")
    if device.type == "cuda":
        print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--runs", type=int, default=50)
    main(p.parse_args())
