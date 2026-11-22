"""Measure hardware throughput for SimCLR pretraining on CIFAR-10."""
import time
import torch
from src.simclr.data import get_pretrain_dataloader
from src.simclr.loss import NTXentLoss
from src.simclr.model import SimCLRModel


def benchmark_throughput():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "Apple Silicon GPU (MPS)"
    else:
        device = torch.device("cpu")
        device_name = "CPU"

    print(f"Device: {device} ({device_name})")

    batch_size = 256
    # Measure on 10% subset (~4,500 images, ~17 batches) to get precise timing
    loader = get_pretrain_dataloader(
        batch_size=batch_size,
        subset_fraction=0.10,
        num_workers=2,
    )

    model = SimCLRModel().to(device)
    criterion = NTXentLoss(temperature=0.5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    model.train()
    # Warmup 2 batches
    it = iter(loader)
    for _ in range(2):
        v1, v2 = next(it)
        v1, v2 = v1.to(device), v2.to(device)
        _, z1 = model(v1)
        _, z2 = model(v2)
        loss = criterion(z1, z2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if device.type == "mps":
        torch.mps.synchronize()

    start_time = time.time()
    total_samples = 0
    num_measured_batches = 0

    for v1, v2 in loader:
        v1, v2 = v1.to(device), v2.to(device)
        _, z1 = model(v1)
        _, z2 = model(v2)
        loss = criterion(z1, z2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_samples += v1.size(0)
        num_measured_batches += 1

    if device.type == "mps":
        torch.mps.synchronize()

    elapsed = time.time() - start_time
    imgs_per_sec = total_samples / elapsed
    time_per_full_epoch = 45000 / imgs_per_sec

    print(f"Measured: {total_samples} samples across {num_measured_batches} batches in {elapsed:.2f}s")
    print(f"Throughput: {imgs_per_sec:.1f} images/sec")
    print(f"Estimated time per full epoch (45,000 images): {time_per_full_epoch:.2f} seconds")
    print(f"Estimated 100 epochs on full dataset: {(time_per_full_epoch * 100) / 60:.1f} minutes")
    print(f"Estimated 200 epochs on full dataset: {(time_per_full_epoch * 200) / 3600:.2f} hours")


if __name__ == "__main__":
    benchmark_throughput()
