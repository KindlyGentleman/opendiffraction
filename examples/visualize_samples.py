"""Example: Visualize paired OpenDiffraction samples."""

import os
import h5py
import matplotlib.pyplot as plt
import numpy as np


def visualize_sample(
    h5_path: str,
    index: int = 0,
    output_path: str = "sample_preview.png",
) -> None:
    """Plot sample: ground truth, measured diffraction, and mask."""
    with h5py.File(h5_path, "r") as f:
        inp = f["input"][index]
        tgt = f["target"][index]
        msk = f["mask"][index]

    if inp.ndim == 3:
        inp = inp[0]
    if tgt.ndim == 3:
        tgt = tgt[0]
    if msk.ndim == 3:
        msk = msk[0]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), dpi=150)

    axes[0].imshow(tgt, cmap="viridis", origin="lower")
    axes[0].set_title("Ground Truth Object $O(r)$", fontsize=10)

    axes[1].imshow(np.log1p(np.maximum(inp, 0.0)), cmap="inferno", origin="lower")
    axes[1].set_title(r"Measured Diffraction $\log(1+I)$", fontsize=10)

    axes[2].imshow(msk, cmap="gray", origin="lower")
    axes[2].set_title("Validity Mask (1=Valid, 0=Defect)", fontsize=10)

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Sample preview saved to {output_path}")


if __name__ == "__main__":
    shard = "output_quick/data/train_000.h5"
    if os.path.exists(shard):
        visualize_sample(shard, index=0)
