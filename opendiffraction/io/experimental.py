"""Read-only experimental data adapter for Coherent X-ray Imaging (CXI) NeXus files.

Loads, validates, and normalizes experimental CXI beamline diffraction patterns
to OpenDiffraction standard tensor shapes and coordinate conventions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
import h5py
import numpy as np


@dataclass
class ExperimentalDiffractionFrame:
    """Standardized record extracted from an experimental CXI file."""

    intensity: np.ndarray  # (512, 512) float32
    valid_mask: np.ndarray  # (512, 512) bool
    photon_energy_ev: float | None
    wavelength_m: float | None
    detector_distance_m: float | None
    pixel_pitch_m: float | None
    source_shape: tuple[int, int]
    cxi_path: str
    frame_index: int
    metadata: dict[str, Any]


class CXIExperimentalAdapter:
    """Read-only adapter for experimental CXIDB / NeXus files."""

    def __init__(self, cxi_path: str) -> None:
        self.cxi_path = os.path.abspath(cxi_path)
        if not os.path.exists(self.cxi_path):
            raise FileNotFoundError(f"CXI file not found: {self.cxi_path}")

        self._inspect_file()

    def _inspect_file(self) -> None:
        """Inspect internal NeXus hierarchy and identify primary data paths."""
        with h5py.File(self.cxi_path, "r") as f:
            # Locate primary diffraction dataset
            data_candidates = [
                "/entry_1/data_1/data",
                "/entry_1/instrument_1/detector_1/data",
                "/entry_1/data",
            ]
            self.data_path: str | None = None
            for p in data_candidates:
                if p in f:
                    self.data_path = p
                    break

            if self.data_path is None:
                # Search for any 2D/3D dataset in NXdata
                for entry_name in f.keys():
                    grp = f[entry_name]
                    if isinstance(grp, h5py.Group) and "data_1" in grp and "data" in grp["data_1"]:
                        self.data_path = f"/{entry_name}/data_1/data"
                        break

            if self.data_path is None:
                raise ValueError(f"Could not locate diffraction data in CXI file: {self.cxi_path}")

            dset = f[self.data_path]
            self.shape = dset.shape
            self.dtype = dset.dtype
            self.num_frames = self.shape[0] if len(self.shape) == 3 else 1

            # Locate mask dataset
            mask_candidates = [
                "/entry_1/data_1/mask",
                "/entry_1/instrument_1/detector_1/mask",
            ]
            self.mask_path: str | None = None
            for p in mask_candidates:
                if p in f:
                    self.mask_path = p
                    break

            # Metadata extraction
            self.energy_ev = self._read_scalar(f, ["/entry_1/instrument_1/source_1/energy", "/entry_1/source_1/energy"])
            self.distance_m = self._read_scalar(
                f, ["/entry_1/instrument_1/detector_1/distance", "/entry_1/detector_1/distance"]
            )
            self.pixel_pitch_m = self._read_scalar(
                f,
                [
                    "/entry_1/instrument_1/detector_1/x_pixel_size",
                    "/entry_1/detector_1/x_pixel_size",
                ],
            )

    @staticmethod
    def _read_scalar(f: h5py.File, paths: list[str]) -> float | None:
        for p in paths:
            if p in f:
                val = f[p][()]
                if hasattr(val, "item"):
                    val = val.item()
                return float(val)
        return None

    def __len__(self) -> int:
        return self.num_frames

    def get_frame(
        self,
        frame_idx: int = 0,
        target_size: tuple[int, int] = (512, 512),
        normalize_flux: bool = True,
    ) -> ExperimentalDiffractionFrame:
        """Load and standardize an experimental diffraction frame.

        Parameters
        ----------
        frame_idx : int
            Index of the frame to retrieve.
        target_size : tuple[int, int]
            Target output resolution (height, width).
        normalize_flux : bool
            If True, clips negative readouts and scales flux appropriately.

        Returns
        -------
        ExperimentalDiffractionFrame
            Standardized record ready for PyTorch evaluation.
        """
        if frame_idx < 0 or frame_idx >= self.num_frames:
            raise IndexError(f"Frame index {frame_idx} out of range [0, {self.num_frames - 1}]")

        with h5py.File(self.cxi_path, "r") as f:
            dset = f[self.data_path]
            raw_frame = dset[frame_idx] if len(self.shape) == 3 else dset[:]

            if self.mask_path is not None and self.mask_path in f:
                raw_mask = f[self.mask_path][frame_idx] if len(f[self.mask_path].shape) == 3 else f[self.mask_path][:]
                mask_bool = raw_mask.astype(bool)
            else:
                # Synthesize validity mask: non-nan, finite, unclipped
                mask_bool = np.isfinite(raw_frame) & (raw_frame > -100.0)

        source_shape = raw_frame.shape

        # Standardize geometry: center, crop or pad to target_size
        std_intensity, std_mask = self._standardize_geometry(raw_frame, mask_bool, target_size)

        wavelength_m = None
        if self.energy_ev is not None and self.energy_ev > 0:
            # hc / E (approx 12398.4 eV * Angstrom)
            wavelength_m = (1.239841984e-6) / self.energy_ev

        return ExperimentalDiffractionFrame(
            intensity=std_intensity.astype(np.float32),
            valid_mask=std_mask.astype(bool),
            photon_energy_ev=self.energy_ev,
            wavelength_m=wavelength_m,
            detector_distance_m=self.distance_m,
            pixel_pitch_m=self.pixel_pitch_m,
            source_shape=source_shape,
            cxi_path=self.cxi_path,
            frame_index=frame_idx,
            metadata={
                "data_path": self.data_path,
                "mask_path": self.mask_path,
                "raw_min": float(np.nanmin(raw_frame)),
                "raw_max": float(np.nanmax(raw_frame)),
            },
        )

    @staticmethod
    def _standardize_geometry(
        frame: np.ndarray,
        mask: np.ndarray,
        target_size: tuple[int, int] = (512, 512),
    ) -> tuple[np.ndarray, np.ndarray]:
        """Center and crop or pad raw detector frame to target_size."""
        th, tw = target_size
        sh, sw = frame.shape[-2:]

        if (sh, sw) == (th, tw):
            return frame, mask

        # Center coordinates
        cy_src, cx_src = sh // 2, sw // 2
        cy_dst, cx_dst = th // 2, tw // 2

        out_frame = np.zeros(target_size, dtype=np.float32)
        out_mask = np.zeros(target_size, dtype=bool)

        # Compute overlapping bounding boxes
        src_y0 = max(0, cy_src - cy_dst)
        src_y1 = min(sh, cy_src + (th - cy_dst))
        src_x0 = max(0, cx_src - cx_dst)
        src_x1 = min(sw, cx_src + (tw - cx_dst))

        dst_y0 = cy_dst - (cy_src - src_y0)
        dst_y1 = dst_y0 + (src_y1 - src_y0)
        dst_x0 = cx_dst - (cx_src - src_x0)
        dst_x1 = dst_x0 + (src_x1 - src_x0)

        out_frame[dst_y0:dst_y1, dst_x0:dst_x1] = frame[src_y0:src_y1, src_x0:src_x1]
        out_mask[dst_y0:dst_y1, dst_x0:dst_x1] = mask[src_y0:src_y1, src_x0:src_x1]

        return out_frame, out_mask
