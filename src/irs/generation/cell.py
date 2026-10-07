"""Tang unit-cell normalization and one-sided angular amplitude weights."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]


@dataclass(frozen=True)
class TangCellModel:
    """Parameters of one rectangular reflecting cell."""

    width_m: float
    height_m: float
    reflection_amplitude: float = 0.9
    pattern_exponent: float = 3.0

    def __post_init__(self) -> None:
        if self.width_m <= 0.0 or self.height_m <= 0.0:
            raise ValueError("Cell width and height must be positive.")
        if not 0.0 <= self.reflection_amplitude <= 1.0:
            raise ValueError("reflection_amplitude must be in [0, 1].")
        if self.pattern_exponent < 0.0:
            raise ValueError("pattern_exponent cannot be negative.")

    @property
    def area_m2(self) -> float:
        """Return the physical area of one unit cell."""
        return self.width_m * self.height_m

    @property
    def ideal_gain_linear(self) -> float:
        """Return the ideal gain of a one-sided cosine-power pattern."""
        return 2.0 * (self.pattern_exponent + 1.0)

    def cascaded_amplitude_scale(self, wavelength_m: float) -> float:
        """Return the physical correction for two isotropic Sionna hops.

        The returned dimensionless scale includes the fixed reflection
        amplitude. It must be applied exactly once to the cascaded channel,
        not once per hop.
        """
        if wavelength_m <= 0.0:
            raise ValueError("wavelength_m must be positive.")
        return (
            self.reflection_amplitude
            * 2.0
            * sqrt(pi * self.ideal_gain_linear * self.area_m2)
            / wavelength_m
        )

    def path_amplitude_from_cosine(self, cosine: FloatArray) -> FloatArray:
        """Return ``sqrt(F)`` for direction cosines relative to the normal."""
        front_cosine: FloatArray = np.asarray(
            np.clip(cosine, 0.0, 1.0),
            dtype=np.float32,
        )
        return np.asarray(
            np.power(front_cosine, self.pattern_exponent / 2.0),
            dtype=np.float32,
        )

    def metadata(self, wavelength_m: float) -> dict[str, float | str]:
        """Return serializable parameters and the applied normalization."""
        return {
            "reference": "Tang et al., arXiv:1911.05326, Theorem 1",
            "power_pattern": (f"max(0, cos(theta))^{self.pattern_exponent:g}"),
            "cell_width_m": self.width_m,
            "cell_height_m": self.height_m,
            "cell_area_m2": self.area_m2,
            "reflection_amplitude": self.reflection_amplitude,
            "ideal_gain_linear": self.ideal_gain_linear,
            "wavelength_m": wavelength_m,
            "cascaded_amplitude_scale": self.cascaded_amplitude_scale(wavelength_m),
        }


def arrival_amplitude_weights(
    theta_rad: FloatArray,
    phi_rad: FloatArray,
    receiver_normals: FloatArray,
    model: TangCellModel,
) -> FloatArray:
    """Return IRS receive-side ``sqrt(F)`` weights with shape ``[R,T,L]``.

    Sionna's arrival angles point from the receiver toward the preceding path
    vertex. ``receiver_normals[r]`` is the outward normal of IRS receiver
    ``r``.
    """
    _validate_angles(theta_rad, phi_rad)
    normals: FloatArray = _validated_normals(
        receiver_normals,
        expected_count=theta_rad.shape[0],
    )
    directions: FloatArray = _directions(theta_rad, phi_rad)
    cosine: FloatArray = np.asarray(
        np.einsum("rtlc,rc->rtl", directions, normals),
        dtype=np.float32,
    )
    return model.path_amplitude_from_cosine(cosine)


def departure_amplitude_weights(
    theta_rad: FloatArray,
    phi_rad: FloatArray,
    transmitter_normals: FloatArray,
    model: TangCellModel,
) -> FloatArray:
    """Return IRS transmit-side ``sqrt(F)`` weights with shape ``[R,T,L]``.

    Sionna's departure angles point from the transmitter toward the following
    path vertex. ``transmitter_normals[t]`` is the outward normal of IRS
    transmitter ``t``.
    """
    _validate_angles(theta_rad, phi_rad)
    normals: FloatArray = _validated_normals(
        transmitter_normals,
        expected_count=theta_rad.shape[1],
    )
    directions: FloatArray = _directions(theta_rad, phi_rad)
    cosine: FloatArray = np.asarray(
        np.einsum("rtlc,tc->rtl", directions, normals),
        dtype=np.float32,
    )
    return model.path_amplitude_from_cosine(cosine)


def _directions(theta_rad: FloatArray, phi_rad: FloatArray) -> FloatArray:
    """Convert Sionna zenith/azimuth angles to Cartesian unit vectors."""
    sin_theta: FloatArray = np.asarray(np.sin(theta_rad), dtype=np.float32)
    return np.asarray(
        np.stack(
            (
                sin_theta * np.cos(phi_rad),
                sin_theta * np.sin(phi_rad),
                np.cos(theta_rad),
            ),
            axis=-1,
        ),
        dtype=np.float32,
    )


def _validate_angles(theta_rad: FloatArray, phi_rad: FloatArray) -> None:
    """Validate aligned Sionna path-angle tensors."""
    if theta_rad.ndim != 3 or phi_rad.ndim != 3:
        raise ValueError("Path angles must have shape [receiver, transmitter, path].")
    if theta_rad.shape != phi_rad.shape:
        raise ValueError("theta_rad and phi_rad must have matching shapes.")
    if not np.isfinite(theta_rad).all() or not np.isfinite(phi_rad).all():
        raise ValueError("Path angles must contain only finite values.")


def _validated_normals(normals: FloatArray, *, expected_count: int) -> FloatArray:
    """Validate and return unit surface normals."""
    values: FloatArray = np.asarray(normals, dtype=np.float32)
    if values.shape != (expected_count, 3):
        raise ValueError(f"normals must have shape ({expected_count}, 3).")
    if not np.isfinite(values).all():
        raise ValueError("normals must contain only finite values.")
    lengths: FloatArray = np.asarray(
        np.linalg.norm(values, axis=1),
        dtype=np.float32,
    )
    if not np.allclose(lengths, 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("Every surface normal must have unit length.")
    return values
