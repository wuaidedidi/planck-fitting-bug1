"""
黑体辐射光谱实验数据管理模块

管理三组不同温度黑体辐射源的光谱实验数据，提供数据校验、加载与预处理功能。
辐射源:
  - 源 1: 白炽灯丝 (~3000 K)
  - 源 2: 中温恒星 (~4500 K)
  - 源 3: 太阳表面 (~6000 K)
"""

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, field_validator

from src.utils.logger import get_logger

logger = get_logger("data.spectrum")


class SpectralDataPoint(BaseModel):
    """单个光谱数据点的验证模型"""

    wavelength_um: float
    spectral_radiance: float

    @field_validator("wavelength_um")
    @classmethod
    def validate_wavelength(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"波长必须为正值，当前值: {v} μm")
        if v < 0.1 or v > 100:
            raise ValueError(f"波长超出合理范围 (0.1-100 μm)，当前值: {v} μm")
        return v

    @field_validator("spectral_radiance")
    @classmethod
    def validate_radiance(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"光谱辐射度必须为正值，当前值: {v}")
        return v


@dataclass
class PhysicalConstants:
    """基本物理常数"""

    h: float = 6.62607015e-34       # Planck 常数，J·s
    c: float = 2.99792458e8         # 光速，m/s
    k_B: float = 1.380649e-23       # Boltzmann 常数，J/K
    b_wien: float = 2.8977729e-3    # Wien 位移常数，m·K
    sigma: float = 5.670374419e-8   # Stefan-Boltzmann 常数，W/(m²·K⁴)

    def validate(self) -> None:
        """校验物理常数的合理性"""
        if self.h <= 0:
            raise ValueError(f"Planck 常数必须为正值: {self.h}")
        if self.c <= 0:
            raise ValueError(f"光速必须为正值: {self.c}")
        if self.k_B <= 0:
            raise ValueError(f"Boltzmann 常数必须为正值: {self.k_B}")
        if self.sigma <= 0:
            raise ValueError(f"Stefan-Boltzmann 常数必须为正值: {self.sigma}")
        logger.info(
            "物理常数校验通过: h=%.3e, c=%.3e, k_B=%.3e, σ=%.3e",
            self.h, self.c, self.k_B, self.sigma,
        )

    @property
    def C1(self) -> float:
        """第一辐射常数 2hc² (W·m²)"""
        return 2.0 * self.h * self.c ** 2

    @property
    def C2(self) -> float:
        """第二辐射常数 hc/k_B (m·K)"""
        return self.h * self.c / self.k_B


@dataclass
class RadiationSource:
    """单个辐射源数据集"""

    source_id: str
    description: str
    data_points: List[SpectralDataPoint] = field(default_factory=list)

    @property
    def wavelengths(self) -> np.ndarray:
        """返回波长数组 (μm)"""
        return np.array([p.wavelength_um for p in self.data_points])

    @property
    def radiances(self) -> np.ndarray:
        """返回光谱辐射度数组 (W/(m²·sr·μm))"""
        return np.array([p.spectral_radiance for p in self.data_points])

    @property
    def size(self) -> int:
        """数据点数量"""
        return len(self.data_points)

    def to_dataframe(self) -> pd.DataFrame:
        """将数据集转换为 Pandas DataFrame"""
        return pd.DataFrame({
            "wavelength_um": self.wavelengths,
            "spectral_radiance": self.radiances,
            "inv_wavelength": 1.0 / self.wavelengths,
            "ln_radiance": np.log10(self.radiances),
        })


@dataclass
class SpectralDataset:
    """黑体辐射光谱数据集（包含多个辐射源）"""

    sources: List[RadiationSource] = field(default_factory=list)

    @staticmethod
    def load_default() -> "SpectralDataset":
        """
        加载三组黑体辐射源的默认实验数据。

        数据基于 Planck 辐射定律生成的模拟实验值（含测量噪声）。
        - 源 1: 白炽灯丝 (~3000 K, ε~0.92)
        - 源 2: 中温恒星 (~4500 K, ε~0.85)
        - 源 3: 太阳表面 (~6000 K, ε~0.95)

        Returns:
            包含完整实验数据的 SpectralDataset 实例
        """
        source_1_data: List[Tuple[float, float]] = [
            (0.50, 2.342e+05),
            (0.80, 8.527e+05),
            (1.00, 9.187e+05),
            (1.20, 8.103e+05),
            (1.50, 6.241e+05),
            (2.00, 3.501e+05),
            (2.50, 1.978e+05),
            (3.00, 1.158e+05),
            (4.00, 4.728e+04),
            (5.00, 2.194e+04),
            (6.00, 1.171e+04),
            (7.00, 6.758e+03),
            (8.00, 4.151e+03),
            (9.00, 2.614e+03),
            (10.0, 1.818e+03),
        ]

        source_2_data: List[Tuple[float, float]] = [
            (0.30, 9.93e+05),
            (0.40, 3.42e+06),
            (0.50, 5.52e+06),
            (0.60, 6.41e+06),
            (0.70, 6.35e+06),
            (0.80, 5.81e+06),
            (1.00, 4.33e+06),
            (1.20, 3.08e+06),
            (1.50, 1.82e+06),
            (2.00, 8.08e+05),
            (2.50, 4.05e+05),
            (3.00, 2.24e+05),
            (4.00, 8.19e+04),
            (5.00, 3.65e+04),
            (6.00, 1.87e+04),
        ]

        source_3_data: List[Tuple[float, float]] = [
            (0.20, 2.24e+06),
            (0.30, 1.59e+07),
            (0.35, 2.32e+07),
            (0.40, 2.81e+07),
            (0.45, 2.97e+07),
            (0.50, 3.04e+07),
            (0.55, 2.92e+07),
            (0.60, 2.71e+07),
            (0.70, 2.25e+07),
            (0.80, 1.80e+07),
            (1.00, 1.13e+07),
            (1.20, 7.18e+06),
            (1.50, 3.80e+06),
            (2.00, 1.54e+06),
            (4.00, 1.36e+05),
        ]

        def _build_source(
            sid: str, desc: str, raw: List[Tuple[float, float]]
        ) -> RadiationSource:
            source = RadiationSource(source_id=sid, description=desc)
            for wl, rad in raw:
                point = SpectralDataPoint(wavelength_um=wl, spectral_radiance=rad)
                source.data_points.append(point)
            return source

        s1 = _build_source("SRC-1", "白炽灯丝 (~3000 K)", source_1_data)
        s2 = _build_source("SRC-2", "中温恒星 (~4500 K)", source_2_data)
        s3 = _build_source("SRC-3", "太阳表面 (~6000 K)", source_3_data)

        dataset = SpectralDataset(sources=[s1, s2, s3])

        total_points = sum(s.size for s in dataset.sources)
        logger.info("已加载 %d 个辐射源，共 %d 个数据点", len(dataset.sources), total_points)
        for s in dataset.sources:
            logger.info(
                "  %s (%s): %d 点, λ 范围 %.2f ~ %.2f μm",
                s.source_id, s.description, s.size,
                s.wavelengths[0], s.wavelengths[-1],
            )

        return dataset

    @property
    def total_points(self) -> int:
        """数据点总数"""
        return sum(s.size for s in self.sources)

    def summary(self) -> str:
        """返回数据集的文本摘要"""
        lines = [
            "=" * 65,
            "黑体辐射光谱实验数据集",
            "=" * 65,
        ]
        for src in self.sources:
            lines.append(f"\n{src.source_id}: {src.description}")
            lines.append(f"  {'波长 (μm)':>12s}  {'辐射度 (W/(m²·sr·μm))':>25s}")
            lines.append("  " + "-" * 42)
            for p in src.data_points:
                lines.append(f"  {p.wavelength_um:>12.2f}  {p.spectral_radiance:>25.4e}")
        lines.append("")
        lines.append(f"共 {len(self.sources)} 个辐射源, {self.total_points} 个数据点")
        lines.append("=" * 65)
        return "\n".join(lines)
