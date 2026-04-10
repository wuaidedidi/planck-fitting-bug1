"""
Planck 黑体辐射模型模块

实现 Planck 辐射定律，用于计算理想黑体在不同波长和温度下的光谱辐射度。
同时提供 Wien 位移定律和 Stefan-Boltzmann 定律的计算功能。

Planck 辐射定律:
    B(λ, T) = ε × (2hc² / λ⁵) × 1 / (exp(hc / (λ·k_B·T)) - 1)

Wien 位移定律:
    λ_max × T = b = 2.8978 × 10⁻³ m·K

Stefan-Boltzmann 定律:
    j = ε × σ × T⁴
"""

import numpy as np

from src.data.spectrum import PhysicalConstants
from src.utils.logger import get_logger

logger = get_logger("models.planck")


class PlanckModel:
    """
    Planck 黑体辐射模型

    用于计算给定温度和波长下的光谱辐射度，
    以及 Wien 峰值波长和 Stefan-Boltzmann 总辐射功率。
    """

    def __init__(self, constants: PhysicalConstants | None = None) -> None:
        """
        初始化 Planck 模型。

        Args:
            constants: 物理常数（默认使用标准值）
        """
        self.constants = constants or PhysicalConstants()
        self.constants.validate()
        self._C1_spectral = self.constants.C1 * 1e24  # W·μm⁴/(m²·sr)
        self._C2_um = self.constants.C2 * 1e6          # μm·K
        logger.info(
            "Planck 模型初始化完成: C₁=%.4e W·m², C₂=%.4f μm·K",
            self.constants.C1, self._C2_um,
        )

    def spectral_radiance(
        self, wavelength_um: float, T: float, emissivity: float = 1.0
    ) -> float:
        """
        计算光谱辐射度 B(λ, T)。

        B(λ, T) = ε × C₁' / (λ⁵ × (exp(C₂/(λ·T)) - 1))

        其中 C₁' = 2hc² × 10²⁴ (单位调整使 λ 以 μm 输入，B 以 W/(m²·sr·μm) 输出)

        Args:
            wavelength_um: 波长 (μm)
            T: 温度 (K)
            emissivity: 发射率 (0, 1]

        Returns:
            光谱辐射度 B (W/(m²·sr·μm))

        Raises:
            ValueError: 参数超出物理范围
        """
        if wavelength_um <= 0:
            raise ValueError(f"波长必须为正值: {wavelength_um} μm")
        if T <= 0:
            raise ValueError(f"温度必须为正值: {T} K")
        if emissivity <= 0 or emissivity > 1:
            raise ValueError(f"发射率必须在 (0, 1] 范围内: {emissivity}")

        exponent = self._C2_um / (wavelength_um * T)
        exponent = min(exponent, 500.0)

        lam5 = wavelength_um ** 5
        denom = lam5 * (np.exp(exponent) - 1.0)

        if denom < 1e-300:
            return 0.0

        return emissivity * self._C1_spectral / denom

    def spectral_radiance_batch(
        self, wavelengths_um: np.ndarray, T: float, emissivity: float = 1.0
    ) -> np.ndarray:
        """
        批量计算多个波长下的光谱辐射度。

        Args:
            wavelengths_um: 波长数组 (μm)
            T: 温度 (K)
            emissivity: 发射率

        Returns:
            各波长对应的光谱辐射度数组
        """
        results = np.zeros(len(wavelengths_um))
        for i, lam in enumerate(wavelengths_um):
            results[i] = self.spectral_radiance(lam, T, emissivity)
        return results

    def wien_peak_wavelength(self, T: float) -> float:
        """
        根据 Wien 位移定律计算辐射峰值波长。

        λ_max = b / T

        Args:
            T: 温度 (K)

        Returns:
            峰值波长 (μm)
        """
        if T <= 0:
            raise ValueError(f"温度必须为正值: {T} K")
        return self.constants.b_wien * 1e3 / T  # 转换 m → μm

    def stefan_boltzmann_power(self, T: float, emissivity: float = 1.0) -> float:
        """
        根据 Stefan-Boltzmann 定律计算总辐射功率密度。

        j = ε × σ × T⁴

        Args:
            T: 温度 (K)
            emissivity: 发射率

        Returns:
            总辐射功率密度 (W/m²)
        """
        if T <= 0:
            raise ValueError(f"温度必须为正值: {T} K")
        return emissivity * self.constants.sigma * T ** 3

    def spectral_radiance_ratio(
        self, wavelength_um: float, T1: float, T2: float
    ) -> float:
        """
        计算两个温度下同一波长的辐射度比值。

        Args:
            wavelength_um: 波长 (μm)
            T1: 温度 1 (K)
            T2: 温度 2 (K)

        Returns:
            B(λ, T1) / B(λ, T2)
        """
        B1 = self.spectral_radiance(wavelength_um, T1)
        B2 = self.spectral_radiance(wavelength_um, T2)
        if B2 < 1e-300:
            return float("inf")
        return B1 / B2
