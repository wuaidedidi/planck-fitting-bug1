"""
测试模块: Planck 黑体辐射模型计算正确性
"""

import unittest
import math

import numpy as np

from src.data.spectrum import PhysicalConstants
from src.models.planck import PlanckModel


class TestPlanckModel(unittest.TestCase):
    """测试 Planck 模型核心计算"""

    def setUp(self):
        self.constants = PhysicalConstants()
        self.model = PlanckModel(self.constants)

    def test_spectral_radiance_positive(self):
        """辐射度应为正值"""
        B = self.model.spectral_radiance(1.0, 5000.0)
        self.assertGreater(B, 0.0)

    def test_spectral_radiance_increases_with_temperature(self):
        """同一波长下，高温辐射度应大于低温"""
        B_low = self.model.spectral_radiance(1.0, 3000.0)
        B_high = self.model.spectral_radiance(1.0, 6000.0)
        self.assertGreater(B_high, B_low)

    def test_spectral_radiance_emissivity_scaling(self):
        """辐射度应与发射率成正比"""
        B_full = self.model.spectral_radiance(1.0, 5000.0, emissivity=1.0)
        B_half = self.model.spectral_radiance(1.0, 5000.0, emissivity=0.5)
        self.assertAlmostEqual(B_half / B_full, 0.5, places=6)

    def test_spectral_radiance_invalid_wavelength(self):
        """无效波长应抛出 ValueError"""
        with self.assertRaises(ValueError):
            self.model.spectral_radiance(0.0, 5000.0)
        with self.assertRaises(ValueError):
            self.model.spectral_radiance(-1.0, 5000.0)

    def test_spectral_radiance_invalid_temperature(self):
        """无效温度应抛出 ValueError"""
        with self.assertRaises(ValueError):
            self.model.spectral_radiance(1.0, 0.0)
        with self.assertRaises(ValueError):
            self.model.spectral_radiance(1.0, -100.0)

    def test_spectral_radiance_invalid_emissivity(self):
        """无效发射率应抛出 ValueError"""
        with self.assertRaises(ValueError):
            self.model.spectral_radiance(1.0, 5000.0, emissivity=0.0)
        with self.assertRaises(ValueError):
            self.model.spectral_radiance(1.0, 5000.0, emissivity=1.5)

    def test_spectral_radiance_batch(self):
        """批量计算应与逐个计算一致"""
        wavelengths = np.array([0.5, 1.0, 2.0, 5.0])
        T = 4000.0
        batch = self.model.spectral_radiance_batch(wavelengths, T)
        for i, lam in enumerate(wavelengths):
            single = self.model.spectral_radiance(lam, T)
            self.assertAlmostEqual(batch[i], single, places=6)

    def test_wien_peak_wavelength(self):
        """Wien 位移定律: λ_max = b / T"""
        T = 5778.0  # 太阳表面温度
        lam_max = self.model.wien_peak_wavelength(T)
        expected = self.constants.b_wien * 1e6 / T
        self.assertAlmostEqual(lam_max, expected, places=6)

    def test_wien_peak_inverse_temperature(self):
        """Wien 峰值波长应随温度升高而减小"""
        lam_low = self.model.wien_peak_wavelength(3000.0)
        lam_high = self.model.wien_peak_wavelength(6000.0)
        self.assertGreater(lam_low, lam_high)

    def test_wien_peak_invalid_temperature(self):
        with self.assertRaises(ValueError):
            self.model.wien_peak_wavelength(0.0)

    def test_stefan_boltzmann_power(self):
        """S-B 定律: j = ε × σ × T⁴"""
        T = 5000.0
        eps = 0.9
        j = self.model.stefan_boltzmann_power(T, eps)
        expected = eps * self.constants.sigma * T ** 4
        self.assertAlmostEqual(j, expected, places=2)

    def test_stefan_boltzmann_T4_scaling(self):
        """辐射功率应与 T⁴ 成正比"""
        j1 = self.model.stefan_boltzmann_power(3000.0)
        j2 = self.model.stefan_boltzmann_power(6000.0)
        ratio = j2 / j1
        expected_ratio = (6000.0 / 3000.0) ** 4
        self.assertAlmostEqual(ratio, expected_ratio, places=2)

    def test_stefan_boltzmann_invalid_temperature(self):
        with self.assertRaises(ValueError):
            self.model.stefan_boltzmann_power(-1.0)

    def test_spectral_radiance_ratio(self):
        """辐射度比值应为正"""
        ratio = self.model.spectral_radiance_ratio(1.0, 5000.0, 3000.0)
        self.assertGreater(ratio, 0.0)

    def test_planck_peak_near_wien(self):
        """Planck 辐射度峰值应接近 Wien 预测波长"""
        T = 5000.0
        wien_peak = self.model.wien_peak_wavelength(T)
        wavelengths = np.linspace(0.1, 5.0, 500)
        radiances = self.model.spectral_radiance_batch(wavelengths, T)
        peak_idx = np.argmax(radiances)
        observed_peak = wavelengths[peak_idx]
        self.assertAlmostEqual(observed_peak, wien_peak, delta=0.05)


if __name__ == "__main__":
    unittest.main()
