"""
测试模块: Planck 参数拟合优化器
"""

import unittest

import numpy as np

from src.data.spectrum import SpectralDataset, PhysicalConstants
from src.fitting.optimizer import PlanckOptimizer, FittingResult, SourceFittingResult


class TestPlanckOptimizer(unittest.TestCase):
    """测试优化器初始化与拟合"""

    def setUp(self):
        self.dataset = SpectralDataset.load_default()
        self.constants = PhysicalConstants()

    def test_optimizer_init(self):
        optimizer = PlanckOptimizer(dataset=self.dataset, constants=self.constants)
        self.assertIsNotNone(optimizer.model)
        self.assertEqual(len(self.dataset.sources), 3)

    def test_optimizer_init_insufficient_data(self):
        """数据不足时应抛出异常"""
        from src.data.spectrum import RadiationSource, SpectralDataPoint
        bad_source = RadiationSource(
            source_id="BAD", description="test",
            data_points=[SpectralDataPoint(wavelength_um=1.0, spectral_radiance=1e5)],
        )
        bad_dataset = SpectralDataset(sources=[bad_source])
        with self.assertRaises(ValueError):
            PlanckOptimizer(dataset=bad_dataset, constants=self.constants)

    def test_fit_single_source(self):
        """拟合单个辐射源应返回 SourceFittingResult"""
        optimizer = PlanckOptimizer(dataset=self.dataset, constants=self.constants)
        sr = optimizer.fit_source(
            self.dataset.sources[0],
            de_seed=42, de_maxiter=200, de_popsize=10,
            local_maxiter=500,
        )
        self.assertIsInstance(sr, SourceFittingResult)
        self.assertGreater(sr.temperature, 1000.0)
        self.assertLess(sr.temperature, 10000.0)
        self.assertGreater(sr.emissivity, 0.0)
        self.assertLessEqual(sr.emissivity, 1.0)

    def test_fit_all_returns_result(self):
        """拟合所有辐射源应返回完整 FittingResult"""
        optimizer = PlanckOptimizer(dataset=self.dataset, constants=self.constants)
        result = optimizer.fit_all(
            de_seed=42, de_maxiter=200, de_popsize=10,
            local_maxiter=500,
        )
        self.assertIsInstance(result, FittingResult)
        self.assertEqual(len(result.source_results), 3)

    def test_fit_quality_aard(self):
        """各辐射源 AARD 应 < 15%"""
        optimizer = PlanckOptimizer(dataset=self.dataset, constants=self.constants)
        result = optimizer.fit_all(
            de_seed=42, de_maxiter=500, de_popsize=15,
            local_maxiter=2000,
        )
        for sr in result.source_results:
            self.assertLess(
                sr.aard_percent, 15.0,
                f"{sr.source_id} AARD {sr.aard_percent:.2f}% 超过 15% 阈值"
            )

    def test_fit_quality_r_squared(self):
        """各辐射源 R² 应 > 0.8"""
        optimizer = PlanckOptimizer(dataset=self.dataset, constants=self.constants)
        result = optimizer.fit_all(
            de_seed=42, de_maxiter=500, de_popsize=15,
            local_maxiter=2000,
        )
        for sr in result.source_results:
            self.assertGreater(
                sr.r_squared, 0.8,
                f"{sr.source_id} R² {sr.r_squared:.4f} 低于 0.8 阈值"
            )

    def test_fit_temperatures_in_expected_range(self):
        """拟合温度应在合理范围"""
        optimizer = PlanckOptimizer(dataset=self.dataset, constants=self.constants)
        result = optimizer.fit_all(
            de_seed=42, de_maxiter=500, de_popsize=15,
            local_maxiter=2000,
        )
        expected_ranges = [
            (2000, 4000),   # SRC-1 ~3000 K
            (3500, 5500),   # SRC-2 ~4500 K
            (5000, 7000),   # SRC-3 ~6000 K
        ]
        for sr, (t_min, t_max) in zip(result.source_results, expected_ranges):
            self.assertGreater(sr.temperature, t_min,
                               f"{sr.source_id} T={sr.temperature:.0f} K 低于预期")
            self.assertLess(sr.temperature, t_max,
                            f"{sr.source_id} T={sr.temperature:.0f} K 高于预期")

    def test_fit_arrays_correct_length(self):
        """结果数组长度应与数据点数一致"""
        optimizer = PlanckOptimizer(dataset=self.dataset, constants=self.constants)
        result = optimizer.fit_all(
            de_seed=42, de_maxiter=200, de_popsize=10,
        )
        for sr in result.source_results:
            self.assertEqual(len(sr.wavelengths), 15)
            self.assertEqual(len(sr.exp_radiance), 15)
            self.assertEqual(len(sr.calc_radiance), 15)
            self.assertEqual(len(sr.residuals), 15)
            self.assertEqual(len(sr.relative_deviations), 15)

    def test_fit_summary_not_empty(self):
        """summary() 应返回可读文本"""
        optimizer = PlanckOptimizer(dataset=self.dataset, constants=self.constants)
        result = optimizer.fit_all(
            de_seed=42, de_maxiter=200, de_popsize=10,
        )
        summary = result.summary()
        self.assertIn("Planck", summary)
        self.assertIn("AARD", summary)
        self.assertIn("SRC-1", summary)


if __name__ == "__main__":
    unittest.main()
