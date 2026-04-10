"""
测试模块: 光谱实验数据加载与校验
"""

import unittest

import numpy as np
from pydantic import ValidationError

from src.data.spectrum import (
    SpectralDataPoint,
    RadiationSource,
    SpectralDataset,
    PhysicalConstants,
)


class TestSpectralDataPoint(unittest.TestCase):
    """测试单个数据点校验"""

    def test_valid_data_point(self):
        point = SpectralDataPoint(wavelength_um=1.0, spectral_radiance=9.2e5)
        self.assertAlmostEqual(point.wavelength_um, 1.0)
        self.assertAlmostEqual(point.spectral_radiance, 9.2e5)

    def test_invalid_wavelength_negative(self):
        with self.assertRaises(ValidationError):
            SpectralDataPoint(wavelength_um=-1.0, spectral_radiance=1e5)

    def test_invalid_wavelength_out_of_range(self):
        with self.assertRaises(ValidationError):
            SpectralDataPoint(wavelength_um=200.0, spectral_radiance=1e5)

    def test_invalid_radiance_zero(self):
        with self.assertRaises(ValidationError):
            SpectralDataPoint(wavelength_um=1.0, spectral_radiance=0.0)

    def test_invalid_radiance_negative(self):
        with self.assertRaises(ValidationError):
            SpectralDataPoint(wavelength_um=1.0, spectral_radiance=-100.0)


class TestPhysicalConstants(unittest.TestCase):
    """测试物理常数校验"""

    def test_default_constants(self):
        c = PhysicalConstants()
        self.assertAlmostEqual(c.h, 6.62607015e-34)
        self.assertAlmostEqual(c.c, 2.99792458e8)
        self.assertAlmostEqual(c.k_B, 1.380649e-23)

    def test_validate_passes(self):
        c = PhysicalConstants()
        c.validate()

    def test_C1_property(self):
        c = PhysicalConstants()
        expected = 2.0 * c.h * c.c ** 2
        self.assertAlmostEqual(c.C1, expected, places=30)

    def test_C2_property(self):
        c = PhysicalConstants()
        expected = c.h * c.c / c.k_B
        self.assertAlmostEqual(c.C2, expected, places=10)

    def test_invalid_planck_constant(self):
        c = PhysicalConstants(h=-1.0)
        with self.assertRaises(ValueError):
            c.validate()


class TestSpectralDataset(unittest.TestCase):
    """测试数据集加载"""

    def test_load_default_source_count(self):
        dataset = SpectralDataset.load_default()
        self.assertEqual(len(dataset.sources), 3)

    def test_load_default_total_points(self):
        dataset = SpectralDataset.load_default()
        self.assertEqual(dataset.total_points, 45)

    def test_each_source_has_15_points(self):
        dataset = SpectralDataset.load_default()
        for src in dataset.sources:
            self.assertEqual(src.size, 15)

    def test_wavelengths_positive(self):
        dataset = SpectralDataset.load_default()
        for src in dataset.sources:
            self.assertTrue(np.all(src.wavelengths > 0))

    def test_radiances_positive(self):
        dataset = SpectralDataset.load_default()
        for src in dataset.sources:
            self.assertTrue(np.all(src.radiances > 0))

    def test_wavelengths_monotonically_increasing(self):
        dataset = SpectralDataset.load_default()
        for src in dataset.sources:
            diffs = np.diff(src.wavelengths)
            self.assertTrue(np.all(diffs > 0))

    def test_source_ids_unique(self):
        dataset = SpectralDataset.load_default()
        ids = [s.source_id for s in dataset.sources]
        self.assertEqual(len(ids), len(set(ids)))

    def test_summary_not_empty(self):
        dataset = SpectralDataset.load_default()
        summary = dataset.summary()
        self.assertIn("SRC-1", summary)
        self.assertIn("SRC-2", summary)
        self.assertIn("SRC-3", summary)

    def test_source_to_dataframe(self):
        dataset = SpectralDataset.load_default()
        df = dataset.sources[0].to_dataframe()
        self.assertEqual(len(df), 15)
        for col in ["wavelength_um", "spectral_radiance", "inv_wavelength", "ln_radiance"]:
            self.assertIn(col, df.columns)


if __name__ == "__main__":
    unittest.main()
