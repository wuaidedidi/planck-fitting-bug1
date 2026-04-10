"""
测试模块: 报告生成与可视化输出
"""

import json
import os
import shutil
import unittest
from pathlib import Path

import numpy as np

from src.data.spectrum import SpectralDataset, PhysicalConstants
from src.fitting.optimizer import PlanckOptimizer
from src.report.generator import ReportGenerator
from src.visualization.plotter import ResultPlotter


class _FittingResultMixin:
    """提供共享的拟合结果"""

    _result = None

    @classmethod
    def _get_result(cls):
        if cls._result is None:
            dataset = SpectralDataset.load_default()
            constants = PhysicalConstants()
            optimizer = PlanckOptimizer(dataset=dataset, constants=constants)
            cls._result = optimizer.fit_all(
                de_seed=42, de_maxiter=200, de_popsize=10, local_maxiter=500,
            )
        return cls._result


class TestReportGenerator(_FittingResultMixin, unittest.TestCase):
    """测试报告生成"""

    TEST_OUTPUT = "output/_test_report"

    def setUp(self):
        os.makedirs(self.TEST_OUTPUT, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.TEST_OUTPUT):
            shutil.rmtree(self.TEST_OUTPUT)

    def test_generate_text_report(self):
        result = self._get_result()
        gen = ReportGenerator(output_dir=self.TEST_OUTPUT)
        gen.generate_text_report(result)
        filepath = Path(self.TEST_OUTPUT) / "fitting_report.txt"
        self.assertTrue(filepath.exists())
        content = filepath.read_text(encoding="utf-8")
        self.assertIn("Planck", content)
        self.assertIn("SRC-1", content)

    def test_generate_json_report(self):
        result = self._get_result()
        gen = ReportGenerator(output_dir=self.TEST_OUTPUT)
        gen.generate_json_report(result)
        filepath = Path(self.TEST_OUTPUT) / "fitting_result.json"
        self.assertTrue(filepath.exists())
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("source_results", data)
        self.assertIn("overall_quality", data)
        self.assertEqual(len(data["source_results"]), 3)

    def test_generate_csv_report(self):
        result = self._get_result()
        gen = ReportGenerator(output_dir=self.TEST_OUTPUT)
        gen.generate_csv_report(result)
        filepath = Path(self.TEST_OUTPUT) / "fitting_data.csv"
        self.assertTrue(filepath.exists())
        content = filepath.read_text(encoding="utf-8")
        lines = content.strip().split("\n")
        self.assertEqual(len(lines), 46)  # header + 45 data points

    def test_generate_all(self):
        result = self._get_result()
        gen = ReportGenerator(output_dir=self.TEST_OUTPUT)
        gen.generate_all(result)
        self.assertTrue((Path(self.TEST_OUTPUT) / "fitting_report.txt").exists())
        self.assertTrue((Path(self.TEST_OUTPUT) / "fitting_result.json").exists())
        self.assertTrue((Path(self.TEST_OUTPUT) / "fitting_data.csv").exists())


class TestResultPlotter(_FittingResultMixin, unittest.TestCase):
    """测试图表生成"""

    TEST_OUTPUT = "output/_test_figures"

    def setUp(self):
        os.makedirs(self.TEST_OUTPUT, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.TEST_OUTPUT):
            shutil.rmtree(self.TEST_OUTPUT)

    def test_plot_spectral_fitting(self):
        result = self._get_result()
        plotter = ResultPlotter(output_dir=self.TEST_OUTPUT)
        path = plotter.plot_spectral_fitting(result)
        self.assertTrue(os.path.exists(path))
        self.assertGreater(os.path.getsize(path), 0)

    def test_plot_multi_source_comparison(self):
        result = self._get_result()
        plotter = ResultPlotter(output_dir=self.TEST_OUTPUT)
        path = plotter.plot_multi_source_comparison(result)
        self.assertTrue(os.path.exists(path))

    def test_plot_wien_verification(self):
        result = self._get_result()
        plotter = ResultPlotter(output_dir=self.TEST_OUTPUT)
        path = plotter.plot_wien_verification(result)
        self.assertTrue(os.path.exists(path))

    def test_plot_fitting_dashboard(self):
        result = self._get_result()
        plotter = ResultPlotter(output_dir=self.TEST_OUTPUT)
        path = plotter.plot_fitting_dashboard(result)
        self.assertTrue(os.path.exists(path))
        self.assertGreater(os.path.getsize(path), 10000)

    def test_plot_all(self):
        result = self._get_result()
        plotter = ResultPlotter(output_dir=self.TEST_OUTPUT)
        plotter.plot_all(result)
        expected_files = [
            "spectral_fitting.png",
            "multi_source_comparison.png",
            "wien_verification.png",
            "fitting_dashboard.png",
        ]
        for fname in expected_files:
            self.assertTrue(
                os.path.exists(os.path.join(self.TEST_OUTPUT, fname)),
                f"{fname} not generated",
            )


if __name__ == "__main__":
    unittest.main()
