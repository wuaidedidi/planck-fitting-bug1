"""
测试模块: 数据分析功能 (Pandas)
"""

import os
import shutil
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.analyzer import (
    SpectrumAnalyzer,
    AnalysisReport,
    WienVerification,
    StefanBoltzmannVerification,
)
from src.data.spectrum import SpectralDataset, PhysicalConstants
from src.fitting.optimizer import PlanckOptimizer
from src.visualization.plotter import ResultPlotter


class _SharedFittingMixin:
    """共享的拟合结果，避免重复计算"""

    _result = None
    _dataset = None
    _constants = None

    @classmethod
    def _ensure_fitted(cls):
        if cls._result is None:
            cls._dataset = SpectralDataset.load_default()
            cls._constants = PhysicalConstants()
            optimizer = PlanckOptimizer(dataset=cls._dataset, constants=cls._constants)
            cls._result = optimizer.fit_all(
                de_seed=42, de_maxiter=200, de_popsize=10, local_maxiter=500,
            )


class TestSpectrumAnalyzerInit(_SharedFittingMixin, unittest.TestCase):
    """测试分析器初始化与主数据表构建"""

    def setUp(self):
        self._ensure_fitted()
        self.analyzer = SpectrumAnalyzer(
            dataset=self._dataset, result=self._result, constants=self._constants,
        )

    def test_master_df_is_dataframe(self):
        df = self.analyzer.master_df
        self.assertIsInstance(df, pd.DataFrame)

    def test_master_df_row_count(self):
        df = self.analyzer.master_df
        self.assertEqual(len(df), 45)

    def test_master_df_columns_exist(self):
        expected_cols = [
            "source_id", "wavelength_um", "B_exp", "B_calc",
            "residual", "rd_percent", "temperature", "emissivity", "wave_group",
        ]
        for col in expected_cols:
            self.assertIn(col, self.analyzer.master_df.columns, f"缺少列: {col}")

    def test_master_df_cached(self):
        df1 = self.analyzer.master_df
        df2 = self.analyzer.master_df
        self.assertIs(df1, df2)


class TestDescriptiveStatistics(_SharedFittingMixin, unittest.TestCase):
    """测试描述性统计"""

    def setUp(self):
        self._ensure_fitted()
        self.analyzer = SpectrumAnalyzer(
            dataset=self._dataset, result=self._result, constants=self._constants,
        )

    def test_descriptive_stats_type(self):
        desc = self.analyzer.descriptive_statistics()
        self.assertIsInstance(desc, pd.DataFrame)

    def test_descriptive_stats_has_extra_rows(self):
        desc = self.analyzer.descriptive_statistics()
        self.assertIn("range", desc.index)
        self.assertIn("cv_%", desc.index)


class TestWienVerification(_SharedFittingMixin, unittest.TestCase):
    """测试 Wien 位移验证"""

    def setUp(self):
        self._ensure_fitted()
        self.analyzer = SpectrumAnalyzer(
            dataset=self._dataset, result=self._result, constants=self._constants,
        )

    def test_returns_wien_verification(self):
        wien = self.analyzer.wien_displacement_verification()
        self.assertIsInstance(wien, WienVerification)

    def test_wien_has_3_sources(self):
        wien = self.analyzer.wien_displacement_verification()
        self.assertEqual(len(wien.source_ids), 3)

    def test_wien_peaks_positive(self):
        wien = self.analyzer.wien_displacement_verification()
        self.assertTrue(np.all(wien.wien_peak_calc > 0))


class TestStefanBoltzmannVerification(_SharedFittingMixin, unittest.TestCase):
    """测试 Stefan-Boltzmann 验证"""

    def setUp(self):
        self._ensure_fitted()
        self.analyzer = SpectrumAnalyzer(
            dataset=self._dataset, result=self._result, constants=self._constants,
        )

    def test_returns_sb_verification(self):
        sb = self.analyzer.stefan_boltzmann_verification()
        self.assertIsInstance(sb, StefanBoltzmannVerification)

    def test_sb_r_squared_high(self):
        sb = self.analyzer.stefan_boltzmann_verification()
        self.assertGreater(sb.r_squared, 0.9)


class TestResidualAnalysis(_SharedFittingMixin, unittest.TestCase):
    """测试残差分析"""

    def setUp(self):
        self._ensure_fitted()
        self.analyzer = SpectrumAnalyzer(
            dataset=self._dataset, result=self._result, constants=self._constants,
        )

    def test_residual_stats_keys(self):
        stats = self.analyzer.residual_analysis()
        expected_keys = [
            "mean_rd_percent", "std_rd_percent", "median_rd_percent",
            "max_abs_rd_percent", "skewness", "kurtosis", "shapiro_p_value",
        ]
        for key in expected_keys:
            self.assertIn(key, stats)

    def test_max_deviation_less_than_50(self):
        stats = self.analyzer.residual_analysis()
        self.assertLess(stats["max_abs_rd_percent"], 50.0)


class TestOutlierDetection(_SharedFittingMixin, unittest.TestCase):
    """测试异常值检测"""

    def setUp(self):
        self._ensure_fitted()
        self.analyzer = SpectrumAnalyzer(
            dataset=self._dataset, result=self._result, constants=self._constants,
        )

    def test_outliers_is_list(self):
        outliers = self.analyzer.detect_outliers()
        self.assertIsInstance(outliers, list)

    def test_outlier_indices_valid(self):
        outliers = self.analyzer.detect_outliers()
        for idx in outliers:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, 45)


class TestParameterSensitivity(_SharedFittingMixin, unittest.TestCase):
    """测试参数灵敏度分析"""

    def setUp(self):
        self._ensure_fitted()
        self.analyzer = SpectrumAnalyzer(
            dataset=self._dataset, result=self._result, constants=self._constants,
        )

    def test_sensitivity_returns_lists(self):
        sens_T, sens_eps = self.analyzer.parameter_sensitivity(n_points=11)
        self.assertEqual(len(sens_T), 3)
        self.assertEqual(len(sens_eps), 3)

    def test_sensitivity_aard_array_length(self):
        sens_T, _ = self.analyzer.parameter_sensitivity(n_points=11)
        for s in sens_T:
            self.assertEqual(len(s.aard_values), 11)

    def test_sensitivity_min_at_center(self):
        """基准参数处 AARD 应接近最小值"""
        sens_T, _ = self.analyzer.parameter_sensitivity(n_points=11)
        s = sens_T[0]
        center_idx = len(s.aard_values) // 2
        self.assertAlmostEqual(
            s.aard_values[center_idx],
            min(s.aard_values),
            delta=2.0,
        )


class TestGroupAnalysis(_SharedFittingMixin, unittest.TestCase):
    """测试波长分组分析"""

    def setUp(self):
        self._ensure_fitted()
        self.analyzer = SpectrumAnalyzer(
            dataset=self._dataset, result=self._result, constants=self._constants,
        )

    def test_group_analysis_type(self):
        groups = self.analyzer.group_analysis()
        self.assertIsInstance(groups, pd.DataFrame)

    def test_group_analysis_has_groups(self):
        groups = self.analyzer.group_analysis()
        self.assertGreater(len(groups), 0)
        self.assertLessEqual(len(groups), 3)


class TestCorrelationAnalysis(_SharedFittingMixin, unittest.TestCase):
    """测试相关性分析"""

    def setUp(self):
        self._ensure_fitted()
        self.analyzer = SpectrumAnalyzer(
            dataset=self._dataset, result=self._result, constants=self._constants,
        )

    def test_correlation_is_square(self):
        corr = self.analyzer.correlation_analysis()
        self.assertEqual(corr.shape[0], corr.shape[1])

    def test_correlation_diagonal_is_one(self):
        corr = self.analyzer.correlation_analysis()
        for i in range(len(corr)):
            self.assertAlmostEqual(corr.iloc[i, i], 1.0, places=4)


class TestFullAnalysis(_SharedFittingMixin, unittest.TestCase):
    """测试完整分析流程"""

    def setUp(self):
        self._ensure_fitted()
        self.analyzer = SpectrumAnalyzer(
            dataset=self._dataset, result=self._result, constants=self._constants,
        )

    def test_full_analysis_returns_report(self):
        report = self.analyzer.run_full_analysis()
        self.assertIsInstance(report, AnalysisReport)

    def test_full_analysis_all_fields(self):
        report = self.analyzer.run_full_analysis()
        self.assertIsInstance(report.master_df, pd.DataFrame)
        self.assertIsInstance(report.descriptive_stats, pd.DataFrame)
        self.assertIsInstance(report.correlation_matrix, pd.DataFrame)
        self.assertIsInstance(report.group_analysis, pd.DataFrame)
        self.assertIsInstance(report.residual_stats, dict)
        self.assertIsInstance(report.outlier_indices, list)


class TestExportFunctions(_SharedFittingMixin, unittest.TestCase):
    """测试数据导出"""

    TEST_OUTPUT = "output/_test_analysis"

    def setUp(self):
        self._ensure_fitted()
        self.analyzer = SpectrumAnalyzer(
            dataset=self._dataset, result=self._result, constants=self._constants,
        )
        os.makedirs(self.TEST_OUTPUT, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.TEST_OUTPUT):
            shutil.rmtree(self.TEST_OUTPUT)

    def test_export_master_csv(self):
        path = self.analyzer.export_master_csv(self.TEST_OUTPUT)
        self.assertTrue(os.path.exists(path))
        df = pd.read_csv(path)
        self.assertEqual(len(df), 45)

    def test_export_analysis_report_json(self):
        path = self.analyzer.export_analysis_report(self.TEST_OUTPUT)
        self.assertTrue(os.path.exists(path))
        self.assertGreater(os.path.getsize(path), 100)


class TestAnalysisPlots(_SharedFittingMixin, unittest.TestCase):
    """测试分析图表生成"""

    TEST_OUTPUT = "output/_test_analysis_figures"

    def setUp(self):
        self._ensure_fitted()
        self.analyzer = SpectrumAnalyzer(
            dataset=self._dataset, result=self._result, constants=self._constants,
        )
        self.report = self.analyzer.run_full_analysis()
        os.makedirs(self.TEST_OUTPUT, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.TEST_OUTPUT):
            shutil.rmtree(self.TEST_OUTPUT)

    def test_plot_correlation_heatmap(self):
        plotter = ResultPlotter(output_dir=self.TEST_OUTPUT)
        path = plotter.plot_correlation_heatmap(self.report)
        self.assertTrue(os.path.exists(path))

    def test_plot_sensitivity(self):
        plotter = ResultPlotter(output_dir=self.TEST_OUTPUT)
        path = plotter.plot_sensitivity(self.report)
        self.assertTrue(os.path.exists(path))

    def test_plot_residual_distribution(self):
        plotter = ResultPlotter(output_dir=self.TEST_OUTPUT)
        path = plotter.plot_residual_distribution(self.report)
        self.assertTrue(os.path.exists(path))

    def test_plot_analysis_dashboard(self):
        plotter = ResultPlotter(output_dir=self.TEST_OUTPUT)
        path = plotter.plot_analysis_dashboard(self.report)
        self.assertTrue(os.path.exists(path))
        self.assertGreater(os.path.getsize(path), 10000)

    def test_plot_analysis_all(self):
        plotter = ResultPlotter(output_dir=self.TEST_OUTPUT)
        plotter.plot_analysis_all(self.report)
        expected = [
            "correlation_heatmap.png",
            "parameter_sensitivity.png",
            "residual_distribution.png",
            "analysis_dashboard.png",
        ]
        for fname in expected:
            self.assertTrue(
                os.path.exists(os.path.join(self.TEST_OUTPUT, fname)),
                f"{fname} 未生成",
            )


if __name__ == "__main__":
    unittest.main()
