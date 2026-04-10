"""
拟合结果报告生成模块

将 Planck 黑体辐射参数拟合结果输出为结构化的文本报告和 JSON 数据文件，
支持后续自动化处理与归档。
"""

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

from src.data.spectrum import PhysicalConstants
from src.fitting.optimizer import FittingResult
from src.models.planck import PlanckModel
from src.utils.logger import get_logger

logger = get_logger("report.generator")


class ReportGenerator:
    """拟合结果报告生成器"""

    def __init__(self, output_dir: str = "output") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._model = PlanckModel(PhysicalConstants())
        logger.info("报告输出目录: %s", self.output_dir.resolve())

    def generate_all(self, result: FittingResult) -> None:
        """生成全部报告文件。"""
        self.generate_text_report(result)
        self.generate_json_report(result)
        self.generate_csv_report(result)
        logger.info("所有报告文件已生成完毕")

    def generate_text_report(self, result: FittingResult) -> str:
        """生成文本格式的详细报告。"""
        tz_cst = timezone(timedelta(hours=8))
        now = datetime.now(tz_cst).strftime("%Y-%m-%d %H:%M:%S CST")

        lines = [
            "╔" + "═" * 68 + "╗",
            "║" + "Planck 黑体辐射模型参数拟合报告".center(48) + "║",
            "╚" + "═" * 68 + "╝",
            "",
            f"报告生成时间: {now}",
            "",
            "=" * 70,
            "1. 物理模型",
            "=" * 70,
            "  模型: Planck 黑体辐射定律",
            "  B(λ, T) = ε × (2hc² / λ⁵) × 1 / (exp(hc/(λ·k_B·T)) - 1)",
            "",
            "  物理常数:",
            "    h     = 6.626×10⁻³⁴ J·s      (Planck 常数)",
            "    c     = 2.998×10⁸ m/s         (光速)",
            "    k_B   = 1.381×10⁻²³ J/K       (Boltzmann 常数)",
            "    b     = 2.898×10⁻³ m·K         (Wien 位移常数)",
            "    σ     = 5.670×10⁻⁸ W/(m²·K⁴)  (Stefan-Boltzmann 常数)",
            "",
        ]

        for sr in result.source_results:
            lines.extend([
                "=" * 70,
                f"2-{sr.source_id}: {sr.description}",
                "=" * 70,
                f"  拟合温度  T = {sr.temperature:.2f} K",
                f"  发射率    ε = {sr.emissivity:.4f}",
                f"  Wien 峰值   = {sr.wien_peak_calc:.4f} μm",
                f"  S-B 功率    = {sr.stefan_boltzmann_power:.2f} W/m²",
                "",
                "  拟合质量:",
                f"    目标函数值 = {sr.objective_value:.6e}",
                f"    RMSD       = {sr.rmsd:.6e}",
                f"    AARD       = {sr.aard_percent:.4f} %",
                f"    R²         = {sr.r_squared:.8f}",
                "",
                f"  {'序号':>4s}  {'λ (μm)':>10s}  {'B_exp':>14s}  {'B_calc':>14s}  {'RD (%)':>10s}",
                "  " + "-" * 60,
            ])
            for i in range(len(sr.wavelengths)):
                lines.append(
                    f"  {i + 1:>4d}"
                    f"  {sr.wavelengths[i]:>10.2f}"
                    f"  {sr.exp_radiance[i]:>14.4e}"
                    f"  {sr.calc_radiance[i]:>14.4e}"
                    f"  {sr.relative_deviations[i] * 100:>10.4f}"
                )
            lines.append("  " + "-" * 60)
            lines.append("")

        lines.extend([
            "=" * 70,
            "全局拟合质量",
            "=" * 70,
            f"  总体 AARD = {result.overall_aard:.4f} %",
            f"  总体 R²   = {result.overall_r_squared:.8f}",
            "",
            "=" * 70,
            f"报告结束 — {now}",
            "=" * 70,
        ])

        filepath = self.output_dir / "fitting_report.txt"
        filepath.write_text("\n".join(lines), encoding="utf-8")
        logger.info("文本报告已保存: %s", filepath)
        return str(filepath)

    def generate_json_report(self, result: FittingResult) -> str:
        """生成 JSON 格式的结构化报告。"""
        tz_cst = timezone(timedelta(hours=8))
        now = datetime.now(tz_cst).isoformat()

        report: Dict[str, Any] = {
            "metadata": {
                "title": "Planck Black-Body Radiation Fitting Report",
                "model": "Planck Radiation Law",
                "generated_at": now,
            },
            "physical_constants": {
                "h_J_s": 6.62607015e-34,
                "c_m_per_s": 2.99792458e8,
                "k_B_J_per_K": 1.380649e-23,
                "b_wien_m_K": 2.8977729e-3,
                "sigma_W_per_m2_K4": 5.670374419e-8,
            },
            "source_results": [],
            "overall_quality": {
                "AARD_percent": round(result.overall_aard, 6),
                "R_squared": round(result.overall_r_squared, 10),
            },
        }

        for sr in result.source_results:
            src_data = {
                "source_id": sr.source_id,
                "description": sr.description,
                "fitted_temperature_K": round(sr.temperature, 4),
                "fitted_emissivity": round(sr.emissivity, 6),
                "wien_peak_um": round(sr.wien_peak_calc, 6),
                "stefan_boltzmann_power_W_per_m2": round(sr.stefan_boltzmann_power, 2),
                "fitting_quality": {
                    "objective_value": float(sr.objective_value),
                    "RMSD": float(sr.rmsd),
                    "AARD_percent": round(sr.aard_percent, 6),
                    "R_squared": round(sr.r_squared, 10),
                },
                "data_comparison": [],
            }
            for i in range(len(sr.wavelengths)):
                src_data["data_comparison"].append({
                    "wavelength_um": float(sr.wavelengths[i]),
                    "B_experimental": float(sr.exp_radiance[i]),
                    "B_calculated": float(sr.calc_radiance[i]),
                    "relative_deviation_percent": round(
                        float(sr.relative_deviations[i] * 100), 6
                    ),
                })
            report["source_results"].append(src_data)

        filepath = self.output_dir / "fitting_result.json"
        filepath.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("JSON 报告已保存: %s", filepath)
        return str(filepath)

    def generate_csv_report(self, result: FittingResult) -> str:
        """使用 Pandas 生成 CSV 格式的对比数据表。"""
        rows = []
        for sr in result.source_results:
            for i in range(len(sr.wavelengths)):
                rows.append({
                    "source_id": sr.source_id,
                    "wavelength_um": sr.wavelengths[i],
                    "B_experimental": sr.exp_radiance[i],
                    "B_calculated": sr.calc_radiance[i],
                    "relative_deviation_percent": sr.relative_deviations[i],
                    "temperature_K": sr.temperature,
                    "emissivity": sr.emissivity,
                })

        df = pd.DataFrame(rows)
        filepath = self.output_dir / "fitting_data.csv"
        df.to_csv(filepath, index=False, float_format="%.6e")
        logger.info("CSV 数据已保存 (Pandas): %s", filepath)
        return str(filepath)

    def generate_analysis_summary(
        self, analysis_report: Any, result: FittingResult
    ) -> str:
        """生成数据分析文本摘要报告。"""
        tz_cst = timezone(timedelta(hours=8))
        now = datetime.now(tz_cst).strftime("%Y-%m-%d %H:%M:%S CST")
        resid = analysis_report.residual_stats
        wv = analysis_report.wien_verification
        sb = analysis_report.sb_verification

        lines = [
            "╔" + "═" * 68 + "╗",
            "║" + "黑体辐射数据分析报告".center(54) + "║",
            "╚" + "═" * 68 + "╝",
            "",
            f"报告生成时间: {now}",
            "",
            "=" * 70,
            "1. 描述性统计",
            "=" * 70,
            analysis_report.descriptive_stats.to_string(),
            "",
            "=" * 70,
            "2. Wien 位移定律验证",
            "=" * 70,
        ]

        for i, sid in enumerate(wv.source_ids):
            lines.append(
                f"  {sid}: T={wv.fitted_temperatures[i]:.2f} K, "
                f"λ_peak(Wien)={wv.wien_peak_calc[i]:.4f} μm, "
                f"λ_peak(obs)={wv.observed_peak_approx[i]:.4f} μm, "
                f"偏差={wv.deviations_percent[i]:.2f}%"
            )

        lines.extend([
            "",
            "=" * 70,
            "3. Stefan-Boltzmann 定律验证",
            "=" * 70,
            f"  j/ε vs T⁴ 线性回归 R² = {sb.r_squared:.6f}",
        ])

        for i, sid in enumerate(sb.source_ids):
            lines.append(
                f"  {sid}: T={sb.fitted_temperatures[i]:.2f} K, "
                f"j={sb.sb_power[i]:.2f} W/m², "
                f"ε={sb.emissivities[i]:.4f}"
            )

        lines.extend([
            "",
            "=" * 70,
            "4. 残差统计分析",
            "=" * 70,
            f"  平均相对偏差:    {resid['mean_rd_percent']:.4f} %",
            f"  偏差标准差:      {resid['std_rd_percent']:.4f} %",
            f"  中位数偏差:      {resid['median_rd_percent']:.4f} %",
            f"  最大绝对偏差:    {resid['max_abs_rd_percent']:.4f} %",
            f"  偏度 (skewness): {resid['skewness']:.4f}",
            f"  峰度 (kurtosis): {resid['kurtosis']:.4f}",
            f"  Shapiro-Wilk p:  {resid['shapiro_p_value']:.4f}",
            "",
            "  残差正态性: "
            + ("通过 (p > 0.05)" if resid["shapiro_p_value"] > 0.05 else "未通过 (p ≤ 0.05)"),
            "",
            "=" * 70,
            "5. 异常值检测 (Z-score > 2.0)",
            "=" * 70,
        ])

        if analysis_report.outlier_indices:
            for idx in analysis_report.outlier_indices:
                row = analysis_report.master_df.iloc[idx]
                lines.append(
                    f"  点 {idx + 1}: {row['source_id']}, "
                    f"λ={row['wavelength_um']:.2f} μm, "
                    f"RD={row['rd_percent']:.4f}%"
                )
        else:
            lines.append("  未检测到异常点")

        lines.extend([
            "",
            "=" * 70,
            "6. 波长分组分析",
            "=" * 70,
            analysis_report.group_analysis.to_string(),
            "",
            "=" * 70,
            "7. 变量相关性矩阵",
            "=" * 70,
            analysis_report.correlation_matrix.to_string(),
            "",
            "=" * 70,
            "8. 参数灵敏度分析",
            "=" * 70,
        ])

        for s_T in analysis_report.sensitivity_T:
            lines.append(
                f"  {s_T.source_id} T 基准值: {s_T.base_value:.2f} K, "
                f"梯度: {s_T.gradient:.6f}"
            )
        for s_eps in analysis_report.sensitivity_eps:
            lines.append(
                f"  {s_eps.source_id} ε 基准值: {s_eps.base_value:.4f}, "
                f"梯度: {s_eps.gradient:.6f}"
            )

        lines.extend([
            "",
            "=" * 70,
            f"分析报告结束 — {now}",
            "=" * 70,
        ])

        filepath = self.output_dir / "analysis_summary.txt"
        filepath.write_text("\n".join(lines), encoding="utf-8")
        logger.info("数据分析摘要已保存: %s", filepath)
        return str(filepath)
