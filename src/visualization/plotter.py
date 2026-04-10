"""
结果可视化模块

生成 Planck 黑体辐射拟合结果的专业图表，包括:
1. 光谱拟合曲线图 (实验值 vs Planck 模型)
2. 多源对比图 (三组辐射源叠加)
3. Wien 位移验证图
4. 综合仪表板
5. 相关性热力图 (Pandas 分析)
6. 参数灵敏度图 (Pandas 分析)
7. 残差分布直方图 (Pandas 分析)
8. 分析综合仪表板 (Pandas 分析)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data.spectrum import PhysicalConstants
from src.fitting.optimizer import FittingResult
from src.models.planck import PlanckModel
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.analysis.analyzer import AnalysisReport

logger = get_logger("visualization.plotter")

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

SOURCE_COLORS = ["#E74C3C", "#2E86C1", "#27AE60"]
SOURCE_MARKERS = ["o", "s", "D"]


class ResultPlotter:
    """拟合结果可视化器"""

    def __init__(self, output_dir: str = "output/figures") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._model = PlanckModel(PhysicalConstants())
        logger.info("图表输出目录: %s", self.output_dir.resolve())

    def plot_all(self, result: FittingResult) -> None:
        """生成所有拟合图表。"""
        self.plot_spectral_fitting(result)
        self.plot_multi_source_comparison(result)
        self.plot_wien_verification(result)
        self.plot_fitting_dashboard(result)
        logger.info("所有拟合图表已生成完毕")

    def plot_spectral_fitting(self, result: FittingResult) -> str:
        """绘制各辐射源的光谱拟合曲线图。"""
        n = len(result.source_results)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
        if n == 1:
            axes = [axes]

        for idx, sr in enumerate(result.source_results):
            ax = axes[idx]
            ax.scatter(
                sr.wavelengths, sr.exp_radiance,
                marker=SOURCE_MARKERS[idx % 3], s=80,
                c=SOURCE_COLORS[idx % 3], edgecolors="black",
                linewidth=1, zorder=5, label="Experimental",
            )

            lam_smooth = np.linspace(
                sr.wavelengths.min() * 0.8,
                sr.wavelengths.max() * 1.1,
                300,
            )
            B_smooth = self._model.spectral_radiance_batch(
                lam_smooth, sr.temperature, sr.emissivity
            )
            ax.plot(
                lam_smooth, B_smooth,
                "-", color="black", linewidth=2, label="Planck Fit",
            )

            ax.set_xlabel("Wavelength (μm)")
            ax.set_ylabel("Spectral Radiance (W/(m²·sr·μm))")
            ax.set_title(
                f"{sr.source_id}: T={sr.temperature:.0f} K, ε={sr.emissivity:.3f}\n"
                f"AARD={sr.aard_percent:.2f}%"
            )
            ax.legend(fontsize=9)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        fig.suptitle("Planck Black-Body Spectral Fitting", fontsize=16, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.94])

        filepath = self.output_dir / "spectral_fitting.png"
        fig.savefig(filepath)
        plt.close(fig)
        logger.info("光谱拟合图已保存: %s", filepath)
        return str(filepath)

    def plot_multi_source_comparison(self, result: FittingResult) -> str:
        """绘制多源叠加对比图。"""
        fig, ax = plt.subplots(figsize=(12, 8))

        for idx, sr in enumerate(result.source_results):
            color = SOURCE_COLORS[idx % 3]
            marker = SOURCE_MARKERS[idx % 3]

            ax.scatter(
                sr.wavelengths, sr.exp_radiance,
                marker=marker, s=60, c=color, edgecolors="black",
                linewidth=0.8, zorder=5,
                label=f"{sr.source_id} Exp (T≈{sr.temperature:.0f} K)",
            )

            lam_smooth = np.linspace(
                sr.wavelengths.min() * 0.8,
                sr.wavelengths.max() * 1.1,
                300,
            )
            B_smooth = self._model.spectral_radiance_batch(
                lam_smooth, sr.temperature, sr.emissivity
            )
            ax.plot(
                lam_smooth, B_smooth,
                "-", color=color, linewidth=2, alpha=0.7,
            )

        ax.set_xlabel("Wavelength (μm)")
        ax.set_ylabel("Spectral Radiance (W/(m²·sr·μm))")
        ax.set_title("Multi-Source Black-Body Radiation Comparison")
        ax.legend(loc="upper right", framealpha=0.9)
        ax.set_xlim(left=0)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        filepath = self.output_dir / "multi_source_comparison.png"
        fig.savefig(filepath)
        plt.close(fig)
        logger.info("多源对比图已保存: %s", filepath)
        return str(filepath)

    def plot_wien_verification(self, result: FittingResult) -> str:
        """绘制 Wien 位移验证图。"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        temps = np.array([sr.temperature for sr in result.source_results])
        wien_peaks = np.array([sr.wien_peak_calc for sr in result.source_results])

        for idx, sr in enumerate(result.source_results):
            ax1.scatter(
                sr.temperature, sr.wien_peak_calc,
                marker=SOURCE_MARKERS[idx % 3], s=150,
                c=SOURCE_COLORS[idx % 3], edgecolors="black",
                linewidth=1.5, zorder=5,
                label=f"{sr.source_id}",
            )

        T_line = np.linspace(temps.min() * 0.8, temps.max() * 1.2, 100)
        lam_line = self._model.constants.b_wien * 1e6 / T_line
        ax1.plot(T_line, lam_line, "--", color="gray", linewidth=2, label="Wien Law")
        ax1.set_xlabel("Temperature (K)")
        ax1.set_ylabel("Peak Wavelength (μm)")
        ax1.set_title("Wien Displacement Law Verification")
        ax1.legend()

        rd_percent = []
        for idx, sr in enumerate(result.source_results):
            peak_idx = np.argmax(sr.exp_radiance)
            obs_peak = sr.wavelengths[peak_idx]
            deviation = (sr.wien_peak_calc - obs_peak) / obs_peak * 100
            rd_percent.append(deviation)

            color = SOURCE_COLORS[idx % 3]
            bars = ax2.bar(
                idx, deviation, color=color, edgecolor="black", width=0.6,
            )

        ax2.set_xticks(range(len(result.source_results)))
        ax2.set_xticklabels([sr.source_id for sr in result.source_results])
        ax2.set_ylabel("Wien Peak Deviation (%)")
        ax2.set_title("Wien Prediction vs Observed Peak")
        ax2.axhline(y=0, color="black", linewidth=0.8)

        fig.suptitle("Wien Displacement Law Analysis", fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.94])

        filepath = self.output_dir / "wien_verification.png"
        fig.savefig(filepath)
        plt.close(fig)
        logger.info("Wien 位移验证图已保存: %s", filepath)
        return str(filepath)

    def plot_fitting_dashboard(self, result: FittingResult) -> str:
        """绘制综合仪表板 (2x2 子图)。"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            "Planck Black-Body Fitting Dashboard",
            fontsize=16, fontweight="bold", y=0.98,
        )

        # (0,0) 多源光谱对比
        ax = axes[0, 0]
        for idx, sr in enumerate(result.source_results):
            color = SOURCE_COLORS[idx % 3]
            ax.scatter(
                sr.wavelengths, sr.exp_radiance,
                marker=SOURCE_MARKERS[idx % 3], s=40, c=color,
                edgecolors="black", linewidth=0.5, zorder=5,
                label=f"{sr.source_id} ({sr.temperature:.0f} K)",
            )
            ax.plot(sr.wavelengths, sr.calc_radiance, "--", color=color, linewidth=1.5)
        ax.set_xlabel("λ (μm)")
        ax.set_ylabel("B (W/(m²·sr·μm))")
        ax.set_title("Spectral Comparison")
        ax.legend(fontsize=8)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        # (0,1) 相对偏差
        ax = axes[0, 1]
        offset = 0
        for idx, sr in enumerate(result.source_results):
            color = SOURCE_COLORS[idx % 3]
            rd_pct = sr.relative_deviations * 100
            x_pos = range(offset, offset + len(rd_pct))
            ax.bar(x_pos, rd_pct, color=color, edgecolor="black", linewidth=0.3, width=0.7)
            offset += len(rd_pct) + 1
        ax.set_xlabel("Data Point Index")
        ax.set_ylabel("RD (%)")
        ax.set_title("Relative Deviation")
        ax.axhline(y=0, color="black", linewidth=0.8)

        # (1,0) Wien 验证
        ax = axes[1, 0]
        temps_arr = np.array([sr.temperature for sr in result.source_results])
        peaks_arr = np.array([sr.wien_peak_calc for sr in result.source_results])
        for idx, sr in enumerate(result.source_results):
            ax.scatter(
                sr.temperature, sr.wien_peak_calc,
                marker=SOURCE_MARKERS[idx % 3], s=100,
                c=SOURCE_COLORS[idx % 3], edgecolors="black", zorder=5,
            )
        T_line = np.linspace(temps_arr.min() * 0.8, temps_arr.max() * 1.2, 100)
        ax.plot(T_line, self._model.constants.b_wien * 1e3 / T_line, "--", color="gray", linewidth=2)
        ax.set_xlabel("T (K)")
        ax.set_ylabel("λ_max (μm)")
        ax.set_title("Wien Displacement")

        # (1,1) 拟合摘要
        ax = axes[1, 1]
        ax.axis("off")
        lines = [
            f"{'━' * 40}",
            f"  Planck Fitting Summary",
            f"{'━' * 40}",
            "",
        ]
        for sr in result.source_results:
            lines.extend([
                f"  {sr.source_id}: {sr.description}",
                f"    T = {sr.temperature:.2f} K",
                f"    ε = {sr.emissivity:.4f}",
                f"    AARD = {sr.aard_percent:.4f}%",
                f"    R² = {sr.r_squared:.6f}",
                "",
            ])
        lines.extend([
            f"  Overall AARD = {result.overall_aard:.4f}%",
            f"  Overall R²   = {result.overall_r_squared:.8f}",
            f"{'━' * 40}",
        ])
        ax.text(
            0.05, 0.95, "\n".join(lines),
            transform=ax.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.8", facecolor="#F8F9FA", edgecolor="#BDC3C7"),
        )

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        filepath = self.output_dir / "fitting_dashboard.png"
        fig.savefig(filepath)
        plt.close(fig)
        logger.info("综合仪表板已保存: %s", filepath)
        return str(filepath)

    # ──────────────────────────────────────────────────
    # 数据分析图表 (基于 Pandas AnalysisReport)
    # ──────────────────────────────────────────────────

    def plot_analysis_all(self, report: AnalysisReport) -> None:
        """生成全部分析图表"""
        self.plot_correlation_heatmap(report)
        self.plot_sensitivity(report)
        self.plot_residual_distribution(report)
        self.plot_analysis_dashboard(report)
        logger.info("所有分析图表已生成完毕")

    def plot_correlation_heatmap(self, report: AnalysisReport) -> str:
        """绘制变量间 Pearson 相关性热力图"""
        corr = report.correlation_matrix
        fig, ax = plt.subplots(figsize=(9, 8))

        im = ax.imshow(corr.values, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=10)
        ax.set_yticklabels(corr.columns, fontsize=10)

        for i in range(len(corr)):
            for j in range(len(corr)):
                val = corr.iloc[i, j]
                color = "white" if abs(val) > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, color=color)

        fig.colorbar(im, ax=ax, label="Pearson r", shrink=0.8)
        ax.set_title("Correlation Heatmap — Spectral Analysis Variables")

        filepath = self.output_dir / "correlation_heatmap.png"
        fig.savefig(filepath, bbox_inches="tight")
        plt.close(fig)
        logger.info("相关性热力图已保存: %s", filepath)
        return str(filepath)

    def plot_sensitivity(self, report: AnalysisReport) -> str:
        """绘制参数灵敏度分析图"""
        n_sources = len(report.sensitivity_T)
        fig, axes = plt.subplots(n_sources, 2, figsize=(14, 5 * n_sources))
        if n_sources == 1:
            axes = axes.reshape(1, -1)

        for idx in range(n_sources):
            s_T = report.sensitivity_T[idx]
            s_eps = report.sensitivity_eps[idx]

            ax = axes[idx, 0]
            ax.plot(
                s_T.perturbations, s_T.aard_values,
                "o-", color="#E74C3C", linewidth=2, markersize=4,
            )
            ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
            ax.set_xlabel("T Perturbation (%)")
            ax.set_ylabel("AARD (%)")
            ax.set_title(f"{s_T.source_id}: T Sensitivity (base={s_T.base_value:.0f} K)")

            ax = axes[idx, 1]
            ax.plot(
                s_eps.perturbations, s_eps.aard_values,
                "s-", color="#2E86C1", linewidth=2, markersize=4,
            )
            ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
            ax.set_xlabel("ε Perturbation (%)")
            ax.set_ylabel("AARD (%)")
            ax.set_title(f"{s_eps.source_id}: ε Sensitivity (base={s_eps.base_value:.4f})")

        fig.suptitle("Parameter Sensitivity Analysis", fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        filepath = self.output_dir / "parameter_sensitivity.png"
        fig.savefig(filepath)
        plt.close(fig)
        logger.info("灵敏度分析图已保存: %s", filepath)
        return str(filepath)

    def plot_residual_distribution(self, report: AnalysisReport) -> str:
        """绘制残差分布直方图与正态拟合"""
        rd = report.master_df["rd_percent"].values
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        ax1.hist(rd, bins=15, color="#3498DB", edgecolor="black",
                 alpha=0.7, density=True, label="Observed")
        mu, sigma = rd.mean(), rd.std()
        if sigma > 0:
            x_fit = np.linspace(rd.min() - 1, rd.max() + 1, 100)
            from scipy.stats import norm
            ax1.plot(x_fit, norm.pdf(x_fit, mu, sigma), "r-", linewidth=2,
                     label=f"Normal($\\mu$={mu:.2f}, $\\sigma$={sigma:.2f})")
        ax1.set_xlabel("Relative Deviation (%)")
        ax1.set_ylabel("Density")
        ax1.set_title("Residual Distribution")
        ax1.legend(fontsize=9)

        from scipy import stats as sp_stats
        sp_stats.probplot(rd, dist="norm", plot=ax2)
        ax2.set_title("Q-Q Plot (Normal)")
        ax2.get_lines()[0].set(marker="o", color="#E74C3C", markersize=6)
        ax2.get_lines()[1].set(color="#2E86C1", linewidth=2)

        fig.suptitle(
            "Residual Analysis — Distribution & Normality",
            fontsize=14, fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0, 1, 0.94])

        filepath = self.output_dir / "residual_distribution.png"
        fig.savefig(filepath)
        plt.close(fig)
        logger.info("残差分布图已保存: %s", filepath)
        return str(filepath)

    def plot_analysis_dashboard(self, report: AnalysisReport) -> str:
        """绘制数据分析综合仪表板 (2x2)"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            "Data Analysis Dashboard — Black-Body Radiation",
            fontsize=16, fontweight="bold", y=0.98,
        )

        # (0,0) Wien 位移验证
        ax = axes[0, 0]
        wv = report.wien_verification
        for idx, sid in enumerate(wv.source_ids):
            ax.scatter(
                wv.fitted_temperatures[idx], wv.wien_peak_calc[idx],
                marker=SOURCE_MARKERS[idx % 3], s=100,
                c=SOURCE_COLORS[idx % 3], edgecolors="black", zorder=5,
                label=sid,
            )
        T_line = np.linspace(wv.fitted_temperatures.min() * 0.8,
                             wv.fitted_temperatures.max() * 1.2, 100)
        ax.plot(T_line, self._model.constants.b_wien * 1e6 / T_line,
                "--", color="gray", linewidth=2, label="Wien Law")
        ax.set_xlabel("T (K)")
        ax.set_ylabel("λ_max (μm)")
        ax.set_title("Wien Displacement Verification")
        ax.legend(fontsize=9)

        # (0,1) 波长分组箱线图
        ax = axes[0, 1]
        df = report.master_df
        groups_data = []
        group_labels = []
        for name, grp in df.dropna(subset=["wave_group"]).groupby("wave_group", observed=True):
            groups_data.append(grp["abs_rd_percent"].values)
            group_labels.append(str(name))
        if groups_data:
            bp = ax.boxplot(groups_data, labels=group_labels, patch_artist=True)
            colors_box = ["#AED6F1", "#A9DFBF", "#F9E79F"]
            for patch, c in zip(bp["boxes"], colors_box[:len(bp["boxes"])]):
                patch.set_facecolor(c)
        ax.set_xlabel("Wavelength Group")
        ax.set_ylabel("|Relative Deviation| (%)")
        ax.set_title("Fitting Quality by Wavelength Group")

        # (1,0) 灵敏度对比 (第一个辐射源)
        ax = axes[1, 0]
        if report.sensitivity_T:
            s_T = report.sensitivity_T[0]
            s_eps = report.sensitivity_eps[0]
            ax.plot(s_T.perturbations, s_T.aard_values, "o-",
                    color="#E74C3C", linewidth=1.5, markersize=3, label="T")
            ax.plot(s_eps.perturbations, s_eps.aard_values, "s-",
                    color="#2E86C1", linewidth=1.5, markersize=3, label="ε")
            ax.set_xlabel("Perturbation (%)")
            ax.set_ylabel("AARD (%)")
            ax.set_title(f"Parameter Sensitivity ({s_T.source_id})")
            ax.legend(fontsize=9)

        # (1,1) 分析指标摘要
        ax = axes[1, 1]
        ax.axis("off")
        resid = report.residual_stats
        sb = report.sb_verification
        info_lines = [
            f"{'━' * 40}",
            f"  Data Analysis Summary",
            f"{'━' * 40}",
            "",
            f"  ▸ Wien Displacement Verification",
        ]
        for i, sid in enumerate(wv.source_ids):
            info_lines.append(
                f"    {sid}: λ_peak={wv.wien_peak_calc[i]:.4f} μm "
                f"(dev={wv.deviations_percent[i]:.1f}%)"
            )
        info_lines.extend([
            "",
            f"  ▸ Stefan-Boltzmann R² = {sb.r_squared:.6f}",
            "",
            f"  ▸ Residual Statistics",
            f"    Mean RD = {resid['mean_rd_percent']:.4f}%",
            f"    Std RD  = {resid['std_rd_percent']:.4f}%",
            f"    Shapiro p = {resid['shapiro_p_value']:.4f}",
            "",
            f"  ▸ Outliers: {len(report.outlier_indices)} point(s)",
            f"  ▸ Data Points: {len(report.master_df)}",
            f"{'━' * 40}",
        ])
        ax.text(
            0.05, 0.95, "\n".join(info_lines), transform=ax.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.8", facecolor="#F8F9FA", edgecolor="#BDC3C7"),
        )

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        filepath = self.output_dir / "analysis_dashboard.png"
        fig.savefig(filepath)
        plt.close(fig)
        logger.info("分析综合仪表板已保存: %s", filepath)
        return str(filepath)
