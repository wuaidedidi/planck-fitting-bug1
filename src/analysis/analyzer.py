"""
黑体辐射数据分析模块

基于 Pandas 对黑体辐射光谱拟合结果进行深度分析，包括:
1. 描述性统计 — 实验数据的分布特征
2. Wien 位移定律验证 — 拟合温度 vs Wien 预测峰值波长
3. Stefan-Boltzmann 定律验证 — 总辐射功率 vs T⁴ 关系
4. 残差与异常值分析 — 残差统计、Z-score 异常检测
5. 参数灵敏度分析 — T/ε 微扰对 AARD 的影响
6. 波长分组分析 — 短波/中波/长波段的拟合表现差异
7. 变量相关性分析 — Pearson 相关系数矩阵
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from scipy import stats

from src.data.spectrum import SpectralDataset, PhysicalConstants
from src.fitting.optimizer import FittingResult, SourceFittingResult
from src.models.planck import PlanckModel
from src.utils.logger import get_logger

logger = get_logger("analysis.analyzer")


@dataclass
class WienVerification:
    """Wien 位移定律验证结果"""

    source_ids: List[str]
    fitted_temperatures: np.ndarray
    wien_peak_calc: np.ndarray        # λ_max = b / T
    observed_peak_approx: np.ndarray  # 从数据中估计的峰值波长
    deviations_percent: np.ndarray    # 偏差百分比


@dataclass
class StefanBoltzmannVerification:
    """Stefan-Boltzmann 定律验证结果"""

    source_ids: List[str]
    fitted_temperatures: np.ndarray
    sb_power: np.ndarray              # ε × σ × T⁴
    emissivities: np.ndarray
    T4_values: np.ndarray             # T⁴
    r_squared: float                  # j vs T⁴ 线性回归 R²


@dataclass
class SensitivityResult:
    """单参数灵敏度分析结果"""

    parameter_name: str
    source_id: str
    base_value: float
    perturbations: np.ndarray
    perturbed_values: np.ndarray
    aard_values: np.ndarray
    gradient: float


@dataclass
class AnalysisReport:
    """完整数据分析报告"""

    master_df: pd.DataFrame
    descriptive_stats: pd.DataFrame
    wien_verification: WienVerification
    sb_verification: StefanBoltzmannVerification
    residual_stats: Dict[str, float]
    outlier_indices: List[int]
    sensitivity_T: List[SensitivityResult]
    sensitivity_eps: List[SensitivityResult]
    group_analysis: pd.DataFrame
    correlation_matrix: pd.DataFrame


class SpectrumAnalyzer:
    """
    黑体辐射数据分析器

    整合实验数据与 Planck 拟合结果，执行多维度数据分析并输出 Pandas DataFrame。
    """

    def __init__(
        self,
        dataset: SpectralDataset,
        result: FittingResult,
        constants: PhysicalConstants | None = None,
    ) -> None:
        self.dataset = dataset
        self.result = result
        self.constants = constants or PhysicalConstants()
        self.model = PlanckModel(self.constants)
        self._master_df: pd.DataFrame | None = None
        total = sum(s.size for s in dataset.sources)
        logger.info("分析器初始化完成，共 %d 个数据点", total)

    @property
    def master_df(self) -> pd.DataFrame:
        """构建主数据表，合并所有辐射源的实验值、计算值与推导量"""
        if self._master_df is not None:
            return self._master_df

        rows = []
        for sr in self.result.source_results:
            for i in range(len(sr.wavelengths)):
                rows.append({
                    "source_id": sr.source_id,
                    "description": sr.description,
                    "wavelength_um": sr.wavelengths[i],
                    "inv_wavelength": 1.0 / sr.wavelengths[i],
                    "B_exp": sr.exp_radiance[i],
                    "B_calc": sr.calc_radiance[i],
                    "ln_B_exp": np.log10(sr.exp_radiance[i]),
                    "ln_B_calc": np.log10(sr.calc_radiance[i]),
                    "residual": sr.residuals[i],
                    "abs_residual": abs(sr.residuals[i]),
                    "relative_deviation": sr.relative_deviations[i],
                    "rd_percent": sr.relative_deviations[i],
                    "abs_rd_percent": abs(sr.relative_deviations[i]),
                    "temperature": sr.temperature,
                    "emissivity": sr.emissivity,
                })

        df = pd.DataFrame(rows)

        bins = pd.cut(
            df["wavelength_um"],
            bins=[0, 1.0, 3.0, 100.0],
            labels=["长波(MIR, >3μm)", "中波(NIR, 1-3μm)", "短波(UV-Vis, <1μm)"],
        )
        df["wave_group"] = bins

        self._master_df = df
        logger.info("主数据表构建完成: %d 行 × %d 列", len(df), len(df.columns))
        return df

    def descriptive_statistics(self) -> pd.DataFrame:
        """对核心数值列进行描述性统计"""
        cols = ["wavelength_um", "B_exp", "B_calc", "rd_percent", "temperature", "emissivity"]
        desc = self.master_df[cols].describe()
        desc.loc["range"] = desc.loc["max"] - desc.loc["min"]
        desc.loc["cv_%"] = abs((desc.loc["std"] / desc.loc["mean"]) * 100)
        logger.info("描述性统计完成")
        return desc

    def wien_displacement_verification(self) -> WienVerification:
        """
        Wien 位移定律验证: λ_max × T = b

        对每个辐射源，比较拟合温度预测的 Wien 峰值与数据中的近似峰值。
        """
        source_ids = []
        temperatures = []
        wien_peaks = []
        observed_peaks = []

        for sr in self.result.source_results:
            source_ids.append(sr.source_id)
            temperatures.append(sr.temperature)
            wien_peaks.append(sr.wien_peak_calc)
            peak_idx = np.argmax(sr.exp_radiance)
            observed_peaks.append(sr.wavelengths[peak_idx])

        temps = np.array(temperatures)
        w_calc = np.array(wien_peaks)
        w_obs = np.array(observed_peaks)
        devs = (w_obs - w_calc) / w_obs * 100

        result = WienVerification(
            source_ids=source_ids,
            fitted_temperatures=temps,
            wien_peak_calc=w_calc,
            observed_peak_approx=w_obs,
            deviations_percent=devs,
        )

        logger.info("Wien 位移验证完成: 平均偏差 %.2f%%", np.mean(np.abs(devs)))
        return result

    def stefan_boltzmann_verification(self) -> StefanBoltzmannVerification:
        """
        Stefan-Boltzmann 定律验证: j = ε × σ × T⁴

        验证拟合温度是否满足辐射功率与 T⁴ 的线性关系。
        """
        source_ids = []
        temps = []
        powers = []
        emissivities = []

        for sr in self.result.source_results:
            source_ids.append(sr.source_id)
            temps.append(sr.temperature)
            powers.append(sr.stefan_boltzmann_power)
            emissivities.append(sr.emissivity)

        temps_arr = np.array(temps)
        powers_arr = np.array(powers)
        eps_arr = np.array(emissivities)
        T4 = temps_arr ** 4

        normalized_power = powers_arr / eps_arr
        if len(T4) >= 2:
            _, _, r_value, _, _ = stats.linregress(T4, normalized_power)
            r2 = r_value ** 2
        else:
            r2 = 1.0

        result = StefanBoltzmannVerification(
            source_ids=source_ids,
            fitted_temperatures=temps_arr,
            sb_power=powers_arr,
            emissivities=eps_arr,
            T4_values=T4,
            r_squared=r2,
        )

        logger.info("Stefan-Boltzmann 验证完成: R²=%.6f", r2)
        return result

    def residual_analysis(self) -> Dict[str, float]:
        """残差统计分析"""
        rd = self.master_df["rd_percent"]
        residuals = self.master_df["residual"]

        _, shapiro_p = stats.shapiro(rd) if len(rd) >= 3 else (0, 0)

        result = {
            "mean_rd_percent": float(rd.mean()),
            "std_rd_percent": float(rd.std()),
            "median_rd_percent": float(rd.median()),
            "max_abs_rd_percent": float(rd.abs().max()),
            "skewness": float(rd.skew()),
            "kurtosis": float(rd.kurtosis()),
            "shapiro_p_value": float(shapiro_p),
            "mean_abs_residual": float(residuals.abs().mean()),
            "max_abs_residual": float(residuals.abs().max()),
            "rmsd": float(np.sqrt((residuals ** 2).mean())),
        }

        logger.info(
            "残差分析: 均值偏差=%.4f%%, 标准差=%.4f%%, 偏度=%.4f",
            result["mean_rd_percent"], result["std_rd_percent"], result["skewness"],
        )
        return result

    def detect_outliers(self, z_threshold: float = 2.0) -> List[int]:
        """基于 Z-score 的异常值检测"""
        rd = self.master_df["rd_percent"]
        z_scores = np.abs(stats.zscore(rd))
        outlier_mask = z_scores > z_threshold
        outlier_indices = list(self.master_df.index[outlier_mask])

        if outlier_indices:
            logger.warning(
                "检测到 %d 个潜在异常点 (Z>%.1f): 索引 %s",
                len(outlier_indices), z_threshold, outlier_indices,
            )
        else:
            logger.info("未检测到异常点 (Z 阈值=%.1f)", z_threshold)

        return outlier_indices

    def parameter_sensitivity(
        self,
        perturbation_range: float = 0.10,
        n_points: int = 21,
    ) -> tuple[List[SensitivityResult], List[SensitivityResult]]:
        """
        参数灵敏度分析

        对每个辐射源的 T 和 ε 分别施加 ±perturbation_range 的扰动，
        计算每个扰动水平下的 AARD。
        """
        perturbations = np.linspace(-perturbation_range, perturbation_range, n_points)
        sens_T_list = []
        sens_eps_list = []

        for sr in self.result.source_results:
            base_T = sr.temperature
            base_eps = sr.emissivity
            wavelengths = sr.wavelengths
            B_exp = sr.exp_radiance

            def compute_aard(T: float, eps: float) -> float:
                B_calc = self.model.spectral_radiance_batch(wavelengths, T, eps)
                return float(np.mean(np.abs((B_calc - B_exp) / B_exp)) * 100)

            aard_T = np.array([
                compute_aard(base_T * (1 + p), base_eps) for p in perturbations
            ])
            aard_eps = np.array([
                compute_aard(base_T, np.clip(base_eps * (1 + p), 0.01, 1.0))
                for p in perturbations
            ])

            grad_T = float(np.gradient(aard_T, base_T * perturbations).mean()) if base_T != 0 else 0.0
            grad_eps = float(np.gradient(aard_eps, base_eps * perturbations).mean()) if base_eps != 0 else 0.0

            sens_T_list.append(SensitivityResult(
                parameter_name="T",
                source_id=sr.source_id,
                base_value=base_T,
                perturbations=perturbations,
                perturbed_values=base_T * (1 + perturbations),
                aard_values=aard_T,
                gradient=grad_T,
            ))
            sens_eps_list.append(SensitivityResult(
                parameter_name="ε",
                source_id=sr.source_id,
                base_value=base_eps,
                perturbations=perturbations,
                perturbed_values=base_eps * (1 + perturbations),
                aard_values=aard_eps,
                gradient=grad_eps,
            ))

        logger.info("灵敏度分析完成: %d 个辐射源", len(self.result.source_results))
        return sens_T_list, sens_eps_list

    def group_analysis(self) -> pd.DataFrame:
        """按波长区间分组统计拟合表现"""
        df = self.master_df.dropna(subset=["wave_group"])
        grouped = df.groupby("wave_group", observed=True).agg(
            n_points=("wavelength_um", "count"),
            mean_B_exp=("B_exp", "mean"),
            mean_B_calc=("B_calc", "mean"),
            mean_abs_rd=("abs_rd_percent", "mean"),
            max_abs_rd=("abs_rd_percent", "max"),
            mean_wavelength=("wavelength_um", "mean"),
        ).round(6)
        logger.info("波长分组分析完成: %d 个分组", len(grouped))
        return grouped

    def correlation_analysis(self) -> pd.DataFrame:
        """计算核心变量间的 Pearson 相关系数矩阵"""
        cols = ["wavelength_um", "B_exp", "B_calc", "temperature", "emissivity", "rd_percent"]
        corr = self.master_df[cols].corr().round(4)
        logger.info("相关性分析完成: %d × %d 矩阵", *corr.shape)
        return corr

    def run_full_analysis(self) -> AnalysisReport:
        """执行完整分析流程，返回汇总报告"""
        logger.info("=" * 50)
        logger.info("开始执行完整数据分析...")
        logger.info("=" * 50)

        desc = self.descriptive_statistics()
        wien = self.wien_displacement_verification()
        sb = self.stefan_boltzmann_verification()
        resid = self.residual_analysis()
        outliers = self.detect_outliers()
        sens_T, sens_eps = self.parameter_sensitivity()
        groups = self.group_analysis()
        corr = self.correlation_analysis()

        report = AnalysisReport(
            master_df=self.master_df,
            descriptive_stats=desc,
            wien_verification=wien,
            sb_verification=sb,
            residual_stats=resid,
            outlier_indices=outliers,
            sensitivity_T=sens_T,
            sensitivity_eps=sens_eps,
            group_analysis=groups,
            correlation_matrix=corr,
        )

        logger.info("完整数据分析已完成")
        return report

    def export_master_csv(self, output_dir: str = "output") -> str:
        """将主数据表导出为 CSV"""
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        filepath = path / "analysis_master_data.csv"
        self.master_df.to_csv(filepath, index=False, float_format="%.6e")
        logger.info("主数据表已导出: %s", filepath)
        return str(filepath)

    def export_analysis_report(self, output_dir: str = "output") -> str:
        """将分析报告导出为 JSON"""
        import json

        report = self.run_full_analysis()
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)

        data: Dict[str, Any] = {
            "descriptive_statistics": report.descriptive_stats.to_dict(),
            "wien_verification": {
                "source_ids": report.wien_verification.source_ids,
                "fitted_temperatures_K": report.wien_verification.fitted_temperatures.tolist(),
                "wien_peak_calc_um": report.wien_verification.wien_peak_calc.tolist(),
                "observed_peak_approx_um": report.wien_verification.observed_peak_approx.tolist(),
                "deviations_percent": report.wien_verification.deviations_percent.tolist(),
            },
            "stefan_boltzmann_verification": {
                "r_squared": report.sb_verification.r_squared,
                "sb_power_W_per_m2": report.sb_verification.sb_power.tolist(),
            },
            "residual_analysis": report.residual_stats,
            "outlier_indices": report.outlier_indices,
            "group_analysis": report.group_analysis.to_dict(),
            "correlation_matrix": report.correlation_matrix.to_dict(),
            "sensitivity": {
                "temperature": [
                    {
                        "source_id": s.source_id,
                        "base_value": s.base_value,
                        "gradient": s.gradient,
                    }
                    for s in report.sensitivity_T
                ],
                "emissivity": [
                    {
                        "source_id": s.source_id,
                        "base_value": s.base_value,
                        "gradient": s.gradient,
                    }
                    for s in report.sensitivity_eps
                ],
            },
        }

        filepath = path / "analysis_report.json"
        filepath.write_text(
            json.dumps(data, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        logger.info("分析报告 JSON 已导出: %s", filepath)
        return str(filepath)
