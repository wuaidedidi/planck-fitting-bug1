"""
Planck 模型参数拟合优化模块

使用两阶段优化策略（差分进化 + L-BFGS-B 有界局部优化）拟合黑体辐射源的
有效温度 T 和发射率 ε，最小化计算辐射度与实验辐射度之间的偏差。
"""

import logging
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from scipy.optimize import differential_evolution, minimize

from src.data.spectrum import RadiationSource, SpectralDataset, PhysicalConstants
from src.models.planck import PlanckModel
from src.utils.logger import get_logger

logger = get_logger("fitting.optimizer")


@dataclass
class SourceFittingResult:
    """单个辐射源的拟合结果"""

    source_id: str
    description: str
    temperature: float
    emissivity: float
    objective_value: float
    rmsd: float
    aard_percent: float
    r_squared: float
    wavelengths: np.ndarray
    exp_radiance: np.ndarray
    calc_radiance: np.ndarray
    residuals: np.ndarray
    relative_deviations: np.ndarray
    wien_peak_calc: float
    stefan_boltzmann_power: float
    convergence_message: str = ""
    iterations: int = 0


@dataclass
class FittingResult:
    """全局拟合结果（包含所有辐射源）"""

    source_results: List[SourceFittingResult] = field(default_factory=list)
    overall_aard: float = 0.0
    overall_r_squared: float = 0.0

    def summary(self) -> str:
        """生成拟合结果的文本摘要"""
        lines = [
            "",
            "=" * 70,
            "Planck 黑体辐射模型参数拟合结果",
            "=" * 70,
        ]

        for sr in self.source_results:
            lines.extend([
                "",
                f"━━━ {sr.source_id}: {sr.description} ━━━",
                f"  有效温度 T  = {sr.temperature:>10.2f} K",
                f"  发射率   ε  = {sr.emissivity:>10.4f}",
                f"  Wien 峰值   = {sr.wien_peak_calc:>10.4f} μm",
                f"  S-B 功率    = {sr.stefan_boltzmann_power:>10.2f} W/m²",
                "",
                f"  目标函数值  = {sr.objective_value:.6e}",
                f"  RMSD        = {sr.rmsd:.6e}",
                f"  AARD        = {sr.aard_percent:.4f} %",
                f"  R²          = {sr.r_squared:.8f}",
                "",
                f"  {'λ (μm)':>10s}  {'B_exp':>14s}  {'B_calc':>14s}  {'RD (%)':>10s}",
                "  " + "-" * 55,
            ])
            for i in range(len(sr.wavelengths)):
                lines.append(
                    f"  {sr.wavelengths[i]:>10.2f}"
                    f"  {sr.exp_radiance[i]:>14.4e}"
                    f"  {sr.calc_radiance[i]:>14.4e}"
                    f"  {sr.relative_deviations[i] * 100:>10.4f}"
                )

        lines.extend([
            "",
            "━━━ 全局拟合质量 ━━━",
            f"  总体 AARD = {self.overall_aard:.4f} %",
            f"  总体 R²   = {self.overall_r_squared:.8f}",
            "=" * 70,
        ])
        return "\n".join(lines)


class PlanckOptimizer:
    """
    Planck 模型参数优化器

    使用两阶段优化策略:
    1. 差分进化算法 (DE) - 全局搜索，避免局部最优
    2. L-BFGS-B - 有界局部精细优化
    """

    def __init__(
        self,
        dataset: SpectralDataset,
        constants: PhysicalConstants | None = None,
        T_bounds: Tuple[float, float] = (1000.0, 10000.0),
        eps_bounds: Tuple[float, float] = (0.01, 1.0),
    ) -> None:
        """
        初始化优化器。

        Args:
            dataset: 光谱实验数据集
            constants: 物理常数
            T_bounds: 温度搜索范围 (K)
            eps_bounds: 发射率搜索范围
        """
        for src in dataset.sources:
            if src.size < 2:
                raise ValueError(
                    f"辐射源 {src.source_id} 数据点不足，至少需要 2 个点，当前: {src.size}"
                )

        self.dataset = dataset
        self.model = PlanckModel(constants)
        self.T_bounds = T_bounds
        self.eps_bounds = eps_bounds
        self._eval_count = 0

        logger.info(
            "优化器初始化: %d 个辐射源, T 范围 %s, ε 范围 %s",
            len(dataset.sources), T_bounds, eps_bounds,
        )

    def _objective_function(
        self, params: np.ndarray, source: RadiationSource
    ) -> float:
        """
        目标函数: 最小化相对偏差的平方和。

        OF = Σ [(B_calc - B_exp) / B_exp]²

        Args:
            params: [T, ε]
            source: 辐射源数据

        Returns:
            目标函数值
        """
        T, emissivity = params
        self._eval_count += 1

        try:
            B_calc = self.model.spectral_radiance_batch(
                source.wavelengths, T, emissivity
            )
            relative_errors = (B_calc - source.radiances) / source.radiances
            obj_value = np.sum(relative_errors ** 2)

            if np.isnan(obj_value) or np.isinf(obj_value):
                return 1e20

            return obj_value

        except (ValueError, OverflowError, FloatingPointError) as e:
            logger.debug("目标函数计算异常 (T=%.1f, ε=%.4f): %s", T, emissivity, e)
            return 1e20

    def fit_source(
        self,
        source: RadiationSource,
        de_seed: int = 42,
        de_maxiter: int = 1000,
        de_tol: float = 1e-12,
        de_popsize: int = 25,
        local_maxiter: int = 5000,
        local_tol: float = 1e-14,
    ) -> SourceFittingResult:
        """
        对单个辐射源执行两阶段参数拟合。

        Args:
            source: 辐射源数据
            de_seed: 差分进化随机种子
            de_maxiter: DE 最大迭代次数
            de_tol: DE 收敛容差
            de_popsize: DE 种群大小
            local_maxiter: 局部优化最大迭代次数
            local_tol: 局部优化收敛容差

        Returns:
            SourceFittingResult 拟合结果
        """
        logger.info("开始拟合辐射源 %s: %s", source.source_id, source.description)

        bounds = [self.T_bounds, self.eps_bounds]
        self._eval_count = 0

        planck_logger = logging.getLogger("planck_fitting.models.planck")
        original_level = planck_logger.level
        planck_logger.setLevel(logging.ERROR)

        logger.info("  阶段 1: 差分进化全局搜索")
        de_result = differential_evolution(
            self._objective_function,
            bounds=bounds,
            args=(source,),
            seed=de_seed,
            maxiter=de_maxiter,
            tol=de_tol,
            popsize=de_popsize,
            strategy="best1bin",
            mutation=(0.5, 1.5),
            recombination=0.9,
            polish=False,
        )
        logger.info(
            "  DE 完成: T=%.2f K, ε=%.4f, OF=%.6e, 评估=%d",
            de_result.x[0], de_result.x[1], de_result.fun, self._eval_count,
        )

        logger.info("  阶段 2: L-BFGS-B 局部精细优化")
        eval_before = self._eval_count
        local_result = minimize(
            self._objective_function,
            x0=de_result.x,
            args=(source,),
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": local_maxiter,
                "ftol": local_tol,
                "gtol": 1e-10,
            },
        )

        planck_logger.setLevel(original_level)

        logger.info(
            "  局部优化完成: T=%.2f K, ε=%.4f, OF=%.6e, 额外评估=%d",
            local_result.x[0], local_result.x[1],
            local_result.fun, self._eval_count - eval_before,
        )

        T_opt, eps_opt = local_result.x
        return self._build_source_result(source, T_opt, eps_opt, local_result)

    def fit_all(
        self,
        de_seed: int = 42,
        de_maxiter: int = 1000,
        de_tol: float = 1e-12,
        de_popsize: int = 25,
        local_maxiter: int = 5000,
        local_tol: float = 1e-14,
    ) -> FittingResult:
        """
        对所有辐射源执行参数拟合。

        Returns:
            FittingResult 包含所有辐射源的拟合结果
        """
        source_results = []
        for i, src in enumerate(self.dataset.sources):
            sr = self.fit_source(
                src,
                de_seed=de_seed + i,
                de_maxiter=de_maxiter,
                de_tol=de_tol,
                de_popsize=de_popsize,
                local_maxiter=local_maxiter,
                local_tol=local_tol,
            )
            source_results.append(sr)

        all_rd = np.concatenate([sr.relative_deviations for sr in source_results])
        all_exp = np.concatenate([sr.exp_radiance for sr in source_results])
        all_calc = np.concatenate([sr.calc_radiance for sr in source_results])

        overall_aard = float(np.mean(np.abs(all_rd)) * 100)
        ss_res = np.sum((all_calc - all_exp) ** 2)
        ss_tot = np.sum((all_exp - np.mean(all_exp)) ** 2)
        overall_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        result = FittingResult(
            source_results=source_results,
            overall_aard=overall_aard,
            overall_r_squared=overall_r2,
        )

        logger.info("全部拟合完成: 总体 AARD=%.4f%%, R²=%.8f", overall_aard, overall_r2)
        return result

    def _build_source_result(
        self, source: RadiationSource, T: float, emissivity: float, opt_result
    ) -> SourceFittingResult:
        """构建单个辐射源的完整拟合结果。"""
        B_calc = self.model.spectral_radiance_batch(source.wavelengths, T, emissivity)
        B_exp = source.radiances

        residuals = B_calc - B_exp
        relative_devs = residuals / B_exp

        rmsd = float(np.sqrt(np.mean(residuals ** 2)))
        aard = float(np.mean(np.abs(relative_devs)) * 100)

        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((B_exp - np.mean(B_exp)) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        wien_peak = self.model.wien_peak_wavelength(T)
        sb_power = self.model.stefan_boltzmann_power(T, emissivity)

        result = SourceFittingResult(
            source_id=source.source_id,
            description=source.description,
            temperature=T,
            emissivity=emissivity,
            objective_value=opt_result.fun,
            rmsd=rmsd,
            aard_percent=aard,
            r_squared=r_squared,
            wavelengths=source.wavelengths,
            exp_radiance=B_exp,
            calc_radiance=B_calc,
            residuals=residuals,
            relative_deviations=relative_devs,
            wien_peak_calc=wien_peak,
            stefan_boltzmann_power=sb_power,
            convergence_message=str(opt_result.message) if hasattr(opt_result, "message") else "完成",
            iterations=opt_result.nit if hasattr(opt_result, "nit") else 0,
        )

        logger.info(
            "  %s 拟合质量: T=%.2f K, ε=%.4f, AARD=%.4f%%, R²=%.8f",
            source.source_id, T, emissivity, aard, r_squared,
        )
        return result
