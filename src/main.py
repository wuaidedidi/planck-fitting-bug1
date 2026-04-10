"""
黑体辐射 (Black-Body Radiation) Planck 模型参数拟合与数据分析 — 主入口

完整工作流程:
1. 加载并校验三组辐射源实验光谱数据
2. 初始化物理常数与 Planck 模型
3. 对每个辐射源执行两阶段参数拟合 (差分进化 + L-BFGS-B)
4. 生成拟合可视化图表
5. Pandas 数据分析 (统计/Wien 验证/S-B 验证/灵敏度/异常检测)
6. 生成分析图表 (热力图/灵敏度/残差分布)
7. 输出详细报告 (TXT / JSON / CSV)
"""

import sys
import time

from src.analysis.analyzer import SpectrumAnalyzer
from src.data.spectrum import SpectralDataset, PhysicalConstants
from src.fitting.optimizer import PlanckOptimizer
from src.report.generator import ReportGenerator
from src.utils.logger import setup_logger, get_logger
from src.visualization.plotter import ResultPlotter


def main() -> int:
    """
    主函数：执行黑体辐射 Planck 模型参数拟合全流程。

    Returns:
        0 表示成功，1 表示失败
    """
    root_logger = setup_logger()
    logger = get_logger("main")

    logger.info("=" * 70)
    logger.info("黑体辐射 (Black-Body Radiation) 光谱分析系统")
    logger.info("Planck 辐射定律参数拟合程序启动")
    logger.info("=" * 70)

    start_time = time.time()

    try:
        # ──────────────────────────────────────────────
        # Step 1: 加载实验数据
        # ──────────────────────────────────────────────
        logger.info("━━━ Step 1: 加载实验数据 ━━━")
        dataset = SpectralDataset.load_default()
        logger.info("\n%s", dataset.summary())

        # ──────────────────────────────────────────────
        # Step 2: 初始化物理常数
        # ──────────────────────────────────────────────
        logger.info("━━━ Step 2: 初始化物理常数 ━━━")
        constants = PhysicalConstants()
        constants.validate()

        # ──────────────────────────────────────────────
        # Step 3: 执行参数拟合
        # ──────────────────────────────────────────────
        logger.info("━━━ Step 3: 执行 Planck 参数拟合 ━━━")
        optimizer = PlanckOptimizer(
            dataset=dataset,
            constants=constants,
            T_bounds=(1000.0, 10000.0),
            eps_bounds=(0.01, 1.0),
        )
        result = optimizer.fit_all(
            de_seed=42,
            de_maxiter=1000,
            de_tol=1e-12,
            de_popsize=25,
            local_maxiter=5000,
            local_tol=1e-14,
        )

        logger.info(result.summary())

        # ──────────────────────────────────────────────
        # Step 4: 生成可视化图表
        # ──────────────────────────────────────────────
        logger.info("━━━ Step 4: 生成可视化图表 ━━━")
        plotter = ResultPlotter(output_dir="output/figures")
        plotter.plot_all(result)

        # ──────────────────────────────────────────────
        # Step 5: 数据分析 (Pandas)
        # ──────────────────────────────────────────────
        logger.info("━━━ Step 5: 数据分析 (Pandas) ━━━")
        analyzer = SpectrumAnalyzer(
            dataset=dataset, result=result, constants=constants,
        )
        analysis_report = analyzer.run_full_analysis()
        analyzer.export_master_csv("output")
        analyzer.export_analysis_report("output")

        logger.info("  DataFrame 形状: %s", str(analysis_report.master_df.shape))
        logger.info(
            "  Wien 验证: %s",
            ", ".join(
                f"{sid}: λ={lam:.4f}μm"
                for sid, lam in zip(
                    analysis_report.wien_verification.source_ids,
                    analysis_report.wien_verification.wien_peak_calc,
                )
            ),
        )

        # ──────────────────────────────────────────────
        # Step 6: 生成分析图表
        # ──────────────────────────────────────────────
        logger.info("━━━ Step 6: 生成分析图表 ━━━")
        plotter.plot_analysis_all(analysis_report)

        # ──────────────────────────────────────────────
        # Step 7: 生成报告
        # ──────────────────────────────────────────────
        logger.info("━━━ Step 7: 生成报告 ━━━")
        reporter = ReportGenerator(output_dir="output")
        reporter.generate_all(result)
        reporter.generate_analysis_summary(analysis_report, result)

        elapsed = time.time() - start_time

        logger.info("")
        logger.info("=" * 70)
        logger.info("拟合与分析完成！总耗时: %.2f 秒", elapsed)
        logger.info("=" * 70)
        logger.info("拟合结果:")
        for sr in result.source_results:
            logger.info(
                "  %s: T=%.2f K, ε=%.4f, AARD=%.4f%%, R²=%.8f",
                sr.source_id, sr.temperature, sr.emissivity,
                sr.aard_percent, sr.r_squared,
            )
        logger.info("  总体 AARD = %.4f %%", result.overall_aard)
        logger.info("  总体 R²   = %.8f", result.overall_r_squared)
        logger.info("")
        logger.info("输出文件:")
        logger.info("  报告:     output/fitting_report.txt")
        logger.info("  分析摘要: output/analysis_summary.txt")
        logger.info("  数据:     output/fitting_result.json")
        logger.info("  分析JSON: output/analysis_report.json")
        logger.info("  CSV:      output/fitting_data.csv")
        logger.info("  主数据表: output/analysis_master_data.csv")
        logger.info("  图表:     output/figures/")
        logger.info("  日志:     output/logs/planck_fitting.log")
        logger.info("=" * 70)

        return 0

    except ValueError as e:
        logger.error("数据校验错误: %s", e)
        return 1
    except RuntimeError as e:
        logger.error("运行时错误: %s", e)
        return 1
    except Exception as e:
        logger.error("未预期的错误: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
