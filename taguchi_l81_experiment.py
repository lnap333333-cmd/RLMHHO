#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RL-Chaotic-HHO算法L81田口正交实验设计
针对100×5×3规模，总机器数40的MO-DHFSP问题
4个参数9个水平的全面参数调优实验
评价指标：超体积:IGD:GD = 5:3:2
"""

import numpy as np
import pandas as pd
import time
import json
import pickle
from datetime import datetime
from typing import List, Dict, Tuple, Any
import logging
import os

# 导入算法和问题定义
from problem.mo_dhfsp import MO_DHFSP_Problem
from algorithm.rl_chaotic_hho import RL_ChaoticHHO_Optimizer
from algorithm.nsga2 import NSGA2_Optimizer
from algorithm.moead import MOEAD_Optimizer
from algorithm.mopso import MOPSO_Optimizer
from utils.data_generator import DataGenerator
from utils.performance_metrics import PerformanceEvaluator

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('taguchi_l81_experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TaguchiL81Experiment:
    """L81田口正交实验主控制类 - 4参数9水平"""
    
    def __init__(self):
        self.problem_config = {
            'n_jobs': 100,
            'n_factories': 5,
            'n_stages': 3,
            'total_machines': 40,
            'processing_time_range': (1, 10),
            'due_date_tightness': 1.5,
            'random_seed': 2025
        }
        
        # L81正交表参数配置 - 4参数9水平
        self.factor_levels = self._initialize_factor_levels()
        self.l81_design = self._generate_l81_design()
        
        # 实验控制参数
        self.runs_per_experiment = 5  # 每组实验重复5次
        self.max_iterations = 50     # 算法迭代次数
        
        # 结果存储
        self.results_dir = f"taguchi_l81_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 性能评估器
        self.metrics_evaluator = MetricsEvaluator()
        
    def _initialize_factor_levels(self) -> Dict:
        """初始化4因子9水平参数"""
        return {
            'A_learning_rate': {
                1: 0.00001,  # 极低学习率
                2: 0.00005,  # 很低学习率
                3: 0.0001,   # 低学习率
                4: 0.0005,   # 中低学习率
                5: 0.001,    # 中等学习率
                6: 0.002,    # 中高学习率
                7: 0.005,    # 高学习率
                8: 0.01,     # 很高学习率
                9: 0.02      # 极高学习率
            },
            'B_epsilon_decay': {
                1: 0.985,    # 快速衰减
                2: 0.988,    # 较快衰减
                3: 0.990,    # 中快衰减
                4: 0.993,    # 中等衰减
                5: 0.995,    # 标准衰减
                6: 0.997,    # 慢衰减
                7: 0.999,    # 很慢衰减
                8: 0.9995,   # 极慢衰减
                9: 0.9999    # 最慢衰减
            },
            'C_group_ratios': {
                1: [0.80, 0.10, 0.07, 0.03],  # 极端探索主导
                2: [0.70, 0.15, 0.10, 0.05],  # 超级探索主导
                3: [0.60, 0.20, 0.15, 0.05],  # 强探索主导
                4: [0.50, 0.30, 0.15, 0.05],  # 探索主导
                5: [0.45, 0.25, 0.20, 0.10],  # 基准平衡
                6: [0.35, 0.40, 0.20, 0.05],  # 开发主导
                7: [0.25, 0.45, 0.20, 0.10],  # 强开发主导
                8: [0.20, 0.50, 0.20, 0.10],  # 超级开发主导
                9: [0.15, 0.55, 0.20, 0.10]   # 极端开发主导
            },
            'D_gamma': {
                1: 0.75,     # 短期记忆
                2: 0.80,     # 较短记忆
                3: 0.85,     # 中短记忆
                4: 0.90,     # 中等记忆
                5: 0.95,     # 标准记忆
                6: 0.98,     # 长记忆
                7: 0.99,     # 很长记忆
                8: 0.995,    # 极长记忆
                9: 0.999     # 最长记忆
            }
        }
    
    def _generate_l81_design(self) -> List[Dict]:
        """生成L81(9^4)正交表设计"""
        l81_experiments = []
        exp_id = 1
        
        # 生成L81正交表（9^4设计）
        for a in range(1, 10):  # A因子：9个水平
            for b in range(1, 10):  # B因子：9个水平
                if exp_id > 81:  # 限制为81组实验
                    break
                c = ((a - 1) + (b - 1)) % 9 + 1  # C因子：基于A和B计算
                d = ((a - 1) * 2 + (b - 1) * 3) % 9 + 1  # D因子：基于A和B的复合计算
                
                l81_experiments.append({
                    'exp_id': exp_id,
                    'A': a,
                    'B': b,
                    'C': c,
                    'D': d
                })
                exp_id += 1
            
            if exp_id > 81:  # 限制为81组实验
                break
        
        return l81_experiments
    
    def generate_problem_instance(self) -> MO_DHFSP_Problem:
        """生成标准化的问题实例"""
        logger.info("生成100×5×3规模问题实例，总机器数40")
        
        generator = DataGenerator(seed=self.problem_config['random_seed'])
        
        # 分配各工厂各阶段的机器数量，确保总数为40
        machines_config = self._distribute_machines()
        
        problem_data = generator.generate_problem(
            n_jobs=self.problem_config['n_jobs'],
            n_factories=self.problem_config['n_factories'],
            n_stages=self.problem_config['n_stages'],
            machines_per_stage=machines_config[0],  # 使用第一个工厂的配置作为基准
            processing_time_range=self.problem_config['processing_time_range'],
            due_date_tightness=self.problem_config['due_date_tightness']
        )
        
        # 添加多工厂机器配置信息（使用整数键）
        problem_data['factory_machines'] = {
            i: machines_config[i] for i in range(len(machines_config))
        }
        
        problem = MO_DHFSP_Problem(problem_data)
        
        # 保存问题实例
        with open(f"{self.results_dir}/problem_instance.pkl", 'wb') as f:
            pickle.dump(problem, f)
        
        logger.info(f"问题实例已保存，总机器数: {sum(sum(stage) for stage in machines_config)}")
        return problem
    
    def _distribute_machines(self) -> List[List[int]]:
        """分配40台机器到5个工厂3个阶段"""
        # 确保异构配置，每个工厂每个阶段至少1台机器
        machines_config = [
            [3, 2, 2],  # 工厂1: 7台机器
            [3, 3, 2],  # 工厂2: 8台机器  
            [2, 3, 3],  # 工厂3: 8台机器
            [3, 2, 3],  # 工厂4: 8台机器
            [3, 3, 3]   # 工厂5: 9台机器
        ]
        # 总计: 7+8+8+8+9 = 40台机器
        return machines_config
    
    def run_single_experiment(self, exp_config: Dict, run_id: int, problem: MO_DHFSP_Problem) -> Dict:
        """运行单次实验"""
        exp_id = exp_config['exp_id']
        
        # 获取实验参数
        params = self._get_experiment_parameters(exp_config)
        
        logger.info(f"实验{exp_id}-运行{run_id}: LR={params['learning_rate']:.5f}, "
                   f"Decay={params['epsilon_decay']:.4f}, "
                   f"Groups={params['group_ratios']}, "
                   f"Gamma={params['gamma']:.3f}")
        
        start_time = time.time()
        
        try:
            # 创建优化器
            optimizer = RL_ChaoticHHO_Optimizer(problem, **params)
            
            # 运行优化
            pareto_solutions, convergence_data = optimizer.optimize()
            runtime = time.time() - start_time
            
            # 评估性能
            metrics = self.metrics_evaluator.evaluate_performance(
                pareto_solutions, problem, runtime
            )
            
            result = {
                'exp_id': exp_id,
                'run_id': run_id,
                'exp_config': exp_config,
                'parameters': params,
                'pareto_solutions': pareto_solutions,
                'metrics': metrics,
                'convergence_data': convergence_data,
                'runtime': runtime,
                'timestamp': datetime.now().isoformat(),
                'success': True
            }
            
            logger.info(f"实验{exp_id}-运行{run_id}完成: HV={metrics['hypervolume']:.4f}, "
                       f"IGD={metrics['igd']:.4f}, GD={metrics['gd']:.4f}, "
                       f"时间={runtime:.2f}s")
            
        except Exception as e:
            logger.error(f"实验{exp_id}-运行{run_id}失败: {str(e)}")
            result = {
                'exp_id': exp_id,
                'run_id': run_id,
                'exp_config': exp_config,
                'parameters': params,
                'error': str(e),
                'success': False
            }
        
        return result
    
    def _get_experiment_parameters(self, exp_config: Dict) -> Dict:
        """根据实验配置获取具体参数值"""
        params = {
            'learning_rate': self.factor_levels['A_learning_rate'][exp_config['A']],
            'epsilon_decay': self.factor_levels['B_epsilon_decay'][exp_config['B']],
            'group_ratios': self.factor_levels['C_group_ratios'][exp_config['C']],
            'gamma': self.factor_levels['D_gamma'][exp_config['D']],
            # 固定参数
            'population_size': 100,
            'max_iterations': self.max_iterations,
            'epsilon': 0.9,
            'epsilon_min': 0.01
        }
        return params
    
    def run_experiment_group(self, exp_config: Dict, problem: MO_DHFSP_Problem) -> Dict:
        """运行单组实验（5次重复）"""
        exp_id = exp_config['exp_id']
        logger.info(f"开始实验组{exp_id} ({exp_id}/81)")
        
        group_results = []
        
        # 运行5次重复实验
        for run_id in range(1, self.runs_per_experiment + 1):
            result = self.run_single_experiment(exp_config, run_id, problem)
            group_results.append(result)
            
            # 保存单次实验结果
            with open(f"{self.results_dir}/exp_{exp_id:02d}_run_{run_id}.json", 'w') as f:
                json.dump(result, f, indent=2, default=str)
        
        # 统计分析
        statistics = self._analyze_group_results(group_results)
        
        group_summary = {
            'exp_id': exp_id,
            'exp_config': exp_config,
            'individual_results': group_results,
            'statistics': statistics,
            'timestamp': datetime.now().isoformat()
        }
        
        # 保存组汇总结果
        with open(f"{self.results_dir}/exp_{exp_id:02d}_summary.json", 'w') as f:
            json.dump(group_summary, f, indent=2, default=str)
        
        logger.info(f"实验组{exp_id}完成: 平均HV={statistics['hv_mean']:.4f}, "
                   f"平均IGD={statistics['igd_mean']:.4f}, "
                   f"平均GD={statistics['gd_mean']:.4f}, "
                   f"综合得分={statistics['comprehensive_mean']:.4f}, "
                   f"SNR={statistics['snr_value']:.2f}")
        
        return group_summary
    
    def _analyze_group_results(self, group_results: List[Dict]) -> Dict:
        """分析组实验结果"""
        successful_results = [r for r in group_results if r.get('success', False)]
        
        if not successful_results:
            return {
                'success_count': 0,
                'hv_mean': 0.0, 'hv_std': 0.0,
                'igd_mean': float('inf'), 'igd_std': 0.0,
                'gd_mean': float('inf'), 'gd_std': 0.0,
                'comprehensive_mean': 0.0, 'comprehensive_std': 0.0,
                'runtime_mean': 0.0, 'runtime_std': 0.0,
                'snr_value': -50.0
            }
        
        # 提取指标
        hvs = [r['metrics']['hypervolume'] for r in successful_results]
        igds = [r['metrics']['igd'] for r in successful_results]
        gds = [r['metrics']['gd'] for r in successful_results]
        comprehensives = [r['metrics']['comprehensive'] for r in successful_results]
        runtimes = [r['runtime'] for r in successful_results]
        
        # 计算统计量
        stats = {
            'success_count': len(successful_results),
            'hv_mean': np.mean(hvs), 'hv_std': np.std(hvs),
            'igd_mean': np.mean(igds), 'igd_std': np.std(igds),
            'gd_mean': np.mean(gds), 'gd_std': np.std(gds),
            'comprehensive_mean': np.mean(comprehensives), 'comprehensive_std': np.std(comprehensives),
            'runtime_mean': np.mean(runtimes), 'runtime_std': np.std(runtimes)
        }
        
        # 计算信噪比
        stats['snr_value'] = self.metrics_evaluator.calculate_snr_comprehensive(comprehensives)
        
        return stats
    
    def run_all_experiments(self):
        """运行所有81组实验"""
        logger.info("开始L81田口正交实验")
        
        # 生成问题实例
        problem = self.generate_problem_instance()
        
        # 生成参考前沿
        logger.info("生成参考前沿...")
        self.metrics_evaluator.generate_reference_front(problem)
        
        # 运行所有实验组
        all_results = []
        total_experiments = len(self.l81_design)
        
        for i, exp_config in enumerate(self.l81_design):
            logger.info(f"进度: {i+1}/{total_experiments}")
            
            try:
                group_result = self.run_experiment_group(exp_config, problem)
                all_results.append(group_result)
                
                # 每5组实验保存一次中间结果
                if (i + 1) % 5 == 0:
                    self._save_intermediate_results(all_results[:i+1])
                    
            except Exception as e:
                logger.error(f"实验组{exp_config['exp_id']}运行失败: {str(e)}")
        
        # 保存最终结果
        self._save_final_results(all_results)
        
        # 进行田口分析
        logger.info("开始田口分析...")
        taguchi_results = self._perform_taguchi_analysis(all_results)
        
        logger.info("L81田口实验完成!")
        return all_results, taguchi_results
    
    def _save_intermediate_results(self, results: List[Dict]):
        """保存中间结果"""
        with open(f"{self.results_dir}/intermediate_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    def _save_final_results(self, results: List[Dict]):
        """保存最终结果并导出Excel"""
        # 保存JSON格式
        with open(f"{self.results_dir}/final_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # 导出Excel格式
        self._export_to_excel(results)
    
    def _export_to_excel(self, results: List[Dict]):
        """导出结果到Excel文件"""
        logger.info("开始导出到Excel文件")
        
        # 准备数据
        data_rows = []
        for result in results:
            exp_config = result['exp_config']
            stats = result['statistics']
            
            row = {
                '实验ID': result['exp_id'],
                'A_学习率': self.factor_levels['A_learning_rate'][exp_config['A']],
                'B_衰减率': self.factor_levels['B_epsilon_decay'][exp_config['B']],
                'C_分组比例': str(self.factor_levels['C_group_ratios'][exp_config['C']]),
                'D_折扣因子': self.factor_levels['D_gamma'][exp_config['D']],
                '成功次数': stats['success_count'],
                '平均超体积': stats['hv_mean'],
                '超体积标准差': stats['hv_std'],
                '平均IGD': stats['igd_mean'],
                'IGD标准差': stats['igd_std'],
                '平均GD': stats['gd_mean'],
                'GD标准差': stats['gd_std'],
                '平均综合得分': stats['comprehensive_mean'],
                '综合得分标准差': stats['comprehensive_std'],
                '平均运行时间': stats['runtime_mean'],
                'SNR值': stats['snr_value']
            }
            data_rows.append(row)
        
        # 创建DataFrame并保存
        df = pd.DataFrame(data_rows)
        excel_filename = f"{self.results_dir}/L81_experiment_results.xlsx"
        df.to_excel(excel_filename, index=False, sheet_name='实验结果')
        
        logger.info(f"结果已导出到Excel: {excel_filename}")
    
    def _perform_taguchi_analysis(self, results: List[Dict]) -> Dict:
        """执行田口分析"""
        analyzer = TaguchiL81Analyzer(self.factor_levels)
        taguchi_results = analyzer.analyze(results)
        
        # 保存田口分析结果
        with open(f"{self.results_dir}/taguchi_analysis.json", 'w') as f:
            json.dump(taguchi_results, f, indent=2, default=str)
        
        return taguchi_results

class MetricsEvaluator:
    """性能指标评估器"""
    
    def __init__(self):
        self.reference_front = None
    
    def generate_reference_front(self, problem: MO_DHFSP_Problem):
        """生成参考前沿"""
        logger.info("生成参考前沿，运行多种算法...")
        
        all_solutions = []
        algorithms = [
            ('NSGA2', NSGA2_Optimizer),
            ('MOEA/D', MOEAD_Optimizer),
            ('MOPSO', MOPSO_Optimizer),
            ('RL-Chaotic-HHO', RL_ChaoticHHO_Optimizer)
        ]
        
        for name, AlgorithmClass in algorithms:
            try:
                logger.info(f"运行{name}算法...")
                if name == 'RL-Chaotic-HHO':
                    optimizer = AlgorithmClass(
                        problem, 
                        learning_rate=0.001,
                        epsilon_decay=0.997,
                        group_ratios=[0.45, 0.25, 0.20, 0.10],
                        gamma=0.95,
                        population_size=100,
                        max_iterations=30
                    )
                else:
                    optimizer = AlgorithmClass(problem, population_size=100, max_iterations=30)
                
                solutions, _ = optimizer.optimize()
                all_solutions.extend(solutions)
                logger.info(f"{name}贡献{len(solutions)}个解")
                
            except Exception as e:
                logger.warning(f"{name}算法运行失败: {str(e)}")
        
        # 提取帕累托前沿作为参考前沿
        self.reference_front = self._extract_pareto_front(all_solutions)
        logger.info(f"参考前沿包含{len(self.reference_front)}个解")
        
        # 保存参考前沿
        reference_data = [
            {'makespan': sol.makespan, 'total_tardiness': sol.total_tardiness}
            for sol in self.reference_front
        ]
        with open("reference_front.json", 'w') as f:
            json.dump(reference_data, f, indent=2)
    
    def _extract_pareto_front(self, solutions: List) -> List:
        """提取帕累托前沿"""
        if not solutions:
            return []
        
        pareto_front = []
        for candidate in solutions:
            is_dominated = False
            
            for other in solutions:
                if (other.makespan <= candidate.makespan and 
                    other.total_tardiness <= candidate.total_tardiness and
                    (other.makespan < candidate.makespan or 
                     other.total_tardiness < candidate.total_tardiness)):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_front.append(candidate)
        
        return pareto_front
    
    def evaluate_performance(self, pareto_solutions: List, problem: MO_DHFSP_Problem, runtime: float) -> Dict:
        """评估算法性能"""
        if not pareto_solutions:
            return {
                'hypervolume': 0.0,
                'igd': float('inf'),
                'gd': float('inf'),
                'comprehensive': 0.0,
                'runtime': runtime
            }
        
        # 计算各项指标
        hv = self._calculate_hypervolume(pareto_solutions)
        igd = self._calculate_igd(pareto_solutions)
        gd = self._calculate_gd(pareto_solutions)
        
        # 综合评价（5:3:2权重）
        comprehensive = self.comprehensive_evaluation_5_3_2(hv, igd, gd)
        
        return {
            'hypervolume': hv,
            'igd': igd,
            'gd': gd,
            'comprehensive': comprehensive,
            'runtime': runtime
        }
    
    def _calculate_hypervolume(self, pareto_solutions: List) -> float:
        """计算超体积指标"""
        if not pareto_solutions:
            return 0.0
        
        # 提取目标值
        objectives = np.array([[sol.makespan, sol.total_tardiness] for sol in pareto_solutions])
        
        # 设置参考点（稍大于最大值）
        max_makespan = np.max(objectives[:, 0])
        max_tardiness = np.max(objectives[:, 1])
        reference_point = np.array([max_makespan * 1.1, max_tardiness * 1.1])
        
        # 简化的超体积计算
        normalized_objectives = objectives / reference_point
        volumes = []
        
        for obj in normalized_objectives:
            if all(obj < 1.0):
                volume = np.prod(1.0 - obj)
                volumes.append(volume)
        
        return sum(volumes) / len(objectives) if volumes else 0.0
    
    def _calculate_igd(self, pareto_solutions: List) -> float:
        """计算反向世代距离"""
        if not self.reference_front or not pareto_solutions:
            return float('inf')
        
        distances = []
        for ref_sol in self.reference_front:
            min_dist = float('inf')
            for sol in pareto_solutions:
                dist = self._euclidean_distance(
                    [ref_sol.makespan, ref_sol.total_tardiness],
                    [sol.makespan, sol.total_tardiness]
                )
                min_dist = min(min_dist, dist)
            distances.append(min_dist)
        
        return np.mean(distances)
    
    def _calculate_gd(self, pareto_solutions: List) -> float:
        """计算世代距离"""
        if not self.reference_front or not pareto_solutions:
            return float('inf')
        
        distances = []
        for sol in pareto_solutions:
            min_dist = float('inf')
            for ref_sol in self.reference_front:
                dist = self._euclidean_distance(
                    [sol.makespan, sol.total_tardiness],
                    [ref_sol.makespan, ref_sol.total_tardiness]
                )
                min_dist = min(min_dist, dist)
            distances.append(min_dist)
        
        return np.mean(distances)
    
    def _euclidean_distance(self, point1: List[float], point2: List[float]) -> float:
        """计算欧几里得距离"""
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))
    
    def comprehensive_evaluation_5_3_2(self, hv: float, igd: float, gd: float) -> float:
        """5:3:2权重综合评价"""
        # 归一化处理
        normalized_hv = hv  # 超体积越大越好，直接使用
        normalized_igd = 1.0 / (1.0 + igd) if igd != float('inf') else 0.0  # IGD越小越好
        normalized_gd = 1.0 / (1.0 + gd) if gd != float('inf') else 0.0    # GD越小越好
        
        # 加权综合
        comprehensive = (0.5 * normalized_hv + 0.3 * normalized_igd + 0.2 * normalized_gd)
        return comprehensive
    
    def calculate_snr_comprehensive(self, scores: List[float]) -> float:
        """计算综合得分的信噪比"""
        if not scores or all(s == 0 for s in scores):
            return -50.0
        
        mean_score = np.mean(scores)
        if mean_score <= 0:
            return -50.0
        
        # 信噪比计算：SNR = -10 * log10(1/mean^2)
        snr = -10 * np.log10(1.0 / (mean_score ** 2))
        return snr

class TaguchiL81Analyzer:
    """田口L81分析器"""
    
    def __init__(self, factor_levels: Dict):
        self.factor_levels = factor_levels
    
    def analyze(self, results: List[Dict]) -> Dict:
        """执行田口分析"""
        logger.info("开始田口分析...")
        
        # 提取信噪比数据
        snr_data = self._extract_snr_data(results)
        
        # 因子效应分析
        factor_effects = self._calculate_factor_effects(snr_data)
        
        # 确定最优水平组合
        optimal_combination = self._determine_optimal_combination(factor_effects)
        
        # 方差分析
        anova_results = self._perform_anova(snr_data, factor_effects)
        
        # 预测最优性能
        predicted_snr = self._predict_optimal_snr(factor_effects, optimal_combination)
        
        logger.info("田口分析完成")
        
        return {
            'factor_effects': factor_effects,
            'optimal_combination': optimal_combination,
            'anova_results': anova_results,
            'predicted_snr': predicted_snr,
            'snr_data': snr_data.tolist()
        }
    
    def _extract_snr_data(self, results: List[Dict]) -> np.ndarray:
        """提取信噪比数据"""
        snr_values = []
        for result in results:
            snr = result['statistics']['snr_value']
            snr_values.append(snr if snr != -float('inf') else -50.0)
        
        return np.array(snr_values)
    
    def _calculate_factor_effects(self, snr_data: np.ndarray) -> Dict:
        """计算因子效应"""
        factor_effects = {}
        
        for factor in ['A', 'B', 'C', 'D']:
            factor_effects[factor] = {}
            
            for level in range(1, 10):  # 9个水平
                # 找到该因子该水平对应的实验索引
                level_indices = self._get_level_indices(factor, level)
                
                # 计算该水平的平均SNR
                if level_indices:
                    level_snr = np.mean(snr_data[level_indices])
                else:
                    level_snr = -50.0
                
                factor_effects[factor][level] = level_snr
        
        return factor_effects
    
    def _get_level_indices(self, factor: str, level: int) -> List[int]:
        """获取指定因子水平对应的实验索引"""
        indices = []
        
        if factor == 'A':
            # A因子按顺序分布
            start_idx = (level - 1) * 9
            end_idx = min(start_idx + 9, 81)
            indices = list(range(start_idx, end_idx))
        elif factor == 'B':
            # B因子每9个实验循环一次
            for i in range(81):
                if (i % 9) + 1 == level:
                    indices.append(i)
        elif factor == 'C':
            # C因子基于A和B计算
            for i in range(81):
                a = i // 9 + 1
                b = (i % 9) + 1
                c = ((a - 1) + (b - 1)) % 9 + 1
                if c == level:
                    indices.append(i)
        elif factor == 'D':
            # D因子基于A和B的复合计算
            for i in range(81):
                a = i // 9 + 1
                b = (i % 9) + 1
                d = ((a - 1) * 2 + (b - 1) * 3) % 9 + 1
                if d == level:
                    indices.append(i)
        
        return indices
    
    def _determine_optimal_combination(self, factor_effects: Dict) -> Dict:
        """确定最优参数组合"""
        optimal = {}
        for factor in ['A', 'B', 'C', 'D']:
            # 选择信噪比最大的水平
            best_level = max(
                range(1, 10), 
                key=lambda level: factor_effects[factor][level]
            )
            optimal[factor] = best_level
        
        return optimal
    
    def _perform_anova(self, snr_data: np.ndarray, factor_effects: Dict) -> Dict:
        """执行方差分析"""
        grand_mean = np.mean(snr_data)
        sst = np.sum((snr_data - grand_mean) ** 2)  # 总平方和
        
        anova = {}
        for factor in ['A', 'B', 'C', 'D']:
            # 计算因子平方和
            ss_factor = 0
            for level in range(1, 10):
                level_indices = self._get_level_indices(factor, level)
                if level_indices:
                    level_mean = np.mean(snr_data[level_indices])
                    ss_factor += len(level_indices) * (level_mean - grand_mean) ** 2
            
            # 计算F值
            df_factor = 8  # 自由度 = 水平数 - 1
            df_error = 81 - 9  # 简化的误差自由度
            ms_factor = ss_factor / df_factor
            ms_error = (sst - ss_factor) / df_error if df_error > 0 else 1
            f_value = ms_factor / ms_error if ms_error > 0 else 0
            
            anova[factor] = {
                'sum_of_squares': ss_factor,
                'mean_square': ms_factor,
                'f_value': f_value,
                'contribution': ss_factor / sst * 100 if sst > 0 else 0  # 贡献率%
            }
        
        return anova
    
    def _predict_optimal_snr(self, factor_effects: Dict, optimal_combination: Dict) -> float:
        """预测最优组合的信噪比"""
        grand_mean = np.mean([
            np.mean(list(factor_effects[factor].values()))
            for factor in ['A', 'B', 'C', 'D']
        ])
        
        predicted_snr = grand_mean
        for factor in ['A', 'B', 'C', 'D']:
            optimal_level = optimal_combination[factor]
            level_effect = factor_effects[factor][optimal_level] - np.mean(
                list(factor_effects[factor].values())
            )
            predicted_snr += level_effect
        
        return predicted_snr

def main():
    """主函数"""
    print("🚀 开始RL-Chaotic-HHO L81田口正交实验")
    print("=" * 60)
    print("📊 实验配置:")
    print("   - 正交表: L81(9^4)")
    print("   - 参数数量: 4个")
    print("   - 水平数量: 9个")
    print("   - 问题规模: 100×5×3")
    print("   - 总机器数: 40台")
    print("   - 实验组数: 81组")
    print("   - 每组重复: 5次")
    print("   - 总实验量: 405次")
    print("   - 评价指标: 超体积:反向世代距离:世代距离 = 5:3:2加权综合")
    print("=" * 60)
    
    print(f"\n📈 L81设计相比L49的改进:")
    print(f"   - 参数水平: 7 → 9 (增加28.6%)")
    print(f"   - 实验组数: 49 → 81 (增加65.3%)")
    print(f"   - 每组重复: 10 → 5 (减少50%，平衡总实验量)")
    print(f"   - 总实验量: 490 → 405 (减少17.3%，提高效率)")
    print(f"   - 参数覆盖: 更全面的参数空间探索")
    print(f"   - 学习率范围: 0.00005~0.005 → 0.00001~0.02")
    print(f"   - 衰减率精度: 0.988~0.9995 → 0.985~0.9999")
    print(f"   - 分组策略: 7种 → 9种更多样化的探索-开发平衡")
    print(f"   - 折扣因子: 0.80~0.995 → 0.75~0.999")
    
    # 创建实验控制器
    experiment = TaguchiL81Experiment()
    
    # 运行实验
    start_time = time.time()
    try:
        all_results, taguchi_results = experiment.run_all_experiments()
        
        total_time = time.time() - start_time
        print(f"\n🎉 实验完成! 总耗时: {total_time/3600:.2f}小时")
        print(f"📁 结果保存在: {experiment.results_dir}")
        
        # 输出关键结果
        print("\n📈 田口分析结果:")
        optimal = taguchi_results['optimal_combination']
        print(f"   最优学习率: {experiment.factor_levels['A_learning_rate'][optimal['A']]}")
        print(f"   最优衰减率: {experiment.factor_levels['B_epsilon_decay'][optimal['B']]}")
        print(f"   最优分组比例: {experiment.factor_levels['C_group_ratios'][optimal['C']]}")
        print(f"   最优折扣因子: {experiment.factor_levels['D_gamma'][optimal['D']]}")
        print(f"   预测SNR: {taguchi_results['predicted_snr']:.2f}")
        
        # 输出因子贡献率
        print("\n📊 因子贡献率:")
        for factor in ['A', 'B', 'C', 'D']:
            contribution = taguchi_results['anova_results'][factor]['contribution']
            print(f"   {factor}因子: {contribution:.2f}%")
        
    except KeyboardInterrupt:
        print("\n⚠️ 实验被用户中断")
    except Exception as e:
        print(f"\n❌ 实验运行出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 