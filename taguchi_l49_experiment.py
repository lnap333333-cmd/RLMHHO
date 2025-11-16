#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RL-Chaotic-HHO算法L49田口正交实验设计
针对100×5×3规模，总机器数40的MO-DHFSP问题
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
        logging.FileHandler('taguchi_l49_experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TaguchiL49Experiment:
    """L49田口正交实验主控制类"""
    
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
        
        # L49正交表参数配置
        self.factor_levels = self._initialize_factor_levels()
        self.l49_design = self._generate_l49_design()
        
        # 实验控制参数
        self.runs_per_experiment = 10  # 每组实验重复10次
        self.max_iterations = 50     # 算法迭代次数
        
        # 结果存储
        self.results_dir = f"taguchi_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 性能评估器
        self.metrics_evaluator = MetricsEvaluator()
        
    def _initialize_factor_levels(self) -> Dict:
        """初始化4因子7水平参数"""
        return {
            'A_learning_rate': {
                1: 0.00005, 2: 0.0001, 3: 0.0002, 4: 0.0005,
                5: 0.001, 6: 0.002, 7: 0.005
            },
            'B_epsilon_decay': {
                1: 0.988, 2: 0.990, 3: 0.993, 4: 0.995,
                5: 0.997, 6: 0.999, 7: 0.9995
            },
            'C_group_ratios': {
                1: [0.70, 0.15, 0.10, 0.05],  # 超级探索主导
                2: [0.60, 0.20, 0.15, 0.05],  # 极端探索主导
                3: [0.50, 0.30, 0.15, 0.05],  # 探索主导
                4: [0.45, 0.25, 0.20, 0.10],  # 基准平衡
                5: [0.35, 0.40, 0.20, 0.05],  # 开发主导
                6: [0.25, 0.45, 0.20, 0.10],  # 极端开发主导
                7: [0.20, 0.50, 0.20, 0.10]   # 超级开发主导
            },
            'D_gamma': {
                1: 0.80, 2: 0.85, 3: 0.90, 4: 0.95,
                5: 0.98, 6: 0.99, 7: 0.995
            }
        }
    
    def _generate_l49_design(self) -> List[Dict]:
        """生成L49(7^4)正交表设计"""
        l49_experiments = [
            # 实验1-7: A水平1的组合
            {'exp_id': 1,  'A': 1, 'B': 1, 'C': 1, 'D': 1},
            {'exp_id': 2,  'A': 1, 'B': 2, 'C': 2, 'D': 2},
            {'exp_id': 3,  'A': 1, 'B': 3, 'C': 3, 'D': 3},
            {'exp_id': 4,  'A': 1, 'B': 4, 'C': 4, 'D': 4},
            {'exp_id': 5,  'A': 1, 'B': 5, 'C': 5, 'D': 5},
            {'exp_id': 6,  'A': 1, 'B': 6, 'C': 6, 'D': 6},
            {'exp_id': 7,  'A': 1, 'B': 7, 'C': 7, 'D': 7},
            
            # 实验8-14: A水平2的组合
            {'exp_id': 8,  'A': 2, 'B': 1, 'C': 2, 'D': 3},
            {'exp_id': 9,  'A': 2, 'B': 2, 'C': 3, 'D': 4},
            {'exp_id': 10, 'A': 2, 'B': 3, 'C': 4, 'D': 5},
            {'exp_id': 11, 'A': 2, 'B': 4, 'C': 5, 'D': 6},
            {'exp_id': 12, 'A': 2, 'B': 5, 'C': 6, 'D': 7},
            {'exp_id': 13, 'A': 2, 'B': 6, 'C': 7, 'D': 1},
            {'exp_id': 14, 'A': 2, 'B': 7, 'C': 1, 'D': 2},
            
            # 实验15-21: A水平3的组合
            {'exp_id': 15, 'A': 3, 'B': 1, 'C': 3, 'D': 5},
            {'exp_id': 16, 'A': 3, 'B': 2, 'C': 4, 'D': 6},
            {'exp_id': 17, 'A': 3, 'B': 3, 'C': 5, 'D': 7},
            {'exp_id': 18, 'A': 3, 'B': 4, 'C': 6, 'D': 1},
            {'exp_id': 19, 'A': 3, 'B': 5, 'C': 7, 'D': 2},
            {'exp_id': 20, 'A': 3, 'B': 6, 'C': 1, 'D': 3},
            {'exp_id': 21, 'A': 3, 'B': 7, 'C': 2, 'D': 4},
            
            # 实验22-28: A水平4的组合 (基准学习率)
            {'exp_id': 22, 'A': 4, 'B': 1, 'C': 4, 'D': 7},
            {'exp_id': 23, 'A': 4, 'B': 2, 'C': 5, 'D': 1},
            {'exp_id': 24, 'A': 4, 'B': 3, 'C': 6, 'D': 2},
            {'exp_id': 25, 'A': 4, 'B': 4, 'C': 7, 'D': 3},
            {'exp_id': 26, 'A': 4, 'B': 5, 'C': 1, 'D': 4},
            {'exp_id': 27, 'A': 4, 'B': 6, 'C': 2, 'D': 5},
            {'exp_id': 28, 'A': 4, 'B': 7, 'C': 3, 'D': 6},
            
            # 实验29-35: A水平5的组合
            {'exp_id': 29, 'A': 5, 'B': 1, 'C': 5, 'D': 2},
            {'exp_id': 30, 'A': 5, 'B': 2, 'C': 6, 'D': 3},
            {'exp_id': 31, 'A': 5, 'B': 3, 'C': 7, 'D': 4},
            {'exp_id': 32, 'A': 5, 'B': 4, 'C': 1, 'D': 5},
            {'exp_id': 33, 'A': 5, 'B': 5, 'C': 2, 'D': 6},
            {'exp_id': 34, 'A': 5, 'B': 6, 'C': 3, 'D': 7},
            {'exp_id': 35, 'A': 5, 'B': 7, 'C': 4, 'D': 1},
            
            # 实验36-42: A水平6的组合
            {'exp_id': 36, 'A': 6, 'B': 1, 'C': 6, 'D': 4},
            {'exp_id': 37, 'A': 6, 'B': 2, 'C': 7, 'D': 5},
            {'exp_id': 38, 'A': 6, 'B': 3, 'C': 1, 'D': 6},
            {'exp_id': 39, 'A': 6, 'B': 4, 'C': 2, 'D': 7},
            {'exp_id': 40, 'A': 6, 'B': 5, 'C': 3, 'D': 1},
            {'exp_id': 41, 'A': 6, 'B': 6, 'C': 4, 'D': 2},
            {'exp_id': 42, 'A': 6, 'B': 7, 'C': 5, 'D': 3},
            
            # 实验43-49: A水平7的组合
            {'exp_id': 43, 'A': 7, 'B': 1, 'C': 7, 'D': 6},
            {'exp_id': 44, 'A': 7, 'B': 2, 'C': 1, 'D': 7},
            {'exp_id': 45, 'A': 7, 'B': 3, 'C': 2, 'D': 1},
            {'exp_id': 46, 'A': 7, 'B': 4, 'C': 3, 'D': 2},
            {'exp_id': 47, 'A': 7, 'B': 5, 'C': 4, 'D': 3},
            {'exp_id': 48, 'A': 7, 'B': 6, 'C': 5, 'D': 4},
            {'exp_id': 49, 'A': 7, 'B': 7, 'C': 6, 'D': 5}
        ]
        
        return l49_experiments
    
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
        
        # 获取参数配置
        params = self._get_experiment_parameters(exp_config)
        
        logger.info(f"实验{exp_id}-运行{run_id}: LR={params['learning_rate']:.5f}, "
                   f"Decay={params['epsilon_decay']:.4f}, "
                   f"Groups={params['group_ratios']}, "
                   f"Gamma={params['gamma']:.3f}")
        
        # 创建算法实例
        optimizer = RL_ChaoticHHO_Optimizer(
            problem=problem,
            max_iterations=self.max_iterations,
            **params
        )
        
        # 运行优化
        start_time = time.time()
        try:
            pareto_solutions, convergence_data = optimizer.optimize()
            runtime = time.time() - start_time
            
            # 计算性能指标
            metrics = self.metrics_evaluator.evaluate_performance(
                pareto_solutions, problem, runtime
            )
            
            result = {
                'exp_id': exp_id,
                'run_id': run_id,
                'parameters': params,
                'pareto_solutions': pareto_solutions,
                'metrics': metrics,
                'convergence_data': convergence_data,
                'runtime': runtime,
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
            'population_size_override': 50,  # 强制设置种群大小
            'epsilon': 0.9,
            'epsilon_min': 0.01
        }
        return params
    
    def run_experiment_group(self, exp_config: Dict, problem: MO_DHFSP_Problem) -> Dict:
        """运行单组实验（10次重复）"""
        exp_id = exp_config['exp_id']
        logger.info(f"开始实验组{exp_id} ({exp_id}/49)")
        
        group_results = []
        
        # 运行10次重复实验
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
        """分析实验组结果"""
        successful_results = [r for r in group_results if r.get('success', False)]
        
        if not successful_results:
            return {'error': 'No successful runs', 'snr_value': 0.0}
        
        # 提取性能指标
        hv_values = [r['metrics']['hypervolume'] for r in successful_results]
        igd_values = [r['metrics']['igd'] for r in successful_results]
        gd_values = [r['metrics']['gd'] for r in successful_results]
        
        # 计算综合得分 (5:3:2权重)
        comprehensive_scores = []
        for hv, igd, gd in zip(hv_values, igd_values, gd_values):
            score = self.metrics_evaluator.comprehensive_evaluation_5_3_2(hv, igd, gd)
            comprehensive_scores.append(score)
        
        # 计算信噪比
        snr = self.metrics_evaluator.calculate_snr_comprehensive(comprehensive_scores)
        
        statistics = {
            'n_successful_runs': len(successful_results),
            'hv_mean': np.mean(hv_values),
            'hv_std': np.std(hv_values),
            'igd_mean': np.mean(igd_values),
            'igd_std': np.std(igd_values),
            'gd_mean': np.mean(gd_values),
            'gd_std': np.std(gd_values),
            'comprehensive_mean': np.mean(comprehensive_scores),
            'comprehensive_std': np.std(comprehensive_scores),
            'snr_value': snr,
            'runtime_mean': np.mean([r['runtime'] for r in successful_results])
        }
        
        return statistics
    
    def run_all_experiments(self):
        """运行所有49组实验"""
        logger.info("开始L49田口正交实验")
        
        # 生成问题实例
        problem = self.generate_problem_instance()
        
        # 生成参考前沿
        logger.info("生成参考前沿...")
        self.metrics_evaluator.generate_reference_front(problem)
        
        # 运行所有实验组
        all_results = []
        total_experiments = len(self.l49_design)
        
        for i, exp_config in enumerate(self.l49_design):
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
        
        logger.info("L49田口实验完成!")
        return all_results, taguchi_results
    
    def _save_intermediate_results(self, results: List[Dict]):
        """保存中间结果"""
        with open(f"{self.results_dir}/intermediate_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    def _save_final_results(self, results: List[Dict]):
        """保存最终结果"""
        with open(f"{self.results_dir}/final_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # 保存为Excel格式
        self._export_to_excel(results)
    
    def _export_to_excel(self, results: List[Dict]):
        """导出结果到Excel"""
        # 创建汇总数据框
        summary_data = []
        for result in results:
            if 'statistics' in result:
                row = {
                    'Exp_ID': result['exp_id'],
                    'A_LearningRate': self.factor_levels['A_learning_rate'][result['exp_config']['A']],
                    'B_EpsilonDecay': self.factor_levels['B_epsilon_decay'][result['exp_config']['B']],
                    'C_GroupRatios': str(self.factor_levels['C_group_ratios'][result['exp_config']['C']]),
                    'D_Gamma': self.factor_levels['D_gamma'][result['exp_config']['D']],
                    'HV_Mean': result['statistics']['hv_mean'],
                    'HV_Std': result['statistics']['hv_std'],
                    'IGD_Mean': result['statistics']['igd_mean'],
                    'IGD_Std': result['statistics']['igd_std'],
                    'GD_Mean': result['statistics']['gd_mean'],
                    'GD_Std': result['statistics']['gd_std'],
                    'Comprehensive_Mean': result['statistics']['comprehensive_mean'],
                    'Comprehensive_Std': result['statistics']['comprehensive_std'],
                    'SNR_Value': result['statistics']['snr_value'],
                    'Runtime_Mean': result['statistics']['runtime_mean'],
                    'Successful_Runs': result['statistics']['n_successful_runs']
                }
                summary_data.append(row)
        
        df = pd.DataFrame(summary_data)
        df.to_excel(f"{self.results_dir}/l49_results_summary.xlsx", index=False)
        logger.info("结果已导出到Excel文件")
    
    def _perform_taguchi_analysis(self, results: List[Dict]) -> Dict:
        """执行田口分析"""
        analyzer = TaguchiAnalyzer(self.factor_levels)
        taguchi_results = analyzer.analyze(results)
        
        # 保存田口分析结果
        with open(f"{self.results_dir}/taguchi_analysis.json", 'w') as f:
            json.dump(taguchi_results, f, indent=2, default=str)
        
        return taguchi_results


class MetricsEvaluator:
    """性能指标评估器"""
    
    def __init__(self):
        self.reference_front = None
        self.reference_point = None
    
    def generate_reference_front(self, problem: MO_DHFSP_Problem):
        """生成参考前沿"""
        logger.info("使用经典算法生成参考前沿...")
        
        all_solutions = []
        algorithms = ['NSGA2', 'MOEAD', 'MOPSO']
        
        for alg_name in algorithms:
            logger.info(f"运行{alg_name}算法...")
            try:
                if alg_name == 'NSGA2':
                    optimizer = NSGA2_Optimizer(
                        problem, population_size=50, max_generations=50
                    )
                elif alg_name == 'MOEAD':
                    optimizer = MOEAD_Optimizer(
                        problem, population_size=50, max_generations=50
                    )
                elif alg_name == 'MOPSO':
                    optimizer = MOPSO_Optimizer(
                        problem, swarm_size=50, max_iterations=50
                    )
                
                solutions, _ = optimizer.optimize()
                all_solutions.extend(solutions)
                logger.info(f"{alg_name}完成，获得{len(solutions)}个解")
                
            except Exception as e:
                logger.warning(f"{alg_name}运行失败: {str(e)}")
        
        # 提取非支配解作为参考前沿
        self.reference_front = self._extract_pareto_front(all_solutions)
        
        # 设置参考点（比最差解稍差一些）
        if self.reference_front:
            max_makespan = max(sol.makespan for sol in self.reference_front)
            max_tardiness = max(sol.total_tardiness for sol in self.reference_front)
            self.reference_point = [max_makespan * 1.1, max_tardiness * 1.1]
        else:
            self.reference_point = [1000.0, 1000.0]  # 默认参考点
        
        logger.info(f"参考前沿生成完成：{len(self.reference_front)}个解")
        logger.info(f"参考点设置为：{self.reference_point}")
    
    def _extract_pareto_front(self, solutions: List) -> List:
        """提取帕累托前沿"""
        if not solutions:
            return []
        
        pareto_front = []
        for sol in solutions:
            is_dominated = False
            for other_sol in solutions:
                if (other_sol.makespan <= sol.makespan and 
                    other_sol.total_tardiness <= sol.total_tardiness and
                    (other_sol.makespan < sol.makespan or other_sol.total_tardiness < sol.total_tardiness)):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_front.append(sol)
        
        return pareto_front
    
    def evaluate_performance(self, pareto_solutions: List, problem: MO_DHFSP_Problem, runtime: float) -> Dict:
        """评估算法性能"""
        if not pareto_solutions:
            return {
                'hypervolume': 0.0,
                'igd': float('inf'),
                'gd': float('inf'),
                'pareto_size': 0,
                'runtime': runtime
            }
        
        # 计算超体积
        hv = self._calculate_hypervolume(pareto_solutions)
        
        # 计算IGD和GD
        igd = self._calculate_igd(pareto_solutions)
        gd = self._calculate_gd(pareto_solutions)
        
        return {
            'hypervolume': hv,
            'igd': igd,
            'gd': gd,
            'pareto_size': len(pareto_solutions),
            'runtime': runtime
        }
    
    def _calculate_hypervolume(self, pareto_solutions: List) -> float:
        """计算超体积指标"""
        if not pareto_solutions or not self.reference_point:
            return 0.0
        
        # 标准化目标值
        normalized_solutions = []
        for sol in pareto_solutions:
            norm_makespan = sol.makespan / self.reference_point[0]
            norm_tardiness = sol.total_tardiness / self.reference_point[1]
            normalized_solutions.append([norm_makespan, norm_tardiness])
        
        # 简化的超体积计算（2维情况）
        ref_point = [1.1, 1.1]
        
        # 排序并计算
        normalized_solutions.sort(key=lambda x: x[0])
        
        hv = 0.0
        prev_x = 0.0
        
        for point in normalized_solutions:
            if point[0] < ref_point[0] and point[1] < ref_point[1]:
                width = min(point[0], ref_point[0]) - prev_x
                height = ref_point[1] - point[1]
                hv += width * height
                prev_x = min(point[0], ref_point[0])
        
        return max(0.0, min(hv, ref_point[0] * ref_point[1]))
    
    def _calculate_igd(self, pareto_solutions: List) -> float:
        """计算反向世代距离"""
        if not self.reference_front or not pareto_solutions:
            return float('inf')
        
        distances = []
        for ref_sol in self.reference_front:
            min_distance = min([
                self._euclidean_distance(
                    [ref_sol.makespan, ref_sol.total_tardiness],
                    [sol.makespan, sol.total_tardiness]
                ) for sol in pareto_solutions
            ])
            distances.append(min_distance)
        
        return np.mean(distances) if distances else float('inf')
    
    def _calculate_gd(self, pareto_solutions: List) -> float:
        """计算世代距离"""
        if not self.reference_front or not pareto_solutions:
            return float('inf')
        
        distances = []
        for sol in pareto_solutions:
            min_distance = min([
                self._euclidean_distance(
                    [sol.makespan, sol.total_tardiness],
                    [ref_sol.makespan, ref_sol.total_tardiness]
                ) for ref_sol in self.reference_front
            ])
            distances.append(min_distance)
        
        return np.mean(distances) if distances else float('inf')
    
    def _euclidean_distance(self, point1: List[float], point2: List[float]) -> float:
        """计算欧几里得距离"""
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))
    
    def comprehensive_evaluation_5_3_2(self, hv: float, igd: float, gd: float) -> float:
        """5:3:2权重综合评价"""
        # 归一化处理
        norm_hv = min(hv / 1.21, 1.0)  # 假设最大HV为1.21
        norm_igd = 1.0 / (1.0 + igd)   # 转换为越大越好
        norm_gd = 1.0 / (1.0 + gd)     # 转换为越大越好
        
        # 加权综合
        comprehensive_score = 0.5 * norm_hv + 0.3 * norm_igd + 0.2 * norm_gd
        return comprehensive_score
    
    def calculate_snr_comprehensive(self, scores: List[float]) -> float:
        """计算综合得分的信噪比"""
        if not scores or len(scores) == 0:
            return 0.0
        
        # 田口方法望大特性信噪比
        snr = -10 * np.log10(np.mean(1.0 / np.array(scores) ** 2))
        return snr


class TaguchiAnalyzer:
    """田口分析器"""
    
    def __init__(self, factor_levels: Dict):
        self.factor_levels = factor_levels
    
    def analyze(self, results: List[Dict]) -> Dict:
        """执行田口分析"""
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
        
        return {
            'factor_effects': factor_effects,
            'optimal_combination': optimal_combination,
            'anova_results': anova_results,
            'predicted_snr': predicted_snr,
            'snr_data': snr_data
        }
    
    def _extract_snr_data(self, results: List[Dict]) -> np.ndarray:
        """提取信噪比数据"""
        snr_values = []
        for result in results:
            if 'statistics' in result and 'snr_value' in result['statistics']:
                snr_values.append(result['statistics']['snr_value'])
            else:
                snr_values.append(0.0)  # 失败实验的SNR设为0
        
        return np.array(snr_values)
    
    def _calculate_factor_effects(self, snr_data: np.ndarray) -> Dict:
        """计算因子效应"""
        effects = {}
        factors = ['A', 'B', 'C', 'D']
        
        for factor in factors:
            effects[factor] = {}
            for level in range(1, 8):  # 7个水平
                # 找到该因子该水平对应的实验
                level_indices = self._get_level_indices(factor, level)
                level_snr_values = snr_data[level_indices]
                effects[factor][level] = np.mean(level_snr_values)
            
            # 计算效应范围
            level_means = list(effects[factor].values())
            effects[factor]['range'] = max(level_means) - min(level_means)
            effects[factor]['rank'] = 0  # 将在后面计算排名
        
        # 计算重要性排名
        ranges = [(factor, effects[factor]['range']) for factor in factors]
        ranges.sort(key=lambda x: x[1], reverse=True)
        
        for rank, (factor, _) in enumerate(ranges, 1):
            effects[factor]['rank'] = rank
        
        return effects
    
    def _get_level_indices(self, factor: str, level: int) -> List[int]:
        """获取指定因子水平对应的实验索引"""
        indices = []
        
        # 根据L49正交表的实际设计确定索引
        for exp_id in range(49):
            if factor == 'A':
                exp_level = (exp_id // 7) + 1
            elif factor == 'B':
                exp_level = ((exp_id % 7) + (exp_id // 7)) % 7 + 1
            elif factor == 'C':
                exp_level = ((exp_id % 7) * 2 + (exp_id // 7)) % 7 + 1
            else:  # factor == 'D'
                exp_level = ((exp_id % 7) * 3 + (exp_id // 7)) % 7 + 1
            
            if exp_level == level:
                indices.append(exp_id)
        
        return indices
    
    def _determine_optimal_combination(self, factor_effects: Dict) -> Dict:
        """确定最优参数组合"""
        optimal = {}
        for factor in ['A', 'B', 'C', 'D']:
            # 选择信噪比最大的水平
            best_level = max(
                range(1, 8), 
                key=lambda level: factor_effects[factor][level]
            )
            optimal[factor] = best_level
        
        return optimal
    
    def _perform_anova(self, snr_data: np.ndarray, factor_effects: Dict) -> Dict:
        """执行方差分析"""
        # 简化的方差分析
        grand_mean = np.mean(snr_data)
        sst = np.sum((snr_data - grand_mean) ** 2)  # 总平方和
        
        anova = {}
        for factor in ['A', 'B', 'C', 'D']:
            # 计算因子平方和
            ss_factor = 0
            for level in range(1, 8):
                level_indices = self._get_level_indices(factor, level)
                level_mean = np.mean(snr_data[level_indices])
                ss_factor += len(level_indices) * (level_mean - grand_mean) ** 2
            
            # 计算F值（简化）
            ms_factor = ss_factor / 6  # 自由度 = 水平数 - 1
            ms_error = (sst - ss_factor) / (49 - 7)  # 简化的误差均方
            f_value = ms_factor / ms_error if ms_error > 0 else 0
            
            anova[factor] = {
                'sum_of_squares': ss_factor,
                'mean_square': ms_factor,
                'f_value': f_value,
                'contribution': ss_factor / sst * 100  # 贡献率%
            }
        
        return anova
    
    def _predict_optimal_snr(self, factor_effects: Dict, optimal_combination: Dict) -> float:
        """预测最优组合的信噪比"""
        grand_mean = np.mean([
            np.mean(list(factor_effects[factor].values())[:7])  # 前7个是水平均值
            for factor in ['A', 'B', 'C', 'D']
        ])
        
        predicted_snr = grand_mean
        for factor in ['A', 'B', 'C', 'D']:
            optimal_level = optimal_combination[factor]
            level_effect = factor_effects[factor][optimal_level] - np.mean(
                list(factor_effects[factor].values())[:7]
            )
            predicted_snr += level_effect
        
        return predicted_snr


def main():
    """主函数"""
    print("🚀 开始RL-Chaotic-HHO L49田口正交实验")
    print("=" * 60)
    print("📊 实验配置:")
    print("   - 问题规模: 100×5×3")
    print("   - 总机器数: 40台")
    print("   - 实验组数: 49组")
    print("   - 每组重复: 10次")
    print("   - 总实验量: 490次")
    print("   - 评价指标: 超体积:反向世代距离:世代距离 = 5:3:2加权综合")
    print("=" * 60)
    
    # 创建实验控制器
    experiment = TaguchiL49Experiment()
    
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
        
    except KeyboardInterrupt:
        print("\n⚠️ 实验被用户中断")
    except Exception as e:
        print(f"\n❌ 实验运行出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 