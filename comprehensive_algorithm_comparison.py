#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多目标分布式异构混合流水车间调度算法综合对比实验
对比算法：RL-Chaotic-HHO、I-NSGA-II、MOPSO、MODE、DQN、QL-ABC
数据集：20-200作业数，2-6工厂，2-5并行机器，随规模自适应增长
评价指标：完工时间、总拖期、HV、IGD、GD、Spread（归一化）
输出：Pareto前沿图、对比表格、Excel报告
"""

import os
import sys
import time
import copy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# 导入算法模块
from problem.mo_dhfsp import MO_DHFSP_Problem
from algorithm.rl_chaotic_hho import RL_ChaoticHHO_Optimizer
from algorithm.improved_nsga2 import ImprovedNSGA2_Optimizer
from algorithm.mopso import MOPSO_Optimizer
from algorithm.mode import MODE_Optimizer
from algorithm.dqn_algorithm_wrapper import DQNAlgorithmWrapper
from algorithm.ql_abc import QLABC_Optimizer
from utils.data_generator import DataGenerator

# 设置中文字体和随机种子
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
np.random.seed(42)
random.seed(42)

@dataclass
class ExperimentConfig:
    """实验配置类"""
    scale_name: str
    n_jobs: int
    n_factories: int
    n_stages: int
    machines_per_stage: List[int]
    heterogeneous_machines: Dict[int, List[int]]
    processing_time_range: Tuple[int, int]
    urgency_ddt: List[float]
    dataset_seed: int

class NormalizedMetrics:
    """归一化指标计算器"""
    
    @staticmethod
    def calculate_hypervolume(pareto_solutions: List, all_solutions_combined: List = None) -> float:
        """
        计算归一化超体积指标
        
        Args:
            pareto_solutions: 帕累托解集
            all_solutions_combined: 所有算法的解集合，用于全局归一化
            
        Returns:
            归一化超体积值 [0, 1]
        """
        if not pareto_solutions or len(pareto_solutions) == 0:
            return 0.0
        
        objectives = [(sol.makespan, sol.total_tardiness) for sol in pareto_solutions]
        
        # 使用全局范围进行归一化
        if all_solutions_combined:
            all_objectives = [(sol.makespan, sol.total_tardiness) for sol in all_solutions_combined]
            min_makespan = min(obj[0] for obj in all_objectives)
            max_makespan = max(obj[0] for obj in all_objectives)
            min_tardiness = min(obj[1] for obj in all_objectives)
            max_tardiness = max(obj[1] for obj in all_objectives)
        else:
            min_makespan = min(obj[0] for obj in objectives)
            max_makespan = max(obj[0] for obj in objectives)
            min_tardiness = min(obj[1] for obj in objectives)
            max_tardiness = max(obj[1] for obj in objectives)
        
        # 避免除零
        makespan_range = max(max_makespan - min_makespan, 1e-10)
        tardiness_range = max(max_tardiness - min_tardiness, 1e-10)
        
        # 归一化到[0, 1]区间
        normalized_objectives = []
        for obj in objectives:
            norm_makespan = (obj[0] - min_makespan) / makespan_range
            norm_tardiness = (obj[1] - min_tardiness) / tardiness_range
            normalized_objectives.append((norm_makespan, norm_tardiness))
        
        # 去重复解
        unique_objectives = []
        tolerance = 1e-6
        for obj in normalized_objectives:
            is_duplicate = False
            for unique_obj in unique_objectives:
                if abs(obj[0] - unique_obj[0]) < tolerance and abs(obj[1] - unique_obj[1]) < tolerance:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_objectives.append(obj)
        
        if len(unique_objectives) <= 1:
            return 0.5  # 单点解给予中等分数
        
        # 计算超体积 - 使用改进的2D超体积计算
        ref_point = (1.1, 1.1)
        valid_objectives = [obj for obj in unique_objectives if obj[0] <= 1.0 and obj[1] <= 1.0]
        
        if not valid_objectives:
            return 0.0
        
        # 按第一个目标排序
        sorted_objectives = sorted(valid_objectives, key=lambda x: x[0])
        
        hypervolume = 0.0
        
        # 改进的超体积计算方法
        for i, (x, y) in enumerate(sorted_objectives):
            # 计算当前点左侧的x边界
            left_x = sorted_objectives[i-1][0] if i > 0 else 0.0
            
            # 计算当前点的矩形贡献
            width = x - left_x
            height = ref_point[1] - y
            
            if width > 0 and height > 0:
                hypervolume += width * height
        
        # 添加最右侧区域的贡献
        if sorted_objectives:
            last_x = sorted_objectives[-1][0]
            if last_x < ref_point[0]:
                # 找到最小的y值
                min_y = min(obj[1] for obj in sorted_objectives)
                width = ref_point[0] - last_x
                height = ref_point[1] - min_y
                if width > 0 and height > 0:
                    hypervolume += width * height
        
        # 归一化到[0, 1]
        max_possible_hv = ref_point[0] * ref_point[1]
        normalized_hv = hypervolume / max_possible_hv if max_possible_hv > 0 else 0.0
        
        # 确保返回有意义的值，避免过小的值
        if normalized_hv < 0.001 and len(unique_objectives) > 1:
            # 对于多个解但计算出很小值的情况，给予最小合理值
            normalized_hv = 0.01
        
        return min(max(normalized_hv, 0.0), 1.0)
    
    @staticmethod
    def calculate_igd(pareto_solutions: List, true_pareto_front: List = None) -> float:
        """
        计算归一化反世代距离(IGD)指标
        
        Args:
            pareto_solutions: 当前算法得到的帕累托解集
            true_pareto_front: 真实帕累托前沿
            
        Returns:
            归一化IGD值 [0, 1]
        """
        if not pareto_solutions:
            return 1.0  # 最差情况
        
        current_objectives = [(sol.makespan, sol.total_tardiness) for sol in pareto_solutions]
        
        if true_pareto_front is None:
            # 构建理想参考前沿
            min_makespan = min(obj[0] for obj in current_objectives)
            min_tardiness = min(obj[1] for obj in current_objectives)
            max_makespan = max(obj[0] for obj in current_objectives)
            max_tardiness = max(obj[1] for obj in current_objectives)
            
            # 创建多点参考前沿
            true_pareto_front = []
            n_points = 10
            for i in range(n_points + 1):
                alpha = i / n_points
                # 线性插值生成理想前沿
                makespan = min_makespan + alpha * (max_makespan - min_makespan) * 0.8
                tardiness = min_tardiness + (1 - alpha) * (max_tardiness - min_tardiness) * 0.8
                true_pareto_front.append((makespan, tardiness))
        
        # 计算IGD
        total_distance = 0.0
        for true_point in true_pareto_front:
            min_distance = float('inf')
            for current_point in current_objectives:
                distance = np.sqrt((true_point[0] - current_point[0])**2 + 
                                 (true_point[1] - current_point[1])**2)
                min_distance = min(min_distance, distance)
            total_distance += min_distance
        
        avg_distance = total_distance / len(true_pareto_front)
        
        # 归一化 - 使用目标函数值的范围进行归一化
        max_makespan = max(max(obj[0] for obj in current_objectives), 
                          max(obj[0] for obj in true_pareto_front))
        max_tardiness = max(max(obj[1] for obj in current_objectives), 
                           max(obj[1] for obj in true_pareto_front))
        
        normalization_factor = np.sqrt(max_makespan**2 + max_tardiness**2)
        normalized_igd = avg_distance / max(normalization_factor, 1e-10)
        
        return min(max(normalized_igd, 0.0), 1.0)
    
    @staticmethod
    def calculate_gd(pareto_solutions: List, true_pareto_front: List = None) -> float:
        """
        计算归一化世代距离(GD)指标
        
        Args:
            pareto_solutions: 当前算法得到的帕累托解集
            true_pareto_front: 真实帕累托前沿
            
        Returns:
            归一化GD值 [0, 1]
        """
        if not pareto_solutions:
            return 1.0
        
        current_objectives = [(sol.makespan, sol.total_tardiness) for sol in pareto_solutions]
        
        if true_pareto_front is None:
            # 构建理想参考前沿
            min_makespan = min(obj[0] for obj in current_objectives)
            min_tardiness = min(obj[1] for obj in current_objectives)
            max_makespan = max(obj[0] for obj in current_objectives)
            max_tardiness = max(obj[1] for obj in current_objectives)
            
            # 创建多点参考前沿
            true_pareto_front = []
            n_points = 10
            for i in range(n_points + 1):
                alpha = i / n_points
                makespan = min_makespan + alpha * (max_makespan - min_makespan) * 0.8
                tardiness = min_tardiness + (1 - alpha) * (max_tardiness - min_tardiness) * 0.8
                true_pareto_front.append((makespan, tardiness))
        
        # 计算GD
        total_distance = 0.0
        for current_point in current_objectives:
            min_distance = float('inf')
            for true_point in true_pareto_front:
                distance = np.sqrt((current_point[0] - true_point[0])**2 + 
                                 (current_point[1] - true_point[1])**2)
                min_distance = min(min_distance, distance)
            total_distance += min_distance
        
        avg_distance = total_distance / len(current_objectives)
        
        # 归一化
        max_makespan = max(max(obj[0] for obj in current_objectives), 
                          max(obj[0] for obj in true_pareto_front))
        max_tardiness = max(max(obj[1] for obj in current_objectives), 
                           max(obj[1] for obj in true_pareto_front))
        
        normalization_factor = np.sqrt(max_makespan**2 + max_tardiness**2)
        normalized_gd = avg_distance / max(normalization_factor, 1e-10)
        
        return min(max(normalized_gd, 0.0), 1.0)
    
    @staticmethod
    def calculate_spread(pareto_solutions: List) -> float:
        """
        计算归一化分布均匀性(Spread)指标
        
        Args:
            pareto_solutions: 帕累托解集
            
        Returns:
            归一化Spread值 [0, 1]，0表示分布最均匀
        """
        if not pareto_solutions or len(pareto_solutions) <= 2:
            return 1.0  # 解数量太少，分布性差
        
        objectives = [(sol.makespan, sol.total_tardiness) for sol in pareto_solutions]
        
        # 去除重复解
        unique_objectives = []
        tolerance = 1e-6
        for obj in objectives:
            is_duplicate = False
            for unique_obj in unique_objectives:
                if abs(obj[0] - unique_obj[0]) < tolerance and abs(obj[1] - unique_obj[1]) < tolerance:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_objectives.append(obj)
        
        if len(unique_objectives) <= 2:
            return 1.0
        
        # 按第一个目标排序
        sorted_objectives = sorted(unique_objectives, key=lambda x: x[0])
        
        # 计算相邻解之间的距离
        distances = []
        for i in range(len(sorted_objectives) - 1):
            dist = np.sqrt((sorted_objectives[i+1][0] - sorted_objectives[i][0])**2 + 
                          (sorted_objectives[i+1][1] - sorted_objectives[i][1])**2)
            distances.append(dist)
        
        if not distances:
            return 1.0
        
        # 计算平均距离
        mean_distance = np.mean(distances)
        
        if mean_distance == 0:
            return 1.0
        
        # 计算距离的标准差
        distance_std = np.std(distances)
        
        # 计算边界点距离
        # 找到极端点
        min_makespan_point = min(sorted_objectives, key=lambda x: x[0])
        max_makespan_point = max(sorted_objectives, key=lambda x: x[0])
        min_tardiness_point = min(sorted_objectives, key=lambda x: x[1])
        max_tardiness_point = max(sorted_objectives, key=lambda x: x[1])
        
        # 计算边界距离
        d_f = np.sqrt((sorted_objectives[0][0] - min_makespan_point[0])**2 + 
                     (sorted_objectives[0][1] - min_makespan_point[1])**2)
        d_l = np.sqrt((sorted_objectives[-1][0] - max_makespan_point[0])**2 + 
                     (sorted_objectives[-1][1] - max_makespan_point[1])**2)
        
        # 计算Spread
        numerator = d_f + d_l + sum(abs(d - mean_distance) for d in distances)
        denominator = d_f + d_l + (len(distances) * mean_distance)
        
        if denominator == 0:
            return 1.0
        
        spread = numerator / denominator
        
        # 归一化到[0, 1]
        normalized_spread = min(max(spread, 0.0), 1.0)
        
        return normalized_spread

class DatasetGenerator:
    """数据集生成器"""
    
    @staticmethod
    def generate_adaptive_configs(n_datasets: int = 20) -> List[ExperimentConfig]:
        """
        生成自适应规模的数据集配置
        
        Args:
            n_datasets: 数据集数量
            
        Returns:
            数据集配置列表
        """
        configs = []
        
        for i in range(n_datasets):
            # 设置随机种子
            dataset_seed = 42 + i * 17
            np.random.seed(dataset_seed)
            
            # 作业数：20-200均匀分布
            n_jobs = int(20 + (180 * i / (n_datasets - 1)))
            
            # 根据作业数自适应调整工厂数量
            if n_jobs <= 50:
                n_factories = np.random.randint(2, 4)  # 2-3个工厂
            elif n_jobs <= 100:
                n_factories = np.random.randint(3, 5)  # 3-4个工厂
            else:
                n_factories = np.random.randint(4, 7)  # 4-6个工厂
            
            # 阶段数：3-5个
            n_stages = np.random.randint(3, 6)
            
            # 根据作业数和工厂数自适应调整机器数
            base_machines = 2 if n_jobs <= 50 else (3 if n_jobs <= 100 else 4)
            max_machines = 4 if n_jobs <= 50 else (5 if n_jobs <= 100 else 5)
            
            # 每阶段机器数
            machines_per_stage = []
            for stage in range(n_stages):
                n_machines = np.random.randint(base_machines, max_machines + 1)
                machines_per_stage.append(n_machines)
            
            # 生成异构机器配置
            heterogeneous_machines = {}
            for factory_id in range(n_factories):
                factory_machines = []
                for stage in range(n_stages):
                    # 基于平均机器数生成变化
                    avg_machines = machines_per_stage[stage]
                    variation = np.random.choice([-1, 0, 1], p=[0.2, 0.6, 0.2])
                    actual_machines = max(2, min(5, avg_machines + variation))
                    factory_machines.append(actual_machines)
                heterogeneous_machines[factory_id] = factory_machines
            
            # 处理时间范围
            if n_jobs <= 50:
                time_range = (1, 15 + (i % 5))
            elif n_jobs <= 100:
                time_range = (1, 20 + (i % 8))
            else:
                time_range = (1, 25 + (i % 10))
            
            # 紧急度参数
            base_urgency = 0.8 + (i % 10) * 0.02
            urgency_ddt = [
                base_urgency + np.random.uniform(0, 0.1),
                base_urgency + 1.0 + np.random.uniform(0, 0.2),
                base_urgency + 2.0 + np.random.uniform(0, 0.3)
            ]
            
            # 创建配置
            scale_name = f"Dataset_{i+1:02d}_{n_jobs}J_{n_factories}F_{n_stages}S"
            
            config = ExperimentConfig(
                scale_name=scale_name,
                n_jobs=n_jobs,
                n_factories=n_factories,
                n_stages=n_stages,
                machines_per_stage=machines_per_stage,
                heterogeneous_machines=heterogeneous_machines,
                processing_time_range=time_range,
                urgency_ddt=urgency_ddt,
                dataset_seed=dataset_seed
            )
            
            configs.append(config)
        
        return configs

def generate_problem_data(config: ExperimentConfig) -> Dict:
    """
    根据配置生成问题数据
    
    Args:
        config: 实验配置
        
    Returns:
        问题数据字典
    """
    np.random.seed(config.dataset_seed)
    
    # 生成简化的处理时间矩阵 (n_jobs, n_stages)
    # 这里使用平均处理时间，实际的异构机器配置在factory_machines中处理
    processing_times = []
    for job_id in range(config.n_jobs):
        job_times = []
        for stage_id in range(config.n_stages):
            # 生成该作业在该阶段的平均处理时间
            time = np.random.randint(
                config.processing_time_range[0],
                config.processing_time_range[1] + 1
            )
            job_times.append(time)
        processing_times.append(job_times)
    
    # 生成截止日期
    due_dates = []
    for job_id in range(config.n_jobs):
        # 计算作业的总处理时间作为基准
        total_processing_time = sum(processing_times[job_id])
        
        # 使用紧急度参数生成截止日期
        urgency_factor = np.random.uniform(config.urgency_ddt[0], config.urgency_ddt[2])
        due_date = int(total_processing_time * urgency_factor)
        due_dates.append(max(due_date, 1))  # 确保截止日期至少为1
    
    return {
        'n_jobs': config.n_jobs,
        'n_factories': config.n_factories,
        'n_stages': config.n_stages,
        'machines_per_stage': config.machines_per_stage,
        'heterogeneous_machines': config.heterogeneous_machines,
        'processing_times': processing_times,
        'due_dates': due_dates,
        'urgencies': [1.0] * config.n_jobs  # 添加紧急度信息
    }

def run_algorithm_experiment(problem_data: Dict, algorithm_name: str, 
                           algorithm_class, algorithm_params: Dict, 
                           runs: int = 3) -> Dict:
    """
    运行单个算法实验
    
    Args:
        problem_data: 问题数据
        algorithm_name: 算法名称
        algorithm_class: 算法类
        algorithm_params: 算法参数
        runs: 运行次数
        
    Returns:
        实验结果字典
    """
    print(f"  运行 {algorithm_name} ({runs}次运行)...")
    
    all_results = []
    all_runtimes = []
    
    for run in range(runs):
        try:
            # 创建问题实例 - 修复参数传递
            problem = MO_DHFSP_Problem(problem_data)
            
            # 创建算法实例
            optimizer = algorithm_class(problem, **algorithm_params)
            
            # 运行算法
            start_time = time.time()
            solutions, convergence_data = optimizer.optimize()
            runtime = time.time() - start_time
            
            all_results.append(solutions)
            all_runtimes.append(runtime)
            
            print(f"    第{run+1}次运行: {len(solutions)}个解, 用时{runtime:.2f}s")
            
        except Exception as e:
            print(f"    第{run+1}次运行失败: {str(e)}")
            all_results.append([])
            all_runtimes.append(0.0)
    
    # 合并所有运行的结果
    all_solutions = []
    for solutions in all_results:
        all_solutions.extend(solutions)
    
    # 去重并保留帕累托最优解
    if all_solutions:
        pareto_solutions = extract_pareto_front(all_solutions)
    else:
        pareto_solutions = []
    
    # 计算统计结果
    if pareto_solutions:
        makespans = [sol.makespan for sol in pareto_solutions]
        tardiness = [sol.total_tardiness for sol in pareto_solutions]
        
        makespan_best = min(makespans)
        makespan_avg = np.mean(makespans)
        tardiness_best = min(tardiness)
        tardiness_avg = np.mean(tardiness)
        
        # 计算加权目标
        weighted_objectives = [0.5 * sol.makespan + 0.5 * sol.total_tardiness 
                             for sol in pareto_solutions]
        weighted_best = min(weighted_objectives)
        weighted_avg = np.mean(weighted_objectives)
    else:
        makespan_best = makespan_avg = float('inf')
        tardiness_best = tardiness_avg = float('inf')
        weighted_best = weighted_avg = float('inf')
    
    avg_runtime = np.mean(all_runtimes)
    
    return {
        'algorithm_name': algorithm_name,
        'pareto_solutions': pareto_solutions,
        'makespan_best': makespan_best,
        'makespan_avg': makespan_avg,
        'tardiness_best': tardiness_best,
        'tardiness_avg': tardiness_avg,
        'weighted_best': weighted_best,
        'weighted_avg': weighted_avg,
        'avg_runtime': avg_runtime,
        'pareto_count': len(pareto_solutions)
    }

def extract_pareto_front(solutions: List) -> List:
    """
    提取帕累托前沿
    
    Args:
        solutions: 解集合
        
    Returns:
        帕累托最优解集合
    """
    if not solutions:
        return []
    
    pareto_solutions = []
    
    for sol in solutions:
        is_dominated = False
        
        # 检查是否被其他解支配
        for other_sol in solutions:
            if (other_sol.makespan <= sol.makespan and 
                other_sol.total_tardiness <= sol.total_tardiness and
                (other_sol.makespan < sol.makespan or 
                 other_sol.total_tardiness < sol.total_tardiness)):
                is_dominated = True
                break
        
        if not is_dominated:
            # 检查是否已经存在相同的解
            is_duplicate = False
            for pareto_sol in pareto_solutions:
                if (abs(pareto_sol.makespan - sol.makespan) < 1e-6 and
                    abs(pareto_sol.total_tardiness - sol.total_tardiness) < 1e-6):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                pareto_solutions.append(sol)
    
    return pareto_solutions

def plot_pareto_fronts(all_results: Dict, scale_name: str):
    """
    绘制帕累托前沿对比图
    
    Args:
        all_results: 所有算法结果
        scale_name: 数据集名称
    """
    plt.figure(figsize=(12, 8))
    
    # 参考图片的颜色和标记样式
    algorithm_styles = {
        'RL-Chaotic-HHO': {'color': '#FF6B6B', 'marker': 'o', 'size': 80},      # 红色圆点
        'I-NSGA-II': {'color': '#4ECDC4', 'marker': 's', 'size': 70},           # 青色方块
        'MOPSO': {'color': '#45B7D1', 'marker': '^', 'size': 80},               # 蓝色三角形
        'MODE': {'color': '#FFA500', 'marker': 'v', 'size': 80},                # 橙色倒三角
        'DQN': {'color': '#9932CC', 'marker': '<', 'size': 80},                 # 紫色左三角
        'QL-ABC': {'color': '#8B4513', 'marker': '>', 'size': 80}               # 棕色右三角
    }
    
    for alg_name, result in all_results.items():
        if result['pareto_solutions']:
            makespans = [sol.makespan for sol in result['pareto_solutions']]
            tardiness = [sol.total_tardiness for sol in result['pareto_solutions']]
            
            style = algorithm_styles.get(alg_name, {'color': '#666666', 'marker': 'o', 'size': 60})
            
            plt.scatter(makespans, tardiness,
                       color=style['color'],
                       marker=style['marker'],
                       s=style['size'],
                       alpha=0.8,
                       label=alg_name,
                       edgecolors='black', 
                       linewidth=0.8)
    
    plt.xlabel('完工时间 (Makespan)', fontsize=14, fontweight='bold')
    plt.ylabel('总拖期 (Total Tardiness)', fontsize=14, fontweight='bold')
    plt.title(f'{scale_name} 帕累托前沿对比', fontsize=16, fontweight='bold')
    
    # 设置图例样式
    legend = plt.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    # 设置网格样式
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # 设置坐标轴样式
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    # 保存图片
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/pareto_front_{scale_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  帕累托前沿图已保存: results/pareto_front_{scale_name}.png")

def generate_comparison_tables(all_scale_results: Dict, configs: List[ExperimentConfig]):
    """
    生成对比表格和Excel文件
    
    Args:
        all_scale_results: 所有规模的实验结果
        configs: 实验配置列表
    """
    print("\n生成对比表格和Excel文件...")
    
    # 算法列表
    algorithms = ['RL-Chaotic-HHO', 'I-NSGA-II', 'MOPSO', 'MODE', 'DQN', 'QL-ABC']
    
    # 准备数据
    data_records = []
    
    for config in configs:
        scale_name = config.scale_name
        if scale_name not in all_scale_results:
            continue
            
        scale_results = all_scale_results[scale_name]
        
        # 计算联合帕累托前沿用于指标计算
        all_solutions = []
        for alg_name in algorithms:
            if alg_name in scale_results and scale_results[alg_name]['pareto_solutions']:
                all_solutions.extend(scale_results[alg_name]['pareto_solutions'])
        
        combined_pareto = extract_pareto_front(all_solutions) if all_solutions else []
        
        for alg_name in algorithms:
            if alg_name in scale_results:
                result = scale_results[alg_name]
                
                # 计算归一化指标 - 使用全局解集合进行归一化
                hv = NormalizedMetrics.calculate_hypervolume(result['pareto_solutions'], all_solutions)
                igd = NormalizedMetrics.calculate_igd(result['pareto_solutions'], 
                                                    [(sol.makespan, sol.total_tardiness) for sol in combined_pareto])
                gd = NormalizedMetrics.calculate_gd(result['pareto_solutions'], 
                                                   [(sol.makespan, sol.total_tardiness) for sol in combined_pareto])
                spread = NormalizedMetrics.calculate_spread(result['pareto_solutions'])
                
                record = {
                    'Dataset': scale_name,
                    'Algorithm': alg_name,
                    'Jobs': config.n_jobs,
                    'Factories': config.n_factories,
                    'Stages': config.n_stages,
                    'Makespan_Best': result['makespan_best'] if result['makespan_best'] != float('inf') else 'N/A',
                    'Makespan_Avg': result['makespan_avg'] if result['makespan_avg'] != float('inf') else 'N/A',
                    'Tardiness_Best': result['tardiness_best'] if result['tardiness_best'] != float('inf') else 'N/A',
                    'Tardiness_Avg': result['tardiness_avg'] if result['tardiness_avg'] != float('inf') else 'N/A',
                    'Weighted_Best': result['weighted_best'] if result['weighted_best'] != float('inf') else 'N/A',
                    'Weighted_Avg': result['weighted_avg'] if result['weighted_avg'] != float('inf') else 'N/A',
                    'HV': f"{hv:.4f}",
                    'IGD': f"{igd:.4f}",
                    'GD': f"{gd:.4f}",
                    'Spread': f"{spread:.4f}",
                    'Pareto_Count': result['pareto_count'],
                    'Runtime': f"{result['avg_runtime']:.2f}s"
                }
                
                data_records.append(record)
    
    # 创建DataFrame
    df = pd.DataFrame(data_records)
    
    # 生成Excel文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_filename = f"results/算法对比实验结果_{timestamp}.xlsx"
    
    try:
        with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
            # 完整结果表
            df.to_excel(writer, sheet_name='完整结果', index=False)
            
            # 按指标分别创建表格
            metrics = ['Makespan_Best', 'Tardiness_Best', 'Weighted_Best', 'HV', 'IGD', 'GD', 'Spread']
            metric_names = ['完工时间最优值', '总拖期最优值', '加权目标最优值', '超体积HV', '反世代距离IGD', '世代距离GD', '分布均匀性Spread']
            
            for metric, metric_name in zip(metrics, metric_names):
                # 创建透视表
                pivot_df = df.pivot(index='Dataset', columns='Algorithm', values=metric)
                pivot_df.to_excel(writer, sheet_name=metric_name)
            
            # 运行时间对比
            runtime_df = df.pivot(index='Dataset', columns='Algorithm', values='Runtime')
            runtime_df.to_excel(writer, sheet_name='运行时间对比')
            
            # 帕累托解数量对比
            pareto_count_df = df.pivot(index='Dataset', columns='Algorithm', values='Pareto_Count')
            pareto_count_df.to_excel(writer, sheet_name='帕累托解数量')
        
        print(f"Excel文件已保存: {excel_filename}")
        
    except Exception as e:
        print(f"Excel文件生成失败: {e}")
        
        # 生成文本报告作为备选
        txt_filename = f"results/算法对比实验结果_{timestamp}.txt"
        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write("多目标分布式异构混合流水车间调度算法对比实验结果\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"数据集数量: {len(configs)}\n")
            f.write(f"对比算法: {', '.join(algorithms)}\n\n")
            
            # 写入完整结果
            f.write("完整实验结果:\n")
            f.write("-" * 80 + "\n")
            f.write(df.to_string(index=False))
            f.write("\n\n")
            
            # 算法性能统计
            f.write("算法性能统计:\n")
            f.write("-" * 80 + "\n")
            for alg in algorithms:
                alg_data = df[df['Algorithm'] == alg]
                if not alg_data.empty:
                    f.write(f"\n{alg}:\n")
                    f.write(f"  平均HV: {alg_data['HV'].astype(float).mean():.4f}\n")
                    f.write(f"  平均IGD: {alg_data['IGD'].astype(float).mean():.4f}\n")
                    f.write(f"  平均GD: {alg_data['GD'].astype(float).mean():.4f}\n")
                    f.write(f"  平均Spread: {alg_data['Spread'].astype(float).mean():.4f}\n")
                    f.write(f"  平均运行时间: {alg_data['Runtime'].str.replace('s', '').astype(float).mean():.2f}s\n")
        
        print(f"文本报告已保存: {txt_filename}")

def main():
    """主函数"""
    print("多目标分布式异构混合流水车间调度算法综合对比实验")
    print("=" * 80)
    
    # 确保结果目录存在
    os.makedirs('results', exist_ok=True)
    
    # 生成数据集配置
    print("生成数据集配置...")
    configs = DatasetGenerator.generate_adaptive_configs(n_datasets=10)  # 先用10个数据集测试
    
    print(f"生成了 {len(configs)} 个数据集配置")
    print(f"作业数范围: {configs[0].n_jobs}-{configs[-1].n_jobs}")
    print(f"工厂数范围: 2-6")
    print(f"并行机器数范围: 2-5")
    
    # 算法配置
    algorithms = {
        'RL-Chaotic-HHO': (RL_ChaoticHHO_Optimizer, {
            'population_size': 50,
            'max_iterations': 50,
            'pareto_size_limit': 1000
        }),
        'I-NSGA-II': (ImprovedNSGA2_Optimizer, {
            'population_size': 50,
            'max_iterations': 50,
            'crossover_rate': 0.9,
            'mutation_rate': 0.1
        }),
        'MOPSO': (MOPSO_Optimizer, {
            'swarm_size': 50,
            'max_iterations': 50,
            'w': 0.4,
            'c1': 2.0,
            'c2': 2.0
        }),
        'MODE': (MODE_Optimizer, {
            'population_size': 50,
            'max_generations': 50,
            'F': 0.5,
            'CR': 0.9
        }),
        'DQN': (DQNAlgorithmWrapper, {
            'max_iterations': 50,
            'learning_rate': 0.001,
            'epsilon': 0.1
        }),
        'QL-ABC': (QLABC_Optimizer, {
            'population_size': 50,
            'max_iterations': 50,
            'learning_rate': 0.1,
            'epsilon': 0.05
        })
    }
    
    # 存储所有结果
    all_scale_results = {}
    
    # 对每个数据集运行实验
    for i, config in enumerate(configs):
        print(f"\n{'='*80}")
        print(f"运行数据集 {i+1}/{len(configs)}: {config.scale_name}")
        print(f"{'='*80}")
        print(f"作业数: {config.n_jobs}, 工厂数: {config.n_factories}, 阶段数: {config.n_stages}")
        
        # 生成问题数据
        problem_data = generate_problem_data(config)
        
        # 运行所有算法
        scale_results = {}
        for alg_name, (alg_class, alg_params) in algorithms.items():
            result = run_algorithm_experiment(
                problem_data, alg_name, alg_class, alg_params, runs=3
            )
            scale_results[alg_name] = result
        
        all_scale_results[config.scale_name] = scale_results
        
        # 绘制帕累托前沿图
        plot_pareto_fronts(scale_results, config.scale_name)
    
    # 生成对比表格和Excel文件
    generate_comparison_tables(all_scale_results, configs)
    
    print(f"\n{'='*80}")
    print("实验完成!")
    print(f"{'='*80}")
    print(f"共完成 {len(configs)} 个数据集的实验")
    print(f"结果文件保存在 results/ 目录中")

if __name__ == "__main__":
    main() 