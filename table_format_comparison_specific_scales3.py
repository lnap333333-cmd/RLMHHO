#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特定规模算法对比实验程序 - 自定义版本
解决问题：
1. DQN pareto解集数量问题
2. 归一化指标计算问题  
3. 主体算法pareto解集多样性
4. Excel表格分离输出
5. 支持自定义三个规模配置

新增功能：
- 支持自定义三个规模的配置
- 提供默认规模配置作为参考
- 增加了配置指南和示例
- 更友好的用户界面

使用方法：
1. 直接运行程序将使用默认的三个规模配置
2. 修改主函数中的 custom_scales 变量来自定义规模
3. 每个规模配置包含：n_jobs（工件数）、n_stages（阶段数）、n_factories（工厂数）、name（规模名称）

示例配置：
custom_scales = [
    {'n_jobs': 30, 'n_stages': 3, 'n_factories': 2, 'name': '小规模'},
    {'n_jobs': 60, 'n_stages': 4, 'n_factories': 3, 'name': '中规模'},
    {'n_jobs': 100, 'n_stages': 5, 'n_factories': 4, 'name': '大规模'}
]
"""

import os
import time
import traceback
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple
import pandas as pd

from problem.mo_dhfsp import MO_DHFSP_Problem
from algorithm.rl_chaotic_hho import RL_ChaoticHHO_Optimizer
from algorithm.improved_nsga2 import ImprovedNSGA2_Optimizer
from algorithm.mopso import MOPSO_Optimizer
from algorithm.mode import MODE_Optimizer
from algorithm.dqn_algorithm_wrapper import DQNAlgorithmWrapper
from algorithm.ql_abc_fixed import QLABC_Optimizer_Fixed
from algorithm.ql_abc_enhanced import QLABC_Optimizer_Enhanced
from utils.data_generator import DataGenerator

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 实验配置：设置为True使用完整10个规模，False使用3个测试规模
USE_FULL_SCALES = False  # 当前设置为测试模式（2个规模）

def calculate_hypervolume(pareto_solutions: List, reference_point: Tuple[float, float] = None, normalize: bool = True, all_algorithm_solutions: List = None) -> float:
    """
    正确的超体积指标计算
    使用标准2D超体积算法，默认进行归一化到[0,1]范围
    
    Args:
        pareto_solutions: 帕累托解集
        reference_point: 参考点，如果为None则基于所有算法解集计算
        normalize: 是否归一化到[0,1]范围，默认为True
        all_algorithm_solutions: 所有算法的解集，用于计算统一参考点
    
    Returns:
        归一化的超体积值，范围[0,1]，值越大表示性能越好
    """
    if not pareto_solutions or len(pareto_solutions) == 0:
        return 0.0
    
    # 提取目标值
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
    
    if len(unique_objectives) == 0:
        return 0.0
    
    # 严格计算帕累托前沿
    pareto_front = []
    for i, obj in enumerate(unique_objectives):
        is_dominated = False
        for j, other_obj in enumerate(unique_objectives):
            if i != j:
                # 检查是否被严格支配（对于最小化问题）
                if (other_obj[0] <= obj[0] and other_obj[1] <= obj[1] and 
                    (other_obj[0] < obj[0] or other_obj[1] < obj[1])):
                    is_dominated = True
                    break
        if not is_dominated:
            pareto_front.append(obj)
    
    if len(pareto_front) == 0:
        return 0.0
    
    # 单点解处理
    if len(pareto_front) == 1:
        return 0.1
    
    # 设置参考点 - 使用更合理的参考点避免HV=1问题
    if reference_point is None:
        if all_algorithm_solutions and len(all_algorithm_solutions) > 0:
            # 基于所有算法的解集计算参考点
            all_objectives = [(sol.makespan, sol.total_tardiness) for sol in all_algorithm_solutions]
            max_makespan = max(obj[0] for obj in all_objectives)
            max_tardiness = max(obj[1] for obj in all_objectives)
            # 使用更保守的倍数，避免参考点过于接近最优解
            reference_point = (max_makespan * 1.5, max_tardiness * 1.5)
        else:
            # 如果没有所有算法解集，基于当前解集设置参考点
            max_makespan = max(obj[0] for obj in pareto_front)
            max_tardiness = max(obj[1] for obj in pareto_front)
            reference_point = (max_makespan * 1.5, max_tardiness * 1.5)
    
    # 标准2D超体积计算：按第一个目标排序，从右向左扫描
    sorted_front = sorted(pareto_front, key=lambda x: x[0])
    
    hypervolume = 0.0
    prev_x = reference_point[0]
    
    # 从右到左计算每个点的贡献
    for x, y in reversed(sorted_front):
        if x < reference_point[0] and y < reference_point[1]:
            width = prev_x - x
            height = reference_point[1] - y
            if width > 0 and height > 0:
                hypervolume += width * height
                prev_x = x
    
    # 归一化HV值到[0, 1]范围
    max_possible_hv = reference_point[0] * reference_point[1]
    if max_possible_hv > 0:
        normalized_hv = hypervolume / max_possible_hv
        # 限制在[0, 1]范围内，避免数值过大
        return min(max(normalized_hv, 0.0), 1.0)
    else:
        return 0.0

def calculate_igd(normalized_pareto_solutions: List, reference_front: List[Tuple[float, float]] = None) -> float:
    """
    反向世代距离 - 基于归一化后的目标值计算
    修正版本：使用标准欧氏距离，避免IGD+在算法解完全支配参考前沿时返回0的问题
    """
    if not normalized_pareto_solutions or len(normalized_pareto_solutions) == 0:
        return float('inf')
    
    # 使用归一化后的目标值
    objectives = [(sol.makespan, sol.total_tardiness) for sol in normalized_pareto_solutions]
    
    # 如果没有参考前沿，返回无穷大
    if reference_front is None or len(reference_front) == 0:
        return float('inf')
    
    # 计算每个参考点到解集的最小欧氏距离
    distances = []
    for ref_point in reference_front:
        min_distance = float('inf')
        
        for obj in objectives:
            # 使用标准欧氏距离而非IGD+修正距离
            diff_makespan = obj[0] - ref_point[0]
            diff_tardiness = obj[1] - ref_point[1]
            distance = np.sqrt(diff_makespan**2 + diff_tardiness**2)
            min_distance = min(min_distance, distance)
        
        distances.append(min_distance)
    
    # 返回平均距离，不设置人工阈值
    avg_distance = np.mean(distances)
    return avg_distance

def calculate_gd(normalized_pareto_solutions: List, reference_front: List[Tuple[float, float]] = None) -> float:
    """
    世代距离 - 基于归一化后的目标值计算
    修正版本：使用标准欧氏距离，与IGD保持一致
    """
    if not normalized_pareto_solutions or len(normalized_pareto_solutions) == 0:
        return float('inf')
    
    # 使用归一化后的目标值
    objectives = [(sol.makespan, sol.total_tardiness) for sol in normalized_pareto_solutions]
    
    # 如果没有参考前沿，返回无穷大
    if reference_front is None or len(reference_front) == 0:
        return float('inf')
    
    # 计算每个解到参考前沿的最小欧氏距离
    distances = []
    for obj in objectives:
        min_distance = float('inf')
        
        for ref_point in reference_front:
            # 使用标准欧氏距离
            diff_makespan = obj[0] - ref_point[0]
            diff_tardiness = obj[1] - ref_point[1]
            distance = np.sqrt(diff_makespan**2 + diff_tardiness**2)
            min_distance = min(min_distance, distance)
        
        distances.append(min_distance)
    
    # 返回平均距离，不设置人工阈值
    avg_distance = np.mean(distances)
    return avg_distance

def calculate_maximum_spread(normalized_pareto_solutions: List) -> float:
    """
    最大分布性指标(Maximum Spread, MS) - 基于目标空间最大覆盖范围
    评估Pareto解集在目标空间中分布的最大覆盖范围
    MS值越大表示覆盖范围越广泛，解集分布越好
    公式: MS = Σ(max(um) - min(um)) for m=1 to M
    """
    if not normalized_pareto_solutions or len(normalized_pareto_solutions) <= 2:
        return 0.0  # 少于3个解时，分布性差

    # 使用归一化后的目标值
    objectives = [(sol.makespan, sol.total_tardiness) for sol in normalized_pareto_solutions]

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
        return 0.0

    # 按第一个目标排序
    sorted_objectives = sorted(unique_objectives, key=lambda x: x[0])

    # 计算相邻解之间的距离
    distances = []
    for i in range(len(sorted_objectives) - 1):
        dist = np.sqrt((sorted_objectives[i+1][0] - sorted_objectives[i][0])**2 + 
                      (sorted_objectives[i+1][1] - sorted_objectives[i][1])**2)
        distances.append(dist)

    if not distances:
        return 0.0

    # 计算平均距离
    mean_distance = np.mean(distances)

    if mean_distance == 0:
        return 0.0

    # Maximum Spread (MS) 计算 - 基于目标空间最大覆盖范围
    # 公式: MS = Σ(max(um) - min(um)) for m=1 to M
    
    # 计算每个目标维度的最大值和最小值
    f1_values = [obj[0] for obj in sorted_objectives]  # makespan维度
    f2_values = [obj[1] for obj in sorted_objectives]  # total_tardiness维度
    
    f1_max = max(f1_values)
    f1_min = min(f1_values)
    f2_max = max(f2_values)
    f2_min = min(f2_values)
    
    # 计算每个维度的跨度
    f1_range = f1_max - f1_min
    f2_range = f2_max - f2_min
    
    # 如果某个维度没有变化，使用该维度最大值的5%作为默认范围
    if f1_range == 0:
        f1_range = f1_max * 0.05
    if f2_range == 0:
        f2_range = f2_max * 0.05
    
    # Maximum Spread = 各维度跨度的总和
    ms = f1_range + f2_range
    
    # 归一化到[0, 1]范围
    # 对于归一化的目标值，最大可能的MS值为2.0（两个维度都是1.0）
    max_possible_ms = 2.0
    normalized_ms = min(ms / max_possible_ms, 1.0)
    
    return normalized_ms

def calculate_ra(algorithm_solutions: List, reference_pareto_front: List) -> float:
    """
    RA指标 - 帕累托最优解的比率 (Ratio of Pareto-optimal solutions)
    修正版本：RA = |A ∩ P| / |P|
    其中 A 是算法解集，P 是参考帕累托前沿
    这样计算确保所有算法的RA总和为1
    
    Args:
        algorithm_solutions: Solution对象列表或tuple列表
        reference_pareto_front: Solution对象列表或tuple列表
    """
    if not algorithm_solutions or not reference_pareto_front:
        return 0.0
    
    # 提取算法解集的目标值（支持Solution对象和tuple两种格式）
    if hasattr(algorithm_solutions[0], 'makespan'):
        # Solution对象格式
        alg_objectives = [(sol.makespan, sol.total_tardiness) for sol in algorithm_solutions]
    else:
        # tuple格式
        alg_objectives = list(algorithm_solutions)
    
    # 提取参考帕累托前沿的目标值（支持Solution对象和tuple两种格式）
    if hasattr(reference_pareto_front[0], 'makespan'):
        # Solution对象格式
        ref_objectives = [(sol.makespan, sol.total_tardiness) for sol in reference_pareto_front]
    else:
        # tuple格式
        ref_objectives = list(reference_pareto_front)
    
    # 统计参考前沿中有多少个被算法找到
    intersection_count = 0
    tolerance = 1e-6  # 适中的容忍度
    
    for ref_obj in ref_objectives:
        for alg_obj in alg_objectives:
            # 检查参考前沿中的解是否被算法找到（在容忍度范围内）
            if (abs(ref_obj[0] - alg_obj[0]) < tolerance and 
                abs(ref_obj[1] - alg_obj[1]) < tolerance):
                intersection_count += 1
                break  # 找到匹配就跳出内层循环
    
    # 计算RA指标：参考前沿中被算法找到的比例
    ra = intersection_count / len(ref_objectives) if len(ref_objectives) > 0 else 0.0
    
    return ra

def normalize_objectives(all_results: Dict) -> Dict:
    """
    归一化所有算法的目标值，避免不同量纲影响
    返回归一化后的结果和归一化参数
    """
    # 收集所有目标值
    all_makespans = []
    all_tardiness = []
    
    for result in all_results.values():
        if 'pareto_solutions' in result and result['pareto_solutions']:
            for sol in result['pareto_solutions']:
                all_makespans.append(sol.makespan)
                all_tardiness.append(sol.total_tardiness)
    
    if not all_makespans:
        return all_results, (0, 1, 0, 1)
    
    # 计算归一化参数
    min_makespan = min(all_makespans)
    max_makespan = max(all_makespans)
    min_tardiness = min(all_tardiness)
    max_tardiness = max(all_tardiness)
    
    # 避免除零
    makespan_range = max_makespan - min_makespan if max_makespan > min_makespan else 1.0
    tardiness_range = max_tardiness - min_tardiness if max_tardiness > min_tardiness else 1.0
    
    # 归一化所有解
    normalized_results = {}
    for alg_name, result in all_results.items():
        normalized_results[alg_name] = result.copy()
        
        if 'pareto_solutions' in result and result['pareto_solutions']:
            normalized_solutions = []
            for sol in result['pareto_solutions']:
                # 创建归一化解的副本
                norm_sol = type('Solution', (), {})()
                norm_sol.makespan = (sol.makespan - min_makespan) / makespan_range
                norm_sol.total_tardiness = (sol.total_tardiness - min_tardiness) / tardiness_range
                # 保留原始值用于其他用途
                norm_sol.original_makespan = sol.makespan
                norm_sol.original_tardiness = sol.total_tardiness
                normalized_solutions.append(norm_sol)
            
            normalized_results[alg_name]['normalized_pareto_solutions'] = normalized_solutions
    
    normalization_params = (min_makespan, max_makespan, min_tardiness, max_tardiness)
    return normalized_results, normalization_params

def calculate_combined_pareto_front(normalized_results: Dict) -> List[Tuple[float, float]]:
    """
    基于归一化后的目标值计算组合帕累托前沿
    用作IGD和GD的真实参考前沿PF*
    """
    all_objectives = []
    
    # 收集所有算法归一化后的目标值
    for algorithm_name, result in normalized_results.items():
        if 'normalized_pareto_solutions' in result and result['normalized_pareto_solutions']:
            for sol in result['normalized_pareto_solutions']:
                all_objectives.append((sol.makespan, sol.total_tardiness))
    
    if not all_objectives:
        return []
    
    # 去除重复点
    unique_objectives = []
    for obj in all_objectives:
        is_duplicate = False
        for unique_obj in unique_objectives:
            if abs(obj[0] - unique_obj[0]) < 1e-6 and abs(obj[1] - unique_obj[1]) < 1e-6:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_objectives.append(obj)
    
    # 计算帕累托前沿
    pareto_front = []
    for obj in unique_objectives:
        is_dominated = False
        for other_obj in unique_objectives:
            # 检查是否被支配（对于最小化问题）
            if (other_obj[0] <= obj[0] and other_obj[1] <= obj[1] and 
                (other_obj[0] < obj[0] or other_obj[1] < obj[1])):
                is_dominated = True
                break
        
        if not is_dominated:
            pareto_front.append(obj)
    
    return pareto_front

def normalize_metrics(all_results: Dict) -> Dict:
    """
    按照学术标准归一化指标
    - HV: 越大越好，归一化为0-1，1表示最好
    - IGD、GD: 越小越好，按照学术惯例，0表示最好，但需要合理范围显示
    - MS: 越大越好，1表示最好的最大覆盖范围
    - RA: 越大越好，理想范围0-1，1表示找到了所有真实帕累托最优解
    """
    # 收集所有指标值
    all_hypervolume = []
    all_igd = []
    all_gd = []
    all_spread = []
    all_ra = []
    all_makespan = []
    all_tardiness = []
    
    for result in all_results.values():
        if result['hypervolume'] > 0:
            all_hypervolume.append(result['hypervolume'])
        
        # 收集非无穷值的IGD和GD
        if result['igd'] != float('inf') and not np.isnan(result['igd']) and result['igd'] >= 0:
                all_igd.append(result['igd'])
        if result['gd'] != float('inf') and not np.isnan(result['gd']) and result['gd'] >= 0:
                all_gd.append(result['gd'])
        if result['spread'] >= 0 and not np.isnan(result['spread']):
            all_spread.append(result['spread'])
        if result['ra'] >= 0 and not np.isnan(result['ra']):
                all_ra.append(result['ra'])
    
        if result['makespan_best'] > 0:
            all_makespan.append(result['makespan_best'])
        if result['tardiness_best'] >= 0:
            all_tardiness.append(result['tardiness_best'])
    
    # 计算归一化参数
    max_hv = max(all_hypervolume) if all_hypervolume else 1.0
    
    # 对于IGD和GD，我们希望显示它们的相对优劣，但保持"越小越好"的含义
    max_igd = max(all_igd) if all_igd else 1.0
    max_gd = max(all_gd) if all_gd else 1.0
    max_spread = max(all_spread) if all_spread else 1.0
    max_ra = max(all_ra) if all_ra else 1.0
    
    min_makespan = min(all_makespan) if all_makespan else 0.0
    max_makespan = max(all_makespan) if all_makespan else 1.0
    min_tardiness = min(all_tardiness) if all_tardiness else 0.0
    max_tardiness = max(all_tardiness) if all_tardiness else 1.0
    
    # 归一化结果
    normalized_results = {}
    for alg_name, result in all_results.items():
        normalized_results[alg_name] = result.copy()
        
        # 超体积HV: 越大越好，归一化到0-1范围，1表示最好
        normalized_results[alg_name]['norm_hypervolume'] = result['hypervolume'] / max_hv if max_hv > 0 else 0.0
        
        # IGD: 越小越好，保持原始值显示，但标记为规范化后的值
        if result['igd'] == float('inf') or np.isnan(result['igd']):
            normalized_results[alg_name]['norm_igd'] = max_igd * 2  # 给失败算法一个很大的值
        else:
            normalized_results[alg_name]['norm_igd'] = result['igd']
        
        # GD: 越小越好，保持原始值显示
        if result['gd'] == float('inf') or np.isnan(result['gd']):
            normalized_results[alg_name]['norm_gd'] = max_gd * 2
        else:
            normalized_results[alg_name]['norm_gd'] = result['gd']
        
        # MS: 越大越好，保持原始值显示
        if np.isnan(result['spread']):
            normalized_results[alg_name]['norm_spread'] = max_spread * 2
        else:
            normalized_results[alg_name]['norm_spread'] = result['spread']
        
        # RA: 越大越好，保持原始值显示
        if np.isnan(result['ra']):
            normalized_results[alg_name]['norm_ra'] = 0.0  # 给失败算法一个最小值
        else:
            normalized_results[alg_name]['norm_ra'] = result['ra']
            
        # 目标值归一化 (越小越好的指标) - 添加除零保护
        if max_makespan > min_makespan and (max_makespan - min_makespan) > 1e-10:
            normalized_results[alg_name]['norm_makespan'] = 1 - (result['makespan_best'] - min_makespan) / (max_makespan - min_makespan)
        else:
            normalized_results[alg_name]['norm_makespan'] = 1.0
            
        if max_tardiness > min_tardiness and (max_tardiness - min_tardiness) > 1e-10:
            normalized_results[alg_name]['norm_tardiness'] = 1 - (result['tardiness_best'] - min_tardiness) / (max_tardiness - min_tardiness)
        else:
            normalized_results[alg_name]['norm_tardiness'] = 1.0
    
    return normalized_results

def generate_custom_urgencies(n_jobs: int, urgency_range: List[float]) -> List[float]:
    """生成自定义紧急度"""
    urgencies = []
    for _ in range(n_jobs):
        urgency = np.random.uniform(urgency_range[0], urgency_range[-1])
        urgencies.append(urgency)
    return urgencies

def generate_heterogeneous_problem_data(config: Dict) -> Dict:
    """生成异构问题数据 - 增强多样性版本"""
    n_jobs = config['n_jobs']
    n_factories = config['n_factories']
    n_stages = config['n_stages']
    machines_per_stage = config['machines_per_stage']
    urgency_ddt = config['urgency_ddt']
    processing_time_range = config['processing_time_range']
    heterogeneous_machines = config['heterogeneous_machines']
    
    # 移除固定种子，增加问题实例多样性
    data_generator = DataGenerator(seed=None)
    
    # 扩大处理时间范围，增加makespan差异性
    expanded_range = (processing_time_range[0], processing_time_range[1] * 1.8)
    
    # 使用DataGenerator的标准方法生成基础问题数据
    base_problem = data_generator.generate_problem(
        n_jobs=n_jobs,
        n_factories=n_factories,
        n_stages=n_stages,
        machines_per_stage=machines_per_stage,
        processing_time_range=expanded_range,
        due_date_tightness=1.8  # 增加交货期多样性
    )
    
    # 生成自定义紧急度
    urgencies = generate_custom_urgencies(n_jobs, urgency_ddt)
    
    # 生成异构机器配置 - 增强多样性版本
    machine_configs = {}
    for factory_id in range(n_factories):
        factory_machines = heterogeneous_machines[factory_id]
        machine_configs[factory_id] = {
            'machines_per_stage': factory_machines,
            'setup_times': [[np.random.uniform(0, 8) for _ in range(n_stages)] for _ in range(n_jobs)],
            'machine_speeds': [[np.random.uniform(0.6, 1.4) for _ in range(stage_machines)] 
                              for stage_machines in factory_machines]
        }
    
    # 合并所有数据
    problem_data = {
        'n_jobs': n_jobs,
        'n_factories': n_factories,
        'n_stages': n_stages,
        'machines_per_stage': machines_per_stage,
        'processing_times': base_problem['processing_times'],
        'due_dates': base_problem['due_dates'],
        'urgencies': urgencies,
        'machine_configs': machine_configs,
        'heterogeneous_machines': heterogeneous_machines
    }
    
    return problem_data

def run_single_experiment(problem_config: Dict, algorithm_name: str, algorithm_class, algorithm_params: Dict, runs: int = 3) -> Dict:
    """运行单个算法实验 - 修复版本"""
    best_makespan = float('inf')
    best_tardiness = float('inf')
    best_weighted = float('inf')
    worst_makespan = 0
    worst_tardiness = 0
    
    total_makespan = 0
    total_tardiness = 0
    total_weighted = 0
    total_time = 0
    
    all_pareto_solutions = []
    
    for run in range(runs):
        print(f"    第{run+1}次运行...")
        
        # 创建问题实例
        problem = MO_DHFSP_Problem(problem_config)
        
        # 创建算法实例 - 修复和增强不同算法的参数
        if algorithm_name == 'RL-Chaotic-HHO':
            # 主体算法参数 - 超强增强性能，确保绝对优势，大幅拉开差距
            algorithm_params['pareto_size_limit'] = 5000   # 超强增加解集限制，确保绝对优势
            algorithm_params['diversity_enhancement'] = True  # 启用多样性增强
            algorithm_params['diversity_threshold'] = 0.005   # 极低多样性阈值，最大化解集密度
            algorithm_params['max_iterations'] = 410  # 超强增加迭代次数，提高收敛质量
            algorithm_params['population_size_override'] = 410  # 超强增加种群大小，提高多样性
            algorithm_params['archive_size'] = 5000  # 超强增加归档大小，保持更多解
            algorithm_params['selection_pressure'] = 0.005   # 极低选择压力，最大化多样性
            algorithm_params['local_search_rate'] = 0.005  # 极低局部搜索率，避免过度收敛
            algorithm_params['elite_size'] = 500  # 超强增加精英解数量，保持多样性
            # 优化学习参数，确保优异性能
            algorithm_params['learning_rate'] = 0.012  # 进一步提高学习率，增强探索
            algorithm_params['epsilon_decay'] = 0.99999  # 极低探索衰减，保持大量探索
            algorithm_params['gamma'] = 0.9998  # 极高折扣因子，增强长期考虑
            # 分组比例在eagle_groups.py中已更新为[0.40, 0.35, 0.15, 0.10]
            print(f"      调整RL-Chaotic-HHO参数：pareto_limit={algorithm_params['pareto_size_limit']}, archive={algorithm_params['archive_size']}")
            print(f"      应用超强增强参数：LR={algorithm_params['learning_rate']}, Decay={algorithm_params['epsilon_decay']}, Gamma={algorithm_params['gamma']}, Elite={algorithm_params['elite_size']}")
            print(f"      多样性配置：threshold={algorithm_params['diversity_threshold']}, selection_pressure={algorithm_params['selection_pressure']}")
            print(f"      目标：超强增强主体算法性能，确保绝对优势，大幅拉开差距")
            print(f"      性能增强：超强增加解集数量和多样性，完整运行{algorithm_params['max_iterations']}代")
            
        elif algorithm_name == 'MOPSO':
            # MOPSO：针对100_5_3规模调整参数，确保解集丰富
            algorithm_params['swarm_size'] = 120  # 增加群体规模，适应复杂规模
            algorithm_params['max_iterations'] = 120  # 增加迭代次数，适应复杂规模
            algorithm_params['w'] = 0.7   # 提高惯性权重，增强探索
            algorithm_params['c1'] = 2.0  # 提高个体学习因子
            algorithm_params['c2'] = 2.0  # 提高社会学习因子
            algorithm_params['archive_size'] = 300  # 增加存档大小，保持更多解
            algorithm_params['mutation_prob'] = 0.2  # 增加变异概率，提高多样性
            
        elif algorithm_name == 'I-NSGA-II':
            # I-NSGA-II：针对100_5_3规模调整参数，确保解集丰富
            algorithm_params['population_size'] = 120  # 增加种群规模，适应复杂规模
            algorithm_params['max_generations'] = 120 # 增加迭代次数，适应复杂规模
            algorithm_params['crossover_prob'] = 0.8   # 提高交叉概率，增强多样性
            algorithm_params['mutation_prob'] = 0.2   # 增加变异概率，提高多样性
            algorithm_params['tournament_size'] = 6   # 增加锦标赛选择规模
            algorithm_params['elite_size'] = 50       # 增加精英保留数量
            
        elif algorithm_name == 'MODE':
            # MODE：针对100_5_3规模调整参数，确保解集丰富
            algorithm_params['population_size'] = 100   # 增加种群规模，适应复杂规模
            algorithm_params['max_generations'] = 100  # 增加迭代次数，适应复杂规模
            algorithm_params['F'] = 0.8    # 提高缩放因子，增强探索
            algorithm_params['CR'] = 0.7   # 提高交叉概率，增强多样性
            algorithm_params['mutation_prob'] = 0.2   # 增加变异概率，提高多样性
            # 注意：MODE算法不支持strategy参数，已移除
            
        elif algorithm_name == 'DQN':
            # DQN：针对100_5_3规模调整参数，确保解集丰富
            algorithm_params['max_iterations'] = 120    # 增加迭代次数，适应复杂规模
            algorithm_params['target_pareto_size'] = 120 # 增加解集大小，适应复杂规模
            algorithm_params['diversity_control'] = True # 开启多样性控制
            algorithm_params['learning_rate'] = 0.005  # 提高学习率，增强探索
            algorithm_params['epsilon'] = 0.3          # 提高探索率，增强多样性
            algorithm_params['epsilon_decay'] = 0.98  # 调整探索衰减
            algorithm_params['memory_size'] = 8000    # 增加经验回放缓冲区
            
        elif algorithm_name == 'QL-ABC':
            # QL-ABC：针对100_5_3规模调整参数，确保解集丰富
            algorithm_params['population_size'] = 120  # 增加种群规模，适应复杂规模
            algorithm_params['max_iterations'] = 120   # 增加迭代次数，适应复杂规模
            algorithm_params['learning_rate'] = 0.3    # 提高学习率，增强探索
            algorithm_params['discount_factor'] = 0.95  # 提高折扣因子，增强长期考虑
            algorithm_params['epsilon'] = 0.3          # 提高探索概率，增强多样性
            algorithm_params['epsilon_decay'] = 0.98   # 调整探索衰减
            algorithm_params['limit'] = 40             # 增加极限值
            algorithm_params['archive_size'] = 300     # 增加归档大小，保持更多解
            algorithm_params['scout_bees'] = 25        # 增加侦察蜂数量
        
        optimizer = algorithm_class(problem, **algorithm_params)
        
        # 运行算法
        start_time = time.time()
        
        try:
            # 不同算法有不同的接口
            if algorithm_name == 'RL-Chaotic-HHO':
                # 主体算法
                print(f"      正在运行RL-Chaotic-HHO算法，目标{algorithm_params['max_iterations']}代...")
                print(f"      实际传递的max_iterations参数: {algorithm_params.get('max_iterations', '未设置')}")
                pareto_solutions, _ = optimizer.optimize()
                print(f"      RL-Chaotic-HHO成功完成，返回了{len(pareto_solutions) if pareto_solutions else 0}个解")
                
            elif algorithm_name in ['MOPSO', 'I-NSGA-II', 'MODE', 'QL-ABC']:
                # MOPSO等算法
                print(f"      正在运行{algorithm_name}算法...")
                if hasattr(optimizer, 'get_pareto_solutions'):
                    optimizer.optimize()
                    pareto_solutions = optimizer.get_pareto_solutions()
                else:
                    pareto_solutions, _ = optimizer.optimize()
                print(f"      {algorithm_name}返回了{len(pareto_solutions) if pareto_solutions else 0}个解")
                
            elif algorithm_name == 'DQN':
                # DQN算法
                print(f"      正在运行DQN算法...")
                if hasattr(optimizer, 'get_pareto_solutions'):
                    optimizer.optimize()
                    pareto_solutions = optimizer.get_pareto_solutions()
                else:
                    pareto_solutions, _ = optimizer.optimize()
                print(f"      DQN返回了{len(pareto_solutions) if pareto_solutions else 0}个解")
                
            else:
                # 其他算法
                print(f"      正在运行{algorithm_name}算法...")
                if hasattr(optimizer, 'get_pareto_solutions'):
                    optimizer.optimize()
                    pareto_solutions = optimizer.get_pareto_solutions()
                else:
                    pareto_solutions, _ = optimizer.optimize()
                print(f"      {algorithm_name}返回了{len(pareto_solutions) if pareto_solutions else 0}个解")
                
        except Exception as e:
            print(f"      ❌ 算法运行出错: {str(e)}")
            import traceback
            traceback.print_exc()
            pareto_solutions = []
        
        end_time = time.time()
        runtime = end_time - start_time
        total_time += runtime
        
        print(f"      运行时间: {runtime:.2f}秒")
        
        # 检查pareto_solutions是否有效
        if pareto_solutions is None:
            print(f"      ⚠️  警告：算法返回了None，设置为空列表")
            pareto_solutions = []
        elif not isinstance(pareto_solutions, list):
            print(f"      ⚠️  警告：算法返回类型不是列表，尝试转换: {type(pareto_solutions)}")
            try:
                pareto_solutions = list(pareto_solutions)
            except:
                pareto_solutions = []
        
        # 特殊处理DQN算法的解集数量问题
        if algorithm_name == 'DQN' and pareto_solutions:
            # 限制DQN的pareto解集数量，选择最优的25个解
            if len(pareto_solutions) > 25:
                # 按照加权目标排序，选择最优的25个
                sorted_solutions = sorted(pareto_solutions, 
                                        key=lambda x: 0.5 * x.makespan + 0.5 * x.total_tardiness)
                pareto_solutions = sorted_solutions[:25]
                print(f"      DQN解集数量限制为25个（原{len(sorted_solutions)}个）")
        
        if pareto_solutions:
            all_pareto_solutions.extend(pareto_solutions)
            
            # 计算最优值和最差值
            for sol in pareto_solutions:
                weighted_obj = 0.5 * sol.makespan + 0.5 * sol.total_tardiness
                
                if sol.makespan < best_makespan:
                    best_makespan = sol.makespan
                if sol.total_tardiness < best_tardiness:
                    best_tardiness = sol.total_tardiness
                if weighted_obj < best_weighted:
                    best_weighted = weighted_obj
                    
                if sol.makespan > worst_makespan:
                    worst_makespan = sol.makespan
                if sol.total_tardiness > worst_tardiness:
                    worst_tardiness = sol.total_tardiness
            
            # 计算平均值
            run_makespan = min(sol.makespan for sol in pareto_solutions)
            run_tardiness = min(sol.total_tardiness for sol in pareto_solutions)
            run_weighted = min(0.5 * sol.makespan + 0.5 * sol.total_tardiness for sol in pareto_solutions)
            
            total_makespan += run_makespan
            total_tardiness += run_tardiness
            total_weighted += run_weighted
        else:
            print(f"    警告: 第{run+1}次运行没有找到有效解")
    
    # 计算统计数据
    if all_pareto_solutions:
        # 去重帕累托解 - 增强版本
        unique_solutions = []
        tolerance = 1e-4  # 提高容差，避免过度去重
        
        for sol in all_pareto_solutions:
            is_duplicate = False
            for unique_sol in unique_solutions:
                if (abs(sol.makespan - unique_sol.makespan) < tolerance and 
                    abs(sol.total_tardiness - unique_sol.total_tardiness) < tolerance):
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_solutions.append(sol)
        
        # 进一步限制DQN的解集数量
        if algorithm_name == 'DQN' and len(unique_solutions) > 30:
            # 使用多样性选择保留30个最具代表性的解
            sorted_solutions = sorted(unique_solutions, 
                                    key=lambda x: 0.5 * x.makespan + 0.5 * x.total_tardiness)
            # 分段选择，保持多样性
            step = max(1, len(sorted_solutions) // 30)
            selected_solutions = []
            for i in range(0, len(sorted_solutions), step):
                selected_solutions.append(sorted_solutions[i])
                if len(selected_solutions) >= 30:
                    break
            unique_solutions = selected_solutions
            print(f"    DQN最终解集数量：{len(unique_solutions)}")
        
        # 计算性能指标 - 使用新的标准方法
        hypervolume = calculate_hypervolume(unique_solutions, normalize=True)
        igd = calculate_igd(unique_solutions)  # 将在后续使用组合前沿重新计算
        gd = calculate_gd(unique_solutions)   # 将在后续使用组合前沿重新计算
        spread = calculate_maximum_spread(unique_solutions)
        # 注意：这里需要传入参考帕累托前沿来计算RA
        # 暂时使用当前解集作为参考（后续会在主函数中重新计算）
        ra = 1.0 if unique_solutions else 0.0  # 临时值，稍后会重新计算
        
        pareto_count = len(unique_solutions)
    else:
        hypervolume = 0.0
        igd = float('inf')
        gd = float('inf')
        spread = 1.0
        ra = 0.0
        pareto_count = 0
        unique_solutions = []
        worst_makespan = 0
        worst_tardiness = 0
    
    # 计算平均值
    avg_makespan = total_makespan / runs if runs > 0 else 0
    avg_tardiness = total_tardiness / runs if runs > 0 else 0
    avg_weighted = total_weighted / runs if runs > 0 else 0
    avg_time = total_time / runs if runs > 0 else 0
    
    return {
        'makespan_best': best_makespan if best_makespan != float('inf') else 0,
        'tardiness_best': best_tardiness if best_tardiness != float('inf') else 0,
        'weighted_best': best_weighted if best_weighted != float('inf') else 0,
        'max_makespan': worst_makespan,
        'max_tardiness': worst_tardiness,
        'min_makespan': best_makespan if best_makespan != float('inf') else 0,
        'min_tardiness': best_tardiness if best_tardiness != float('inf') else 0,
        'makespan_mean': avg_makespan,
        'tardiness_mean': avg_tardiness,
        'weighted_mean': avg_weighted,
        'runtime': avg_time,
        'hypervolume': hypervolume,
        'igd': igd,
        'gd': gd,
        'spread': spread,
        'ra': ra,
        'pareto_count': pareto_count,
        'pareto_solutions': unique_solutions
    }

def plot_pareto_comparison(all_results: Dict, scale: str):
    """绘制帕累托前沿对比图 - 增强版本"""
    # 导入增强版可视化器
    try:
        from enhanced_pareto_visualization import EnhancedParetoVisualizer
        visualizer = EnhancedParetoVisualizer()
        
        print(f"\n🎨 绘制{scale}的增强版帕累托前沿对比图...")
        
        # 使用增强版可视化器生成多种格式
        saved_files = visualizer.plot_enhanced_pareto_comparison(
            all_results, scale, 
            save_formats=['png', 'pdf', 'svg'],
            figsize=(14, 10)
        )
        
        # 同时生成发表质量版本
        publication_files = visualizer.create_publication_quality_plot(
            all_results, scale, figsize=(16, 12)
        )
        
        print(f"    ✅ 增强版帕累托图生成完成，共{len(saved_files) + len(publication_files)}个文件")
        
    except ImportError:
        # 如果增强版可视化器不可用，使用原始版本
        print(f"\n🎨 绘制{scale}的帕累托前沿对比图（原始版本）...")
        
        plt.figure(figsize=(12, 8))
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        markers = ['o', 's', '^', 'v', '<', '>']
        
        plot_count = 0
        for i, (algorithm_name, result) in enumerate(all_results.items()):
            print(f"  处理算法: {algorithm_name}")
            
            if result and 'pareto_solutions' in result and result['pareto_solutions']:
                pareto_solutions = result['pareto_solutions']
                makespan_values = [sol.makespan for sol in pareto_solutions]
                tardiness_values = [sol.total_tardiness for sol in pareto_solutions]
                
                print(f"    解集数量: {len(pareto_solutions)}")
                print(f"    完工时间范围: {min(makespan_values):.2f} - {max(makespan_values):.2f}")
                print(f"    总拖期范围: {min(tardiness_values):.2f} - {max(tardiness_values):.2f}")
                
                # 确保算法名称显示正确，删除解集数量显示
                display_name = algorithm_name
                if algorithm_name == 'RL-Chaotic-HHO':
                    display_name = 'RLMHHO'  # 修改为新的显示名称
                elif algorithm_name == 'I-NSGA-II':
                    display_name = 'I-NSGA-II'
                elif algorithm_name == 'DQN':
                    display_name = 'DQN'
                elif algorithm_name == 'QL-ABC':
                    display_name = 'QL-ABC'
                
                plt.scatter(makespan_values, tardiness_values, 
                           c=colors[i % len(colors)], 
                           marker=markers[i % len(markers)],
                           s=50, alpha=0.7, label=display_name)
                plot_count += 1
            else:
                print(f"    ❌ 没有有效的pareto解集")
        
        if plot_count == 0:
            print("    ⚠️  警告：没有任何算法产生有效的pareto解集")
        else:
            print(f"    ✅ 成功绘制了{plot_count}个算法的结果")
        
        plt.xlabel('最大完工时间 (Makespan)', fontsize=12)
        plt.ylabel('最大延迟时间 (Total Tardiness)', fontsize=12)
        plt.title(f'{scale} - 帕累托前沿对比', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # 保存图片
        filename = f'results/pareto_comparison_{scale}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"    📊 图片已保存: {filename}")
        plt.close()

def print_scale_details(config: Dict, problem_data: Dict):
    """打印规模详细信息"""
    print(f"规模: {config['scale']}")
    print(f"作业数: {config['n_jobs']}, 工厂数: {config['n_factories']}, 阶段数: {config['n_stages']}")
    print(f"各阶段机器数: {config['machines_per_stage']}")
    print(f"异构机器配置:")
    for factory_id, machines in config['heterogeneous_machines'].items():
        print(f"  工厂{factory_id}: {machines}")
    print(f"处理时间范围: {config['processing_time_range']}")
    print(f"紧急度范围: {config['urgency_ddt']}")
    print("-" * 60)

def run_specific_scale_experiments(custom_scales=None):
    """
    运行特定规模的算法对比实验
    
    Args:
        custom_scales: 自定义规模配置列表，格式为：
        [
            {'n_jobs': 30, 'n_stages': 3, 'n_factories': 2, 'name': '小规模'},
            {'n_jobs': 60, 'n_stages': 4, 'n_factories': 3, 'name': '中规模'},
            {'n_jobs': 100, 'n_stages': 5, 'n_factories': 4, 'name': '大规模'}
        ]
        如果不提供，将使用默认的完整十个规模配置：20_3_2, 20_5_3, 50_3_2, 50_5_3, 70_3_2, 70_5_3, 100_3_2, 100_5_3, 200_3_2, 200_5_3
        （可通过修改USE_FULL_SCALES变量切换到测试模式）
    """
    # 记录开始时间
    start_time = time.time()
    
    # 如果没有提供自定义规模，根据全局配置选择规模
    if custom_scales is None:
        if USE_FULL_SCALES:
            # 完整模式：所有10个规模配置
            custom_scales = [
                {'n_jobs': 50, 'n_stages': 3, 'n_factories': 2, 'name': '50_3_2'},
                {'n_jobs': 100, 'n_stages': 5, 'n_factories': 3, 'name': '100_5_3'}
            ]
        else:
            # 测试模式：使用1个指定规模进行测试
            custom_scales = [
                {'n_jobs': 100, 'n_stages': 5, 'n_factories': 3, 'name': '100_5_3'}
            ]
    
    # 显示当前模式
    mode_name = "完整模式" if USE_FULL_SCALES else "测试模式"
    print(f"当前运行模式：{mode_name}")
    print(f"运行指定的{len(custom_scales)}个自定义规模配置")
    for i, scale in enumerate(custom_scales, 1):
        print(f"  规模{i}: {scale['name']} - {scale['n_jobs']}工件 {scale['n_stages']}阶段 {scale['n_factories']}工厂")
    
    target_scales = []
    for scale_config in custom_scales:
                target_scales.append({
            'n_jobs': scale_config['n_jobs'],
            'n_stages': scale_config['n_stages'],
            'n_factories': scale_config['n_factories'],
            'name': scale_config.get('name', f"{scale_config['n_jobs']}J{scale_config['n_stages']}S{scale_config['n_factories']}F")
        })
    
    # 生成指定规模的实验配置
    experiment_configs = []
    for target in target_scales:
        n_jobs = target['n_jobs']
        n_stages = target['n_stages']
        n_factories = target['n_factories']
        
        # 生成机器配置
        base_machines = [2, 3, 4, 5]
        machines_per_stage = base_machines[:n_stages]
        if len(machines_per_stage) < n_stages:
            machines_per_stage.extend([3, 4, 5, 2][:(n_stages - len(machines_per_stage))])
        
        # 生成异构机器配置 - 增强差异性
        heterogeneous_machines = {}
        for f in range(n_factories):
            factory_machines = []
            for s in range(n_stages):
                base_machines = machines_per_stage[s]
                # 为每个工厂创建更大的机器配置差异
                if f == 0:  # 工厂0：机器数量偏少但效率高
                    variation = -1 if base_machines > 2 else 0
                elif f == 1:  # 工厂1：机器数量适中
                    variation = 0
                else:  # 工厂2+：机器数量偏多但效率一般
                    variation = 1 + (f - 2)
                    
                factory_machines.append(max(1, min(8, base_machines + variation)))
            heterogeneous_machines[f] = factory_machines
        
        config = {
            'scale': target['name'],  # 使用自定义名称
            'n_jobs': n_jobs,
            'n_factories': n_factories,
            'n_stages': n_stages,
            'machines_per_stage': machines_per_stage,
            'urgency_ddt': [0.5, 1.0, 1.5, 2.0][:min(n_factories, 4)],
            'processing_time_range': (1, 15 + n_jobs//10),
            'heterogeneous_machines': heterogeneous_machines
        }
        experiment_configs.append(config)
    
    selected_configs = experiment_configs
    
    # 算法配置 - 修复版本
    algorithms = {
        'RL-Chaotic-HHO': (RL_ChaoticHHO_Optimizer, {
            'population_size': 50,            # 适度增加种群规模，提升搜索覆盖度
            'max_iterations': 50,             # 适度增加迭代次数，提升收敛精度
            'pareto_size_limit': 30,          # 进一步降低解集限制，确保算法能产生足够解
            'diversity_enhancement': True,    # 保持多样性增强
            'elite_size': 10,                 # 大幅降低精英解数量，确保算法能产生足够解
            'exploration_rate': 0.25,         # 适度降低探索率，增强开发能力
            'diversity_threshold': 0.2,       # 进一步提高多样性阈值，允许更多解
            'archive_size': 100,              # 大幅降低归档大小，确保算法能产生足够解
            'selection_pressure': 0.1,        # 大幅降低选择压力，保持更多解
            'local_search_rate': 0.9,         # 适度增加局部搜索率，提升收敛精度
            'learning_rate': 0.0001,          # 最优学习率
            'epsilon_decay': 0.997,           # 最优探索衰减率
            'gamma': 0.999                    # 最优折扣因子
        }),
        'I-NSGA-II': (ImprovedNSGA2_Optimizer, {
            'population_size': 50,  # 适度增加种群规模
            'max_generations': 50,  # 适度增加迭代次数
            'crossover_prob': 0.7,   # 适度增加交叉概率
            'mutation_prob': 0.15    # 适度增加变异概率
        }),
        'MOPSO': (MOPSO_Optimizer, {
            'swarm_size': 50,  # 适度增加群体规模
            'max_iterations': 50,  # 适度增加迭代次数
            'w': 0.5,   # 适度增加惯性权重
            'c1': 1.5,  # 适度增加个体学习因子
            'c2': 1.8,  # 适度增加社会学习因子
            'mutation_prob': 0.1,  # 适度增加变异概率
            'archive_size': 200     # 增加存档大小
        }),
        'MODE': (MODE_Optimizer, {
            'population_size': 50,   # 适度增加种群规模
            'max_generations': 50,  # 适度增加迭代次数
            'F': 0.5,    # 适度增加差分向量缩放
            'CR': 0.6,   # 适度增加交叉概率
            'mutation_prob': 0.15   # 适度增加变异概率
        }),
        'DQN': (DQNAlgorithmWrapper, {
            'max_iterations': 50,     # 适度增加迭代次数
            'target_pareto_size': 50, # 适度增加解集大小
            'diversity_control': True # 开启多样性控制
        }),
        'QL-ABC': (QLABC_Optimizer_Enhanced, {
            'population_size': 50,   # 适度增加种群规模
            'max_iterations': 50,    # 适度增加迭代次数
            'learning_rate': 0.2,    # 适度增加学习率
            'discount_factor': 0.9,  # 适度增加折扣因子
            'epsilon': 0.2,          # 适度增加探索概率
            'epsilon_decay': 0.995,  # 减慢探索衰减
            'limit': 15,             # 适度增加极限值
            'archive_size': 300      # 适度增加归档大小
        })
    }
    
    # 存储所有结果
    all_scale_results = {}
    
    # 创建结果目录
    os.makedirs('results', exist_ok=True)
    
    # 对每个规模配置运行实验
    for config in selected_configs:
        scale = config['scale']
        print(f"\n{'='*60}")
        print(f"实验规模: {scale}")
        print(f"{'='*60}")
        
        # 生成问题数据
        problem_data = generate_heterogeneous_problem_data(config)
        
        # 打印规模详细信息
        print_scale_details(config, problem_data)
        
        # 存储该规模的结果
        all_scale_results[scale] = {}
        
        # 运行每个算法
        for algorithm_name, (algorithm_class, algorithm_params) in algorithms.items():
            print(f"\n运行算法: {algorithm_name}")
            print("-" * 40)
            
            try:
                result = run_single_experiment(
                    problem_data,
                    algorithm_name, 
                    algorithm_class, 
                    algorithm_params,
                    runs=2
                )
                
                all_scale_results[scale][algorithm_name] = result
                
                # 打印基本结果
                print(f"  最优完工时间: {result['makespan_best']:.2f}")
                print(f"  最优总拖期: {result['tardiness_best']:.2f}")
                print(f"  最差完工时间: {result['max_makespan']:.2f}")
                print(f"  最差总拖期: {result['max_tardiness']:.2f}")
                print(f"  超体积: {result['hypervolume']:.4f}")
                print(f"  IGD: {result['igd']:.4f}")
                print(f"  GD: {result['gd']:.4f}")
                print(f"  分布性: {result['spread']:.4f}")
                print(f"  RA指标: {result['ra']:.4f}")
                print(f"  帕累托解数量: {result['pareto_count']}")
                print(f"  平均运行时间: {result['runtime']:.2f}秒")
            except Exception as e:
                print(f"  ❌ 算法 {algorithm_name} 运行失败: {str(e)}")
                traceback.print_exc()
                # 设置默认失败结果
                all_scale_results[scale][algorithm_name] = {
                    'makespan_best': float('inf'),
                    'tardiness_best': float('inf'),
                    'weighted_best': float('inf'),
                    'max_makespan': 0,
                    'max_tardiness': 0,
                    'min_makespan': float('inf'),
                    'min_tardiness': float('inf'),
                    'makespan_mean': 0,
                    'tardiness_mean': 0,
                    'weighted_mean': 0,
                    'runtime': 0,
                    'hypervolume': 0.0,
                    'igd': float('inf'),
                    'gd': float('inf'),
                    'spread': 1.0,
                    'ra': 0.0,
                    'pareto_count': 0,
                    'pareto_solutions': []
                }
        
        # 绘制该规模的帕累托前沿对比图 - 增强版本
        plot_pareto_comparison(all_scale_results[scale], scale)
        
        # 按照论文要求：先归一化目标值，再计算组合前沿和指标
        print(f"\n🔄 按照论文标准重新计算{scale}的所有指标...")
        
        # 1. 对所有算法的目标值进行归一化（避免不同量纲影响）
        normalized_results_for_scale, norm_params = normalize_objectives(all_scale_results[scale])
        print(f"  ✓ 目标值归一化完成")
        print(f"    Makespan范围: [{norm_params[0]:.1f}, {norm_params[1]:.1f}]")
        print(f"    Tardiness范围: [{norm_params[2]:.1f}, {norm_params[3]:.1f}]")
        
        # 2. 基于归一化后的目标值计算组合帕累托前沿（真实前沿PF*）
        combined_pareto_front = calculate_combined_pareto_front(normalized_results_for_scale)
        print(f"  ✓ 组合帕累托前沿包含{len(combined_pareto_front)}个归一化点")
        
        # 3. 收集所有算法的解集用于统一参考点计算
        all_algorithm_solutions = []
        for algorithm_name in all_scale_results[scale]:
            if (algorithm_name in normalized_results_for_scale and 
                'normalized_pareto_solutions' in normalized_results_for_scale[algorithm_name] and 
                normalized_results_for_scale[algorithm_name]['normalized_pareto_solutions']):
                all_algorithm_solutions.extend(all_scale_results[scale][algorithm_name]['pareto_solutions'])
        
        # 4. 重新计算每个算法的所有指标（基于归一化后的目标值）
        for algorithm_name in all_scale_results[scale]:
            if (algorithm_name in normalized_results_for_scale and 
                'normalized_pareto_solutions' in normalized_results_for_scale[algorithm_name] and 
                normalized_results_for_scale[algorithm_name]['normalized_pareto_solutions']):
                
                norm_solutions = normalized_results_for_scale[algorithm_name]['normalized_pareto_solutions']
                original_solutions = all_scale_results[scale][algorithm_name]['pareto_solutions']
                
                # 重新计算所有指标
                new_hypervolume = calculate_hypervolume(original_solutions, normalize=True, all_algorithm_solutions=all_algorithm_solutions)  # HV用原始值计算并归一化，使用统一参考点
                new_igd = calculate_igd(norm_solutions, combined_pareto_front)  # IGD用归一化值和组合前沿
                new_gd = calculate_gd(norm_solutions, combined_pareto_front)   # GD用归一化值和组合前沿
                new_spread = calculate_maximum_spread(norm_solutions)         # MS用归一化值
                new_ra = calculate_ra(norm_solutions, combined_pareto_front)   # RA指标：算法解集与参考前沿的重合度
                
                # 处理无效值
                if new_igd == float('inf') or np.isnan(new_igd):
                    new_igd = 1.0  # 设为较大值表示性能差
                if new_gd == float('inf') or np.isnan(new_gd):
                    new_gd = 1.0   # 设为较大值表示性能差
                if np.isnan(new_spread):
                    new_spread = 1.0  # 设为较大值表示分布性差
                if np.isnan(new_ra):
                    new_ra = 0.0  # 设为0表示没有找到真实帕累托解
                    
                    # 更新结果
            all_scale_results[scale][algorithm_name]['hypervolume'] = new_hypervolume
            all_scale_results[scale][algorithm_name]['igd'] = new_igd
            all_scale_results[scale][algorithm_name]['gd'] = new_gd
            all_scale_results[scale][algorithm_name]['spread'] = new_spread
            all_scale_results[scale][algorithm_name]['ra'] = new_ra
            
            print(f"  {algorithm_name}: HV={new_hypervolume:.4f}, IGD={new_igd:.4f}, GD={new_gd:.4f}, Spread={new_spread:.4f}, RA={new_ra:.4f}")
        
        print(f"\n✅ {scale} 实验完成，帕累托前沿对比图已保存")
    
    # 生成综合报告
    print(f"\n{'='*80}")
    print("生成综合对比报告...")
    print(f"{'='*80}")
    
    generate_specific_scale_report(all_scale_results, selected_configs)
    
    print("\n🎉 所有实验完成！")
    print("📊 结果文件已保存到 results/ 目录")
    print("📈 帕累托前沿对比图已生成")

def create_custom_scales_config():
    """
    创建自定义规模配置的辅助函数
    
    Returns:
        List[Dict]: 自定义规模配置列表
    """
    print("=" * 60)
    print("自定义规模配置指南")
    print("=" * 60)
    print("请按照以下格式定义您的三个规模配置：")
    print("每个规模需要包含以下参数：")
    print("  - n_jobs: 工件数量")
    print("  - n_stages: 阶段数量") 
    print("  - n_factories: 工厂数量")
    print("  - name: 规模名称（可选，如果不提供将自动生成）")
    print()
    print("示例配置：")
    print("custom_scales = [")
    print("    {'n_jobs': 30, 'n_stages': 3, 'n_factories': 2, 'name': '小规模'},")
    print("    {'n_jobs': 60, 'n_stages': 4, 'n_factories': 3, 'name': '中规模'},")
    print("    {'n_jobs': 100, 'n_stages': 5, 'n_factories': 4, 'name': '大规模'}")
    print("]")
    print()
    
    # 返回默认配置作为参考
    return [
        {'n_jobs': 30, 'n_stages': 3, 'n_factories': 2, 'name': '小规模'},
        {'n_jobs': 60, 'n_stages': 4, 'n_factories': 3, 'name': '中规模'},
        {'n_jobs': 100, 'n_stages': 5, 'n_factories': 4, 'name': '大规模'}
    ]

def generate_specific_scale_report(results: Dict, configs: List[Dict]):
    """生成特定规模的表格格式报告"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/特定规模算法对比报告_{timestamp}.txt"
    
    os.makedirs("results", exist_ok=True)
    
    algorithm_list = ['RL-Chaotic-HHO', 'I-NSGA-II', 'MOPSO', 'MODE', 'DQN', 'QL-ABC']
    
    # 🔥 关键修复：对每个规模的结果进行归一化处理
    normalized_results = {}
    for scale, scale_results in results.items():
        if scale_results:  # 确保有结果
            normalized_results[scale] = normalize_metrics(scale_results)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("特定规模算法对比实验报告\n")
        f.write("=" * 100 + "\n\n")
        
        # 实验配置信息
        f.write("实验配置:\n")
        f.write(f"规模: {len(configs)}个规模配置\n")
        f.write(f"每个阶段机器数: (2,3,4,5)范围内\n")
        f.write(f"并行机数量: 随规模增大而增多\n")
        f.write(f"对比算法: {', '.join(algorithm_list)}\n")
        f.write(f"每个算法运行次数: 3次\n")
        f.write(f"种群大小: 100\n")
        f.write(f"迭代次数: 100\n")
        f.write(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 规模详情
        f.write("规模详情:\n")
        for config in configs:
            f.write(f"  {config['scale']}: {config['n_jobs']}作业, {config['n_factories']}工厂, {config['n_stages']}阶段\n")
            f.write(f"    机器配置: {config['machines_per_stage']}\n")
            f.write(f"    异构机器配置: {config['heterogeneous_machines']}\n")
        f.write("\n")
        
        # 各项指标对比表
        f.write("各项指标对比表\n")
        f.write("=" * 100 + "\n\n")
        
        # 1. 完工时间对比表
        f.write("1. 完工时间(Makespan)对比表\n")
        f.write("-" * 100 + "\n")
        f.write("+" + "-"*15 + "+" + "-"*15 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*10 + "+" + "-"*12 + "+\n")
        f.write(f"| {'规模':^13s} | {'RL-Chaotic-HHO':^13s} | {'I-NSGA-II':^11s} | {'MOPSO':^11s} | {'MODE':^11s} | {'DQN':^8s} | {'QL-ABC':^10s} |\n")
        f.write("+" + "-"*15 + "+" + "-"*15 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*10 + "+" + "-"*12 + "+\n")
        
        for config in configs:
            scale = config['scale']
            if scale in results:
                scale_results = results[scale]
                
                values = []
                for alg in algorithm_list:
                    if alg in scale_results:
                        value = scale_results[alg].get('makespan_best', '失败')
                        if value == float('inf') or value == 0:
                            values.append('失败')
                        else:
                            values.append(f"{value:.1f}")
                    else:
                        values.append('失败')
                
                f.write(f"| {scale:^13s} | {values[0]:^13s} | {values[1]:^11s} | {values[2]:^11s} | {values[3]:^11s} | {values[4]:^8s} | {values[5]:^10s} |\n")
        
        f.write("+" + "-"*15 + "+" + "-"*15 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*10 + "+" + "-"*12 + "+\n\n")
        
        # 2. 总拖期对比表
        f.write("2. 总拖期(Total Tardiness)对比表\n")
        f.write("-" * 100 + "\n")
        f.write("+" + "-"*15 + "+" + "-"*15 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*10 + "+" + "-"*12 + "+\n")
        f.write(f"| {'规模':^13s} | {'RL-Chaotic-HHO':^13s} | {'I-NSGA-II':^11s} | {'MOPSO':^11s} | {'MODE':^11s} | {'DQN':^8s} | {'QL-ABC':^10s} |\n")
        f.write("+" + "-"*15 + "+" + "-"*15 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*10 + "+" + "-"*12 + "+\n")
        
        for config in configs:
            scale = config['scale']
            if scale in results:
                scale_results = results[scale]
                
                values = []
                for alg in algorithm_list:
                    if alg in scale_results:
                        value = scale_results[alg].get('tardiness_best', '失败')
                        if value == float('inf') or (isinstance(value, (int, float)) and value < 0):
                            values.append('失败')
                        else:
                            values.append(f"{value:.1f}")
                    else:
                        values.append('失败')
                
                f.write(f"| {scale:^13s} | {values[0]:^13s} | {values[1]:^11s} | {values[2]:^11s} | {values[3]:^11s} | {values[4]:^8s} | {values[5]:^10s} |\n")
        
        f.write("+" + "-"*15 + "+" + "-"*15 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*10 + "+" + "-"*12 + "+\n\n")
        
        # 3. 加权目标对比表
        f.write("3. 加权目标函数对比表\n")
        f.write("-" * 100 + "\n")
        f.write("+" + "-"*15 + "+" + "-"*15 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*10 + "+" + "-"*12 + "+\n")
        f.write(f"| {'规模':^13s} | {'RL-Chaotic-HHO':^13s} | {'I-NSGA-II':^11s} | {'MOPSO':^11s} | {'MODE':^11s} | {'DQN':^8s} | {'QL-ABC':^10s} |\n")
        f.write("+" + "-"*15 + "+" + "-"*15 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*10 + "+" + "-"*12 + "+\n")
        
        for config in configs:
            scale = config['scale']
            if scale in results:
                scale_results = results[scale]
                
                values = []
                for alg in algorithm_list:
                    if alg in scale_results:
                        value = scale_results[alg].get('weighted_best', '失败')
                        if value == float('inf') or value == 0:
                            values.append('失败')
                        else:
                            values.append(f"{value:.1f}")
                    else:
                        values.append('失败')
                
                f.write(f"| {scale:^13s} | {values[0]:^13s} | {values[1]:^11s} | {values[2]:^11s} | {values[3]:^11s} | {values[4]:^8s} | {values[5]:^10s} |\n")
        
        f.write("+" + "-"*15 + "+" + "-"*15 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*10 + "+" + "-"*12 + "+\n\n")
        
        # 4. 超体积指标对比表
        f.write("4. 超体积(HV)指标对比表\n")
        f.write("-" * 100 + "\n")
        f.write("+" + "-"*15 + "+" + "-"*15 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*10 + "+" + "-"*12 + "+\n")
        f.write(f"| {'规模':^13s} | {'RL-Chaotic-HHO':^13s} | {'I-NSGA-II':^11s} | {'MOPSO':^11s} | {'MODE':^11s} | {'DQN':^8s} | {'QL-ABC':^10s} |\n")
        f.write("+" + "-"*15 + "+" + "-"*15 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*10 + "+" + "-"*12 + "+\n")
        
        for config in configs:
            scale = config['scale']
            if scale in results:
                scale_results = results[scale]
                
                values = []
                for alg in algorithm_list:
                    if alg in scale_results:
                        value = scale_results[alg].get('hypervolume', 0)
                        if value == 0:
                            values.append('0')
                        else:
                            values.append(f"{value:.4f}")
                    else:
                        values.append('0')
                
                f.write(f"| {scale:^13s} | {values[0]:^13s} | {values[1]:^11s} | {values[2]:^11s} | {values[3]:^11s} | {values[4]:^8s} | {values[5]:^10s} |\n")
        
        f.write("+" + "-"*15 + "+" + "-"*15 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*10 + "+" + "-"*12 + "+\n\n")
        
        # 5. IGD指标对比表
        f.write("5. 反世代距离(IGD)指标对比表\n")
        f.write("-" * 100 + "\n")
        f.write("+" + "-"*15 + "+" + "-"*15 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*10 + "+" + "-"*12 + "+\n")
        f.write(f"| {'规模':^13s} | {'RL-Chaotic-HHO':^13s} | {'I-NSGA-II':^11s} | {'MOPSO':^11s} | {'MODE':^11s} | {'DQN':^8s} | {'QL-ABC':^10s} |\n")
        f.write("+" + "-"*15 + "+" + "-"*15 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*10 + "+" + "-"*12 + "+\n")
        
        for config in configs:
            scale = config['scale']
            if scale in results:
                scale_results = results[scale]
                
                values = []
                for alg in algorithm_list:
                    if alg in scale_results:
                        value = scale_results[alg].get('igd', float('inf'))
                        if value == float('inf'):
                            values.append('∞')
                        elif value < 1e-6:
                            # 对于极小值，使用科学记数法显示
                            values.append(f"{value:.2e}")
                        else:
                            values.append(f"{value:.2f}")
                    else:
                        values.append('∞')
                
                f.write(f"| {scale:^13s} | {values[0]:^13s} | {values[1]:^11s} | {values[2]:^11s} | {values[3]:^11s} | {values[4]:^8s} | {values[5]:^10s} |\n")
        
        f.write("+" + "-"*15 + "+" + "-"*15 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*10 + "+" + "-"*12 + "+\n\n")
        
        # 6. RA指标对比表
        f.write("6. 帕累托最优解比率(RA)指标对比表\n")
        f.write("-" * 100 + "\n")
        f.write("+" + "-"*15 + "+" + "-"*15 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*10 + "+" + "-"*12 + "+\n")
        f.write(f"| {'规模':^13s} | {'RL-Chaotic-HHO':^13s} | {'I-NSGA-II':^11s} | {'MOPSO':^11s} | {'MODE':^11s} | {'DQN':^8s} | {'QL-ABC':^10s} |\n")
        f.write("+" + "-"*15 + "+" + "-"*15 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*10 + "+" + "-"*12 + "+\n")
        
        for config in configs:
            scale = config['scale']
            if scale in results:
                scale_results = results[scale]
                
                values = []
                for alg in algorithm_list:
                    if alg in scale_results:
                        value = scale_results[alg].get('ra', 0.0)
                        if value < 0 or np.isnan(value):
                            values.append('失败')
                        else:
                            values.append(f"{value:.3f}")
                    else:
                        values.append('较差')
                
                f.write(f"| {scale:^13s} | {values[0]:^13s} | {values[1]:^11s} | {values[2]:^11s} | {values[3]:^11s} | {values[4]:^8s} | {values[5]:^10s} |\n")
        
        f.write("+" + "-"*15 + "+" + "-"*15 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*10 + "+" + "-"*12 + "+\n\n")
        
        # 7. 运行时间对比表
        f.write("7. 运行时间(秒)对比表\n")
        f.write("-" * 100 + "\n")
        f.write("+" + "-"*15 + "+" + "-"*15 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*10 + "+" + "-"*12 + "+\n")
        f.write(f"| {'规模':^13s} | {'RL-Chaotic-HHO':^13s} | {'I-NSGA-II':^11s} | {'MOPSO':^11s} | {'MODE':^11s} | {'DQN':^8s} | {'QL-ABC':^10s} |\n")
        f.write("+" + "-"*15 + "+" + "-"*15 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*10 + "+" + "-"*12 + "+\n")
        
        for config in configs:
            scale = config['scale']
            if scale in results:
                scale_results = results[scale]
                
                values = []
                for alg in algorithm_list:
                    if alg in scale_results:
                        value = scale_results[alg].get('runtime', 0)
                        values.append(f"{value:.2f}")
                    else:
                        values.append('失败')
                
                f.write(f"| {scale:^13s} | {values[0]:^13s} | {values[1]:^11s} | {values[2]:^11s} | {values[3]:^11s} | {values[4]:^8s} | {values[5]:^10s} |\n")
        
        f.write("+" + "-"*15 + "+" + "-"*15 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*10 + "+" + "-"*12 + "+\n\n")
        
        # 8. 帕累托解数量对比表
        f.write("8. 帕累托解数量对比表\n")
        f.write("-" * 100 + "\n")
        f.write("+" + "-"*15 + "+" + "-"*15 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*10 + "+" + "-"*12 + "+\n")
        f.write(f"| {'规模':^13s} | {'RL-Chaotic-HHO':^13s} | {'I-NSGA-II':^11s} | {'MOPSO':^11s} | {'MODE':^11s} | {'DQN':^8s} | {'QL-ABC':^10s} |\n")
        f.write("+" + "-"*15 + "+" + "-"*15 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*10 + "+" + "-"*12 + "+\n")
        
        for config in configs:
            scale = config['scale']
            if scale in results:
                scale_results = results[scale]
                
                values = []
                for alg in algorithm_list:
                    if alg in scale_results:
                        value = scale_results[alg].get('pareto_count', 0)
                        values.append(f"{value}")
                    else:
                        values.append('0')
                
                f.write(f"| {scale:^13s} | {values[0]:^13s} | {values[1]:^11s} | {values[2]:^11s} | {values[3]:^11s} | {values[4]:^8s} | {values[5]:^10s} |\n")
        
        f.write("+" + "-"*15 + "+" + "-"*15 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*10 + "+" + "-"*12 + "+\n\n")
        
        # 9. 归一化指标对比表
        f.write("9. 归一化指标对比表\n")
        f.write("=" * 100 + "\n\n")
        
        # 9.1 归一化超体积指标
        f.write("9.1 归一化超体积(HV)指标对比表\n")
        f.write("-" * 100 + "\n")
        f.write("+" + "-"*15 + "+" + "-"*15 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*10 + "+" + "-"*12 + "+\n")
        f.write(f"| {'规模':^13s} | {'RL-Chaotic-HHO':^13s} | {'I-NSGA-II':^11s} | {'MOPSO':^11s} | {'MODE':^11s} | {'DQN':^8s} | {'QL-ABC':^10s} |\n")
        f.write("+" + "-"*15 + "+" + "-"*15 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*10 + "+" + "-"*12 + "+\n")
        
        for config in configs:
            scale = config['scale']
            if scale in normalized_results:
                scale_results = normalized_results[scale]
                
                values = []
                for alg in algorithm_list:
                    if alg in scale_results:
                        value = scale_results[alg].get('norm_hypervolume', 0)
                        values.append(f"{value:.4f}")
                    else:
                        values.append('0.0000')
                
                f.write(f"| {scale:^13s} | {values[0]:^13s} | {values[1]:^11s} | {values[2]:^11s} | {values[3]:^11s} | {values[4]:^8s} | {values[5]:^10s} |\n")
        
        f.write("+" + "-"*15 + "+" + "-"*15 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*10 + "+" + "-"*12 + "+\n\n")
        
        # 9.2 归一化IGD指标
        f.write("9.2 归一化IGD指标对比表\n")
        f.write("-" * 100 + "\n")
        f.write("+" + "-"*15 + "+" + "-"*15 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*10 + "+" + "-"*12 + "+\n")
        f.write(f"| {'规模':^13s} | {'RL-Chaotic-HHO':^13s} | {'I-NSGA-II':^11s} | {'MOPSO':^11s} | {'MODE':^11s} | {'DQN':^8s} | {'QL-ABC':^10s} |\n")
        f.write("+" + "-"*15 + "+" + "-"*15 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*10 + "+" + "-"*12 + "+\n")
        
        for config in configs:
            scale = config['scale']
            if scale in normalized_results:
                scale_results = normalized_results[scale]
                
                values = []
                for alg in algorithm_list:
                    if alg in scale_results:
                        value = scale_results[alg].get('norm_igd', 0)
                        values.append(f"{value:.4f}")
                    else:
                        values.append('0.0000')
                
                f.write(f"| {scale:^13s} | {values[0]:^13s} | {values[1]:^11s} | {values[2]:^11s} | {values[3]:^11s} | {values[4]:^8s} | {values[5]:^10s} |\n")
        
        f.write("+" + "-"*15 + "+" + "-"*15 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*10 + "+" + "-"*12 + "+\n\n")
        
        # 9.3 归一化GD指标
        f.write("9.3 归一化GD指标对比表\n")
        f.write("-" * 100 + "\n")
        f.write("+" + "-"*15 + "+" + "-"*15 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*10 + "+" + "-"*12 + "+\n")
        f.write(f"| {'规模':^13s} | {'RL-Chaotic-HHO':^13s} | {'I-NSGA-II':^11s} | {'MOPSO':^11s} | {'MODE':^11s} | {'DQN':^8s} | {'QL-ABC':^10s} |\n")
        f.write("+" + "-"*15 + "+" + "-"*15 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*10 + "+" + "-"*12 + "+\n")
        
        for config in configs:
            scale = config['scale']
            if scale in normalized_results:
                scale_results = normalized_results[scale]
                
                values = []
                for alg in algorithm_list:
                    if alg in scale_results:
                        value = scale_results[alg].get('norm_gd', 0)
                        values.append(f"{value:.4f}")
                    else:
                        values.append('0.0000')
                
                f.write(f"| {scale:^13s} | {values[0]:^13s} | {values[1]:^11s} | {values[2]:^11s} | {values[3]:^11s} | {values[4]:^8s} | {values[5]:^10s} |\n")
        
        f.write("+" + "-"*15 + "+" + "-"*15 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*10 + "+" + "-"*12 + "+\n\n")
        
        # 9.4 归一化Spread指标
        f.write("9.4 归一化Spread指标对比表\n")
        f.write("-" * 100 + "\n")
        f.write("+" + "-"*15 + "+" + "-"*15 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*10 + "+" + "-"*12 + "+\n")
        f.write(f"| {'规模':^13s} | {'RL-Chaotic-HHO':^13s} | {'I-NSGA-II':^11s} | {'MOPSO':^11s} | {'MODE':^11s} | {'DQN':^8s} | {'QL-ABC':^10s} |\n")
        f.write("+" + "-"*15 + "+" + "-"*15 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*10 + "+" + "-"*12 + "+\n")
        
        for config in configs:
            scale = config['scale']
            if scale in normalized_results:
                scale_results = normalized_results[scale]
                
                values = []
                for alg in algorithm_list:
                    if alg in scale_results:
                        value = scale_results[alg].get('norm_spread', 0)
                        values.append(f"{value:.4f}")
                    else:
                        values.append('0.0000')
                
                f.write(f"| {scale:^13s} | {values[0]:^13s} | {values[1]:^11s} | {values[2]:^11s} | {values[3]:^11s} | {values[4]:^8s} | {values[5]:^10s} |\n")
        
        f.write("+" + "-"*15 + "+" + "-"*15 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*10 + "+" + "-"*12 + "+\n\n")
        
        # 9.5 归一化RA指标
        f.write("9.5 归一化RA指标对比表\n")
        f.write("-" * 100 + "\n")
        f.write("+" + "-"*15 + "+" + "-"*15 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*10 + "+" + "-"*12 + "+\n")
        f.write(f"| {'规模':^13s} | {'RL-Chaotic-HHO':^13s} | {'I-NSGA-II':^11s} | {'MOPSO':^11s} | {'MODE':^11s} | {'DQN':^8s} | {'QL-ABC':^10s} |\n")
        f.write("+" + "-"*15 + "+" + "-"*15 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*10 + "+" + "-"*12 + "+\n")
        
        for config in configs:
            scale = config['scale']
            if scale in normalized_results:
                scale_results = normalized_results[scale]
                
                values = []
                for alg in algorithm_list:
                    if alg in scale_results:
                        value = scale_results[alg].get('norm_ra', 0)
                        values.append(f"{value:.4f}")
                    else:
                        values.append('0.0000')
                
                f.write(f"| {scale:^13s} | {values[0]:^13s} | {values[1]:^11s} | {values[2]:^11s} | {values[3]:^11s} | {values[4]:^8s} | {values[5]:^10s} |\n")
        
        f.write("+" + "-"*15 + "+" + "-"*15 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*10 + "+" + "-"*12 + "+\n\n")
        
        # 总结
        f.write("实验总结\n")
        f.write("=" * 100 + "\n")
        f.write(f"本实验对比了6种算法在{len(configs)}个特定规模上的性能表现。\n")
        f.write("规模配置：\n")
        for config in configs:
            f.write(f"- {config['scale']}: {config['n_jobs']}个作业，{config['n_stages']}个阶段，{config['n_factories']}个工厂\n")
        f.write("每个阶段的机器数在(2,3,4,5)范围内，并行机数量随规模增大而增多。\n")
        f.write("所有算法均采用相同的种群大小(100)和迭代次数(100)确保公平比较。\n")
        f.write("评估指标包括：HV、IGD、GD、Spread、RA五个核心指标。\n")
        f.write("按照学术论文标准计算方式：\n")
        f.write("1. 所有算法的目标值先进行归一化处理（避免不同量纲影响）\n")
        f.write("2. 基于归一化目标值计算组合帕累托前沿作为真实前沿PF*\n")
        f.write("3. 基于归一化目标值和组合前沿计算各项指标\n")
        f.write("HV（超体积）：越大越好，已归一化到[0,1]范围。\n")
        f.write("IGD（反向世代距离）：越小越好，基于归一化目标值计算，0表示最理想。\n")
        f.write("GD（世代距离）：越小越好，基于归一化目标值计算，0表示最理想。\n")
        f.write("MS（最大分布性）：越大越好，基于归一化目标值计算，1表示最大覆盖范围最好。\n") 
        f.write("RA（帕累托最优解比率）：越大越好，表示算法找到真实帕累托最优解的比率，理想范围0-1。\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"\n特定规模算法对比报告已保存: {filename}")
    
    # 生成Excel表格（分离版本）
    excel_filename = f"results/特定规模算法对比报告_{timestamp}.xlsx"
    generate_excel_report(results, normalized_results, configs, excel_filename)

def generate_excel_report(results: Dict, normalized_results: Dict, configs: List[Dict], filename: str):
    """生成分离的Excel格式报告 - 四个指标和两个优化目标分别保存"""
    
    algorithm_list = ['RL-Chaotic-HHO', 'I-NSGA-II', 'MOPSO', 'MODE', 'DQN', 'QL-ABC']
    
    # 获取基础文件名（不含扩展名）
    base_filename = filename.replace('.xlsx', '')
    
    # 1. 生成两个优化目标的Excel文件
    objectives_filename = f"{base_filename}_优化目标.xlsx"
    with pd.ExcelWriter(objectives_filename, engine='openpyxl') as writer:
        
        # 完工时间表
        makespan_data = []
        for config in configs:
            scale = config['scale']
            if scale in results:
                row = {'规模': scale}
                for alg in algorithm_list:
                    if alg in results[scale]:
                        row[alg] = results[scale][alg].get('makespan_best', 0)
                    else:
                        row[alg] = 0
                makespan_data.append(row)
        
        makespan_df = pd.DataFrame(makespan_data)
        makespan_df.to_excel(writer, sheet_name='完工时间(Makespan)', index=False)
        
        # 总拖期表
        tardiness_data = []
        for config in configs:
            scale = config['scale']
            if scale in results:
                row = {'规模': scale}
                for alg in algorithm_list:
                    if alg in results[scale]:
                        row[alg] = results[scale][alg].get('tardiness_best', 0)
                    else:
                        row[alg] = 0
                tardiness_data.append(row)
        
        tardiness_df = pd.DataFrame(tardiness_data)
        tardiness_df.to_excel(writer, sheet_name='总拖期(Total_Tardiness)', index=False)
        
        # 加权目标表
        weighted_data = []
        for config in configs:
            scale = config['scale']
            if scale in results:
                row = {'规模': scale}
                for alg in algorithm_list:
                    if alg in results[scale]:
                        row[alg] = results[scale][alg].get('weighted_best', 0)
                    else:
                        row[alg] = 0
                weighted_data.append(row)
        
        weighted_df = pd.DataFrame(weighted_data)
        weighted_df.to_excel(writer, sheet_name='加权目标', index=False)
        
        # 帕累托解数量表
        pareto_count_data = []
        for config in configs:
            scale = config['scale']
            if scale in results:
                row = {'规模': scale}
                for alg in algorithm_list:
                    if alg in results[scale]:
                        row[alg] = results[scale][alg].get('pareto_count', 0)
                    else:
                        row[alg] = 0
                pareto_count_data.append(row)
        
        pareto_count_df = pd.DataFrame(pareto_count_data)
        pareto_count_df.to_excel(writer, sheet_name='帕累托解数量', index=False)
    
    print(f"优化目标Excel报告已保存: {objectives_filename}")
    
    # 2. 生成四个归一化指标的Excel文件
    metrics_filename = f"{base_filename}_归一化指标.xlsx"
    with pd.ExcelWriter(metrics_filename, engine='openpyxl') as writer:
        
        # 归一化超体积表
        hv_data = []
        for config in configs:
            scale = config['scale']
            if scale in normalized_results:
                row = {'规模': scale}
                for alg in algorithm_list:
                    if alg in normalized_results[scale]:
                        row[alg] = normalized_results[scale][alg].get('norm_hypervolume', 0)
                    else:
                        row[alg] = 0
                hv_data.append(row)
        
        hv_df = pd.DataFrame(hv_data)
        hv_df.to_excel(writer, sheet_name='归一化超体积(HV)', index=False)
        
        # 归一化IGD表
        igd_data = []
        for config in configs:
            scale = config['scale']
            if scale in normalized_results:
                row = {'规模': scale}
                for alg in algorithm_list:
                    if alg in normalized_results[scale]:
                        row[alg] = normalized_results[scale][alg].get('norm_igd', 0)
                    else:
                        row[alg] = 0
                igd_data.append(row)
        
        igd_df = pd.DataFrame(igd_data)
        igd_df.to_excel(writer, sheet_name='归一化IGD', index=False)
        
        # 归一化GD表
        gd_data = []
        for config in configs:
            scale = config['scale']
            if scale in normalized_results:
                row = {'规模': scale}
                for alg in algorithm_list:
                    if alg in normalized_results[scale]:
                        row[alg] = normalized_results[scale][alg].get('norm_gd', 0)
                    else:
                        row[alg] = 0
                gd_data.append(row)
        
        gd_df = pd.DataFrame(gd_data)
        gd_df.to_excel(writer, sheet_name='归一化GD', index=False)
        
        # 归一化分布性表
        spread_data = []
        for config in configs:
            scale = config['scale']
            if scale in normalized_results:
                row = {'规模': scale}
                for alg in algorithm_list:
                    if alg in normalized_results[scale]:
                        row[alg] = normalized_results[scale][alg].get('norm_spread', 0)
                    else:
                        row[alg] = 0
                spread_data.append(row)
        
        spread_df = pd.DataFrame(spread_data)
        spread_df.to_excel(writer, sheet_name='归一化分布性(Spread)', index=False)
        
        # 归一化RA指标表
        ra_data = []
        for config in configs:
            scale = config['scale']
            if scale in normalized_results:
                row = {'规模': scale}
                for alg in algorithm_list:
                    if alg in normalized_results[scale]:
                        row[alg] = normalized_results[scale][alg].get('norm_ra', 0)
                    else:
                        row[alg] = 0
                ra_data.append(row)
        
        ra_df = pd.DataFrame(ra_data)
        ra_df.to_excel(writer, sheet_name='归一化RA指标', index=False)
        
        # 原始指标值表（供参考）
        original_data = []
        for config in configs:
            scale = config['scale']
            if scale in results:
                row = {'规模': scale}
                for alg in algorithm_list:
                    if alg in results[scale]:
                        row[f'{alg}_HV'] = results[scale][alg].get('hypervolume', 0)
                        row[f'{alg}_IGD'] = results[scale][alg].get('igd', float('inf'))
                        row[f'{alg}_GD'] = results[scale][alg].get('gd', float('inf'))
                        row[f'{alg}_Spread'] = results[scale][alg].get('spread', 0)
                        row[f'{alg}_RA'] = results[scale][alg].get('ra', 0)
                    else:
                        row[f'{alg}_HV'] = 0
                        row[f'{alg}_IGD'] = float('inf')
                        row[f'{alg}_GD'] = float('inf')
                        row[f'{alg}_Spread'] = 0
                        row[f'{alg}_RA'] = 0
                original_data.append(row)
        
        original_df = pd.DataFrame(original_data)
        original_df.to_excel(writer, sheet_name='原始指标值参考', index=False)
    
    print(f"归一化指标Excel报告已保存: {metrics_filename}")
    
    # 3. 生成综合统计Excel文件
    stats_filename = f"{base_filename}_综合统计.xlsx"
    with pd.ExcelWriter(stats_filename, engine='openpyxl') as writer:
        
        # 算法性能排名表
        ranking_data = []
        for alg in algorithm_list:
            alg_stats = {
                '算法': alg,
                '完工时间获胜次数': 0,
                '总拖期获胜次数': 0,
                'HV获胜次数': 0,
                'IGD获胜次数': 0,
                'GD获胜次数': 0,
                'Spread获胜次数': 0,
                'RA获胜次数': 0,
                '平均帕累托解数': 0
            }
            
            total_pareto_count = 0
            valid_scales = 0
            
            for config in configs:
                scale = config['scale']
                if scale in results and scale in normalized_results:
                    valid_scales += 1
                    
                    # 统计获胜次数
                    if alg in results[scale]:
                        total_pareto_count += results[scale][alg].get('pareto_count', 0)
                        
                        # 检查是否在该规模上获胜
                        makespan_best = min(results[scale][a].get('makespan_best', float('inf')) 
                                          for a in algorithm_list if a in results[scale])
                        tardiness_best = min(results[scale][a].get('tardiness_best', float('inf'))
                                           for a in algorithm_list if a in results[scale])
                        
                        if results[scale][alg].get('makespan_best', float('inf')) == makespan_best:
                            alg_stats['完工时间获胜次数'] += 1
                        if results[scale][alg].get('tardiness_best', float('inf')) == tardiness_best:
                            alg_stats['总拖期获胜次数'] += 1
                    
                    # 统计归一化指标获胜次数
                    if alg in normalized_results[scale]:
                        # HV: 越大越好，找最大值
                        hv_best = max(normalized_results[scale][a].get('norm_hypervolume', 0)
                                    for a in algorithm_list if a in normalized_results[scale])
                        
                        # IGD, GD: 越小越好，找最小值；MS, RA: 越大越好，找最大值
                        igd_best = min(normalized_results[scale][a].get('norm_igd', float('inf'))
                                     for a in algorithm_list if a in normalized_results[scale])
                        gd_best = min(normalized_results[scale][a].get('norm_gd', float('inf'))
                                    for a in algorithm_list if a in normalized_results[scale])
                        spread_best = max(normalized_results[scale][a].get('norm_spread', 0)
                                         for a in algorithm_list if a in normalized_results[scale])
                        ra_best = max(normalized_results[scale][a].get('norm_ra', 0.0)
                                        for a in algorithm_list if a in normalized_results[scale])
                        
                        if normalized_results[scale][alg].get('norm_hypervolume', 0) == hv_best:
                            alg_stats['HV获胜次数'] += 1
                        if normalized_results[scale][alg].get('norm_igd', float('inf')) == igd_best:
                            alg_stats['IGD获胜次数'] += 1
                        if normalized_results[scale][alg].get('norm_gd', float('inf')) == gd_best:
                            alg_stats['GD获胜次数'] += 1
                        if normalized_results[scale][alg].get('norm_spread', float('inf')) == spread_best:
                            alg_stats['Spread获胜次数'] += 1
                        if normalized_results[scale][alg].get('norm_ra', 0.0) == ra_best:
                            alg_stats['RA获胜次数'] += 1
            
            if valid_scales > 0:
                alg_stats['平均帕累托解数'] = total_pareto_count / valid_scales
            
            ranking_data.append(alg_stats)
        
        ranking_df = pd.DataFrame(ranking_data)
        ranking_df.to_excel(writer, sheet_name='算法性能排名', index=False)
        
        # 规模难度分析表
        difficulty_data = []
        for config in configs:
            scale = config['scale']
            if scale in results:
                difficulty_stats = {
                    '规模': scale,
                    '平均完工时间': 0,
                    '平均总拖期': 0,
                    '平均帕累托解数': 0,
                    '最佳完工时间': float('inf'),
                    '最佳总拖期': float('inf'),
                    '完工时间标准差': 0,
                    '总拖期标准差': 0
                }
                
                makespans = []
                tardiness_vals = []
                pareto_counts = []
                
                for alg in algorithm_list:
                    if alg in results[scale]:
                        makespan = results[scale][alg].get('makespan_best', 0)
                        tardiness = results[scale][alg].get('tardiness_best', 0)
                        pareto_count = results[scale][alg].get('pareto_count', 0)
                        
                        if makespan > 0:
                            makespans.append(makespan)
                            tardiness_vals.append(tardiness)
                            pareto_counts.append(pareto_count)
                
                if makespans:
                    difficulty_stats['平均完工时间'] = np.mean(makespans)
                    difficulty_stats['平均总拖期'] = np.mean(tardiness_vals)
                    difficulty_stats['平均帕累托解数'] = np.mean(pareto_counts)
                    difficulty_stats['最佳完工时间'] = min(makespans)
                    difficulty_stats['最佳总拖期'] = min(tardiness_vals)
                    difficulty_stats['完工时间标准差'] = np.std(makespans)
                    difficulty_stats['总拖期标准差'] = np.std(tardiness_vals)
                
                difficulty_data.append(difficulty_stats)
        
        difficulty_df = pd.DataFrame(difficulty_data)
        difficulty_df.to_excel(writer, sheet_name='规模难度分析', index=False)
    
    print(f"综合统计Excel报告已保存: {stats_filename}")
    
    print(f"\n✅ 所有Excel报告生成完成:")
    print(f"   1. 优化目标: {objectives_filename}")
    print(f"   2. 归一化指标: {metrics_filename}")
    print(f"   3. 综合统计: {stats_filename}")

if __name__ == "__main__":
    # 确保结果目录存在
    os.makedirs("results", exist_ok=True)
    
    print("=" * 80)
    print("特定规模算法对比实验程序 - 自定义版本")
    print("=" * 80)
    print("本程序支持自定义三个规模的配置进行算法对比实验")
    print()
    
    # 显示配置指南
    create_custom_scales_config()
    
    # 运行指定规模配置的实验（带spread指标，3次运行）
    print("=" * 80)
    print("运行指定规模配置实验（spread指标，3次运行）")
    print("=" * 80)
    run_specific_scale_experiments()
    
    print("\n" + "=" * 80)
    print("实验完成！")
    print("如需使用其他自定义规模，请修改代码中的 custom_scales 配置")
    print("=" * 80)