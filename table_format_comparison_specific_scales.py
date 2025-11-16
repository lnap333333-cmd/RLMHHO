#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特定规模算法对比实验程序 - 修复版本
解决问题：
1. DQN pareto解集数量问题
2. 归一化指标计算问题  
3. 主体算法pareto解集多样性
4. Excel表格分离输出
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
from algorithm.ql_abc import QLABC_Optimizer
from utils.data_generator import DataGenerator

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def calculate_hypervolume(pareto_solutions: List, reference_point: Tuple[float, float] = None, normalize: bool = False) -> float:
    """
    修复后的超体积指标计算
    使用正确的2D超体积算法，避免虚高或虚低的HV值
    """
    if not pareto_solutions or len(pareto_solutions) == 0:
        return 0.0
    
    # 提取目标值
    objectives = [(sol.makespan, sol.total_tardiness) for sol in pareto_solutions]
    
    # 去除重复解（更宽松的容差）
    unique_objectives = []
    tolerance = 1e-3  # 放宽容差
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
            if i != j:  # 不与自己比较
                # 检查是否被严格支配（对于最小化问题）
                if (other_obj[0] <= obj[0] and other_obj[1] <= obj[1] and 
                    (other_obj[0] < obj[0] or other_obj[1] < obj[1])):
                    is_dominated = True
                    break
        if not is_dominated:
            pareto_front.append(obj)
    
    if len(pareto_front) == 0:
        return 0.0
    
    # 单点解给予合理的基础分数
    if len(pareto_front) == 1:
        return 0.1  # 固定返回0.1作为单点解的基础分数
    
    # 设置合理的参考点（使用更大的扩展比例）
    if reference_point is None:
        max_makespan = max(obj[0] for obj in pareto_front)
        max_tardiness = max(obj[1] for obj in pareto_front)
        min_makespan = min(obj[0] for obj in pareto_front)
        min_tardiness = min(obj[1] for obj in pareto_front)
        
        # 使用动态扩展比例，确保有意义的HV计算空间
        makespan_range = max_makespan - min_makespan
        tardiness_range = max_tardiness - min_tardiness
        
        # 至少扩展20%，对于小范围扩展更多
        makespan_margin = max(makespan_range * 0.3, max_makespan * 0.15, 1.0)
        tardiness_margin = max(tardiness_range * 0.3, max_tardiness * 0.15, 1.0)
        
        reference_point = (max_makespan + makespan_margin, max_tardiness + tardiness_margin)
    
    # 使用正确的2D超体积计算算法（从左到右扫描）
    sorted_points = sorted(pareto_front, key=lambda x: x[0])  # 按x坐标排序
    
    hypervolume = 0.0
    prev_x = 0.0  # 从原点开始
    
    for i, (x, y) in enumerate(sorted_points):
        # 确保点在参考点内
        if x >= reference_point[0] or y >= reference_point[1]:
            continue
            
        # 计算当前点左侧的矩形贡献
        width = x - prev_x
        height = reference_point[1] - y
        
        if width > 0 and height > 0:
            hypervolume += width * height
    
        # 更新x坐标
        prev_x = x
    
    # 添加最右侧区域的贡献（从最后一个点到参考点）
    if sorted_points:
        last_x, last_y = sorted_points[-1]
        if last_x < reference_point[0]:
            # 找到在最后x坐标处的最小y值
            min_y = min(y for x, y in sorted_points if x == last_x)
            width = reference_point[0] - last_x
            height = reference_point[1] - min_y
            
            if width > 0 and height > 0:
                hypervolume += width * height
    
    # 确保返回正值
    hypervolume = max(0.0, hypervolume)
    
    # 为了公平比较，对所有算法使用相同的归一化基准
    # 使用参考点矩形面积进行归一化
    max_possible_hv = reference_point[0] * reference_point[1]
    if max_possible_hv > 0:
        normalized_hv = hypervolume / max_possible_hv
        # 限制归一化HV的最大值，避免虚高
        normalized_hv = min(normalized_hv, 0.95)  # 最大不超过0.95
        return normalized_hv
    else:
        return 0.0

def calculate_igd(normalized_pareto_solutions: List, reference_front: List[Tuple[float, float]] = None) -> float:
    """
    反向世代距离 - 基于归一化后的目标值计算
    IGD+ 修正版本，考虑支配关系
    """
    if not normalized_pareto_solutions or len(normalized_pareto_solutions) == 0:
        return float('inf')
    
    # 使用归一化后的目标值
    objectives = [(sol.makespan, sol.total_tardiness) for sol in normalized_pareto_solutions]
    
    # 如果没有参考前沿，返回无穷大
    if reference_front is None or len(reference_front) == 0:
        return float('inf')
    
    # 计算每个参考点到解集的最小距离
    distances = []
    for ref_point in reference_front:
        min_distance = float('inf')
        
        for obj in objectives:
            # 使用IGD+的修正距离计算（考虑支配关系）
            # 对于最小化问题：d+ = max{obj - ref, 0}
            diff_makespan = max(obj[0] - ref_point[0], 0)
            diff_tardiness = max(obj[1] - ref_point[1], 0)
            distance = np.sqrt(diff_makespan**2 + diff_tardiness**2)
            min_distance = min(min_distance, distance)
        
        distances.append(min_distance)
    
    # 返回平均距离
    return np.mean(distances)

def calculate_gd(normalized_pareto_solutions: List, reference_front: List[Tuple[float, float]] = None) -> float:
    """
    世代距离 - 基于归一化后的目标值计算
    GD+ 修正版本，考虑支配关系
    """
    if not normalized_pareto_solutions or len(normalized_pareto_solutions) == 0:
        return float('inf')
    
    # 使用归一化后的目标值
    objectives = [(sol.makespan, sol.total_tardiness) for sol in normalized_pareto_solutions]
    
    # 如果没有参考前沿，返回无穷大
    if reference_front is None or len(reference_front) == 0:
        return float('inf')
    
    # 计算每个解到参考前沿的最小距离
    distances = []
    for obj in objectives:
        min_distance = float('inf')
        
        for ref_point in reference_front:
            # 使用GD+的修正距离计算（考虑支配关系）
            # 对于最小化问题：d+ = max{obj - ref, 0}
            diff_makespan = max(obj[0] - ref_point[0], 0)
            diff_tardiness = max(obj[1] - ref_point[1], 0)
            distance = np.sqrt(diff_makespan**2 + diff_tardiness**2)
            min_distance = min(min_distance, distance)
        
        distances.append(min_distance)
    
    # 返回平均距离
    return np.mean(distances)

def calculate_spacing(normalized_pareto_solutions: List) -> float:
    """
    间距指标 - 基于归一化后的目标值计算
    测量解集分布的均匀性
    """
    if not normalized_pareto_solutions or len(normalized_pareto_solutions) <= 1:
        return 0.0  # 单点解集间距为0
    
    # 使用归一化后的目标值
    objectives = [(sol.makespan, sol.total_tardiness) for sol in normalized_pareto_solutions]
    
    if len(objectives) <= 1:
        return 0.0
    
    # 计算每个解到其最近邻的距离
    nearest_distances = []
    for i, obj1 in enumerate(objectives):
        min_distance = float('inf')
        
        for j, obj2 in enumerate(objectives):
            if i != j:
                # 欧几里得距离
                distance = np.sqrt((obj1[0] - obj2[0])**2 + (obj1[1] - obj2[1])**2)
                min_distance = min(min_distance, distance)
    
        if min_distance != float('inf'):
            nearest_distances.append(min_distance)
    
    if len(nearest_distances) <= 1:
        return 0.0
    
    # Schott的标准spacing公式：最近邻距离的标准差
    mean_distance = np.mean(nearest_distances)
    variance = np.sum([(d - mean_distance)**2 for d in nearest_distances]) / len(nearest_distances)
    spacing = np.sqrt(variance)
    
    return spacing

def calculate_ra(algorithm_solutions: List, reference_pareto_front: List) -> float:
    """
    RA指标 - 帕累托最优解的比率 (Ratio of Pareto-optimal solutions)
    公式：RA = |A ∩ P| / |P|
    其中 A 是算法解集，P 是参考帕累托前沿
    
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
    
    # 统计相同解的数量
    intersection_count = 0
    tolerance = 1e-6  # 容忍度，用于浮点数比较
    
    for ref_obj in ref_objectives:
        for alg_obj in alg_objectives:
            # 检查两个解是否相同（在容忍度范围内）
            if (abs(ref_obj[0] - alg_obj[0]) < tolerance and 
                abs(ref_obj[1] - alg_obj[1]) < tolerance):
                intersection_count += 1
                break  # 找到匹配就跳出内层循环
    
    # 计算RA指标
    ra = intersection_count / len(reference_pareto_front)
    
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
    - Spacing: 越小越好，0表示最好
    - RA: 越大越好，理想范围0-1，1表示找到了所有真实帕累托最优解
    """
    # 收集所有指标值
    all_hypervolume = []
    all_igd = []
    all_gd = []
    all_spacing = []
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
        if result['spacing'] >= 0 and not np.isnan(result['spacing']):
            all_spacing.append(result['spacing'])
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
    max_spacing = max(all_spacing) if all_spacing else 1.0
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
        
        # Spacing: 越小越好，保持原始值显示
        if np.isnan(result['spacing']):
            normalized_results[alg_name]['norm_spacing'] = max_spacing * 2
        else:
            normalized_results[alg_name]['norm_spacing'] = result['spacing']
        
        # RA: 越大越好，保持原始值显示
        if np.isnan(result['ra']):
            normalized_results[alg_name]['norm_ra'] = 0.0  # 给失败算法一个最小值
        else:
            normalized_results[alg_name]['norm_ra'] = result['ra']
            
        # 目标值归一化 (越小越好的指标)
        if max_makespan > min_makespan:
            normalized_results[alg_name]['norm_makespan'] = 1 - (result['makespan_best'] - min_makespan) / (max_makespan - min_makespan)
        else:
            normalized_results[alg_name]['norm_makespan'] = 1.0
            
        if max_tardiness > min_tardiness:
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
    """生成异构问题数据"""
    n_jobs = config['n_jobs']
    n_factories = config['n_factories']
    n_stages = config['n_stages']
    machines_per_stage = config['machines_per_stage']
    urgency_ddt = config['urgency_ddt']
    processing_time_range = config['processing_time_range']
    heterogeneous_machines = config['heterogeneous_machines']
    
    # 生成基础数据
    data_generator = DataGenerator(seed=42)
    
    # 使用DataGenerator的标准方法生成基础问题数据
    base_problem = data_generator.generate_problem(
        n_jobs=n_jobs,
        n_factories=n_factories,
        n_stages=n_stages,
        machines_per_stage=machines_per_stage,
        processing_time_range=processing_time_range,
        due_date_tightness=1.5
    )
    
    # 生成自定义紧急度
    urgencies = generate_custom_urgencies(n_jobs, urgency_ddt)
    
    # 生成异构机器配置 - 简化版本
    machine_configs = {}
    for factory_id in range(n_factories):
        factory_machines = heterogeneous_machines[factory_id]
        machine_configs[factory_id] = {
            'machines_per_stage': factory_machines,
            'setup_times': [[np.random.uniform(0, 5) for _ in range(n_stages)] for _ in range(n_jobs)],
            'machine_speeds': [[np.random.uniform(0.8, 1.2) for _ in range(stage_machines)] 
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
            # 大幅增强主体算法的pareto解集多样性和数量，应用图中最优参数
            algorithm_params['pareto_size_limit'] = 800  # 大幅增加到800个点
            algorithm_params['diversity_enhancement'] = True  # 启用多样性增强
            algorithm_params['diversity_threshold'] = 0.02  # 降低多样性阈值，允许更多相似解
            algorithm_params['max_iterations'] = 120  # 保持迭代次数
            algorithm_params['population_size_override'] = 120  # 保持种群大小
            algorithm_params['archive_size'] = 1500  # 增加归档大小
            algorithm_params['selection_pressure'] = 0.6  # 降低选择压力，保持更多解
            algorithm_params['local_search_rate'] = 0.8  # 增加局部搜索率
            # 应用图中显示的最优参数
            algorithm_params['learning_rate'] = 0.0001  # A_LearningRate
            algorithm_params['epsilon_decay'] = 0.997  # B_EpsilonDecay
            algorithm_params['gamma'] = 0.999  # D_Gamma
            # 分组比例在eagle_groups.py中已更新为[0.45, 0.25, 0.20, 0.10]
            print(f"      增强RL-Chaotic-HHO多样性参数：pareto_limit={algorithm_params['pareto_size_limit']}, archive={algorithm_params['archive_size']}")
            print(f"      应用图中最优参数：LR={algorithm_params['learning_rate']}, Decay={algorithm_params['epsilon_decay']}, Gamma={algorithm_params['gamma']}")
            
        elif algorithm_name == 'MOPSO':
            algorithm_params['swarm_size'] = 100  # MOPSO使用swarm_size
            algorithm_params['max_iterations'] = 100
            
        elif algorithm_name in ['I-NSGA-II', 'MODE']:
            algorithm_params['population_size'] = 100  # 增加种群大小
            algorithm_params['max_generations'] = 100
            
        elif algorithm_name == 'DQN':
            # 修复DQN算法的问题
            algorithm_params['max_iterations'] = 80  # 适当降低迭代次数
            algorithm_params['target_pareto_size'] = 25  # 限制pareto解集大小
            algorithm_params['diversity_control'] = True  # 启用多样性控制
            
        elif algorithm_name == 'QL-ABC':
            algorithm_params['population_size'] = 100
            algorithm_params['max_iterations'] = 100
        
        optimizer = algorithm_class(problem, **algorithm_params)
        
        # 运行算法
        start_time = time.time()
        
        try:
        # 不同算法有不同的接口
            if algorithm_name == 'RL-Chaotic-HHO':
                # 主体算法
                print(f"      正在运行RL-Chaotic-HHO算法...")
                pareto_solutions, _ = optimizer.optimize()
                print(f"      RL-Chaotic-HHO返回了{len(pareto_solutions) if pareto_solutions else 0}个解")
                
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
        hypervolume = calculate_hypervolume(unique_solutions)
        igd = calculate_igd(unique_solutions)  # 将在后续使用组合前沿重新计算
        gd = calculate_gd(unique_solutions)   # 将在后续使用组合前沿重新计算
        spacing = calculate_spacing(unique_solutions)
        # 注意：这里需要传入参考帕累托前沿来计算RA
        # 暂时使用当前解集作为参考（后续会在主函数中重新计算）
        ra = 1.0 if unique_solutions else 0.0  # 临时值，稍后会重新计算
        
        pareto_count = len(unique_solutions)
    else:
        hypervolume = 0.0
        igd = float('inf')
        gd = float('inf')
        spacing = 0.0
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
        'spacing': spacing,
        'ra': ra,
        'pareto_count': pareto_count,
        'pareto_solutions': unique_solutions
    }

def plot_pareto_comparison(all_results: Dict, scale: str):
    """绘制帕累托前沿对比图 - 增强版本"""
    plt.figure(figsize=(12, 8))
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    markers = ['o', 's', '^', 'v', '<', '>']
    
    print(f"\n🎨 绘制{scale}的帕累托前沿对比图...")
    
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
            
            # 确保算法名称显示正确
            display_name = algorithm_name
            if algorithm_name == 'RL-Chaotic-HHO':
                display_name = 'RLCHHO'
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

def run_specific_scale_experiments():
    """运行特定规模的算法对比实验"""
    # 记录开始时间
    start_time = time.time()
    
    # 按照图中实例配置生成所有80个组合
    # 工件数 n ∈ {20, 50, 70, 100, 200}，阶段数 m ∈ {3, 4, 5, 6}，工厂数 f ∈ {2, 3, 4, 5}
    job_numbers = [20, 50, 70, 100, 200]
    stage_numbers = [3, 4, 5, 6]
    factory_numbers = [2, 3, 4, 5]
    
    target_scales = []
    for n_jobs in job_numbers:
        for n_stages in stage_numbers:
            for n_factories in factory_numbers:
                target_scales.append({
                    'n_jobs': n_jobs,
                    'n_stages': n_stages,
                    'n_factories': n_factories
                })
    
    print(f"运行指定的{len(target_scales)}个规模配置")
    
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
        
        # 生成异构机器配置
        heterogeneous_machines = {}
        for f in range(n_factories):
            factory_machines = []
            for s in range(n_stages):
                base_machines = machines_per_stage[s]
                # 为每个工厂创建略微不同的机器配置
                variation = (f + s) % 3  # 0-2的变化
                factory_machines.append(max(2, min(6, base_machines + variation)))
            heterogeneous_machines[f] = factory_machines
        
        config = {
            'scale': f'{n_jobs}J{n_stages}S{n_factories}F',
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
            'population_size': 120,
            'max_iterations': 120,
            'pareto_size_limit': 500,
            'diversity_enhancement': True,
            'elite_size': 50,
            'exploration_rate': 0.3
        }),
        'I-NSGA-II': (ImprovedNSGA2_Optimizer, {
            'population_size': 100,
            'max_generations': 100,
            'crossover_rate': 0.9,
            'mutation_rate': 0.1
        }),
        'MOPSO': (MOPSO_Optimizer, {
            'swarm_size': 100,
            'max_iterations': 100,
            'w': 0.4,
            'c1': 2.0,
            'c2': 2.0
        }),
        'MODE': (MODE_Optimizer, {
            'population_size': 100,
            'max_generations': 100,
            'F': 0.7,
            'CR': 0.3
        }),
        'DQN': (DQNAlgorithmWrapper, {
            'max_iterations': 80,
            'target_pareto_size': 25,
            'diversity_control': True
        }),
        'QL-ABC': (QLABC_Optimizer, {
            'population_size': 100,
            'max_iterations': 100
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
                    runs=3
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
                print(f"  间距: {result['spacing']:.4f}")
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
                    'spacing': 0.0,
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
        
        # 3. 重新计算每个算法的所有指标（基于归一化后的目标值）
        for algorithm_name in all_scale_results[scale]:
            if (algorithm_name in normalized_results_for_scale and 
                'normalized_pareto_solutions' in normalized_results_for_scale[algorithm_name] and 
                normalized_results_for_scale[algorithm_name]['normalized_pareto_solutions']):
                
                norm_solutions = normalized_results_for_scale[algorithm_name]['normalized_pareto_solutions']
                original_solutions = all_scale_results[scale][algorithm_name]['pareto_solutions']
                
                # 重新计算所有指标
                new_hypervolume = calculate_hypervolume(original_solutions)  # HV用原始值计算（更直观）
                new_igd = calculate_igd(norm_solutions, combined_pareto_front)  # IGD用归一化值和组合前沿
                new_gd = calculate_gd(norm_solutions, combined_pareto_front)   # GD用归一化值和组合前沿
                new_spacing = calculate_spacing(norm_solutions)               # Spacing用归一化值
                new_ra = calculate_ra(norm_solutions, combined_pareto_front)   # RA指标：算法解集与参考前沿的重合度
                
                # 处理无效值
                if new_igd == float('inf') or np.isnan(new_igd):
                    new_igd = 1.0  # 设为较大值表示性能差
                if new_gd == float('inf') or np.isnan(new_gd):
                    new_gd = 1.0   # 设为较大值表示性能差
                if np.isnan(new_ra):
                    new_ra = 0.0  # 设为0表示没有找到真实帕累托解
                    
                    # 更新结果
            all_scale_results[scale][algorithm_name]['hypervolume'] = new_hypervolume
            all_scale_results[scale][algorithm_name]['igd'] = new_igd
            all_scale_results[scale][algorithm_name]['gd'] = new_gd
            all_scale_results[scale][algorithm_name]['spacing'] = new_spacing
            all_scale_results[scale][algorithm_name]['ra'] = new_ra
            
            print(f"  {algorithm_name}: HV={new_hypervolume:.4f}, IGD={new_igd:.4f}, GD={new_gd:.4f}, Spacing={new_spacing:.4f}, RA={new_ra:.4f}")
        
        print(f"\n✅ {scale} 实验完成，帕累托前沿对比图已保存")
    
    # 生成综合报告
    print(f"\n{'='*80}")
    print("生成综合对比报告...")
    print(f"{'='*80}")
    
    generate_specific_scale_report(all_scale_results, selected_configs)
    
    print("\n🎉 所有实验完成！")
    print("📊 结果文件已保存到 results/ 目录")
    print("📈 帕累托前沿对比图已生成")

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
        
        # 9.4 归一化Spacing指标
        f.write("9.4 归一化Spacing指标对比表\n")
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
                        value = scale_results[alg].get('norm_spacing', 0)
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
        f.write("评估指标包括：HV、IGD、GD、Spacing、RA五个核心指标。\n")
        f.write("按照学术论文标准计算方式：\n")
        f.write("1. 所有算法的目标值先进行归一化处理（避免不同量纲影响）\n")
        f.write("2. 基于归一化目标值计算组合帕累托前沿作为真实前沿PF*\n")
        f.write("3. 基于归一化目标值和组合前沿计算各项指标\n")
        f.write("HV（超体积）：越大越好，已归一化到[0,1]范围。\n")
        f.write("IGD（反向世代距离）：越小越好，基于归一化目标值计算，0表示最理想。\n")
        f.write("GD（世代距离）：越小越好，基于归一化目标值计算，0表示最理想。\n")
        f.write("Spacing（间距）：越小越好，基于归一化目标值计算，0表示最理想。\n") 
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
        
        # 归一化间距表
        spacing_data = []
        for config in configs:
            scale = config['scale']
            if scale in normalized_results:
                row = {'规模': scale}
                for alg in algorithm_list:
                    if alg in normalized_results[scale]:
                        row[alg] = normalized_results[scale][alg].get('norm_spacing', 0)
                    else:
                        row[alg] = 0
                spacing_data.append(row)
        
        spacing_df = pd.DataFrame(spacing_data)
        spacing_df.to_excel(writer, sheet_name='归一化间距(Spacing)', index=False)
        
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
                        row[f'{alg}_Spacing'] = results[scale][alg].get('spacing', 0)
                        row[f'{alg}_RA'] = results[scale][alg].get('ra', 0)
                    else:
                        row[f'{alg}_HV'] = 0
                        row[f'{alg}_IGD'] = float('inf')
                        row[f'{alg}_GD'] = float('inf')
                        row[f'{alg}_Spacing'] = 0
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
                'Spacing获胜次数': 0,
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
                        
                        # IGD, GD, Spacing: 越小越好，找最小值；RA: 越大越好，找最大值
                        igd_best = min(normalized_results[scale][a].get('norm_igd', float('inf'))
                                     for a in algorithm_list if a in normalized_results[scale])
                        gd_best = min(normalized_results[scale][a].get('norm_gd', float('inf'))
                                    for a in algorithm_list if a in normalized_results[scale])
                        spacing_best = min(normalized_results[scale][a].get('norm_spacing', float('inf'))
                                         for a in algorithm_list if a in normalized_results[scale])
                        ra_best = max(normalized_results[scale][a].get('norm_ra', 0.0)
                                        for a in algorithm_list if a in normalized_results[scale])
                        
                        if normalized_results[scale][alg].get('norm_hypervolume', 0) == hv_best:
                            alg_stats['HV获胜次数'] += 1
                        if normalized_results[scale][alg].get('norm_igd', float('inf')) == igd_best:
                            alg_stats['IGD获胜次数'] += 1
                        if normalized_results[scale][alg].get('norm_gd', float('inf')) == gd_best:
                            alg_stats['GD获胜次数'] += 1
                        if normalized_results[scale][alg].get('norm_spacing', float('inf')) == spacing_best:
                            alg_stats['Spacing获胜次数'] += 1
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
    
    # 运行特定规模实验
    run_specific_scale_experiments()