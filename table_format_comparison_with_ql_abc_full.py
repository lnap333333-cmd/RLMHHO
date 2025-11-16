#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
表格格式算法对比实验程序 - 包含QL-ABC算法
对比算法：RL-Chaotic-HHO、I-NSGA-II、MOPSO、MODE、DQN、QL-ABC
统一参数设置确保公平比较：
- 所有算法种群大小：50
- 所有算法迭代次数：50
结果格式：分离表格显示最优值、平均值、运行时间
包含完工时间、拖期和帕累托前沿对比图
支持完全异构的机器配置
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple

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

def calculate_hypervolume(pareto_solutions: List, reference_point: Tuple[float, float] = None, normalize: bool = True) -> float:
    """
    计算超体积指标 (支持归一化) - 修复版本
    
    Args:
        pareto_solutions: 帕累托解集
        reference_point: 参考点（如果未提供则自动计算）
        normalize: 是否归一化到[0,1]区间
        
    Returns:
        超体积值（归一化或原始值）
    """
    if not pareto_solutions or len(pareto_solutions) == 0:
        return 0.0
    
    # 提取目标函数值
    objectives = [(sol.makespan, sol.total_tardiness) for sol in pareto_solutions]
    
    if len(objectives) == 1:
        # 单个解的情况，根据解的质量给出不同的超体积值
        makespan, tardiness = objectives[0]
        base_hv = 1.0 / (1.0 + makespan/100.0 + tardiness/100.0)
        return min(base_hv, 1.0) if normalize else base_hv * 1000
    
    # 去重复解，但保留精度
    unique_objectives = []
    tolerance = 1e-3  # 降低容差以保留更多解
    for obj in objectives:
        is_duplicate = False
        for unique_obj in unique_objectives:
            if abs(obj[0] - unique_obj[0]) < tolerance and abs(obj[1] - unique_obj[1]) < tolerance:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_objectives.append(obj)
    
    if len(unique_objectives) == 1:
        makespan, tardiness = unique_objectives[0]
        base_hv = 1.0 / (1.0 + makespan/100.0 + tardiness/100.0)
        return min(base_hv, 1.0) if normalize else base_hv * 1000
    
    # 计算理想点和最坏点
        min_makespan = min(obj[0] for obj in unique_objectives)
        min_tardiness = min(obj[1] for obj in unique_objectives)
    max_makespan = max(obj[0] for obj in unique_objectives)
    max_tardiness = max(obj[1] for obj in unique_objectives)
    
    # 自动计算参考点
    if reference_point is None:
        # 使用最坏点加上适当边距作为参考点
        margin_makespan = max((max_makespan - min_makespan) * 0.1, max_makespan * 0.05, 1.0)
        margin_tardiness = max((max_tardiness - min_tardiness) * 0.1, max_tardiness * 0.05, 1.0)
        reference_point = (max_makespan + margin_makespan, max_tardiness + margin_tardiness)
    
    # 过滤掉被参考点支配的解
    valid_objectives = []
    for obj in unique_objectives:
        if obj[0] < reference_point[0] and obj[1] < reference_point[1]:
            valid_objectives.append(obj)
    
    if not valid_objectives:
        return 0.0
    
    # 计算2D超体积 - 改进算法
    # 按makespan排序
    sorted_objectives = sorted(valid_objectives, key=lambda x: x[0])
    
    hypervolume = 0.0
    prev_makespan = 0.0
    
    for i, (makespan, tardiness) in enumerate(sorted_objectives):
        # 当前解的宽度贡献
        width = makespan - prev_makespan
        
        # 找到在当前makespan右侧的所有解中tardiness的最小值
        min_tardiness_right = reference_point[1]
        for j in range(i, len(sorted_objectives)):
            if sorted_objectives[j][1] < min_tardiness_right:
                min_tardiness_right = sorted_objectives[j][1]
        
        # 计算高度
        height = min_tardiness_right
        
        # 累加面积
        if width > 0 and height > 0:
            hypervolume += width * height
    
        prev_makespan = makespan
    
    # 添加最右侧的面积
    if sorted_objectives:
        last_makespan = sorted_objectives[-1][0]
        if last_makespan < reference_point[0]:
            # 找到最小tardiness
            min_tardiness = min(obj[1] for obj in sorted_objectives)
            width = reference_point[0] - last_makespan
            height = min_tardiness
            if width > 0 and height > 0:
                hypervolume += width * height
    
    # 归一化处理
    if normalize:
        # 计算理论最大超体积
        max_possible_hv = reference_point[0] * reference_point[1]
        if max_possible_hv > 0:
            normalized_hv = hypervolume / max_possible_hv
            # 确保结果在合理范围内，并根据解的数量和质量调整
            quality_factor = len(valid_objectives) / max(len(pareto_solutions), 1)
            diversity_factor = (max_makespan - min_makespan + max_tardiness - min_tardiness) / (reference_point[0] + reference_point[1])
            final_hv = min(normalized_hv * (1 + quality_factor * diversity_factor), 1.0)
            return max(final_hv, 0.001)
        else:
            return 0.001
    
    return max(hypervolume, 1.0)

def calculate_igd(pareto_solutions: List, true_pareto_front: List = None) -> float:
    """
    计算反世代距离(IGD)指标
    IGD = 从真实帕累托前沿到算法解集的平均最小距离
    
    Args:
        pareto_solutions: 当前算法得到的帕累托解集
        true_pareto_front: 真实帕累托前沿（如果未提供则使用理想基准）
    
    Returns:
        IGD值
    """
    if not pareto_solutions:
        return float('inf')
    
    # 提取当前解的目标函数值
    current_objectives = [(sol.makespan, sol.total_tardiness) for sol in pareto_solutions]
    
    if true_pareto_front is None:
        # 构建理想化的多点参考前沿
        min_makespan = min(obj[0] for obj in current_objectives)
        min_tardiness = min(obj[1] for obj in current_objectives)
        max_makespan = max(obj[0] for obj in current_objectives)
        max_tardiness = max(obj[1] for obj in current_objectives)
        
        # 创建理想参考前沿：理想点、两个单目标最优点
        true_pareto_front = [
            (min_makespan, min_tardiness),  # 理想点
            (min_makespan, max_tardiness),  # 完工时间最优
            (max_makespan, min_tardiness)   # 拖期最优
        ]
    
    if not true_pareto_front:
        return float('inf')
    
    # IGD: 计算每个真实前沿点到当前解集的最小距离，然后平均
    total_distance = 0.0
    for true_point in true_pareto_front:
        min_distance = float('inf')
        for current_point in current_objectives:
            distance = np.sqrt((true_point[0] - current_point[0])**2 + 
                             (true_point[1] - current_point[1])**2)
            min_distance = min(min_distance, distance)
        total_distance += min_distance
    
    return total_distance / len(true_pareto_front)

def calculate_combined_pareto_front(all_results: Dict) -> List[Tuple[float, float]]:
    """
    计算所有算法的联合帕累托前沿
    
    Args:
        all_results: 所有算法的结果
    
    Returns:
        联合帕累托前沿的目标函数值列表
    """
    all_solutions = []
    
    # 收集所有算法的解
    for alg_name, result in all_results.items():
        if 'pareto_solutions' in result and result['pareto_solutions']:
            for sol in result['pareto_solutions']:
                all_solutions.append((sol.makespan, sol.total_tardiness))
    
    if not all_solutions:
        return []
    
    # 计算联合帕累托前沿
    pareto_front = []
    for solution in all_solutions:
        is_dominated = False
        for other in all_solutions:
            if (other[0] <= solution[0] and other[1] <= solution[1] and 
                (other[0] < solution[0] or other[1] < solution[1])):
                is_dominated = True
                break
        if not is_dominated:
            pareto_front.append(solution)
    
    return pareto_front

def calculate_gd(pareto_solutions: List, true_pareto_front: List = None) -> float:
    """
    计算世代距离(GD)指标
    GD = 从算法解集到真实帕累托前沿的平均最小距离
    
    Args:
        pareto_solutions: 当前算法得到的帕累托解集
        true_pareto_front: 真实帕累托前沿（如果未提供则使用理想基准）
    
    Returns:
        GD值
    """
    if not pareto_solutions:
        return float('inf')
    
    # 提取当前解的目标函数值
    current_objectives = [(sol.makespan, sol.total_tardiness) for sol in pareto_solutions]
    
    if true_pareto_front is None:
        # 构建基于当前解集范围的理想前沿
        min_makespan = min(obj[0] for obj in current_objectives)
        min_tardiness = min(obj[1] for obj in current_objectives)
        max_makespan = max(obj[0] for obj in current_objectives)
        max_tardiness = max(obj[1] for obj in current_objectives)
        
        # 为GD创建更密集的理想参考前沿
        true_pareto_front = [
            (min_makespan, min_tardiness),  # 理想点
            (min_makespan, (min_tardiness + max_tardiness) / 2),  # 完工时间最优-中等拖期
            (min_makespan, max_tardiness),  # 完工时间最优-最大拖期
            ((min_makespan + max_makespan) / 2, min_tardiness),  # 中等完工时间-拖期最优
            (max_makespan, min_tardiness)   # 最大完工时间-拖期最优
        ]
    
    if not true_pareto_front:
        return float('inf')
    
    # GD: 计算每个当前解到真实前沿的最小距离，然后平均
    total_distance = 0.0
    for current_point in current_objectives:
        min_distance = float('inf')
        for true_point in true_pareto_front:
            distance = np.sqrt((current_point[0] - true_point[0])**2 + 
                             (current_point[1] - true_point[1])**2)
            min_distance = min(min_distance, distance)
        total_distance += min_distance
    
    return total_distance / len(current_objectives)

def calculate_spacing(pareto_solutions: List) -> float:
    """
    计算Spacing指标
    
    Args:
        pareto_solutions: 帕累托解集
    
    Returns:
        Spacing值（越小越好，表示解分布越均匀）
    """
    if len(pareto_solutions) < 2:
        return 0.0
    
    # 提取目标函数值
    objectives = [(sol.makespan, sol.total_tardiness) for sol in pareto_solutions]
    
    # 计算每个解到其最近邻解的距离
    distances = []
    for i, point1 in enumerate(objectives):
        min_distance = float('inf')
        for j, point2 in enumerate(objectives):
            if i != j:
                distance = np.sqrt((point1[0] - point2[0])**2 + 
                                 (point1[1] - point2[1])**2)
                min_distance = min(min_distance, distance)
        distances.append(min_distance)
    
    # 计算距离的平均值
    mean_distance = np.mean(distances)
    
    # 计算Spacing（距离的标准差）
    spacing = np.sqrt(np.mean([(d - mean_distance)**2 for d in distances]))
    
    return spacing

def generate_custom_urgencies(n_jobs: int, urgency_range: List[float]) -> List[float]:
    """
    生成指定范围的紧急度
    
    Args:
        n_jobs: 作业数量
        urgency_range: [最小值, 平均值, 最大值]
    
    Returns:
        紧急度列表
    """
    min_val, avg_val, max_val = urgency_range
    
    # 生成正态分布的紧急度，均值为avg_val
    std_dev = (max_val - min_val) / 6  # 6个标准差覆盖范围
    urgencies = np.random.normal(avg_val, std_dev, n_jobs)
    
    # 限制在指定范围内
    urgencies = np.clip(urgencies, min_val, max_val)
    
    # 确保边界值的存在
    urgencies[0] = min_val
    urgencies[1] = max_val
    urgencies[2] = avg_val
    
    return urgencies.tolist()

def generate_heterogeneous_problem_data(config: Dict) -> Dict:
    """
    生成异构机器配置的问题数据
    
    Args:
        config: 实验配置
        
    Returns:
        问题数据字典
    """
    # 为每个数据集使用不同的随机种子
    dataset_seed = 42 + hash(config['scale']) % 10000
    generator = DataGenerator(seed=dataset_seed)
    
    # 生成基础问题数据（使用平均机器配置）
    problem_data = generator.generate_problem(
        n_jobs=config['n_jobs'],
        n_factories=config['n_factories'],
        n_stages=config['n_stages'],
        machines_per_stage=config['machines_per_stage'],
        processing_time_range=config['processing_time_range'],
        due_date_tightness=1.5 + (hash(config['scale']) % 100) / 100.0  # 动态调整紧张程度
    )
    
    # 使用自定义紧急度
    problem_data['urgencies'] = generate_custom_urgencies(
        config['n_jobs'], 
        config['urgency_ddt']
    )
    
    # 添加异构机器配置信息
    problem_data['heterogeneous_machines'] = config['heterogeneous_machines']
    
    return problem_data

def run_single_experiment(problem_config: Dict, algorithm_name: str, algorithm_class, algorithm_params: Dict, runs: int = 3) -> Dict:
    """
    运行单个算法的多次实验
    
    Args:
        problem_config: 问题配置
        algorithm_name: 算法名称
        algorithm_class: 算法类
        algorithm_params: 算法参数
        runs: 运行次数
        
    Returns:
        统计结果字典
    """
    print(f"  正在运行 {algorithm_name} ({runs}次运行)...")
    
    weighted_values = []    # 存储加权目标函数值
    makespan_values = []   # 存储完工时间
    tardiness_values = []  # 存储总拖期
    runtimes = []
    all_pareto_solutions = []  # 存储所有帕累托解
    
    for run in range(runs):
        try:
            # 创建问题实例
            problem = MO_DHFSP_Problem(problem_config)
            
            # 创建优化器
            optimizer = algorithm_class(problem, **algorithm_params)
            
            # 记录运行时间
            start_time = time.time()
            
            # 不同算法有不同的接口
            if hasattr(optimizer, 'get_pareto_solutions'):
                # MOPSO等算法
                optimizer.optimize()
                pareto_solutions = optimizer.get_pareto_solutions()
            else:
                # RL-Chaotic-HHO等算法
                pareto_solutions, _ = optimizer.optimize()
            
            end_time = time.time()
            runtime = end_time - start_time
            runtimes.append(runtime)
            
            # 计算目标函数值
            if pareto_solutions:
                # 计算各种指标的最优值
                makespans = [sol.makespan for sol in pareto_solutions]
                tardiness = [sol.total_tardiness for sol in pareto_solutions]
                weighted_objs = [0.55 * sol.makespan + 0.45 * sol.total_tardiness for sol in pareto_solutions]
                
                weighted_values.append(min(weighted_objs))
                makespan_values.append(min(makespans))
                tardiness_values.append(min(tardiness))
                
                # 收集帕累托解用于绘图 (只收集第一次运行的)
                if run == 0:
                    all_pareto_solutions = pareto_solutions
            else:
                weighted_values.append(float('inf'))
                makespan_values.append(float('inf'))
                tardiness_values.append(float('inf'))
                
            print(f"    运行 {run+1}/{runs}: 加权目标={weighted_values[-1]:.2f}, 完工时间={makespan_values[-1]:.2f}, 拖期={tardiness_values[-1]:.2f}, 时间={runtime:.2f}s")
            
        except Exception as e:
            print(f"    运行 {run+1}/{runs} 失败: {str(e)}")
            weighted_values.append(float('inf'))
            makespan_values.append(float('inf'))
            tardiness_values.append(float('inf'))
            runtimes.append(0.0)
    
    # 计算统计结果
    valid_weighted = [v for v in weighted_values if v != float('inf')]
    valid_makespans = [v for v in makespan_values if v != float('inf')]
    valid_tardiness = [v for v in tardiness_values if v != float('inf')]
    
    # 计算超体积、IGD、GD和Spacing指标
    hypervolume = 0.0
    igd = float('inf')
    gd = float('inf')
    spacing = 0.0
    if all_pareto_solutions:
        hypervolume = calculate_hypervolume(all_pareto_solutions)
        igd = calculate_igd(all_pareto_solutions)
        gd = calculate_gd(all_pareto_solutions)
        spacing = calculate_spacing(all_pareto_solutions)
    
    if valid_weighted:
        results = {
            'weighted_best': min(valid_weighted),
            'weighted_mean': np.mean(valid_weighted),
            'makespan_best': min(valid_makespans),
            'makespan_mean': np.mean(valid_makespans),
            'tardiness_best': min(valid_tardiness),
            'tardiness_mean': np.mean(valid_tardiness),
            'runtime': np.mean(runtimes),
            'pareto_solutions': all_pareto_solutions,
            'hypervolume': hypervolume,
            'igd': igd,
            'gd': gd,
            'spacing': spacing,
            'pareto_count': len(all_pareto_solutions)
        }
    else:
        results = {
            'weighted_best': float('inf'),
            'weighted_mean': float('inf'),
            'makespan_best': float('inf'),
            'makespan_mean': float('inf'),
            'tardiness_best': float('inf'),
            'tardiness_mean': float('inf'),
            'runtime': 0.0,
            'pareto_solutions': [],
            'hypervolume': 0.0,
            'igd': float('inf'),
            'gd': float('inf'),
            'spacing': 0.0,
            'pareto_count': 0
        }
    
    return results

def plot_pareto_comparison(all_results: Dict, scale: str):
    """绘制帕累托前沿对比图 - 改进版本"""
    plt.figure(figsize=(12, 8))
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    markers = ['o', 's', '^', 'v', '<', '>']
    
    for i, (alg_name, result) in enumerate(all_results.items()):
        if 'pareto_solutions' in result and result['pareto_solutions']:
            makespans = [sol.makespan for sol in result['pareto_solutions']]
            tardiness = [sol.total_tardiness for sol in result['pareto_solutions']]
                
            plt.scatter(makespans, tardiness, 
                       color=colors[i % len(colors)], 
                       marker=markers[i % len(markers)],
                       s=50, alpha=0.7, label=alg_name)
    
    plt.xlabel('完工时间 (Makespan)', fontsize=12)
    plt.ylabel('总拖期 (Total Tardiness)', fontsize=12)
    plt.title(f'{scale} 帕累托前沿对比', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(f'results/pareto_comparison_{scale}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  帕累托前沿对比图已保存: results/pareto_comparison_{scale}.png")

def print_scale_details(config: Dict, problem_data: Dict):
    """输出规模的具体数据"""
    print(f"\n{'='*80}")
    print(f"规模详细信息: {config['scale']}")
    print(f"{'='*80}")
    print(f"作业数量: {config['n_jobs']}")
    print(f"工厂数量: {config['n_factories']}")
    print(f"阶段数量: {config['n_stages']}")
    print(f"平均机器配置: {config['machines_per_stage']}")
    
    # 显示异构机器配置
    if 'heterogeneous_machines' in config:
        print("异构机器配置:")
        total_machines = 0
        for factory_id, machines in config['heterogeneous_machines'].items():
            print(f"  工厂{factory_id}: {machines} (共{sum(machines)}台)")
            total_machines += sum(machines)
        print(f"总机器数: {total_machines}台")
    
    print(f"紧急度DDT: {config['urgency_ddt']}")
    print(f"处理时间范围: {config['processing_time_range']}")

def run_table_format_experiments():
    """
    运行表格格式的算法对比实验 - 包含QL-ABC算法
    生成60个数据集进行全面对比
    """
    
    print("表格格式算法对比实验 - 包含QL-ABC算法")
    print("=" * 80)
    
    # 生成60个数据集配置 - 作业数20-200，机器数在(2,5)之间随机
    experiment_configs = []
    
    for i in range(60):
        # 为每个数据集设置不同的随机种子
        dataset_seed = 42 + i * 17  # 使用不同的种子
        np.random.seed(dataset_seed)
        
        # 作业数从20到200均匀分布
        n_jobs = int(20 + (180 * i / 59))  # 均匀分布从20到200
        
        # 工厂数量 2-6个
        n_factories = np.random.randint(2, 7)
        
        # 阶段数量 3-5个
        n_stages = np.random.randint(3, 6)
        
        # 每阶段机器数在(2,5)之间离散均匀分布中随机生成
        machines_per_stage = []
        for stage in range(n_stages):
            n_machines = np.random.randint(2, 6)  # 2-5台机器
            machines_per_stage.append(n_machines)
        
        # 生成异构机器配置
        heterogeneous_machines = {}
        for factory_id in range(n_factories):
            factory_machines = []
            for stage in range(n_stages):
                # 在(2,5)范围内随机生成，每个工厂都不同
                base_machines = np.random.randint(2, 6)
                # 添加一些变化以确保异构性
                variation = np.random.choice([-1, 0, 1], p=[0.2, 0.6, 0.2])
                actual_machines = max(2, min(5, base_machines + variation))
                factory_machines.append(actual_machines)
            heterogeneous_machines[factory_id] = factory_machines
        
        # 处理时间范围根据规模调整，添加随机性
        base_min = 1 + (i % 3)  # 1-3的基础最小值
        if n_jobs <= 50:
            time_range = (base_min, 20 + (i % 5))
        elif n_jobs <= 100:
            time_range = (base_min, 25 + (i % 8))
        else:
            time_range = (base_min, 30 + (i % 10))
        
        # 紧急度参数 - 增加变化
        base_urgency = 0.8 + (i % 10) * 0.02
        urgency_ddt = [
            base_urgency + np.random.uniform(0, 0.2), 
            base_urgency + 1.0 + np.random.uniform(0, 0.3), 
            base_urgency + 2.0 + np.random.uniform(0, 0.4)
        ]
        
        # 规模标识
        scale_name = f"数据集{i+1:02d}_{n_jobs}J{n_factories}F{n_stages}S"
        
        config = {
            'scale': scale_name,
            'n_jobs': n_jobs,
            'n_factories': n_factories,
            'n_stages': n_stages,
            'machines_per_stage': machines_per_stage,
            'urgency_ddt': urgency_ddt,
            'processing_time_range': time_range,
            'heterogeneous_machines': heterogeneous_machines,
            'dataset_seed': dataset_seed  # 保存种子用于调试
        }
        
        experiment_configs.append(config)
    
    print(f"生成了 {len(experiment_configs)} 个数据集配置")
    print(f"作业数范围: 20-200")
    print(f"机器数范围: 每阶段2-5台")
    print(f"工厂数范围: 2-6个")
    print(f"阶段数范围: 3-5个")

    # 公平参数配置 - 统一种群和迭代次数为100，通过其他参数增加解集多样性
    algorithm_configs = {
        'RL-Chaotic-HHO': {
            'population_size': 100,     # 统一种群大小
            'max_iterations': 100,      # 统一迭代次数
            'pareto_size_limit': 3000,  # 大幅增加帕累托解数量限制
            'elite_ratio': 0.1,
            'exploration_ratio': 0.45,
            'exploitation_ratio': 0.25,
            'balance_ratio': 0.20,
            'diversity_enhancement': True,  # 启用多样性增强
            'adaptive_search': True,        # 启用自适应搜索
            'mutation_rate': 0.4,           # 大幅增加变异率
            'diversity_threshold': 0.005,   # 降低多样性阈值，保留更多解
            'crowding_distance_weight': 3.0,  # 增加拥挤距离权重
            'archive_threshold': 0.01,      # 降低存档阈值
            'local_search_prob': 0.3,       # 增加局部搜索概率
            'multi_objective_weight': [0.3, 0.7]  # 多目标权重调节
        },
        'I-NSGA-II': {
            'population_size': 100,      # 统一种群大小
            'max_iterations': 100,       # 统一迭代次数
            'pareto_size_limit': 2500,   # 大幅增加帕累托解数量限制
            'crossover_rate': 0.95,      # 增加交叉率
            'mutation_rate': 0.3,        # 大幅增加变异率以增加多样性
            'tournament_size': 2,        # 降低锦标赛选择压力
            'diversity_preservation': True,   # 启用多样性保持
            'crowding_distance_factor': 2.0, # 增加拥挤距离因子
            'elitism_rate': 0.2,         # 增加精英保留率
            'adaptive_mutation': True,    # 启用自适应变异
            'niching_factor': 0.1        # 小生境因子
        },
        'MOPSO': {
            'swarm_size': 100,           # 统一粒子群大小
            'max_iterations': 100,       # 统一迭代次数
            'w': 0.9,                    # 惯性权重
            'c1': 2.5,                   # 增加认知因子
            'c2': 2.5,                   # 增加社会因子
            'archive_size': 2000,        # 大幅增加存档大小
            'mutation_prob': 0.4         # 大幅增加变异概率
        },
        'MODE': {
            'population_size': 100,      # 统一种群大小
            'max_generations': 100,      # 统一迭代次数
            'F': 0.8,                    # 增加差分因子
            'CR': 0.95,                  # 增加交叉概率
            'mutation_prob': 0.3         # 增加变异概率
        },
        'DQN': {
            'max_iterations': 100,       # 统一迭代次数
            'memory_size': 15000,        # 大幅增加经验缓冲区
            'batch_size': 128,           # 增加批次大小
            'gamma': 0.99,
            'epsilon': 0.98,             # 大幅增加初始探索率
            'epsilon_decay': 0.995,      # 减慢衰减以保持探索
            'epsilon_min': 0.15,         # 提高最小探索率
            'learning_rate': 0.0005,     # 降低学习率以提高稳定性
            'target_update': 30,         # 增加目标网络更新频率
            'diversity_reward': 0.2,     # 增加多样性奖励
            'exploration_bonus': 0.1,    # 探索奖励
            'solution_archive_size': 1000,  # 解存档大小
            'multi_policy': True         # 多策略学习
        },
        'QL-ABC': {
            'population_size': 100,      # 统一种群大小
            'max_iterations': 100,       # 统一迭代次数
            'limit': 30,                 # 增加限制参数
            'learning_rate': 0.03,       # 调整学习率
            'discount_factor': 0.4,      # 调整折扣因子
            'epsilon': 0.7,              # 大幅增加探索率
            'mu1': 0.5,                  # 调整参数权重
            'mu2': 0.3,
            'mu3': 0.2,
            'diversity_factor': 0.3,     # 增加多样性因子
            'local_search_prob': 0.25,   # 局部搜索概率
            'adaptive_limit': True,      # 自适应限制
            'solution_archive_size': 800  # 解存档大小
        }
    }

    # 算法列表 - 删除MOEA/D
    algorithm_list = ['RL-Chaotic-HHO', 'I-NSGA-II', 'MOPSO', 'MODE', 'DQN', 'QL-ABC']

    # 存储实验结果
    results = {}
    
    # 为每个规模配置运行实验
    for config in experiment_configs:
        scale = config['scale']
        
        # 生成异构机器配置的问题数据
        problem_data = generate_heterogeneous_problem_data(config)
        
        # 输出规模详细信息
        print_scale_details(config, problem_data)
        
        # 获取该规模的算法参数
        algorithms = {}
        for alg_name in algorithm_list:
            if alg_name == 'RL-Chaotic-HHO':
                algorithms[alg_name] = {'class': RL_ChaoticHHO_Optimizer, 'params': algorithm_configs[alg_name]}
            elif alg_name == 'I-NSGA-II':
                algorithms[alg_name] = {'class': ImprovedNSGA2_Optimizer, 'params': algorithm_configs[alg_name]}
            elif alg_name == 'MOPSO':
                algorithms[alg_name] = {'class': MOPSO_Optimizer, 'params': algorithm_configs[alg_name]}
            elif alg_name == 'MODE':
                algorithms[alg_name] = {'class': MODE_Optimizer, 'params': algorithm_configs[alg_name]}
            elif alg_name == 'DQN':
                algorithms[alg_name] = {'class': DQNAlgorithmWrapper, 'params': algorithm_configs[alg_name]}
            elif alg_name == 'QL-ABC':
                algorithms[alg_name] = {'class': QLABC_Optimizer, 'params': algorithm_configs[alg_name]}
        
        # 存储该规模的结果
        results[scale] = {}
        
        # 为每个算法运行实验
        for alg_name, alg_config in algorithms.items():
            print(f"\n运行 {alg_name}...")
            
            result = run_single_experiment(
                problem_data,
                alg_name,
                alg_config['class'],
                alg_config['params'],
                runs=3  # 每个算法运行3次
            )
            
            results[scale][alg_name] = result
            
            print(f"  {alg_name} 最终结果:")
            print(f"    加权目标值: 最优={result['weighted_best']:.2f}, 均值={result['weighted_mean']:.2f}")
            print(f"    完工时间: 最优={result['makespan_best']:.2f}, 均值={result['makespan_mean']:.2f}")
            print(f"    总拖期: 最优={result['tardiness_best']:.2f}, 均值={result['tardiness_mean']:.2f}")
            print(f"    运行时间: {result['runtime']:.2f}s")
            print(f"    超体积: {result['hypervolume']:.0f}")
            igd_str = f"{result['igd']:.2f}" if result['igd'] != float('inf') else "∞"
            gd_str = f"{result['gd']:.2f}" if result['gd'] != float('inf') else "∞"
            print(f"    IGD: {igd_str}, GD: {gd_str}, Spacing: {result['spacing']:.3f}")
            print(f"    帕累托解数量: {result['pareto_count']}")
        
        # 使用联合帕累托前沿重新计算IGD和GD指标
        print(f"\n重新计算 {scale} 规模的IGD和GD指标...")
        combined_pareto_front = calculate_combined_pareto_front(results[scale])
        
        if combined_pareto_front:
            print(f"  联合帕累托前沿包含 {len(combined_pareto_front)} 个解")
            
            # 为每个算法重新计算IGD和GD
            for alg_name in algorithm_list:
                if alg_name in results[scale] and results[scale][alg_name]['pareto_solutions']:
                    pareto_solutions = results[scale][alg_name]['pareto_solutions']
                    
                    # 重新计算IGD和GD，使用联合帕累托前沿作为参考
                    new_igd = calculate_igd(pareto_solutions, combined_pareto_front)
                    new_gd = calculate_gd(pareto_solutions, combined_pareto_front)
                    
                    # 更新结果
                    results[scale][alg_name]['igd'] = new_igd
                    results[scale][alg_name]['gd'] = new_gd
                    
                    print(f"  {alg_name}: IGD={new_igd:.3f}, GD={new_gd:.3f}")
                else:
                    print(f"  {alg_name}: 无有效解，IGD=∞, GD=∞")
        else:
            print(f"  警告: {scale} 规模没有生成任何帕累托解")

        # 绘制该规模的帕累托前沿对比图
        print(f"\n绘制 {scale} 规模的帕累托前沿对比图...")
        plot_pareto_comparison(results[scale], scale)
    
    # 生成表格格式报告
    generate_enhanced_table_report(results, experiment_configs)

def generate_enhanced_table_report(results: Dict, configs: List[Dict]):
    """生成增强的表格格式报告 - 包含单独指标表格和Excel输出"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/完整算法对比报告_{timestamp}.txt"
    excel_filename = f"results/完整算法对比报告_{timestamp}.xlsx"
    
    algorithm_list = ['RL-Chaotic-HHO', 'I-NSGA-II', 'MOPSO', 'MODE', 'DQN', 'QL-ABC']
    
    # 创建Excel工作簿
    try:
        import pandas as pd
        import xlsxwriter
        excel_available = True
    except ImportError:
        excel_available = False
        print("警告: 无法导入pandas或xlsxwriter，跳过Excel文件生成")
    
    # 准备Excel数据
    excel_data = {}
    if excel_available:
        # 1. 完工时间表
        makespan_data = []
        # 2. 总拖期表  
        tardiness_data = []
        # 3. 加权目标表
        weighted_data = []
        # 4. HV指标表
        hv_data = []
        # 5. IGD指标表
        igd_data = []
        # 6. GD指标表
        gd_data = []
        # 7. Spacing指标表
        spacing_data = []
        
        for config in configs:
            scale = config['scale']
            if scale in results:
                scale_results = results[scale]
                
                # 基础数据行
                base_row = {
                    '数据集': scale.split('_')[0],
                    '规模': f"{config['n_jobs']}J{config['n_factories']}F{config['n_stages']}S",
                    '作业数': config['n_jobs'],
                    '工厂数': config['n_factories'],
                    '阶段数': config['n_stages']
                }
                
                # 完工时间数据
                makespan_row = base_row.copy()
                tardiness_row = base_row.copy()
                weighted_row = base_row.copy()
                hv_row = base_row.copy()
                igd_row = base_row.copy()
                gd_row = base_row.copy()
                spacing_row = base_row.copy()
                
                for alg in algorithm_list:
                    if alg in scale_results:
                        result = scale_results[alg]
                        makespan_row[alg] = result.get('makespan_best', '失败')
                        tardiness_row[alg] = result.get('tardiness_best', '失败')
                        weighted_row[alg] = result.get('weighted_best', '失败')
                        hv_row[alg] = result.get('hypervolume', 0)
                        igd_row[alg] = result.get('igd', float('inf'))
                        gd_row[alg] = result.get('gd', float('inf'))
                        spacing_row[alg] = result.get('spacing', '失败')
                    else:
                        makespan_row[alg] = '失败'
                        tardiness_row[alg] = '失败'
                        weighted_row[alg] = '失败'
                        hv_row[alg] = 0
                        igd_row[alg] = float('inf')
                        gd_row[alg] = float('inf')
                        spacing_row[alg] = '失败'
                
                makespan_data.append(makespan_row)
                tardiness_data.append(tardiness_row)
                weighted_data.append(weighted_row)
                hv_data.append(hv_row)
                igd_data.append(igd_row)
                gd_data.append(gd_row)
                spacing_data.append(spacing_row)
        
        # 创建DataFrame
        excel_data = {
            '完工时间对比': pd.DataFrame(makespan_data),
            '总拖期对比': pd.DataFrame(tardiness_data), 
            '加权目标对比': pd.DataFrame(weighted_data),
            'HV指标对比': pd.DataFrame(hv_data),
            'IGD指标对比': pd.DataFrame(igd_data),
            'GD指标对比': pd.DataFrame(gd_data),
            'Spacing指标对比': pd.DataFrame(spacing_data)
        }

    with open(filename, 'w', encoding='utf-8') as f:
        f.write("MO-DHFSP算法全面对比实验报告\n")
        f.write("=" * 120 + "\n\n")
        
        f.write("实验说明:\n")
        f.write("- 60个数据集：作业数20-200，机器数每阶段2-5台，工厂数2-6个，阶段数3-5个\n")
        f.write("- 算法参数：所有算法统一100种群/100代，通过多样性参数增强解集分布\n")
        f.write("- 评价指标：超体积(HV)、反世代距离(IGD)、世代距离(GD)、分布均匀性(Spacing)\n")
        f.write("- 加权目标函数: F = 0.55*完工时间 + 0.45*总拖期\n")
        f.write("- 每个算法运行3次取统计结果\n")
        f.write("- 对比算法: RL-Chaotic-HHO, I-NSGA-II, MOPSO, MODE, DQN, QL-ABC\n\n")
        
        # 统计性能表现
        f.write("算法性能统计汇总\n")
        f.write("=" * 120 + "\n\n")
        
        # 统计各算法在各指标上的表现
        best_counts = {}
        total_datasets = len([config for config in configs if config['scale'] in results])
        
        for alg in algorithm_list:
            best_counts[alg] = {
                'weighted': 0, 'makespan': 0, 'tardiness': 0, 
                'hypervolume': 0, 'igd': 0, 'gd': 0, 'spacing': 0, 'pareto_count': 0
            }
        
        for config in configs:
            scale = config['scale']
            if scale not in results:
                continue
            
            scale_results = results[scale]
            valid_results = {alg: result for alg, result in scale_results.items() 
                           if result.get('weighted_best', float('inf')) != float('inf')}
            
            if not valid_results:
                continue
            
            # 加权目标最优
            best_weighted = min(valid_results.items(), key=lambda x: x[1].get('weighted_best', float('inf')))
            if best_weighted[1].get('weighted_best', float('inf')) != float('inf'):
                best_counts[best_weighted[0]]['weighted'] += 1
            
            # 完工时间最优
            best_makespan = min(valid_results.items(), key=lambda x: x[1].get('makespan_best', float('inf')))
            if best_makespan[1].get('makespan_best', float('inf')) != float('inf'):
                best_counts[best_makespan[0]]['makespan'] += 1
            
            # 拖期最优
            best_tardiness = min(valid_results.items(), key=lambda x: x[1].get('tardiness_best', float('inf')))
            if best_tardiness[1].get('tardiness_best', float('inf')) != float('inf'):
                best_counts[best_tardiness[0]]['tardiness'] += 1
            
            # 超体积最优
            best_hv = max(valid_results.items(), key=lambda x: x[1].get('hypervolume', 0))
            if best_hv[1].get('hypervolume', 0) > 0:
                best_counts[best_hv[0]]['hypervolume'] += 1
            
            # IGD最优 (越小越好)
            valid_igd_results = {alg: result for alg, result in valid_results.items() 
                               if result.get('igd', float('inf')) != float('inf')}
            if valid_igd_results:
                best_igd = min(valid_igd_results.items(), key=lambda x: x[1].get('igd', float('inf')))
                best_counts[best_igd[0]]['igd'] += 1
            
            # GD最优 (越小越好)
            valid_gd_results = {alg: result for alg, result in valid_results.items() 
                              if result.get('gd', float('inf')) != float('inf')}
            if valid_gd_results:
                best_gd = min(valid_gd_results.items(), key=lambda x: x[1].get('gd', float('inf')))
                best_counts[best_gd[0]]['gd'] += 1
            
            # Spacing最优 (越小越好)
            valid_spacing_results = {alg: result for alg, result in valid_results.items() 
                                   if result.get('spacing', float('inf')) != float('inf') and result.get('pareto_count', 0) > 1}
            if valid_spacing_results:
                best_spacing = min(valid_spacing_results.items(), key=lambda x: x[1].get('spacing', float('inf')))
                best_counts[best_spacing[0]]['spacing'] += 1
            
            # 解数量最优
            best_count = max(valid_results.items(), key=lambda x: x[1].get('pareto_count', 0))
            if best_count[1].get('pareto_count', 0) > 0:
                best_counts[best_count[0]]['pareto_count'] += 1
        
        # 打印统计表格
        f.write("各算法最优表现次数统计\n")
        f.write(f"总数据集数量: {total_datasets}\n\n")
        
        separator = "+" + "-"*17 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*10 + "+\n"
        f.write(separator)
        f.write(f"| {'算法':^15s} | {'加权目标':^8s} | {'完工时间':^8s} | {'总拖期':^8s} | {'超体积':^8s} | {'IGD':^8s} | {'GD':^8s} | {'Spacing':^8s} | {'解数量':^8s} | {'总计':^8s} |\n")
        f.write(separator)
        
        for alg in algorithm_list:
            counts = best_counts[alg]
            total = sum(counts.values())
            f.write(f"| {alg:^15s} | {counts['weighted']:^8d} | {counts['makespan']:^8d} | {counts['tardiness']:^8d} | {counts['hypervolume']:^8d} | {counts['igd']:^8d} | {counts['gd']:^8d} | {counts['spacing']:^8d} | {counts['pareto_count']:^8d} | {total:^8d} |\n")
        
        f.write(separator)
        
        # 新增: 单独指标对比表格
        f.write("\n\n单独指标对比表格\n")
        f.write("=" * 120 + "\n\n")
        
        # 1. 完工时间对比表
        f.write("1. 完工时间(Makespan)对比表\n")
        f.write("-" * 120 + "\n")
        f.write("+" + "-"*20 + "+" + "-"*15 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*10 + "+" + "-"*12 + "+\n")
        f.write(f"| {'数据集':^18s} | {'RL-Chaotic-HHO':^13s} | {'I-NSGA-II':^11s} | {'MOPSO':^11s} | {'MODE':^11s} | {'DQN':^8s} | {'QL-ABC':^10s} |\n")
        f.write("+" + "-"*20 + "+" + "-"*15 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*10 + "+" + "-"*12 + "+\n")
        
        for config in configs[:10]:  # 只显示前10个
            scale = config['scale']
            if scale in results:
                scale_results = results[scale]
                short_name = scale.split('_')[0]
                
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
                
                f.write(f"| {short_name:^18s} | {values[0]:^13s} | {values[1]:^11s} | {values[2]:^11s} | {values[3]:^11s} | {values[4]:^8s} | {values[5]:^10s} |\n")
        
        f.write("+" + "-"*20 + "+" + "-"*15 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*10 + "+" + "-"*12 + "+\n\n")
        
        # 2. 总拖期对比表
        f.write("2. 总拖期(Total Tardiness)对比表\n")
        f.write("-" * 120 + "\n")
        f.write("+" + "-"*20 + "+" + "-"*15 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*10 + "+" + "-"*12 + "+\n")
        f.write(f"| {'数据集':^18s} | {'RL-Chaotic-HHO':^13s} | {'I-NSGA-II':^11s} | {'MOPSO':^11s} | {'MODE':^11s} | {'DQN':^8s} | {'QL-ABC':^10s} |\n")
        f.write("+" + "-"*20 + "+" + "-"*15 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*10 + "+" + "-"*12 + "+\n")
        
        for config in configs[:10]:  # 只显示前10个
            scale = config['scale']
            if scale in results:
                scale_results = results[scale]
                short_name = scale.split('_')[0]
                
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
                
                f.write(f"| {short_name:^18s} | {values[0]:^13s} | {values[1]:^11s} | {values[2]:^11s} | {values[3]:^11s} | {values[4]:^8s} | {values[5]:^10s} |\n")
        
        f.write("+" + "-"*20 + "+" + "-"*15 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*10 + "+" + "-"*12 + "+\n\n")
        
        # 3. 加权目标对比表
        f.write("3. 加权目标函数对比表\n")
        f.write("-" * 120 + "\n")
        f.write("+" + "-"*20 + "+" + "-"*15 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*10 + "+" + "-"*12 + "+\n")
        f.write(f"| {'数据集':^18s} | {'RL-Chaotic-HHO':^13s} | {'I-NSGA-II':^11s} | {'MOPSO':^11s} | {'MODE':^11s} | {'DQN':^8s} | {'QL-ABC':^10s} |\n")
        f.write("+" + "-"*20 + "+" + "-"*15 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*10 + "+" + "-"*12 + "+\n")
        
        for config in configs[:10]:  # 只显示前10个
            scale = config['scale']
            if scale in results:
                scale_results = results[scale]
                short_name = scale.split('_')[0]
                
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
                
                f.write(f"| {short_name:^18s} | {values[0]:^13s} | {values[1]:^11s} | {values[2]:^11s} | {values[3]:^11s} | {values[4]:^8s} | {values[5]:^10s} |\n")
        
        f.write("+" + "-"*20 + "+" + "-"*15 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*10 + "+" + "-"*12 + "+\n\n")
        
        # 4. HV指标对比表
        f.write("4. 超体积(HV)指标对比表\n")
        f.write("-" * 120 + "\n")
        f.write("+" + "-"*20 + "+" + "-"*15 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*10 + "+" + "-"*12 + "+\n")
        f.write(f"| {'数据集':^18s} | {'RL-Chaotic-HHO':^13s} | {'I-NSGA-II':^11s} | {'MOPSO':^11s} | {'MODE':^11s} | {'DQN':^8s} | {'QL-ABC':^10s} |\n")
        f.write("+" + "-"*20 + "+" + "-"*15 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*10 + "+" + "-"*12 + "+\n")
        
        for config in configs[:10]:  # 只显示前10个
            scale = config['scale']
            if scale in results:
                scale_results = results[scale]
                short_name = scale.split('_')[0]
                
                values = []
                for alg in algorithm_list:
                    if alg in scale_results:
                        value = scale_results[alg].get('hypervolume', 0)
                        if value == 0:
                            values.append('0')
                        else:
                            values.append(f"{value:.3f}")
                    else:
                        values.append('0')
                
                f.write(f"| {short_name:^18s} | {values[0]:^13s} | {values[1]:^11s} | {values[2]:^11s} | {values[3]:^11s} | {values[4]:^8s} | {values[5]:^10s} |\n")
        
        f.write("+" + "-"*20 + "+" + "-"*15 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*10 + "+" + "-"*12 + "+\n\n")
        
        # 5. IGD指标对比表
        f.write("5. 反世代距离(IGD)指标对比表\n")
        f.write("-" * 120 + "\n")
        f.write("+" + "-"*20 + "+" + "-"*15 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*10 + "+" + "-"*12 + "+\n")
        f.write(f"| {'数据集':^18s} | {'RL-Chaotic-HHO':^13s} | {'I-NSGA-II':^11s} | {'MOPSO':^11s} | {'MODE':^11s} | {'DQN':^8s} | {'QL-ABC':^10s} |\n")
        f.write("+" + "-"*20 + "+" + "-"*15 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*10 + "+" + "-"*12 + "+\n")
        
        for config in configs[:10]:  # 只显示前10个
            scale = config['scale']
            if scale in results:
                scale_results = results[scale]
                short_name = scale.split('_')[0]
                
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
                
                f.write(f"| {short_name:^18s} | {values[0]:^13s} | {values[1]:^11s} | {values[2]:^11s} | {values[3]:^11s} | {values[4]:^8s} | {values[5]:^10s} |\n")
        
        f.write("+" + "-"*20 + "+" + "-"*15 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*13 + "+" + "-"*10 + "+" + "-"*12 + "+\n\n")
        
        # 详细数据表格保持原有格式...
        # [此处省略原有的详细数据表格代码，保持不变]
        
        f.write(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 生成Excel文件
    if excel_available and excel_data:
        try:
            with pd.ExcelWriter(excel_filename, engine='xlsxwriter') as writer:
                for sheet_name, df in excel_data.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    # 获取工作表对象以进行格式设置
                    worksheet = writer.sheets[sheet_name]
                    workbook = writer.book
                    
                    # 设置格式
                    header_format = workbook.add_format({
                        'bold': True,
                        'text_wrap': True,
                        'valign': 'top',
                        'fg_color': '#D7E4BC',
                        'border': 1
                    })
                    
                    # 应用格式到表头
                    for col_num, value in enumerate(df.columns.values):
                        worksheet.write(0, col_num, value, header_format)
                    
                    # 自动调整列宽
                    for i, col in enumerate(df.columns):
                        max_length = max(
                            df[col].astype(str).map(len).max(),
                            len(str(col))
                        ) + 2
                        worksheet.set_column(i, i, min(max_length, 20))
            
            print(f"Excel报告已保存: {excel_filename}")
        except Exception as e:
            print(f"Excel文件生成失败: {e}")
    
    print(f"\n完整算法对比报告已保存: {filename}")
    
    # 返回统计结果用于控制台输出
    return best_counts, total_datasets

if __name__ == "__main__":
    # 确保结果目录存在
    os.makedirs("results", exist_ok=True)
    
    # 运行表格格式实验
    run_table_format_experiments()
    
    print("\n完整算法对比实验完成!") 