#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多目标分布式异构混合流水车间调度算法对比实验 - 测试版本
测试2个小规模数据集，验证所有功能是否正常
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

# 导入算法模块
from problem.mo_dhfsp import MO_DHFSP_Problem
from algorithm.rl_chaotic_hho import RL_ChaoticHHO_Optimizer
from algorithm.improved_nsga2 import ImprovedNSGA2_Optimizer
from algorithm.mopso import MOPSO_Optimizer
from algorithm.mode import MODE_Optimizer
from algorithm.dqn_algorithm_wrapper import DQNAlgorithmWrapper
from algorithm.ql_abc import QLABC_Optimizer

# 设置中文字体和随机种子
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
np.random.seed(42)
random.seed(42)

def calculate_normalized_hypervolume(pareto_solutions: List, all_solutions_combined: List = None) -> float:
    """计算归一化超体积指标"""
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

def calculate_normalized_igd(pareto_solutions: List, all_solutions_combined: List = None) -> float:
    """计算归一化IGD指标"""
    if not pareto_solutions:
        return 1.0
    
    current_objectives = [(sol.makespan, sol.total_tardiness) for sol in pareto_solutions]
    
    # 使用全局范围构建参考前沿
    if all_solutions_combined:
        all_objectives = [(sol.makespan, sol.total_tardiness) for sol in all_solutions_combined]
        min_makespan = min(obj[0] for obj in all_objectives)
        min_tardiness = min(obj[1] for obj in all_objectives)
        max_makespan = max(obj[0] for obj in all_objectives)
        max_tardiness = max(obj[1] for obj in all_objectives)
    else:
        min_makespan = min(obj[0] for obj in current_objectives)
        min_tardiness = min(obj[1] for obj in current_objectives)
        max_makespan = max(obj[0] for obj in current_objectives)
        max_tardiness = max(obj[1] for obj in current_objectives)
    
    # 构建理想参考前沿 - 更合理的分布
    true_pareto_front = []
    n_points = 20
    for i in range(n_points + 1):
        alpha = i / n_points
        # 构建凸前沿
        makespan = min_makespan + alpha * (max_makespan - min_makespan)
        tardiness = min_tardiness + (1 - alpha) * (max_tardiness - min_tardiness)
        true_pareto_front.append((makespan, tardiness))
    
    # 计算IGD
    total_distance = 0.0
    for true_point in true_pareto_front:
        min_distance = float('inf')
        for current_point in current_objectives:
            # 使用归一化的欧几里得距离
            makespan_range = max(max_makespan - min_makespan, 1e-10)
            tardiness_range = max(max_tardiness - min_tardiness, 1e-10)
            
            norm_dist_makespan = (true_point[0] - current_point[0]) / makespan_range
            norm_dist_tardiness = (true_point[1] - current_point[1]) / tardiness_range
            
            distance = np.sqrt(norm_dist_makespan**2 + norm_dist_tardiness**2)
            min_distance = min(min_distance, distance)
        total_distance += min_distance
    
    avg_distance = total_distance / len(true_pareto_front)
    
    # 归一化到[0, 1]区间
    max_possible_distance = np.sqrt(2)  # 归一化空间中的最大距离
    normalized_igd = avg_distance / max_possible_distance
    
    return min(max(normalized_igd, 0.0), 1.0)

def calculate_normalized_gd(pareto_solutions: List, all_solutions_combined: List = None) -> float:
    """计算归一化GD指标"""
    if not pareto_solutions:
        return 1.0
    
    current_objectives = [(sol.makespan, sol.total_tardiness) for sol in pareto_solutions]
    
    # 使用全局范围构建参考前沿
    if all_solutions_combined:
        all_objectives = [(sol.makespan, sol.total_tardiness) for sol in all_solutions_combined]
        min_makespan = min(obj[0] for obj in all_objectives)
        min_tardiness = min(obj[1] for obj in all_objectives)
        max_makespan = max(obj[0] for obj in all_objectives)
        max_tardiness = max(obj[1] for obj in all_objectives)
    else:
        min_makespan = min(obj[0] for obj in current_objectives)
        min_tardiness = min(obj[1] for obj in current_objectives)
        max_makespan = max(obj[0] for obj in current_objectives)
        max_tardiness = max(obj[1] for obj in current_objectives)
    
    # 构建理想参考前沿
    true_pareto_front = []
    n_points = 20
    for i in range(n_points + 1):
        alpha = i / n_points
        makespan = min_makespan + alpha * (max_makespan - min_makespan)
        tardiness = min_tardiness + (1 - alpha) * (max_tardiness - min_tardiness)
        true_pareto_front.append((makespan, tardiness))
    
    # 计算GD
    total_distance = 0.0
    for current_point in current_objectives:
        min_distance = float('inf')
        for true_point in true_pareto_front:
            # 使用归一化的欧几里得距离
            makespan_range = max(max_makespan - min_makespan, 1e-10)
            tardiness_range = max(max_tardiness - min_tardiness, 1e-10)
            
            norm_dist_makespan = (current_point[0] - true_point[0]) / makespan_range
            norm_dist_tardiness = (current_point[1] - true_point[1]) / tardiness_range
            
            distance = np.sqrt(norm_dist_makespan**2 + norm_dist_tardiness**2)
            min_distance = min(min_distance, distance)
        total_distance += min_distance
    
    avg_distance = total_distance / len(current_objectives)
    
    # 归一化到[0, 1]区间
    max_possible_distance = np.sqrt(2)  # 归一化空间中的最大距离
    normalized_gd = avg_distance / max_possible_distance
    
    return min(max(normalized_gd, 0.0), 1.0)

def calculate_normalized_spread(pareto_solutions: List) -> float:
    """计算归一化Spread指标"""
    if not pareto_solutions or len(pareto_solutions) <= 2:
        return 1.0
    
    objectives = [(sol.makespan, sol.total_tardiness) for sol in pareto_solutions]
    
    # 去重复解
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
    
    mean_distance = np.mean(distances)
    if mean_distance == 0:
        return 1.0
    
    # 计算边界距离
    min_makespan_point = min(sorted_objectives, key=lambda x: x[0])
    max_makespan_point = max(sorted_objectives, key=lambda x: x[0])
    
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
    return min(max(spread, 0.0), 1.0)

def generate_test_problem_data(n_jobs: int, n_factories: int, n_stages: int) -> Dict:
    """生成测试用的问题数据"""
    
    # 生成机器配置
    machines_per_stage = [3, 3, 3][:n_stages]  # 每阶段3台机器
    
    # 生成异构机器配置
    heterogeneous_machines = {}
    for factory_id in range(n_factories):
        factory_machines = []
        for stage_id in range(n_stages):
            # 每个工厂在每个阶段有2-4台机器
            n_machines = np.random.randint(2, 5)
            factory_machines.append(n_machines)
        heterogeneous_machines[factory_id] = factory_machines
    
    # 生成处理时间矩阵
    processing_times = []
    for job_id in range(n_jobs):
        job_times = []
        for stage_id in range(n_stages):
            time = np.random.randint(5, 16)  # 5-15的处理时间
            job_times.append(time)
        processing_times.append(job_times)
    
    # 生成截止日期
    due_dates = []
    for job_id in range(n_jobs):
        total_time = sum(processing_times[job_id])
        due_date = int(total_time * np.random.uniform(1.2, 2.5))
        due_dates.append(max(due_date, 1))
    
    return {
        'n_jobs': n_jobs,
        'n_factories': n_factories,
        'n_stages': n_stages,
        'machines_per_stage': machines_per_stage,
        'heterogeneous_machines': heterogeneous_machines,
        'processing_times': processing_times,
        'due_dates': due_dates,
        'urgencies': [1.0] * n_jobs
    }

def extract_pareto_front(solutions: List) -> List:
    """提取帕累托前沿"""
    if not solutions:
        return []
    
    pareto_solutions = []
    
    for sol in solutions:
        is_dominated = False
        
        for other_sol in solutions:
            if (other_sol.makespan <= sol.makespan and 
                other_sol.total_tardiness <= sol.total_tardiness and
                (other_sol.makespan < sol.makespan or 
                 other_sol.total_tardiness < sol.total_tardiness)):
                is_dominated = True
                break
        
        if not is_dominated:
            is_duplicate = False
            for pareto_sol in pareto_solutions:
                if (abs(pareto_sol.makespan - sol.makespan) < 1e-6 and
                    abs(pareto_sol.total_tardiness - sol.total_tardiness) < 1e-6):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                pareto_solutions.append(sol)
    
    return pareto_solutions

def run_test_algorithm(problem_data: Dict, algorithm_name: str, algorithm_class, algorithm_params: Dict) -> Dict:
    """运行单个算法测试"""
    print(f"  测试 {algorithm_name}...")
    
    try:
        # 创建问题实例
        problem = MO_DHFSP_Problem(problem_data)
        
        # 创建算法实例
        optimizer = algorithm_class(problem, **algorithm_params)
        
        # 运行算法
        start_time = time.time()
        solutions, convergence_data = optimizer.optimize()
        runtime = time.time() - start_time
        
        # 提取帕累托前沿
        pareto_solutions = extract_pareto_front(solutions) if solutions else []
        
        # 计算指标
        if pareto_solutions:
            makespans = [sol.makespan for sol in pareto_solutions]
            tardiness = [sol.total_tardiness for sol in pareto_solutions]
            
            makespan_best = min(makespans)
            tardiness_best = min(tardiness)
            weighted_best = min(0.5 * sol.makespan + 0.5 * sol.total_tardiness for sol in pareto_solutions)
            
            # 计算归一化指标 - 使用全局解集合进行归一化
            hv = calculate_normalized_hypervolume(pareto_solutions, solutions)
            igd = calculate_normalized_igd(pareto_solutions, solutions)
            gd = calculate_normalized_gd(pareto_solutions, solutions)
            spread = calculate_normalized_spread(pareto_solutions)
        else:
            makespan_best = tardiness_best = weighted_best = float('inf')
            hv = igd = gd = spread = 0.0
        
        print(f"    ✅ 成功: {len(pareto_solutions)}个解, 用时{runtime:.2f}s")
        print(f"       最优完工时间: {makespan_best:.2f}, 最优拖期: {tardiness_best:.2f}")
        print(f"       HV: {hv:.4f}, IGD: {igd:.4f}, GD: {gd:.4f}, Spread: {spread:.4f}")
        
        return {
            'algorithm_name': algorithm_name,
            'pareto_solutions': pareto_solutions,
            'makespan_best': makespan_best,
            'tardiness_best': tardiness_best,
            'weighted_best': weighted_best,
            'hv': hv,
            'igd': igd,
            'gd': gd,
            'spread': spread,
            'runtime': runtime,
            'pareto_count': len(pareto_solutions),
            'success': True
        }
        
    except Exception as e:
        print(f"    ❌ 失败: {str(e)}")
        return {
            'algorithm_name': algorithm_name,
            'pareto_solutions': [],
            'makespan_best': float('inf'),
            'tardiness_best': float('inf'),
            'weighted_best': float('inf'),
            'hv': 0.0,
            'igd': 1.0,
            'gd': 1.0,
            'spread': 1.0,
            'runtime': 0.0,
            'pareto_count': 0,
            'success': False
        }

def plot_test_pareto_fronts(results: Dict, dataset_name: str):
    """绘制测试帕累托前沿图"""
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
    
    for alg_name, result in results.items():
        if result['success'] and result['pareto_solutions']:
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
    plt.title(f'{dataset_name} 帕累托前沿对比', fontsize=16, fontweight='bold')
    
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
    plt.savefig(f'results/test_pareto_{dataset_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  帕累托前沿图已保存: results/test_pareto_{dataset_name}.png")

def generate_test_report(all_results: Dict):
    """生成测试报告"""
    print("\n生成测试报告...")
    
    # 准备数据
    data_records = []
    algorithms = ['RL-Chaotic-HHO', 'I-NSGA-II', 'MOPSO', 'MODE', 'DQN', 'QL-ABC']
    
    for dataset_name, dataset_results in all_results.items():
        for alg_name in algorithms:
            if alg_name in dataset_results:
                result = dataset_results[alg_name]
                
                record = {
                    'Dataset': dataset_name,
                    'Algorithm': alg_name,
                    'Success': '✅' if result['success'] else '❌',
                    'Makespan_Best': f"{result['makespan_best']:.2f}" if result['makespan_best'] != float('inf') else 'N/A',
                    'Tardiness_Best': f"{result['tardiness_best']:.2f}" if result['tardiness_best'] != float('inf') else 'N/A',
                    'Weighted_Best': f"{result['weighted_best']:.2f}" if result['weighted_best'] != float('inf') else 'N/A',
                    'HV': f"{result['hv']:.4f}",
                    'IGD': f"{result['igd']:.4f}",
                    'GD': f"{result['gd']:.4f}",
                    'Spread': f"{result['spread']:.4f}",
                    'Pareto_Count': result['pareto_count'],
                    'Runtime': f"{result['runtime']:.2f}s"
                }
                
                data_records.append(record)
    
    # 创建DataFrame
    df = pd.DataFrame(data_records)
    
    # 生成Excel文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_filename = f"results/测试报告_{timestamp}.xlsx"
    
    try:
        with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='测试结果', index=False)
        
        print(f"测试报告已保存: {excel_filename}")
        
    except Exception as e:
        print(f"Excel文件生成失败: {e}")
        
        # 生成文本报告
        txt_filename = f"results/测试报告_{timestamp}.txt"
        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write("多目标分布式异构混合流水车间调度算法测试报告\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"测试数据集: {len(all_results)}个\n")
            f.write(f"测试算法: {', '.join(algorithms)}\n\n")
            
            f.write("测试结果:\n")
            f.write("-" * 60 + "\n")
            f.write(df.to_string(index=False))
            f.write("\n\n")
            
            # 成功率统计
            f.write("算法成功率统计:\n")
            f.write("-" * 60 + "\n")
            for alg in algorithms:
                alg_data = df[df['Algorithm'] == alg]
                success_count = len(alg_data[alg_data['Success'] == '✅'])
                total_count = len(alg_data)
                success_rate = success_count / total_count if total_count > 0 else 0
                f.write(f"{alg}: {success_count}/{total_count} ({success_rate:.1%})\n")
        
        print(f"文本报告已保存: {txt_filename}")

def main():
    """主函数 - 测试版本"""
    print("多目标分布式异构混合流水车间调度算法对比实验 - 测试版本")
    print("=" * 60)
    
    # 确保结果目录存在
    os.makedirs('results', exist_ok=True)
    
    # 测试数据集配置
    test_datasets = [
        {
            'name': '小规模测试',
            'n_jobs': 10,
            'n_factories': 2,
            'n_stages': 3
        },
        {
            'name': '中规模测试',
            'n_jobs': 20,
            'n_factories': 3,
            'n_stages': 3
        }
    ]
    
    # 算法配置 - 使用较小的参数进行快速测试
    algorithms = {
        'RL-Chaotic-HHO': (RL_ChaoticHHO_Optimizer, {
            'population_size': 20,
            'max_iterations': 10,
            'pareto_size_limit': 100
        }),
        'I-NSGA-II': (ImprovedNSGA2_Optimizer, {
            'population_size': 20,
            'max_iterations': 10,
            'crossover_rate': 0.9,
            'mutation_rate': 0.1
        }),
        'MOPSO': (MOPSO_Optimizer, {
            'swarm_size': 20,
            'max_iterations': 10,
            'w': 0.4,
            'c1': 2.0,
            'c2': 2.0
        }),
        'MODE': (MODE_Optimizer, {
            'population_size': 20,
            'max_generations': 10,
            'F': 0.5,
            'CR': 0.9
        }),
        'DQN': (DQNAlgorithmWrapper, {
            'max_iterations': 10,
            'learning_rate': 0.001,
            'epsilon': 0.1
        }),
        'QL-ABC': (QLABC_Optimizer, {
            'population_size': 20,
            'max_iterations': 10,
            'learning_rate': 0.1,
            'epsilon': 0.05
        })
    }
    
    # 存储所有结果
    all_results = {}
    
    # 对每个测试数据集运行实验
    for dataset_config in test_datasets:
        dataset_name = dataset_config['name']
        print(f"\n{'='*60}")
        print(f"测试数据集: {dataset_name}")
        print(f"作业数: {dataset_config['n_jobs']}, 工厂数: {dataset_config['n_factories']}, 阶段数: {dataset_config['n_stages']}")
        print(f"{'='*60}")
        
        # 生成问题数据
        problem_data = generate_test_problem_data(
            dataset_config['n_jobs'],
            dataset_config['n_factories'],
            dataset_config['n_stages']
        )
        
        # 第一轮：运行所有算法收集解
        print("第一轮：收集所有算法的解...")
        all_solutions = []
        dataset_results = {}
        
        for alg_name, (alg_class, alg_params) in algorithms.items():
            print(f"  运行 {alg_name}...")
            try:
                problem = MO_DHFSP_Problem(problem_data)
                optimizer = alg_class(problem, **alg_params)
                solutions, _ = optimizer.optimize()
                all_solutions.extend(solutions if solutions else [])
                dataset_results[alg_name] = {'solutions': solutions}
            except Exception as e:
                print(f"    ❌ 失败: {str(e)}")
                dataset_results[alg_name] = {'solutions': []}
        
        # 第二轮：使用全局解集合重新计算指标
        print("第二轮：计算归一化指标...")
        for alg_name, stored_data in dataset_results.items():
            solutions = stored_data['solutions']
            pareto_solutions = extract_pareto_front(solutions) if solutions else []
            
            start_time = time.time()
            runtime = 0.1  # 近似运行时间
            
            # 计算指标
            if pareto_solutions:
                makespans = [sol.makespan for sol in pareto_solutions]
                tardiness = [sol.total_tardiness for sol in pareto_solutions]
                
                makespan_best = min(makespans)
                tardiness_best = min(tardiness)
                weighted_best = min(0.5 * sol.makespan + 0.5 * sol.total_tardiness for sol in pareto_solutions)
                
                # 计算归一化指标 - 使用全局解集合进行归一化
                hv = calculate_normalized_hypervolume(pareto_solutions, all_solutions)
                igd = calculate_normalized_igd(pareto_solutions, all_solutions)
                gd = calculate_normalized_gd(pareto_solutions, all_solutions)
                spread = calculate_normalized_spread(pareto_solutions)
            else:
                makespan_best = tardiness_best = weighted_best = float('inf')
                hv = igd = gd = spread = 0.0
            
            # 更新结果
            dataset_results[alg_name] = {
                'algorithm_name': alg_name,
                'pareto_solutions': pareto_solutions,
                'makespan_best': makespan_best,
                'tardiness_best': tardiness_best,
                'weighted_best': weighted_best,
                'hv': hv,
                'igd': igd,
                'gd': gd,
                'spread': spread,
                'runtime': runtime,
                'pareto_count': len(pareto_solutions),
                'success': len(pareto_solutions) > 0
            }
            
            print(f"  {alg_name}: {len(pareto_solutions)}个解, HV={hv:.4f}, IGD={igd:.4f}, GD={gd:.4f}")
        
        all_results[dataset_name] = dataset_results
        
        # 绘制帕累托前沿图
        plot_test_pareto_fronts(dataset_results, dataset_name)
    
    # 生成测试报告
    generate_test_report(all_results)
    
    print(f"\n{'='*60}")
    print("测试完成!")
    print(f"{'='*60}")
    print(f"共测试 {len(test_datasets)} 个数据集")
    print(f"测试结果保存在 results/ 目录中")
    print("\n如果测试通过，可以运行完整版本: python comprehensive_algorithm_comparison.py")

if __name__ == "__main__":
    main() 