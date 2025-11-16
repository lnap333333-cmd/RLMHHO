#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多目标算法对比实验
比较RL-Chaotic-HHO和NSGA-II算法在MO-DHFSP问题上的性能
包含详细的数据集信息输出
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple

from problem.mo_dhfsp import MO_DHFSP_Problem
from algorithm.rl_chaotic_hho import RL_ChaoticHHO_Optimizer
from algorithm.nsga2 import NSGA2_Optimizer
from utils.data_generator import DataGenerator
from utils.visualization import ResultVisualizer
from utils.performance_metrics import PerformanceEvaluator

def print_dataset_info(problem: MO_DHFSP_Problem, problem_name: str):
    """输出数据集详细信息"""
    print("=" * 80)
    print(f" {problem_name} 数据集详细信息")
    print("=" * 80)
    
    print("📊 问题规模:")
    print(f"   • 作业数量: {problem.n_jobs}")
    print(f"   • 工厂数量: {problem.n_factories}")
    print(f"   • 阶段数量: {problem.n_stages}")
    print(f"   • 总机器数量: {sum(problem.machines_per_stage)}")
    
    print(f"\n🏭 异构机器配置:")
    for stage in range(problem.n_stages):
        print(f"   • 阶段 {stage}: {problem.machines_per_stage[stage]} 台机器")
    print(f"   • 配置特点: {problem.n_factories} 个异构工厂，各阶段机器数量不等")
    
    # 显示部分处理时间矩阵
    print(f"\n⚙️ 处理时间矩阵 (作业 × 阶段):")
    print("     " + "  ".join([f"阶段 {i}" for i in range(problem.n_stages)]))
    
    # 显示前6个和后6个作业的处理时间
    jobs_to_show = min(6, problem.n_jobs)
    for job in range(jobs_to_show):
        times = [f"{problem.processing_times[job][stage]:5.1f}" for stage in range(problem.n_stages)]
        print(f"作业{job:2d}:" + "".join(times))
    
    if problem.n_jobs > 12:
        print("...")
        for job in range(problem.n_jobs-jobs_to_show, problem.n_jobs):
            times = [f"{problem.processing_times[job][stage]:5.1f}" for stage in range(problem.n_stages)]
            print(f"作业{job:2d}:" + "".join(times))
    elif problem.n_jobs > 6:
        for job in range(jobs_to_show, problem.n_jobs):
            times = [f"{problem.processing_times[job][stage]:5.1f}" for stage in range(problem.n_stages)]
            print(f"作业{job:2d}:" + "".join(times))
    
    # 显示交货期信息（简化显示）
    print(f"\n📅 交货期信息:")
    jobs_to_show = min(10, problem.n_jobs)
    for job in range(jobs_to_show):
        print(f"   • 作业 {job:2d}: 交货期 = {problem.due_dates[job]:7.1f}")
    if problem.n_jobs > jobs_to_show:
        print(f"   • ... (共{problem.n_jobs}个作业)")
    
    # 显示紧急度信息（简化显示）
    print(f"\n⚡ 紧急度信息:")
    urgencies = [problem.urgencies[job] for job in range(problem.n_jobs)]
    print(f"   • 紧急度范围: [{min(urgencies):.2f}, {max(urgencies):.2f}]")
    print(f"   • 平均紧急度: {np.mean(urgencies):.2f}")
    
    # 统计摘要
    all_times = [problem.processing_times[job][stage] 
                for job in range(problem.n_jobs) 
                for stage in range(problem.n_stages)]
    
    print(f"\n📈 统计摘要:")
    print(f"   • 平均处理时间: {np.mean(all_times):.2f}")
    print(f"   • 处理时间范围: [{min(all_times):.1f}, {max(all_times):.1f}]")
    print(f"   • 平均交货期: {np.mean(problem.due_dates):.2f}")
    print(f"   • 交货期范围: [{min(problem.due_dates):.1f}, {max(problem.due_dates):.1f}]")
    
    # 计算理论下界
    min_stage_times = [min(problem.processing_times[job][stage] for job in range(problem.n_jobs)) 
                      for stage in range(problem.n_stages)]
    theoretical_lower_bound = sum(min_stage_times) * problem.n_jobs / max(problem.machines_per_stage)
    print(f"   • 理论完工时间下界: {theoretical_lower_bound:.2f}")
    print("=" * 80)

def calculate_weighted_objective(makespan: float, tardiness: float) -> float:
    """计算加权目标函数值 F = 0.55*F1 + 0.45*F2"""
    return 0.55 * makespan + 0.45 * tardiness

def generate_custom_urgencies(n_jobs: int, urgency_range: List[float]) -> np.ndarray:
    """
    根据指定范围生成紧急度
    
    Args:
        n_jobs: 作业数量
        urgency_range: [最小值, 平均值, 最大值]
    
    Returns:
        紧急度数组
    """
    min_urgency, mean_urgency, max_urgency = urgency_range
    
    # 生成正态分布的紧急度，但限制在指定范围内
    urgencies = np.random.normal(mean_urgency, (max_urgency - min_urgency) / 6, n_jobs)
    
    # 裁剪到指定范围
    urgencies = np.clip(urgencies, min_urgency, max_urgency)
    
    # 确保有一些值接近边界值
    n_min = max(1, n_jobs // 10)
    n_max = max(1, n_jobs // 10)
    
    # 设置一些最小值
    min_indices = np.random.choice(n_jobs, n_min, replace=False)
    urgencies[min_indices] = np.random.uniform(min_urgency, min_urgency + 0.1 * (max_urgency - min_urgency))
    
    # 设置一些最大值
    remaining_indices = [i for i in range(n_jobs) if i not in min_indices]
    max_indices = np.random.choice(remaining_indices, min(n_max, len(remaining_indices)), replace=False)
    urgencies[max_indices] = np.random.uniform(max_urgency - 0.1 * (max_urgency - min_urgency), max_urgency)
    
    return urgencies

def run_comparison_experiment():
    """运行对比实验"""
    print("🚀 开始多目标算法对比实验")
    print(f"⏰ 实验开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 实验配置 - 参考表格中的规模设置
    test_configs = [
        # 小规模测试
        {'name': '20×5×3', 'n_jobs': 20, 'n_factories': 5, 'n_stages': 3, 'seed': 100, 'urgency_range': [0.9, 1.9, 2.9]},
        {'name': '20×5×4', 'n_jobs': 20, 'n_factories': 5, 'n_stages': 4, 'seed': 101, 'urgency_range': [0.8, 1.8, 2.8]},
        # 中规模测试
        {'name': '50×5×3', 'n_jobs': 50, 'n_factories': 5, 'n_stages': 3, 'seed': 200, 'urgency_range': [2.45, 3.45, 4.45]},
        {'name': '50×5×4', 'n_jobs': 50, 'n_factories': 5, 'n_stages': 4, 'seed': 201, 'urgency_range': [2, 3, 4]},
        # 大规模测试
        {'name': '100×5×3', 'n_jobs': 100, 'n_factories': 5, 'n_stages': 3, 'seed': 300, 'urgency_range': [6.4, 7.4, 8.4]},
        {'name': '100×5×4', 'n_jobs': 100, 'n_factories': 5, 'n_stages': 4, 'seed': 301, 'urgency_range': [4.95, 5.95, 6.95]},
    ]
    
    # 算法参数
    algorithm_params = {
        'max_iterations': 50,  # 适当减少迭代次数以加快实验速度
        'max_generations': 50,
        'population_size': 50,
    }
    
    # 存储所有实验结果
    all_results = {}
    summary_results = []
    
    for config in test_configs:
        print(f"\n{'='*80}")
        print(f"🧪 测试配置: {config['name']}")
        print(f"{'='*80}")
        
        # 生成测试数据
        generator = DataGenerator(seed=config['seed'])
        
        # 根据阶段数设置机器配置
        if config['n_stages'] == 3:
            machines_per_stage = [3, 4, 5]
        else:  # n_stages == 4
            machines_per_stage = [2, 3, 4, 3]
        
        problem_data = generator.generate_problem(
            n_jobs=config['n_jobs'],
            n_factories=config['n_factories'],
            n_stages=config['n_stages'],
            machines_per_stage=machines_per_stage,
            processing_time_range=(1, 30),
            due_date_tightness=1.5  # 参考Luo S论文的紧急度设置
        )
        
        # 使用指定的紧急度范围覆盖生成的紧急度
        urgency_range = config['urgency_range']
        custom_urgencies = generate_custom_urgencies(config['n_jobs'], urgency_range)
        problem_data['urgencies'] = custom_urgencies
        
        # 创建问题实例
        problem = MO_DHFSP_Problem(problem_data)
        
        # 输出数据集信息（简化版）
        print_dataset_info(problem, config['name'])
        
        # 使用指定的紧急度DDT统计
        urgency_stats = f"[{config['urgency_range'][0]}, {config['urgency_range'][1]}, {config['urgency_range'][2]}]"
        
        # 存储本次实验结果
        results = {
            'problem': problem,
            'algorithms': {},
            'config': config,
            'urgency_stats': urgency_stats
        }
        
        # 测试算法列表
        algorithms = [
            ('RL-Chaotic-HHO', RL_ChaoticHHO_Optimizer),
            ('NSGA-II', NSGA2_Optimizer)
        ]
        
        for alg_name, AlgorithmClass in algorithms:
            print(f"\n🔬 运行 {alg_name} 算法...")
            
            try:
                # 创建优化器
                optimizer = AlgorithmClass(problem, **algorithm_params)
                
                # 运行优化
                start_time = time.time()
                pareto_solutions, convergence_data = optimizer.optimize()
                end_time = time.time()
                
                execution_time = end_time - start_time
                
                if pareto_solutions:
                    # 计算加权目标函数值
                    weighted_objectives = []
                    for sol in pareto_solutions:
                        weighted_obj = calculate_weighted_objective(sol.makespan, sol.total_tardiness)
                        weighted_objectives.append(weighted_obj)
                    
                    # 统计结果
                    best_weighted = min(weighted_objectives)
                    mean_weighted = np.mean(weighted_objectives)
                    
                    # 原始目标值统计
                    makespans = [sol.makespan for sol in pareto_solutions]
                    tardiness = [sol.total_tardiness for sol in pareto_solutions]
                    
                    best_makespan = min(makespans)
                    best_tardiness = min(tardiness)
                    
                else:
                    best_weighted = float('inf')
                    mean_weighted = float('inf')
                    best_makespan = float('inf')
                    best_tardiness = float('inf')
                
                # 存储结果
                results['algorithms'][alg_name] = {
                    'pareto_solutions': pareto_solutions,
                    'convergence_data': convergence_data,
                    'execution_time': execution_time,
                    'best_weighted': best_weighted,
                    'mean_weighted': mean_weighted,
                    'best_makespan': best_makespan,
                    'best_tardiness': best_tardiness
                }
                
                # 输出算法性能
                print(f"   ✅ {alg_name} 运行完成!")
                print(f"   ⏱️  运行时间: {execution_time:.2f}秒")
                print(f"   📊 帕累托解数量: {len(pareto_solutions)}")
                if pareto_solutions:
                    print(f"   🎯 最优加权目标: {best_weighted:.2f}")
                    print(f"   📈 平均加权目标: {mean_weighted:.2f}")
                    print(f"   🔥 最优完工时间: {best_makespan:.2f}")
                    print(f"   ⏰ 最优总拖期: {best_tardiness:.2f}")
                else:
                    print(f"   ⚠️  未找到可行解")
                    
            except Exception as e:
                print(f"   ❌ {alg_name} 运行出错: {str(e)}")
                results['algorithms'][alg_name] = {
                    'pareto_solutions': [],
                    'execution_time': 0.0,
                    'best_weighted': float('inf'),
                    'mean_weighted': float('inf'),
                    'best_makespan': float('inf'),
                    'best_tardiness': float('inf')
                }
        
        all_results[config['name']] = results
        
        # 收集汇总结果
        for alg_name in algorithms:
            alg_name_str = alg_name[0]
            alg_data = results['algorithms'][alg_name_str]
            summary_results.append({
                'scale': config['name'],
                'urgency_ddt': urgency_stats,
                'algorithm': alg_name_str,
                'best_weighted': alg_data['best_weighted'],
                'mean_weighted': alg_data['mean_weighted'],
                'execution_time': alg_data['execution_time']
            })
    
    # 生成表格格式的综合报告
    generate_table_report(summary_results)
    
    print(f"\n✅ 对比实验完成!")
    print(f"⏰ 实验结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def generate_table_report(summary_results: List[Dict]):
    """生成表格格式的实验报告"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_filename = f"results/table_comparison_report_{timestamp}.txt"
    
    # 按规模和算法组织数据
    scales = ['20×5×3', '20×5×4', '50×5×3', '50×5×4', '100×5×3', '100×5×4']
    algorithms = ['RL-Chaotic-HHO', 'NSGA-II']
    
    # 创建数据字典
    data_dict = {}
    urgency_dict = {}
    
    for result in summary_results:
        scale = result['scale']
        algorithm = result['algorithm']
        
        if scale not in data_dict:
            data_dict[scale] = {}
            urgency_dict[scale] = result['urgency_ddt']
        
        data_dict[scale][algorithm] = (
            result['best_weighted'],
            result['mean_weighted'], 
            result['execution_time']
        )
    
    # 生成报告
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write("=" * 120 + "\n")
        f.write("多目标分布式混合流水车间调度算法对比实验表格报告\n")
        f.write("=" * 120 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("目标函数: F = 0.55*F1 + 0.45*F2 (F1=完工时间, F2=总拖期)\n")
        f.write("结果格式: (最优解, 均值, 收敛时间)\n\n")
        
        # 表头
        f.write(f"{'规模':<12} {'紧急度DDT':<20} {'RL-Chaotic-HHO算法':<35} {'NSGA-II算法':<35}\n")
        f.write("-" * 120 + "\n")
        
        # 数据行
        for scale in scales:
            if scale in data_dict:
                urgency = urgency_dict.get(scale, "N/A")
                
                rlhho_data = data_dict[scale].get('RL-Chaotic-HHO', (float('inf'), float('inf'), 0))
                nsga2_data = data_dict[scale].get('NSGA-II', (float('inf'), float('inf'), 0))
                
                rlhho_str = f"({rlhho_data[0]:.1f},{rlhho_data[1]:.1f},{rlhho_data[2]:.2f})"
                nsga2_str = f"({nsga2_data[0]:.1f},{nsga2_data[1]:.1f},{nsga2_data[2]:.2f})"
                
                f.write(f"{scale:<12} {urgency:<20} {rlhho_str:<35} {nsga2_str:<35}\n")
        
        f.write("\n" + "=" * 120 + "\n")
        f.write("说明:\n")
        f.write("- 规模格式: 作业数×工厂数×阶段数\n")
        f.write("- 紧急度DDT: [最小值, 平均值, 最大值]\n")
        f.write("- 算法结果: (最优加权目标值, 平均加权目标值, 收敛时间(秒))\n")
        f.write("- 目标函数采用加权组合: F = 0.55*完工时间 + 0.45*总拖期\n")
    
    print(f"\n📋 表格格式报告已保存: {report_filename}")
    
    # 同时在控制台输出表格
    print(f"\n📊 实验结果汇总表:")
    print("=" * 120)
    print("多目标分布式混合流水车间调度算法对比实验结果")
    print("目标函数: F = 0.55*F1 + 0.45*F2 (F1=完工时间, F2=总拖期)")
    print("结果格式: (最优解, 均值, 收敛时间)")
    print("-" * 120)
    print(f"{'规模':<12} {'紧急度DDT':<20} {'RL-Chaotic-HHO算法':<35} {'NSGA-II算法':<35}")
    print("-" * 120)
    
    for scale in scales:
        if scale in data_dict:
            urgency = urgency_dict.get(scale, "N/A")
            
            rlhho_data = data_dict[scale].get('RL-Chaotic-HHO', (float('inf'), float('inf'), 0))
            nsga2_data = data_dict[scale].get('NSGA-II', (float('inf'), float('inf'), 0))
            
            rlhho_str = f"({rlhho_data[0]:.1f},{rlhho_data[1]:.1f},{rlhho_data[2]:.2f})"
            nsga2_str = f"({nsga2_data[0]:.1f},{nsga2_data[1]:.1f},{nsga2_data[2]:.2f})"
            
            print(f"{scale:<12} {urgency:<20} {rlhho_str:<35} {nsga2_str:<35}")
    
    print("=" * 120)

if __name__ == "__main__":
    # 设置中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 运行对比实验
    run_comparison_experiment() 