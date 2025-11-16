#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三算法对比实验程序
对比RL-Chaotic-HHO、NSGA-II和MOEA/D在多目标分布式异构混合流水车间调度问题上的性能
"""

import os
import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple

from problem.mo_dhfsp import MO_DHFSP_Problem
from algorithm.rl_chaotic_hho import RL_ChaoticHHO_Optimizer
from algorithm.nsga2 import NSGA2_Optimizer
from algorithm.moead import MOEAD_Optimizer
from utils.data_generator import DataGenerator
from utils.performance_metrics import PerformanceEvaluator
from utils.visualization import ResultVisualizer

def print_dataset_info(problem_data: Dict, problem_name: str):
    """详细输出数据集信息"""
    print(f"\n{'='*80}")
    print(f"数据集详细信息: {problem_name}")
    print(f"{'='*80}")
    
    # 基本信息
    print(f"问题规模: {problem_data['n_jobs']}作业 × {problem_data['n_factories']}工厂 × {problem_data['n_stages']}阶段")
    print(f"机器配置: {problem_data['machines_per_stage']}")
    print(f"总机器数: {sum(problem_data['machines_per_stage'])}")
    
    # 处理时间信息
    processing_times = np.array(problem_data['processing_times'])
    print(f"\n处理时间统计:")
    print(f"  范围: [{processing_times.min():.2f}, {processing_times.max():.2f}]")
    print(f"  均值: {processing_times.mean():.2f}")
    print(f"  标准差: {processing_times.std():.2f}")
    
    # 显示处理时间矩阵
    print(f"\n处理时间矩阵 (作业×阶段):")
    print("作业\\阶段", end="")
    for stage in range(problem_data['n_stages']):
        print(f"{stage:>8}", end="")
    print()
    
    for job in range(min(10, problem_data['n_jobs'])):  # 只显示前10个作业
        print(f"作业{job:2d}", end="")
        for stage in range(problem_data['n_stages']):
            print(f"{processing_times[job][stage]:>8.2f}", end="")
        print()
    
    if problem_data['n_jobs'] > 10:
        print(f"... (共{problem_data['n_jobs']}个作业)")
    
    # 交货期信息
    due_dates = np.array(problem_data['due_dates'])
    print(f"\n交货期统计:")
    print(f"  范围: [{due_dates.min():.2f}, {due_dates.max():.2f}]")
    print(f"  均值: {due_dates.mean():.2f}")
    print(f"  标准差: {due_dates.std():.2f}")
    
    # 紧急度信息
    urgencies = np.array(problem_data['urgencies'])
    print(f"\n紧急度统计:")
    print(f"  范围: [{urgencies.min():.2f}, {urgencies.max():.2f}]")
    print(f"  均值: {urgencies.mean():.2f}")
    print(f"  标准差: {urgencies.std():.2f}")
    
    # 紧急度分布
    high_urgency = np.sum(urgencies >= 1.5)
    medium_urgency = np.sum((urgencies >= 1.0) & (urgencies < 1.5))
    low_urgency = np.sum(urgencies < 1.0)
    
    print(f"  分布: 高紧急度({high_urgency}个, {high_urgency/len(urgencies)*100:.1f}%), "
          f"中等紧急度({medium_urgency}个, {medium_urgency/len(urgencies)*100:.1f}%), "
          f"低紧急度({low_urgency}个, {low_urgency/len(urgencies)*100:.1f}%)")
    
    # 作业复杂度分析
    job_complexities = np.std(processing_times, axis=1) / np.mean(processing_times, axis=1)
    print(f"\n作业复杂度分析:")
    print(f"  复杂度范围: [{job_complexities.min():.3f}, {job_complexities.max():.3f}]")
    print(f"  平均复杂度: {job_complexities.mean():.3f}")
    
    # 阶段负载分析
    stage_loads = np.sum(processing_times, axis=0)
    print(f"\n阶段负载分析:")
    for stage in range(problem_data['n_stages']):
        load_per_machine = stage_loads[stage] / problem_data['machines_per_stage'][stage]
        print(f"  阶段{stage}: 总负载={stage_loads[stage]:.2f}, "
              f"机器数={problem_data['machines_per_stage'][stage]}, "
              f"平均负载={load_per_machine:.2f}")
    
    print(f"{'='*80}\n")

def run_algorithm_comparison():
    """运行三算法对比实验"""
    
    # 生成测试数据
    generator = DataGenerator(seed=42)
    
    # 中规模异构工厂异构机器配置
    problem_data = generator.generate_problem(
        n_jobs=20,
        n_factories=3,
        n_stages=3,
        machines_per_stage=[2, 3, 4],  # 异构机器配置
        processing_time_range=(1, 25),
        due_date_tightness=1.5
    )
    
    problem_name = "中规模异构工厂异构机器"
    
    # 输出数据集详细信息
    print_dataset_info(problem_data, problem_name)
    
    # 创建问题实例
    problem = MO_DHFSP_Problem(problem_data)
    
    # 算法配置
    algorithms = {
        'RL-Chaotic-HHO': {
            'class': RL_ChaoticHHO_Optimizer,
            'params': {
                'max_iterations': 50,
            }
        },
        'NSGA-II': {
            'class': NSGA2_Optimizer,
            'params': {
                'population_size': 100,
                'max_generations': 50,
                'crossover_prob': 0.9,
                'mutation_prob': 0.1
            }
        },
        'MOEA/D': {
            'class': MOEAD_Optimizer,
            'params': {
                'population_size': 100,
                'max_generations': 50,
                'crossover_prob': 0.9,
                'mutation_prob': 0.1,
                'neighbor_size': 10,
                'delta': 0.9,
                'nr': 2
            }
        }
    }
    
    # 运行算法并收集结果
    results = {}
    
    print("开始运行三算法对比实验...")
    print(f"问题配置: {problem_data['n_jobs']}作业, {problem_data['n_factories']}工厂, {problem_data['n_stages']}阶段")
    print(f"机器配置: {problem_data['machines_per_stage']}")
    print("-" * 80)
    
    for alg_name, alg_config in algorithms.items():
        print(f"\n正在运行 {alg_name}...")
        start_time = time.time()
        
        try:
            # 创建优化器
            optimizer = alg_config['class'](problem, **alg_config['params'])
            
            # 运行优化
            pareto_solutions, convergence_data = optimizer.optimize()
            
            end_time = time.time()
            runtime = end_time - start_time
            
            # 存储结果
            results[alg_name] = {
                'pareto_solutions': pareto_solutions,
                'convergence_data': convergence_data,
                'runtime': runtime,
                'optimizer': optimizer
            }
            
            print(f"{alg_name} 完成! 运行时间: {runtime:.2f}秒, 帕累托解数量: {len(pareto_solutions)}")
            
            if pareto_solutions:
                makespans = [sol.makespan for sol in pareto_solutions]
                tardiness = [sol.total_tardiness for sol in pareto_solutions]
                print(f"  完工时间范围: [{min(makespans):.2f}, {max(makespans):.2f}]")
                print(f"  总拖期范围: [{min(tardiness):.2f}, {max(tardiness):.2f}]")
            
        except Exception as e:
            print(f"{alg_name} 运行失败: {str(e)}")
            import traceback
            traceback.print_exc()
            results[alg_name] = None
    
    # 性能分析
    print("\n" + "="*80)
    print("性能分析结果")
    print("="*80)
    
    metrics_calculator = PerformanceEvaluator()
    
    # 收集所有解用于计算参考点
    all_solutions = []
    for alg_name, result in results.items():
        if result and result['pareto_solutions']:
            all_solutions.extend(result['pareto_solutions'])
    
    if all_solutions:
        # 计算参考点
        all_makespans = [sol.makespan for sol in all_solutions]
        all_tardiness = [sol.total_tardiness for sol in all_solutions]
        nadir_point = [max(all_makespans), max(all_tardiness)]
        ideal_point = [min(all_makespans), min(all_tardiness)]
        
        print(f"理想点: ({ideal_point[0]:.2f}, {ideal_point[1]:.2f})")
        print(f"负理想点: ({nadir_point[0]:.2f}, {nadir_point[1]:.2f})")
        print()
        
        # 计算各算法的性能指标
        performance_summary = {}
        
        for alg_name, result in results.items():
            if result and result['pareto_solutions']:
                solutions = result['pareto_solutions']
                
                # 计算性能指标
                igd = metrics_calculator.calculate_igd(solutions, all_solutions)
                hv = metrics_calculator.calculate_hypervolume(solutions, nadir_point)
                spacing = metrics_calculator.calculate_spacing(solutions)
                
                # 最优值
                makespans = [sol.makespan for sol in solutions]
                tardiness = [sol.total_tardiness for sol in solutions]
                best_makespan = min(makespans)
                best_tardiness = min(tardiness)
                
                performance_summary[alg_name] = {
                    'pareto_size': len(solutions),
                    'best_makespan': best_makespan,
                    'best_tardiness': best_tardiness,
                    'igd': igd,
                    'hypervolume': hv,
                    'spacing': spacing,
                    'runtime': result['runtime']
                }
                
                print(f"{alg_name:15s}: 帕累托解={len(solutions):2d}, "
                      f"最优完工时间={best_makespan:7.2f}, "
                      f"最优总拖期={best_tardiness:7.2f}")
                print(f"{'':15s}  IGD={igd:8.4f}, HV={hv:8.4f}, "
                      f"Spacing={spacing:6.4f}, 运行时间={result['runtime']:6.2f}s")
                print()
        
        # 生成可视化结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            results_for_viz = {name: res for name, res in results.items() if res is not None}
            
            # 生成单独的帕累托前沿对比图
            pareto_filename = f"results/{problem_name}_三算法帕累托前沿对比_{timestamp}.png"
            
            # 使用matplotlib直接绘制帕累托前沿对比图
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 9))
            
            # 定义颜色和标记
            colors = ['#FF1744', '#00E676', '#2196F3']
            markers = ['o', 's', '^']
            
            for i, (alg_name, result) in enumerate(results_for_viz.items()):
                if result and result['pareto_solutions']:
                    solutions = result['pareto_solutions']
                    makespans = [sol.makespan for sol in solutions]
                    tardiness = [sol.total_tardiness for sol in solutions]
                    
                    plt.scatter(makespans, tardiness, 
                               c=colors[i % len(colors)], 
                               marker=markers[i % len(markers)],
                               label=alg_name, 
                               alpha=0.8, 
                               s=100,
                               edgecolors='black',
                               linewidth=1.2)
            
            plt.xlabel('完工时间 (Makespan)', fontsize=14, fontweight='bold')
            plt.ylabel('总拖期 (Total Tardiness)', fontsize=14, fontweight='bold')
            plt.title(f'{problem_name} - 三算法帕累托前沿对比', fontsize=16, fontweight='bold')
            plt.legend(fontsize=13, loc='upper right', frameon=True, 
                      fancybox=True, shadow=True, borderpad=1)
            plt.grid(True, alpha=0.4, linestyle='--')
            plt.tick_params(axis='both', which='major', labelsize=12)
            plt.tight_layout()
            plt.savefig(pareto_filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"帕累托前沿对比图已保存: {pareto_filename}")
            
        except Exception as e:
            print(f"可视化生成失败: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # 生成文本报告
        report_filename = f"results/三算法对比报告_{timestamp}.txt"
        generate_comparison_report(results, performance_summary, problem_data, problem_name, report_filename)
        print(f"对比报告已保存: {report_filename}")
    
    else:
        print("没有找到有效的解，无法进行性能分析")

def generate_comparison_report(results: Dict, performance_summary: Dict, 
                             problem_data: Dict, problem_name: str, filename: str):
    """生成对比实验报告"""
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("三算法对比实验报告\n")
        f.write("=" * 50 + "\n\n")
        
        # 实验配置
        f.write("实验配置:\n")
        f.write(f"问题类型: 多目标分布式异构混合流水车间调度\n")
        f.write(f"问题规模: {problem_data['n_jobs']}作业 × {problem_data['n_factories']}工厂 × {problem_data['n_stages']}阶段\n")
        f.write(f"机器配置: {problem_data['machines_per_stage']}\n")
        f.write(f"优化目标: 完工时间 + 总拖期\n\n")
        
        # 算法参数
        f.write("算法参数:\n")
        f.write("RL-Chaotic-HHO: 最大迭代=50\n")
        f.write("NSGA-II: 种群=100, 代数=50, 交叉率=0.9, 变异率=0.1\n")
        f.write("MOEA/D: 种群=100, 代数=50, 交叉率=0.9, 变异率=0.1, 邻域=10\n\n")
        
        # 性能对比
        f.write("性能对比结果:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'算法':15s} {'帕累托解':>8s} {'最优完工时间':>12s} {'最优总拖期':>10s} {'IGD':>8s} {'HV':>8s} {'运行时间':>8s}\n")
        f.write("-" * 80 + "\n")
        
        for alg_name, metrics in performance_summary.items():
            f.write(f"{alg_name:15s} {metrics['pareto_size']:8d} "
                   f"{metrics['best_makespan']:12.2f} {metrics['best_tardiness']:10.2f} "
                   f"{metrics['igd']:8.4f} {metrics['hypervolume']:8.4f} "
                   f"{metrics['runtime']:8.2f}\n")
        
        f.write("-" * 80 + "\n\n")
        
        # 算法分析
        f.write("算法分析:\n")
        
        if performance_summary:
            # 找出最佳算法
            best_igd = min(metrics['igd'] for metrics in performance_summary.values())
            best_hv = max(metrics['hypervolume'] for metrics in performance_summary.values())
            best_makespan = min(metrics['best_makespan'] for metrics in performance_summary.values())
            best_tardiness = min(metrics['best_tardiness'] for metrics in performance_summary.values())
            
            for alg_name, metrics in performance_summary.items():
                f.write(f"\n{alg_name}:\n")
                
                strengths = []
                if metrics['igd'] == best_igd:
                    strengths.append("最佳IGD(收敛性)")
                if metrics['hypervolume'] == best_hv:
                    strengths.append("最佳超体积(多样性)")
                if metrics['best_makespan'] == best_makespan:
                    strengths.append("最佳完工时间")
                if metrics['best_tardiness'] == best_tardiness:
                    strengths.append("最佳总拖期")
                
                if strengths:
                    f.write(f"  优势: {', '.join(strengths)}\n")
                
                f.write(f"  帕累托解数量: {metrics['pareto_size']}\n")
                f.write(f"  收敛性(IGD): {metrics['igd']:.4f}\n")
                f.write(f"  多样性(HV): {metrics['hypervolume']:.4f}\n")
                f.write(f"  计算效率: {metrics['runtime']:.2f}秒\n")
        
        f.write(f"\n报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

if __name__ == "__main__":
    # 确保结果目录存在
    os.makedirs("results", exist_ok=True)
    
    # 运行对比实验
    run_algorithm_comparison()
    
    print("\n三算法对比实验完成!") 