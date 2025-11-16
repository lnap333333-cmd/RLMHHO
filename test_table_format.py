#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整测试新的表格格式 - 包含完工时间和拖期的独立展示
"""

import os
import time
import numpy as np
from datetime import datetime

from problem.mo_dhfsp import MO_DHFSP_Problem
from algorithm.rl_chaotic_hho import RL_ChaoticHHO_Optimizer
from algorithm.nsga2 import NSGA2_Optimizer
from algorithm.moead import MOEAD_Optimizer
from utils.data_generator import DataGenerator

def generate_custom_urgencies(n_jobs: int, urgency_range):
    """生成指定范围的紧急度"""
    min_val, avg_val, max_val = urgency_range
    
    # 生成正态分布的紧急度
    std_dev = (max_val - min_val) / 6
    urgencies = np.random.normal(avg_val, std_dev, n_jobs)
    urgencies = np.clip(urgencies, min_val, max_val)
    
    # 确保边界值存在
    urgencies[0] = min_val
    urgencies[1] = max_val
    urgencies[2] = avg_val
    
    return urgencies.tolist()

def test_single_scale():
    """测试单个规模配置"""
    
    print("测试表格格式实验 - 20×5×3规模")
    print("=" * 50)
    
    # 测试配置 - 小规模快速测试
    config = {
        'scale': '20×5×3',
        'n_jobs': 20,
        'n_factories': 5,
        'n_stages': 3,
        'machines_per_stage': [3, 3, 3],
        'urgency_ddt': [0.9, 1.9, 2.9],
        'processing_time_range': (1, 20)
    }
    
    # 生成问题数据
    generator = DataGenerator(seed=42)
    problem_data = generator.generate_problem(
        n_jobs=config['n_jobs'],
        n_factories=config['n_factories'],
        n_stages=config['n_stages'],
        machines_per_stage=config['machines_per_stage'],
        processing_time_range=config['processing_time_range'],
        due_date_tightness=1.5
    )
    
    # 使用自定义紧急度
    problem_data['urgencies'] = generate_custom_urgencies(
        config['n_jobs'], 
        config['urgency_ddt']
    )
    
    # 验证紧急度
    urgencies = np.array(problem_data['urgencies'])
    print(f"紧急度DDT: {config['urgency_ddt']}")
    print(f"实际紧急度范围: [{urgencies.min():.2f}, {urgencies.max():.2f}]")
    print(f"紧急度均值: {urgencies.mean():.2f}")
    
    # 算法配置 - 减少迭代次数用于快速测试
    algorithms = {
        'RL-Chaotic-HHO': {
            'class': RL_ChaoticHHO_Optimizer,
            'params': {'max_iterations': 20}  # 减少迭代次数
        },
        'NSGA-II': {
            'class': NSGA2_Optimizer,
            'params': {
                'population_size': 50,  # 减少种群大小
                'max_generations': 20,  # 减少代数
                'crossover_prob': 0.9,
                'mutation_prob': 0.1
            }
        },
        'MOEA/D': {
            'class': MOEAD_Optimizer,
            'params': {
                'population_size': 50,  # 减少种群大小
                'max_generations': 20,  # 减少代数
                'crossover_prob': 0.9,
                'mutation_prob': 0.1,
                'neighbor_size': 10,
                'delta': 0.9,
                'nr': 2
            }
        }
    }
    
    # 存储结果
    results = {}
    
    # 测试每个算法
    for alg_name, alg_config in algorithms.items():
        print(f"\n正在测试 {alg_name}...")
        
        try:
            # 创建问题实例
            problem = MO_DHFSP_Problem(problem_data)
            
            # 创建优化器
            optimizer = alg_config['class'](problem, **alg_config['params'])
            
            # 运行优化
            start_time = time.time()
            pareto_solutions, convergence_data = optimizer.optimize()
            end_time = time.time()
            
            runtime = end_time - start_time
            
            # 计算加权目标函数值
            if pareto_solutions:
                best_objective = float('inf')
                for sol in pareto_solutions:
                    weighted_obj = 0.55 * sol.makespan + 0.45 * sol.total_tardiness
                    best_objective = min(best_objective, weighted_obj)
            else:
                best_objective = float('inf')
            
            results[alg_name] = {
                'best': best_objective,
                'runtime': runtime,
                'pareto_size': len(pareto_solutions) if pareto_solutions else 0
            }
            
            print(f"  {alg_name} 完成:")
            print(f"    加权目标值: {best_objective:.2f}")
            print(f"    运行时间: {runtime:.2f}s")
            print(f"    帕累托解数: {len(pareto_solutions) if pareto_solutions else 0}")
            
        except Exception as e:
            print(f"  {alg_name} 运行失败: {str(e)}")
            results[alg_name] = {
                'best': float('inf'),
                'runtime': 0.0,
                'pareto_size': 0
            }
    
    # 输出测试结果表格
    print("\n" + "=" * 80)
    print("测试结果")
    print("=" * 80)
    print("目标函数: F = 0.55*F1 + 0.45*F2")
    print("-" * 80)
    
    print(f"{'算法':^15s} | {'加权目标值':^12s} | {'运行时间(s)':^12s} | {'帕累托解数':^12s}")
    print("-" * 80)
    
    for alg_name, result in results.items():
        print(f"{alg_name:^15s} | {result['best']:^12.2f} | {result['runtime']:^12.2f} | {result['pareto_size']:^12d}")
    
    print("-" * 80)
    
    # 找出最佳算法
    if results:
        best_alg = min(results.items(), key=lambda x: x[1]['best'])
        fastest_alg = min(results.items(), key=lambda x: x[1]['runtime'])
        most_solutions_alg = max(results.items(), key=lambda x: x[1]['pareto_size'])
        
        print(f"\n性能分析:")
        print(f"  最佳目标值: {best_alg[0]} ({best_alg[1]['best']:.2f})")
        print(f"  最快运行: {fastest_alg[0]} ({fastest_alg[1]['runtime']:.2f}s)")
        print(f"  最多解数: {most_solutions_alg[0]} ({most_solutions_alg[1]['pareto_size']}个)")

def test_complete_table_format():
    """测试完整的新表格格式，包含完工时间和拖期"""
    
    # 模拟数据
    scale = "小规模20×3×3"
    rl_result = {
        'weighted_best': 85.2, 'weighted_mean': 87.5,
        'makespan_best': 45.1, 'makespan_mean': 46.8,
        'tardiness_best': 12.3, 'tardiness_mean': 15.2,
        'runtime': 14.5
    }
    nsga_result = {
        'weighted_best': 89.1, 'weighted_mean': 91.2,
        'makespan_best': 48.2, 'makespan_mean': 49.5,
        'tardiness_best': 18.7, 'tardiness_mean': 21.3,
        'runtime': 2.1
    }
    moead_result = {
        'weighted_best': 92.3, 'weighted_mean': 94.8,
        'makespan_best': 50.5, 'makespan_mean': 52.1,
        'tardiness_best': 22.1, 'tardiness_mean': 25.4,
        'runtime': 3.2
    }
    
    print("=" * 120)
    print("完整的表格格式测试 - 包含完工时间和拖期的独立展示")
    print("=" * 120)
    
    # 表格1: 最优值对比（包含加权目标、完工时间、总拖期）
    print("\n🎯 最优值对比表")
    print("+" + "-" * 15 + "+" + "-" * 12 + "+" + "-" * 18 + "+" + "-" * 15 + "+" + "-" * 15 + "+")
    print(f"| {'规模':^13s} | {'指标':^10s} | {'RL-Chaotic-HHO':^16s} | {'NSGA-II':^13s} | {'MOEA/D':^13s} |")
    print("+" + "-" * 15 + "+" + "-" * 12 + "+" + "-" * 18 + "+" + "-" * 15 + "+" + "-" * 15 + "+")
    
    # 加权目标函数最优值行
    print(f"| {scale:^13s} | {'加权目标':^10s} | {rl_result['weighted_best']:^16.1f} | {nsga_result['weighted_best']:^13.1f} | {moead_result['weighted_best']:^13.1f} |")
    
    # 完工时间最优值行  
    print(f"| {'':<13s} | {'完工时间':^10s} | {rl_result['makespan_best']:^16.1f} | {nsga_result['makespan_best']:^13.1f} | {moead_result['makespan_best']:^13.1f} |")
    
    # 总拖期最优值行
    print(f"| {'':<13s} | {'总拖期':^10s} | {rl_result['tardiness_best']:^16.1f} | {nsga_result['tardiness_best']:^13.1f} | {moead_result['tardiness_best']:^13.1f} |")
    
    print("+" + "-" * 15 + "+" + "-" * 12 + "+" + "-" * 18 + "+" + "-" * 15 + "+" + "-" * 15 + "+")
    
    # 表格2: 平均值对比（包含加权目标、完工时间、总拖期）
    print("\n📊 平均值对比表")
    print("+" + "-" * 15 + "+" + "-" * 12 + "+" + "-" * 18 + "+" + "-" * 15 + "+" + "-" * 15 + "+")
    print(f"| {'规模':^13s} | {'指标':^10s} | {'RL-Chaotic-HHO':^16s} | {'NSGA-II':^13s} | {'MOEA/D':^13s} |")
    print("+" + "-" * 15 + "+" + "-" * 12 + "+" + "-" * 18 + "+" + "-" * 15 + "+" + "-" * 15 + "+")
    
    # 加权目标函数平均值行
    print(f"| {scale:^13s} | {'加权目标':^10s} | {rl_result['weighted_mean']:^16.1f} | {nsga_result['weighted_mean']:^13.1f} | {moead_result['weighted_mean']:^13.1f} |")
    
    # 完工时间平均值行
    print(f"| {'':<13s} | {'完工时间':^10s} | {rl_result['makespan_mean']:^16.1f} | {nsga_result['makespan_mean']:^13.1f} | {moead_result['makespan_mean']:^13.1f} |")
    
    # 总拖期平均值行
    print(f"| {'':<13s} | {'总拖期':^10s} | {rl_result['tardiness_mean']:^16.1f} | {nsga_result['tardiness_mean']:^13.1f} | {moead_result['tardiness_mean']:^13.1f} |")
    
    print("+" + "-" * 15 + "+" + "-" * 12 + "+" + "-" * 18 + "+" + "-" * 15 + "+" + "-" * 15 + "+")
    
    # 表格3: 运行时间对比
    print("\n⏱️ 运行时间对比表 (秒)")
    print("+" + "-" * 15 + "+" + "-" * 18 + "+" + "-" * 15 + "+" + "-" * 15 + "+")
    print(f"| {'规模':^13s} | {'RL-Chaotic-HHO':^16s} | {'NSGA-II':^13s} | {'MOEA/D':^13s} |")
    print("+" + "-" * 15 + "+" + "-" * 18 + "+" + "-" * 15 + "+" + "-" * 15 + "+")
    
    # 运行时间行
    print(f"| {scale:^13s} | {rl_result['runtime']:^16.1f} | {nsga_result['runtime']:^13.1f} | {moead_result['runtime']:^13.1f} |")
    
    print("+" + "-" * 15 + "+" + "-" * 18 + "+" + "-" * 15 + "+" + "-" * 15 + "+")
    
    print("\n📋 详细数据分析")
    print("-" * 80)
    print("🔍 完工时间对比:")
    print(f"  - RL-Chaotic-HHO: 最优 {rl_result['makespan_best']:.1f}, 平均 {rl_result['makespan_mean']:.1f}")
    print(f"  - NSGA-II:         最优 {nsga_result['makespan_best']:.1f}, 平均 {nsga_result['makespan_mean']:.1f}")
    print(f"  - MOEA/D:          最优 {moead_result['makespan_best']:.1f}, 平均 {moead_result['makespan_mean']:.1f}")
    
    print("\n🚀 总拖期对比:")
    print(f"  - RL-Chaotic-HHO: 最优 {rl_result['tardiness_best']:.1f}, 平均 {rl_result['tardiness_mean']:.1f}")
    print(f"  - NSGA-II:         最优 {nsga_result['tardiness_best']:.1f}, 平均 {nsga_result['tardiness_mean']:.1f}")
    print(f"  - MOEA/D:          最优 {moead_result['tardiness_best']:.1f}, 平均 {moead_result['tardiness_mean']:.1f}")
    
    print("\n⚡ 运行效率对比:")
    print(f"  - RL-Chaotic-HHO: {rl_result['runtime']:.1f}秒")
    print(f"  - NSGA-II:         {nsga_result['runtime']:.1f}秒")
    print(f"  - MOEA/D:          {moead_result['runtime']:.1f}秒")
    
    print("\n✅ 完整表格格式测试完成！")
    print("✨ 新格式特点：")
    print("  ✓ 取消了括号形式，每个指标单独占用一个单元格")
    print("  ✓ 分为三个独立表格：最优值、平均值、运行时间")
    print("  ✓ 完工时间和拖期作为独立行显示，便于对比")
    print("  ✓ 数据清晰易读，支持横向和纵向对比分析")
    print("  ✓ 包含详细的数据分析总结")

if __name__ == "__main__":
    # 确保结果目录存在
    os.makedirs("results", exist_ok=True)
    
    # 运行测试
    test_single_scale()
    test_complete_table_format()
    
    print("\n测试完成!") 