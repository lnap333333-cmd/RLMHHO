#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于强化学习协调的混沌哈里斯鹰-鹰分组多目标分布式混合流水车间调度算法
Multi-Objective Distributed Hybrid Flow Shop Scheduling using RL-Coordinated Chaotic Harris Hawk with Eagle Grouping

主程序入口
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime

from problem.mo_dhfsp import MO_DHFSP_Problem
from algorithm.rl_chaotic_hho import RL_ChaoticHHO_Optimizer
from utils.data_generator import DataGenerator
from utils.visualization import ResultVisualizer
from utils.performance_metrics import PerformanceEvaluator

def main():
    """主函数"""
    print("="*80)
    print("基于强化学习协调的混沌哈里斯鹰-鹰分组多目标优化算法")
    print("="*80)
    
    # 创建结果目录
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # 测试不同规模的问题（调整为更小规模进行快速测试）
    test_scales = [
        {'name': '小规模', 'jobs': 10, 'factories': 2, 'stages': 3, 'machines_per_stage': [2, 2, 2]},
        {'name': '中规模', 'jobs': 20, 'factories': 2, 'stages': 3, 'machines_per_stage': [2, 3, 2]},
    ]
    
    all_results = {}
    
    for scale in test_scales:
        print(f"\n开始测试 {scale['name']} 问题:")
        print(f"作业数: {scale['jobs']}, 工厂数: {scale['factories']}")
        print(f"阶段数: {scale['stages']}, 各阶段机器数: {scale['machines_per_stage']}")
        
        # 生成测试数据
        data_gen = DataGenerator()
        problem_data = data_gen.generate_problem(
            n_jobs=scale['jobs'],
            n_factories=scale['factories'], 
            n_stages=scale['stages'],
            machines_per_stage=scale['machines_per_stage']
        )
        
        # 创建问题实例
        problem = MO_DHFSP_Problem(problem_data)
        
        # 创建优化器
        optimizer = RL_ChaoticHHO_Optimizer(problem)
        
        # 运行优化
        start_time = time.time()
        best_solutions, convergence_data = optimizer.optimize()
        end_time = time.time()
        
        # 评估性能
        evaluator = PerformanceEvaluator()
        metrics = evaluator.evaluate(best_solutions, convergence_data)
        
        # 保存结果
        result = {
            'scale': scale,
            'best_solutions': best_solutions,
            'convergence_data': convergence_data,
            'metrics': metrics,
            'runtime': end_time - start_time
        }
        all_results[scale['name']] = result
        
        # 输出结果
        print(f"优化完成! 运行时间: {result['runtime']:.2f}秒")
        print(f"找到帕累托解数量: {len(best_solutions)}")
        if best_solutions:
            best_makespan = min(sol.makespan for sol in best_solutions)
            best_tardiness = min(sol.total_tardiness for sol in best_solutions)
            print(f"最优完工时间: {best_makespan:.2f}")
            print(f"最优总拖期: {best_tardiness:.2f}")
        
        # 可视化结果
        visualizer = ResultVisualizer()
        visualizer.plot_pareto_front(best_solutions, f"results/{scale['name']}_pareto.png")
        visualizer.plot_convergence(convergence_data, f"results/{scale['name']}_convergence.png")
    
    # 生成综合报告
    generate_comprehensive_report(all_results)
    
    print("\n" + "="*80)
    print("所有测试完成! 结果已保存到 results/ 目录")
    print("="*80)

def generate_comprehensive_report(all_results):
    """生成综合测试报告"""
    with open('results/comprehensive_report.txt', 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("综合测试报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        for scale_name, result in all_results.items():
            f.write(f"{scale_name} 测试结果:\n")
            f.write("-" * 40 + "\n")
            f.write(f"问题规模: {result['scale']['jobs']}作业, {result['scale']['factories']}工厂\n")
            f.write(f"运行时间: {result['runtime']:.2f}秒\n")
            f.write(f"帕累托解数量: {len(result['best_solutions'])}\n")
            
            if result['best_solutions']:
                best_makespan = min(sol.makespan for sol in result['best_solutions'])
                best_tardiness = min(sol.total_tardiness for sol in result['best_solutions'])
                f.write(f"最优完工时间: {best_makespan:.2f}\n")
                f.write(f"最优总拖期: {best_tardiness:.2f}\n")
            
            f.write(f"性能指标: {result['metrics']}\n")
            f.write("\n")

if __name__ == "__main__":
    main() 