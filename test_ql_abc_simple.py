#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版QL-ABC测试脚本
只测试修正版QL-ABC的基本功能
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from problem.mo_dhfsp import MO_DHFSP_Problem
from algorithm.ql_abc_fixed import QLABC_Optimizer_Fixed


def test_ql_abc_fixed_basic():
    """测试修正版QL-ABC的基本功能"""
    print("=" * 60)
    print("测试修正版QL-ABC基本功能")
    print("=" * 60)
    
    # 创建小规模问题实例
    n_jobs, n_factories = 15, 2
    n_stages = 2
    
    # 生成测试数据
    np.random.seed(42)
    processing_times = np.random.randint(10, 30, size=(n_jobs, n_stages))
    job_total_times = np.sum(processing_times, axis=1)
    due_dates = np.random.uniform(1.5, 2.0, size=n_jobs) * job_total_times
    machines_per_stage = [2, 2]  # 每阶段2台机器
    
    # 构建problem_data字典
    problem_data = {
        'n_jobs': n_jobs,
        'n_factories': n_factories,
        'n_stages': n_stages,
        'machines_per_stage': machines_per_stage,
        'processing_times': processing_times.tolist(),
        'due_dates': due_dates.tolist(),
        'urgencies': [1.0] * n_jobs
    }
    
    problem = MO_DHFSP_Problem(problem_data)
    
    print(f"问题规模: {n_jobs}个作业, {n_factories}个工厂, {n_stages}个阶段")
    print(f"机器配置: {machines_per_stage}")
    print(f"参数设置: 种群大小=30, 最大迭代=50, 学习率=0.4, 折扣因子=0.8")
    
    # 创建修正版QL-ABC优化器
    optimizer = QLABC_Optimizer_Fixed(
        problem=problem,
        population_size=30,
        max_iterations=50,
        learning_rate=0.4,
        discount_factor=0.8,
        epsilon=0.1
    )
    
    # 执行优化
    start_time = time.time()
    pareto_front, convergence_data = optimizer.optimize()
    end_time = time.time()
    
    print(f"\n优化完成!")
    print(f"运行时间: {end_time - start_time:.2f}秒")
    print(f"帕累托前沿解数量: {len(pareto_front)}")
    
    if pareto_front:
        # 计算目标函数值
        makespans = [sol.makespan for sol in pareto_front]
        tardinesses = [sol.total_tardiness for sol in pareto_front]
        
        print(f"完工时间范围: [{min(makespans):.2f}, {max(makespans):.2f}]")
        print(f"总拖期范围: [{min(tardinesses):.2f}, {max(tardinesses):.2f}]")
        
        # 绘制帕累托前沿
        plt.figure(figsize=(10, 6))
        plt.scatter(makespans, tardinesses, c='red', s=50, alpha=0.7, label='QL-ABC修正版')
        plt.xlabel('完工时间 (Makespan)')
        plt.ylabel('总拖期 (Total Tardiness)')
        plt.title('QL-ABC修正版 - 帕累托前沿')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('ql_abc_fixed_basic_pareto.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 绘制收敛曲线
        if convergence_data:
            iterations = [data['iteration'] for data in convergence_data]
            archive_sizes = [data['archive_size'] for data in convergence_data]
            q_table_sizes = [data['q_table_size'] for data in convergence_data]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            ax1.plot(iterations, archive_sizes, 'b-', linewidth=2)
            ax1.set_xlabel('迭代次数')
            ax1.set_ylabel('档案大小')
            ax1.set_title('档案大小收敛曲线')
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(iterations, q_table_sizes, 'g-', linewidth=2)
            ax2.set_xlabel('迭代次数')
            ax2.set_ylabel('Q表大小')
            ax2.set_title('Q表大小增长曲线')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('ql_abc_fixed_basic_convergence.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # 打印Q-learning相关信息
            print(f"\nQ-learning统计信息:")
            print(f"Q表大小: {len(optimizer.q_table)}")
            print(f"状态空间大小: {len(set(state[0] for state in optimizer.q_table.keys()))}")
            
            # 分析Q表内容
            if optimizer.q_table:
                q_values = []
                for state_actions in optimizer.q_table.values():
                    q_values.extend(state_actions.values())
                
                print(f"Q值范围: [{min(q_values):.4f}, {max(q_values):.4f}]")
                print(f"平均Q值: {np.mean(q_values):.4f}")
    
    return pareto_front, convergence_data


def test_ql_abc_fixed_parameters():
    """测试修正版QL-ABC的参数设置"""
    print("\n" + "=" * 60)
    print("测试修正版QL-ABC参数设置")
    print("=" * 60)
    
    # 创建问题实例
    n_jobs, n_factories = 20, 3
    n_stages = 2
    
    # 生成测试数据
    np.random.seed(42)
    processing_times = np.random.randint(10, 40, size=(n_jobs, n_stages))
    job_total_times = np.sum(processing_times, axis=1)
    due_dates = np.random.uniform(1.5, 2.0, size=n_jobs) * job_total_times
    machines_per_stage = [3, 3]
    
    problem_data = {
        'n_jobs': n_jobs,
        'n_factories': n_factories,
        'n_stages': n_stages,
        'machines_per_stage': machines_per_stage,
        'processing_times': processing_times.tolist(),
        'due_dates': due_dates.tolist(),
        'urgencies': [1.0] * n_jobs
    }
    
    problem = MO_DHFSP_Problem(problem_data)
    
    # 测试不同参数设置
    parameter_sets = [
        {'learning_rate': 0.2, 'discount_factor': 0.6, 'epsilon': 0.05},
        {'learning_rate': 0.4, 'discount_factor': 0.8, 'epsilon': 0.1},
        {'learning_rate': 0.6, 'discount_factor': 0.9, 'epsilon': 0.15}
    ]
    
    results = []
    
    for i, params in enumerate(parameter_sets):
        print(f"\n测试参数组 {i+1}: {params}")
        
        optimizer = QLABC_Optimizer_Fixed(
            problem=problem,
            population_size=40,
            max_iterations=30,
            **params
        )
        
        start_time = time.time()
        pareto_front, convergence_data = optimizer.optimize()
        end_time = time.time()
        
        if pareto_front:
            makespans = [sol.makespan for sol in pareto_front]
            tardinesses = [sol.total_tardiness for sol in pareto_front]
            
            result = {
                'params': params,
                'pareto_size': len(pareto_front),
                'best_makespan': min(makespans),
                'best_tardiness': min(tardinesses),
                'runtime': end_time - start_time,
                'q_table_size': len(optimizer.q_table)
            }
            results.append(result)
            
            print(f"  帕累托解数量: {result['pareto_size']}")
            print(f"  最佳完工时间: {result['best_makespan']:.2f}")
            print(f"  最佳总拖期: {result['best_tardiness']:.2f}")
            print(f"  运行时间: {result['runtime']:.2f}秒")
            print(f"  Q表大小: {result['q_table_size']}")
    
    # 绘制参数对比图
    if results:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 帕累托解数量对比
        pareto_sizes = [r['pareto_size'] for r in results]
        axes[0, 0].bar(range(len(results)), pareto_sizes, color=['blue', 'red', 'green'])
        axes[0, 0].set_xlabel('参数组')
        axes[0, 0].set_ylabel('帕累托解数量')
        axes[0, 0].set_title('帕累托解数量对比')
        axes[0, 0].set_xticks(range(len(results)))
        axes[0, 0].set_xticklabels([f'组{i+1}' for i in range(len(results))])
        
        # 最佳完工时间对比
        best_makespans = [r['best_makespan'] for r in results]
        axes[0, 1].bar(range(len(results)), best_makespans, color=['blue', 'red', 'green'])
        axes[0, 1].set_xlabel('参数组')
        axes[0, 1].set_ylabel('最佳完工时间')
        axes[0, 1].set_title('最佳完工时间对比')
        axes[0, 1].set_xticks(range(len(results)))
        axes[0, 1].set_xticklabels([f'组{i+1}' for i in range(len(results))])
        
        # 运行时间对比
        runtimes = [r['runtime'] for r in results]
        axes[1, 0].bar(range(len(results)), runtimes, color=['blue', 'red', 'green'])
        axes[1, 0].set_xlabel('参数组')
        axes[1, 0].set_ylabel('运行时间(秒)')
        axes[1, 0].set_title('运行时间对比')
        axes[1, 0].set_xticks(range(len(results)))
        axes[1, 0].set_xticklabels([f'组{i+1}' for i in range(len(results))])
        
        # Q表大小对比
        q_table_sizes = [r['q_table_size'] for r in results]
        axes[1, 1].bar(range(len(results)), q_table_sizes, color=['blue', 'red', 'green'])
        axes[1, 1].set_xlabel('参数组')
        axes[1, 1].set_ylabel('Q表大小')
        axes[1, 1].set_title('Q表大小对比')
        axes[1, 1].set_xticks(range(len(results)))
        axes[1, 1].set_xticklabels([f'组{i+1}' for i in range(len(results))])
        
        plt.tight_layout()
        plt.savefig('ql_abc_fixed_parameter_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    return results


if __name__ == "__main__":
    # 测试基本功能
    pareto_front, convergence_data = test_ql_abc_fixed_basic()
    
    # 测试参数设置
    parameter_results = test_ql_abc_fixed_parameters()
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
    print("修正版QL-ABC算法严格按照原文要求实现：")
    print("1. 状态特征：使用fa、fv、fm三个特征，权重μ1=0.4, μ2=0.2, μ3=0.2")
    print("2. 动作集：h ∈ [1, H/k]，k=10")
    print("3. 奖励机制：使用Beta分布计算奖励值")
    print("4. 参数设置：学习率=0.4, 折扣因子=0.8, 种群大小=100, 迭代次数=1000")
    print("5. 蜜源更新：严格按照公式(13)实现")
    print("6. 侦察蜂：按照公式(15)生成新蜜源") 