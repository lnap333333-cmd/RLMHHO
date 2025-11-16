#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试增强版QL-ABC算法性能
对比原版和增强版的性能差异
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from problem.mo_dhfsp import MO_DHFSP_Problem
from algorithm.ql_abc_fixed import QLABC_Optimizer_Fixed
from algorithm.ql_abc_enhanced import QLABC_Optimizer_Enhanced
from utils.data_generator import DataGenerator

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def test_ql_abc_comparison():
    """对比原版和增强版QL-ABC算法"""
    print("=" * 60)
    print("QL-ABC算法性能对比测试")
    print("=" * 60)
    
    # 生成测试问题
    problem_config = {
        'n_jobs': 30,
        'n_factories': 2,
        'n_stages': 3,
        'machines_per_stage': [2, 3, 4],
        'urgency_ddt': [0.5, 1.0],
        'processing_time_range': (1, 18),
        'heterogeneous_machines': {
            0: [2, 4, 6],
            1: [3, 5, 4]
        }
    }
    
    # 使用正确的数据生成方法
    data_generator = DataGenerator()
    problem_data = data_generator.generate_problem(
        n_jobs=problem_config['n_jobs'],
        n_factories=problem_config['n_factories'],
        n_stages=problem_config['n_stages'],
        machines_per_stage=problem_config['machines_per_stage'],
        processing_time_range=problem_config['processing_time_range']
    )
    problem = MO_DHFSP_Problem(problem_data)
    
    print(f"测试问题: {problem_config['n_jobs']}工件, {problem_config['n_factories']}工厂, {problem_config['n_stages']}阶段")
    print()
    
    # 测试原版QL-ABC
    print("测试原版QL-ABC算法...")
    start_time = time.time()
    
    original_optimizer = QLABC_Optimizer_Fixed(problem, 
                                             population_size=100,
                                             max_iterations=100,
                                             learning_rate=0.4,
                                             discount_factor=0.8,
                                             epsilon=0.1)
    
    original_solutions, original_convergence = original_optimizer.optimize()
    original_runtime = time.time() - start_time
    
    print(f"  原版QL-ABC: {len(original_solutions)}个解, 耗时{original_runtime:.2f}秒")
    
    # 测试增强版QL-ABC
    print("\n测试增强版QL-ABC算法...")
    start_time = time.time()
    
    enhanced_optimizer = QLABC_Optimizer_Enhanced(problem,
                                                population_size=100,
                                                max_iterations=1000,
                                                learning_rate=0.3,
                                                discount_factor=0.9,
                                                epsilon=0.2,
                                                epsilon_decay=0.995,
                                                limit=15,
                                                archive_size=200)
    
    enhanced_solutions, enhanced_convergence = enhanced_optimizer.optimize()
    enhanced_runtime = time.time() - start_time
    
    print(f"  增强版QL-ABC: {len(enhanced_solutions)}个解, 耗时{enhanced_runtime:.2f}秒")
    
    # 计算性能指标
    print("\n性能指标对比:")
    print("-" * 40)
    
    # 原版指标
    if original_solutions:
        original_makespans = [sol.makespan for sol in original_solutions]
        original_tardiness = [sol.total_tardiness for sol in original_solutions]
        original_best_makespan = min(original_makespans)
        original_best_tardiness = min(original_tardiness)
    else:
        original_best_makespan = float('inf')
        original_best_tardiness = float('inf')
    
    # 增强版指标
    if enhanced_solutions:
        enhanced_makespans = [sol.makespan for sol in enhanced_solutions]
        enhanced_tardiness = [sol.total_tardiness for sol in enhanced_solutions]
        enhanced_best_makespan = min(enhanced_makespans)
        enhanced_best_tardiness = min(enhanced_tardiness)
    else:
        enhanced_best_makespan = float('inf')
        enhanced_best_tardiness = float('inf')
    
    print(f"{'指标':<15} {'原版QL-ABC':<15} {'增强版QL-ABC':<15} {'改进':<10}")
    print("-" * 60)
    print(f"{'解数量':<15} {len(original_solutions):<15} {len(enhanced_solutions):<15} {len(enhanced_solutions) - len(original_solutions):<10}")
    print(f"{'运行时间(秒)':<15} {original_runtime:<15.2f} {enhanced_runtime:<15.2f} {enhanced_runtime - original_runtime:<10.2f}")
    print(f"{'最优完工时间':<15} {original_best_makespan:<15.2f} {enhanced_best_makespan:<15.2f} {original_best_makespan - enhanced_best_makespan:<10.2f}")
    print(f"{'最优总拖期':<15} {original_best_tardiness:<15.2f} {enhanced_best_tardiness:<15.2f} {original_best_tardiness - enhanced_best_tardiness:<10.2f}")
    
    # 绘制对比图
    plt.figure(figsize=(15, 5))
    
    # 帕累托前沿对比
    plt.subplot(1, 3, 1)
    if original_solutions:
        plt.scatter(original_makespans, original_tardiness, c='blue', s=50, alpha=0.7, label='原版QL-ABC')
    if enhanced_solutions:
        plt.scatter(enhanced_makespans, enhanced_tardiness, c='red', s=50, alpha=0.7, label='增强版QL-ABC')
    
    plt.xlabel('完工时间')
    plt.ylabel('总拖期')
    plt.title('帕累托前沿对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 收敛曲线对比
    plt.subplot(1, 3, 2)
    if original_convergence:
        iterations_orig = [data['iteration'] for data in original_convergence]
        archive_sizes_orig = [data['archive_size'] for data in original_convergence]
        plt.plot(iterations_orig, archive_sizes_orig, 'b-', linewidth=2, label='原版QL-ABC')
    
    if enhanced_convergence:
        iterations_enh = [data['iteration'] for data in enhanced_convergence]
        archive_sizes_enh = [data['archive_size'] for data in enhanced_convergence]
        plt.plot(iterations_enh, archive_sizes_enh, 'r-', linewidth=2, label='增强版QL-ABC')
    
    plt.xlabel('迭代次数')
    plt.ylabel('档案大小')
    plt.title('收敛曲线对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 最优解收敛对比
    plt.subplot(1, 3, 3)
    if original_convergence:
        best_makespans_orig = [data['best_makespan'] for data in original_convergence]
        plt.plot(iterations_orig, best_makespans_orig, 'b-', linewidth=2, label='原版QL-ABC')
    
    if enhanced_convergence:
        best_makespans_enh = [data['best_makespan'] for data in enhanced_convergence]
        plt.plot(iterations_enh, best_makespans_enh, 'r-', linewidth=2, label='增强版QL-ABC')
    
    plt.xlabel('迭代次数')
    plt.ylabel('最优完工时间')
    plt.title('最优解收敛对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ql_abc_enhanced_comparison_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n对比图已保存: {filename}")
    
    plt.show()
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结:")
    print("=" * 60)
    
    improvements = []
    if len(enhanced_solutions) > len(original_solutions):
        improvements.append(f"解数量增加: {len(enhanced_solutions) - len(original_solutions)}个")
    
    if enhanced_best_makespan < original_best_makespan:
        improvements.append(f"完工时间改进: {original_best_makespan - enhanced_best_makespan:.2f}")
    
    if enhanced_best_tardiness < original_best_tardiness:
        improvements.append(f"拖期改进: {original_best_tardiness - enhanced_best_tardiness:.2f}")
    
    if improvements:
        print("✅ 增强版QL-ABC的改进:")
        for improvement in improvements:
            print(f"   - {improvement}")
    else:
        print("❌ 增强版QL-ABC没有明显改进")
    
    print(f"\n主要改进措施:")
    print("  1. 增加迭代次数: 100 → 1000")
    print("  2. 优化学习参数: 学习率0.4→0.3, 折扣因子0.8→0.9")
    print("  3. 增加探索率: 0.1→0.2, 并添加衰减机制")
    print("  4. 动态状态空间和权重调整")
    print("  5. 增强的蜜源更新策略")
    print("  6. 锦标赛选择替代轮盘赌")
    print("  7. 基于精英解的新解生成")

if __name__ == "__main__":
    test_ql_abc_comparison() 