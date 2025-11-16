#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速QL-ABC性能测试
简单对比原版和增强版的性能差异
"""

import time
import numpy as np
from datetime import datetime

from problem.mo_dhfsp import MO_DHFSP_Problem
from algorithm.ql_abc_fixed import QLABC_Optimizer_Fixed
from algorithm.ql_abc_enhanced import QLABC_Optimizer_Enhanced
from utils.data_generator import DataGenerator

def quick_ql_abc_test():
    """快速测试QL-ABC算法性能"""
    print("=" * 50)
    print("快速QL-ABC性能测试")
    print("=" * 50)
    
    # 生成小规模测试问题
    data_generator = DataGenerator()
    problem_data = data_generator.generate_problem(
        n_jobs=20,
        n_factories=2,
        n_stages=3,
        machines_per_stage=[2, 3, 2],
        processing_time_range=(1, 15)
    )
    
    problem = MO_DHFSP_Problem(problem_data)
    print(f"测试问题: 20工件, 2工厂, 3阶段")
    print()
    
    # 测试原版QL-ABC（减少迭代次数以加快测试）
    print("测试原版QL-ABC算法...")
    start_time = time.time()
    
    original_optimizer = QLABC_Optimizer_Fixed(problem, 
                                             population_size=50,
                                             max_iterations=50,  # 减少迭代次数
                                             learning_rate=0.4,
                                             discount_factor=0.8,
                                             epsilon=0.1)
    
    original_solutions, _ = original_optimizer.optimize()
    original_runtime = time.time() - start_time
    
    print(f"  原版QL-ABC: {len(original_solutions)}个解, 耗时{original_runtime:.2f}秒")
    
    # 测试增强版QL-ABC（减少迭代次数以加快测试）
    print("\n测试增强版QL-ABC算法...")
    start_time = time.time()
    
    enhanced_optimizer = QLABC_Optimizer_Enhanced(problem,
                                                population_size=50,
                                                max_iterations=100,  # 减少迭代次数
                                                learning_rate=0.3,
                                                discount_factor=0.9,
                                                epsilon=0.2,
                                                epsilon_decay=0.995,
                                                limit=10,
                                                archive_size=100)
    
    enhanced_solutions, _ = enhanced_optimizer.optimize()
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
        original_avg_makespan = np.mean(original_makespans)
        original_avg_tardiness = np.mean(original_tardiness)
    else:
        original_best_makespan = float('inf')
        original_best_tardiness = float('inf')
        original_avg_makespan = 0
        original_avg_tardiness = 0
    
    # 增强版指标
    if enhanced_solutions:
        enhanced_makespans = [sol.makespan for sol in enhanced_solutions]
        enhanced_tardiness = [sol.total_tardiness for sol in enhanced_solutions]
        enhanced_best_makespan = min(enhanced_makespans)
        enhanced_best_tardiness = min(enhanced_tardiness)
        enhanced_avg_makespan = np.mean(enhanced_makespans)
        enhanced_avg_tardiness = np.mean(enhanced_tardiness)
    else:
        enhanced_best_makespan = float('inf')
        enhanced_best_tardiness = float('inf')
        enhanced_avg_makespan = 0
        enhanced_avg_tardiness = 0
    
    print(f"{'指标':<15} {'原版QL-ABC':<15} {'增强版QL-ABC':<15} {'改进':<10}")
    print("-" * 60)
    print(f"{'解数量':<15} {len(original_solutions):<15} {len(enhanced_solutions):<15} {len(enhanced_solutions) - len(original_solutions):<10}")
    print(f"{'运行时间(秒)':<15} {original_runtime:<15.2f} {enhanced_runtime:<15.2f} {enhanced_runtime - original_runtime:<10.2f}")
    print(f"{'最优完工时间':<15} {original_best_makespan:<15.2f} {enhanced_best_makespan:<15.2f} {original_best_makespan - enhanced_best_makespan:<10.2f}")
    print(f"{'最优总拖期':<15} {original_best_tardiness:<15.2f} {enhanced_best_tardiness:<15.2f} {original_best_tardiness - enhanced_best_tardiness:<10.2f}")
    print(f"{'平均完工时间':<15} {original_avg_makespan:<15.2f} {enhanced_avg_makespan:<15.2f} {original_avg_makespan - enhanced_avg_makespan:<10.2f}")
    print(f"{'平均总拖期':<15} {original_avg_tardiness:<15.2f} {enhanced_avg_tardiness:<15.2f} {original_avg_tardiness - enhanced_avg_tardiness:<10.2f}")
    
    # 总结
    print("\n" + "=" * 50)
    print("测试总结:")
    print("=" * 50)
    
    improvements = []
    if len(enhanced_solutions) > len(original_solutions):
        improvements.append(f"解数量增加: {len(enhanced_solutions) - len(original_solutions)}个")
    
    if enhanced_best_makespan < original_best_makespan:
        improvements.append(f"完工时间改进: {original_best_makespan - enhanced_best_makespan:.2f}")
    
    if enhanced_best_tardiness < original_best_tardiness:
        improvements.append(f"拖期改进: {original_best_tardiness - enhanced_best_tardiness:.2f}")
    
    if enhanced_avg_makespan < original_avg_makespan:
        improvements.append(f"平均完工时间改进: {original_avg_makespan - enhanced_avg_makespan:.2f}")
    
    if enhanced_avg_tardiness < original_avg_tardiness:
        improvements.append(f"平均拖期改进: {original_avg_tardiness - enhanced_avg_tardiness:.2f}")
    
    if improvements:
        print("✅ 增强版QL-ABC的改进:")
        for improvement in improvements:
            print(f"   - {improvement}")
    else:
        print("❌ 增强版QL-ABC没有明显改进")
    
    print(f"\n主要改进措施:")
    print("  1. 优化学习参数: 学习率0.4→0.3, 折扣因子0.8→0.9")
    print("  2. 增加探索率: 0.1→0.2, 并添加衰减机制")
    print("  3. 动态状态空间和权重调整")
    print("  4. 增强的蜜源更新策略")
    print("  5. 锦标赛选择替代轮盘赌")
    print("  6. 基于精英解的新解生成")
    
    return len(enhanced_solutions) > len(original_solutions) or enhanced_best_makespan < original_best_makespan

if __name__ == "__main__":
    success = quick_ql_abc_test()
    if success:
        print("\n🎉 增强版QL-ABC测试成功，性能有所改进！")
    else:
        print("\n⚠️ 增强版QL-ABC测试完成，但性能改进不明显") 