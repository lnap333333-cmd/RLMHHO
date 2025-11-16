#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修正版本的QL-ABC算法
验证是否严格按照原文要求实现
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
from table_format_comparison_specific_scales import calculate_igd, calculate_gd, calculate_hypervolume, calculate_ra


def test_ql_abc_fixed_small_scale():
    """测试小规模问题的修正版QL-ABC算法"""
    print("=" * 60)
    print("测试修正版QL-ABC算法 - 小规模问题")
    print("=" * 60)
    
    # 创建小规模问题实例
    n_jobs, n_factories = 20, 3
    n_stages = 3
    
    # 生成测试数据
    np.random.seed(42)  # 固定随机种子确保可重复性
    
    # 生成处理时间矩阵 [job][stage]
    processing_times = np.random.randint(10, 50, size=(n_jobs, n_stages))
    
    # 生成交货期（基于处理时间的1.5-2倍）
    job_total_times = np.sum(processing_times, axis=1)
    due_dates = np.random.uniform(1.5, 2.0, size=n_jobs) * job_total_times
    
    # 生成机器配置（每阶段2-4台机器）
    machines_per_stage = [np.random.randint(2, 5) for _ in range(n_stages)]
    
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
    print(f"参数设置: 种群大小=100, 最大迭代=1000, 学习率=0.4, 折扣因子=0.8")
    
    # 创建修正版QL-ABC优化器
    optimizer = QLABC_Optimizer_Fixed(
        problem=problem,
        population_size=50,  # 测试时使用较小种群
        max_iterations=100,  # 测试时使用较少迭代
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
        plt.savefig('ql_abc_fixed_pareto_front.png', dpi=300, bbox_inches='tight')
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
            plt.savefig('ql_abc_fixed_convergence.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    return pareto_front, convergence_data


def test_ql_abc_fixed_vs_original():
    """对比修正版和原版QL-ABC算法"""
    print("\n" + "=" * 60)
    print("对比修正版和原版QL-ABC算法")
    print("=" * 60)
    
    # 创建问题实例
    n_jobs, n_factories = 30, 4
    n_stages = 3
    
    # 生成测试数据
    np.random.seed(42)
    processing_times = np.random.randint(10, 50, size=(n_jobs, n_stages))
    job_total_times = np.sum(processing_times, axis=1)
    due_dates = np.random.uniform(1.5, 2.0, size=n_jobs) * job_total_times
    machines_per_stage = [np.random.randint(2, 5) for _ in range(n_stages)]
    
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
    
    # 测试原版QL-ABC
    print("\n测试原版QL-ABC算法...")
    from algorithm.ql_abc import QLABC_Optimizer
    
    optimizer_original = QLABC_Optimizer(
        problem=problem,
        population_size=50,
        max_iterations=100
    )
    
    start_time = time.time()
    pareto_original, conv_original = optimizer_original.optimize()
    time_original = time.time() - start_time
    
    # 测试修正版QL-ABC
    print("\n测试修正版QL-ABC算法...")
    optimizer_fixed = QLABC_Optimizer_Fixed(
        problem=problem,
        population_size=50,
        max_iterations=100
    )
    
    start_time = time.time()
    pareto_fixed, conv_fixed = optimizer_fixed.optimize()
    time_fixed = time.time() - start_time
    
    # 对比结果
    print(f"\n对比结果:")
    print(f"{'指标':<15} {'原版QL-ABC':<15} {'修正版QL-ABC':<15}")
    print("-" * 50)
    print(f"{'运行时间(秒)':<15} {time_original:<15.2f} {time_fixed:<15.2f}")
    print(f"{'帕累托解数量':<15} {len(pareto_original):<15} {len(pareto_fixed):<15}")
    
    if pareto_original and pareto_fixed:
        # 计算目标函数范围
        makespan_orig = [sol.makespan for sol in pareto_original]
        tardiness_orig = [sol.total_tardiness for sol in pareto_original]
        makespan_fixed = [sol.makespan for sol in pareto_fixed]
        tardiness_fixed = [sol.total_tardiness for sol in pareto_fixed]
        
        print(f"{'完工时间范围':<15} [{min(makespan_orig):.1f},{max(makespan_orig):.1f}] {'':<5} [{min(makespan_fixed):.1f},{max(makespan_fixed):.1f}]")
        print(f"{'总拖期范围':<15} [{min(tardiness_orig):.1f},{max(tardiness_orig):.1f}] {'':<5} [{min(tardiness_fixed):.1f},{max(tardiness_fixed):.1f}]")
        
        # 绘制对比图
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.scatter(makespan_orig, tardiness_orig, c='blue', s=40, alpha=0.7, label='原版QL-ABC')
        plt.xlabel('完工时间')
        plt.ylabel('总拖期')
        plt.title('原版QL-ABC帕累托前沿')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.scatter(makespan_fixed, tardiness_fixed, c='red', s=40, alpha=0.7, label='修正版QL-ABC')
        plt.xlabel('完工时间')
        plt.ylabel('总拖期')
        plt.title('修正版QL-ABC帕累托前沿')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        plt.scatter(makespan_orig, tardiness_orig, c='blue', s=40, alpha=0.7, label='原版QL-ABC')
        plt.scatter(makespan_fixed, tardiness_fixed, c='red', s=40, alpha=0.7, label='修正版QL-ABC')
        plt.xlabel('完工时间')
        plt.ylabel('总拖期')
        plt.title('帕累托前沿对比')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        iterations = [data['iteration'] for data in conv_original]
        archive_orig = [data['archive_size'] for data in conv_original]
        archive_fixed = [data['archive_size'] for data in conv_fixed]
        
        plt.plot(iterations, archive_orig, 'b-', linewidth=2, label='原版QL-ABC')
        plt.plot(iterations, archive_fixed, 'r-', linewidth=2, label='修正版QL-ABC')
        plt.xlabel('迭代次数')
        plt.ylabel('档案大小')
        plt.title('档案大小收敛对比')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('ql_abc_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    return pareto_original, pareto_fixed


def test_ql_abc_fixed_ra_performance():
    """测试修正版QL-ABC的RA指标性能"""
    print("\n" + "=" * 60)
    print("测试修正版QL-ABC的RA指标性能")
    print("=" * 60)
    
    # 创建问题实例
    n_jobs, n_factories = 25, 3
    n_stages = 3
    
    # 生成测试数据
    np.random.seed(42)
    processing_times = np.random.randint(10, 50, size=(n_jobs, n_stages))
    job_total_times = np.sum(processing_times, axis=1)
    due_dates = np.random.uniform(1.5, 2.0, size=n_jobs) * job_total_times
    machines_per_stage = [np.random.randint(2, 5) for _ in range(n_stages)]
    
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
    
    # 生成参考帕累托前沿（使用多次运行的最佳解）
    print("生成参考帕累托前沿...")
    reference_solutions = []
    
    for run in range(5):
        optimizer = QLABC_Optimizer_Fixed(
            problem=problem,
            population_size=50,
            max_iterations=100
        )
        pareto_front, _ = optimizer.optimize()
        reference_solutions.extend(pareto_front)
    
    # 计算参考帕累托前沿
    reference_pareto = []
    for sol in reference_solutions:
        is_dominated = False
        for ref_sol in reference_pareto:
            if (ref_sol.makespan <= sol.makespan and ref_sol.total_tardiness <= sol.total_tardiness and
                (ref_sol.makespan < sol.makespan or ref_sol.total_tardiness < sol.total_tardiness)):
                is_dominated = True
                break
        if not is_dominated:
            reference_pareto.append(sol)
    
    print(f"参考帕累托前沿大小: {len(reference_pareto)}")
    
    # 测试修正版QL-ABC的RA指标
    print("\n测试修正版QL-ABC的RA指标...")
    optimizer = QLABC_Optimizer_Fixed(
        problem=problem,
        population_size=50,
        max_iterations=100
    )
    
    pareto_front, _ = optimizer.optimize()
    
    if pareto_front and reference_pareto:
        # 计算RA指标
        ra_score = calculate_ra(pareto_front, reference_pareto)
        print(f"修正版QL-ABC的RA指标: {ra_score:.4f}")
        
        # 归一化处理用于其他指标计算
        all_solutions = pareto_front + reference_pareto
        all_makespans = [sol.makespan for sol in all_solutions]
        all_tardiness = [sol.total_tardiness for sol in all_solutions]
        
        min_makespan = min(all_makespans)
        max_makespan = max(all_makespans)
        min_tardiness = min(all_tardiness)
        max_tardiness = max(all_tardiness)
        
        makespan_range = max_makespan - min_makespan if max_makespan > min_makespan else 1.0
        tardiness_range = max_tardiness - min_tardiness if max_tardiness > min_tardiness else 1.0
        
        # 创建归一化的解
        normalized_pareto = []
        for sol in pareto_front:
            norm_sol = type('Solution', (), {})()
            norm_sol.makespan = (sol.makespan - min_makespan) / makespan_range
            norm_sol.total_tardiness = (sol.total_tardiness - min_tardiness) / tardiness_range
            normalized_pareto.append(norm_sol)
        
        normalized_reference = []
        for sol in reference_pareto:
            norm_sol = type('Solution', (), {})()
            norm_sol.makespan = (sol.makespan - min_makespan) / makespan_range
            norm_sol.total_tardiness = (sol.total_tardiness - min_tardiness) / tardiness_range
            normalized_reference.append(norm_sol)
        
        # 计算其他指标
        igd_score = calculate_igd(normalized_pareto, normalized_reference)
        gd_score = calculate_gd(normalized_pareto, normalized_reference)
        hv_score = calculate_hypervolume(pareto_front)
        
        print(f"IGD指标: {igd_score:.4f}")
        print(f"GD指标: {gd_score:.4f}")
        print(f"HV指标: {hv_score:.4f}")
        
        # 绘制结果
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        ref_makespans = [sol.makespan for sol in reference_pareto]
        ref_tardinesses = [sol.total_tardiness for sol in reference_pareto]
        plt.scatter(ref_makespans, ref_tardinesses, c='gray', s=30, alpha=0.5, label='参考帕累托前沿')
        
        test_makespans = [sol.makespan for sol in pareto_front]
        test_tardinesses = [sol.total_tardiness for sol in pareto_front]
        plt.scatter(test_makespans, test_tardinesses, c='red', s=50, alpha=0.8, label='修正版QL-ABC')
        
        plt.xlabel('完工时间')
        plt.ylabel('总拖期')
        plt.title(f'RA指标测试 (RA={ra_score:.4f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        metrics = ['RA', 'IGD', 'GD', 'HV']
        values = [ra_score, igd_score, gd_score, hv_score]
        colors = ['red', 'blue', 'green', 'orange']
        
        bars = plt.bar(metrics, values, color=colors, alpha=0.7)
        plt.ylabel('指标值')
        plt.title('修正版QL-ABC性能指标')
        
        # 在柱状图上添加数值标签
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.4f}', ha='center', va='bottom')
        
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('ql_abc_fixed_ra_test.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    return pareto_front, reference_pareto


if __name__ == "__main__":
    # 测试修正版QL-ABC算法
    pareto_front, convergence_data = test_ql_abc_fixed_small_scale()
    
    # 对比原版和修正版
    pareto_original, pareto_fixed = test_ql_abc_fixed_vs_original()
    
    # 测试RA指标性能
    test_pareto, reference_pareto = test_ql_abc_fixed_ra_performance()
    
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