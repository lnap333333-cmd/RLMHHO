#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import random
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from problem.mo_dhfsp import MO_DHFSP_Problem
from algorithm.mosa import MOSA_Optimizer
from algorithm.nsga2 import NSGA2_Optimizer
import time

def quick_test():
    """快速测试调整参数后的MOSA性能"""
    
    # 创建小规模测试问题
    problem_config = {
        'n_jobs': 20,
        'n_factories': 3,
        'n_stages': 3,
        'machines_per_stage': [2, 3, 3],  # 基础机器配置
        'processing_times': [[random.randint(1, 10) for _ in range(3)] for _ in range(20)],
        'due_dates': [random.randint(20, 40) for _ in range(20)],
        'urgencies': [random.uniform(0.1, 0.9) for _ in range(20)],
        'heterogeneous_machines': {
            0: [2, 2, 2],  # 工厂0: 每阶段2台机器
            1: [2, 3, 3],  # 工厂1: 每阶段2,3,3台机器  
            2: [2, 3, 4]   # 工厂2: 每阶段2,3,4台机器
        }
    }
    
    problem = MO_DHFSP_Problem(problem_config)
    
    print("🧪 快速测试: 调整参数后的MOSA vs NSGA-II")
    print("=" * 60)
    
    # 测试MOSA (新参数)
    print("\n🔥 测试MOSA (新参数)...")
    mosa_params = {
        'initial_temperature': 500.0,
        'final_temperature': 0.1,
        'cooling_rate': 0.98,
        'max_iterations': 800,
        'archive_size': 50,
        'neighborhood_size': 10
    }
    
    mosa_optimizer = MOSA_Optimizer(problem, **mosa_params)
    start_time = time.time()
    mosa_solutions, mosa_info = mosa_optimizer.optimize()
    mosa_time = time.time() - start_time
    
    # 计算MOSA最优解
    mosa_best = min(mosa_solutions, key=lambda s: 0.55*s.makespan + 0.45*s.total_tardiness)
    mosa_weighted = 0.55*mosa_best.makespan + 0.45*mosa_best.total_tardiness
    
    print(f"  ✅ MOSA结果:")
    print(f"     迭代次数: {mosa_info['iterations']}")
    print(f"     帕累托解数: {len(mosa_solutions)}")
    print(f"     最优加权目标: {mosa_weighted:.2f}")
    print(f"     最优完工时间: {mosa_best.makespan:.2f}")
    print(f"     最优总拖期: {mosa_best.total_tardiness:.2f}")
    print(f"     运行时间: {mosa_time:.2f}s")
    
    # 测试NSGA-II (对比)
    print("\n🧬 测试NSGA-II (对比)...")
    nsga2_params = {
        'population_size': 60,
        'max_generations': 60,
        'crossover_prob': 0.9,
        'mutation_prob': 0.1
    }
    
    nsga2_optimizer = NSGA2_Optimizer(problem, **nsga2_params)
    start_time = time.time()
    nsga2_solutions, nsga2_info = nsga2_optimizer.optimize()
    nsga2_time = time.time() - start_time
    
    # 计算NSGA-II最优解
    nsga2_best = min(nsga2_solutions, key=lambda s: 0.55*s.makespan + 0.45*s.total_tardiness)
    nsga2_weighted = 0.55*nsga2_best.makespan + 0.45*nsga2_best.total_tardiness
    
    print(f"  ✅ NSGA-II结果:")
    print(f"     代数: {nsga2_params['max_generations']}")
    print(f"     帕累托解数: {len(nsga2_solutions)}")
    print(f"     最优加权目标: {nsga2_weighted:.2f}")
    print(f"     最优完工时间: {nsga2_best.makespan:.2f}")
    print(f"     最优总拖期: {nsga2_best.total_tardiness:.2f}")
    print(f"     运行时间: {nsga2_time:.2f}s")
    
    # 性能对比
    print("\n📊 性能对比:")
    print("=" * 40)
    improvement = ((nsga2_weighted - mosa_weighted) / nsga2_weighted) * 100
    speed_ratio = nsga2_time / mosa_time
    
    if improvement > 0:
        print(f"  🎯 MOSA比NSGA-II好 {improvement:.1f}%")
    else:
        print(f"  ❌ MOSA比NSGA-II差 {-improvement:.1f}%")
    
    print(f"  ⏱️  速度比: NSGA-II {speed_ratio:.1f}x 倍于MOSA")
    
    if improvement > 50:
        print("  ⚠️  警告: MOSA性能仍然异常优秀!")
        return False
    elif improvement < -20:
        print("  ✅ 正常: MOSA性能在合理范围内")
        return True
    else:
        print("  ✅ 正常: MOSA与NSGA-II性能相当")
        return True

if __name__ == "__main__":
    quick_test() 