#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的MOPSO测试程序
用于调试MOPSO算法
"""

import time
import numpy as np
from problem.mo_dhfsp import MO_DHFSP_Problem
from algorithm.mopso import MOPSO_Optimizer
from utils.data_generator import DataGenerator

def simple_test():
    """简单测试"""
    print("简化MOPSO测试")
    print("=" * 50)
    
    # 生成简单测试问题
    generator = DataGenerator(seed=42)
    problem_data = generator.generate_problem(
        n_jobs=10,
        n_factories=2,
        n_stages=2,
        machines_per_stage=[2, 2],
        processing_time_range=(1, 5),
        due_date_tightness=1.5
    )
    
    # 设置紧急度
    problem_data['urgencies'] = [1.0 + i * 0.2 for i in range(10)]
    
    print(f"问题规模: {problem_data['n_jobs']}作业 × {problem_data['n_factories']}工厂 × {problem_data['n_stages']}阶段")
    print(f"机器配置: {problem_data['machines_per_stage']}")
    print(f"紧急度: {problem_data['urgencies']}")
    
    # 创建问题实例
    problem = MO_DHFSP_Problem(problem_data)
    
    # 测试问题基础功能
    print("\n测试问题基础功能...")
    random_solution = problem.generate_random_solution()
    print(f"随机解: 完工时间={random_solution.makespan:.2f}, 拖期={random_solution.total_tardiness:.2f}")
    
    # 测试create_solution方法
    print("\n测试create_solution方法...")
    factory_assignment = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # 交替分配
    test_solution = problem.create_solution(factory_assignment)
    print(f"测试解: 完工时间={test_solution.makespan:.2f}, 拖期={test_solution.total_tardiness:.2f}")
    print(f"工厂分配: {test_solution.factory_assignment}")
    print(f"作业序列: {test_solution.job_sequences}")
    
    # 创建MOPSO优化器
    print("\n创建MOPSO优化器...")
    optimizer = MOPSO_Optimizer(
        problem=problem,
        swarm_size=10,
        max_iterations=20,
        w=0.9,
        c1=2.0,
        c2=2.0,
        archive_size=20,
        mutation_prob=0.1
    )
    
    # 运行优化
    print("\n开始优化...")
    try:
        start_time = time.time()
        pareto_solutions, convergence_data = optimizer.optimize()
        end_time = time.time()
        
        print(f"\n优化完成!")
        print(f"运行时间: {end_time - start_time:.2f}秒")
        print(f"帕累托解数量: {len(pareto_solutions)}")
        
        if pareto_solutions:
            makespans = [sol.makespan for sol in pareto_solutions]
            tardiness = [sol.total_tardiness for sol in pareto_solutions]
            print(f"完工时间范围: [{min(makespans):.2f}, {max(makespans):.2f}]")
            print(f"总拖期范围: [{min(tardiness):.2f}, {max(tardiness):.2f}]")
            
            # 显示前几个解
            print(f"\n前5个帕累托解:")
            for i, sol in enumerate(pareto_solutions[:5]):
                print(f"  解{i+1}: 完工时间={sol.makespan:.2f}, 拖期={sol.total_tardiness:.2f}")
                print(f"       工厂分配={sol.factory_assignment}")
        
        return True
        
    except Exception as e:
        print(f"优化过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = simple_test()
    if success:
        print("\n✓ MOPSO测试成功!")
    else:
        print("\n✗ MOPSO测试失败!") 