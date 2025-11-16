#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NSGA-II调试测试程序
"""

import time
from problem.mo_dhfsp import MO_DHFSP_Problem
from algorithm.nsga2 import NSGA2_Optimizer
from utils.data_generator import DataGenerator

def test_nsga2_debug():
    """带调试信息的NSGA-II测试"""
    print("🐛 NSGA-II调试测试")
    
    # 生成更小的测试数据
    generator = DataGenerator(seed=42)
    problem_data = generator.generate_problem(
        n_jobs=5,  # 减小到5个作业
        n_factories=2,
        n_stages=2,  # 减小到2个阶段
        machines_per_stage=[1, 1],  # 每阶段1台机器
        processing_time_range=(1, 10),
        due_date_tightness=1.5
    )
    
    # 创建问题实例
    problem = MO_DHFSP_Problem(problem_data)
    
    print(f"📊 测试问题: {problem.n_jobs}作业, {problem.n_factories}工厂, {problem.n_stages}阶段")
    
    try:
        # 创建小规模NSGA-II优化器
        print("\n🔬 开始NSGA-II调试测试...")
        optimizer = NSGA2_Optimizer(problem, population_size=10, max_generations=3)
        
        print("✅ 优化器创建成功")
        
        # 手动测试初始化
        print("🔧 测试初始化...")
        optimizer._initialize_population()
        print(f"✅ 初始化成功，种群大小: {len(optimizer.population)}")
        
        # 测试一代进化
        print("🔧 测试一代进化...")
        offspring = optimizer._generate_offspring()
        print(f"✅ 子代生成成功，子代数量: {len(offspring)}")
        
        # 测试环境选择
        print("🔧 测试环境选择...")
        combined_pop = optimizer.population + offspring
        print(f"📊 合并种群大小: {len(combined_pop)}")
        
        new_pop = optimizer._environmental_selection(combined_pop)
        print(f"✅ 环境选择成功，新种群大小: {len(new_pop)}")
        
        # 运行完整优化（小规模）
        print("🔧 运行完整优化...")
        start_time = time.time()
        pareto_solutions, convergence_data = optimizer.optimize()
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        print(f"✅ NSGA-II运行成功!")
        print(f"⏱️  运行时间: {execution_time:.2f}秒")
        print(f"📊 帕累托解数量: {len(pareto_solutions)}")
        
        if pareto_solutions:
            best_makespan = min(sol.makespan for sol in pareto_solutions)
            best_tardiness = min(sol.total_tardiness for sol in pareto_solutions)
            print(f"🎯 最优完工时间: {best_makespan:.2f}")
            print(f"📈 最优总拖期: {best_tardiness:.2f}")
        
    except Exception as e:
        print(f"❌ NSGA-II运行失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_nsga2_debug() 