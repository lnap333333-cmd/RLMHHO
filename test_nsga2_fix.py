#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NSGA-II修复测试程序
"""

import time
from problem.mo_dhfsp import MO_DHFSP_Problem
from algorithm.nsga2 import NSGA2_Optimizer
from utils.data_generator import DataGenerator

def test_nsga2_fix():
    """测试NSGA-II修复效果"""
    print("🧪 测试NSGA-II修复效果")
    
    # 生成测试数据
    generator = DataGenerator(seed=42)
    problem_data = generator.generate_problem(
        n_jobs=10,
        n_factories=2,
        n_stages=3,
        machines_per_stage=[2, 2, 2],
        processing_time_range=(1, 20),
        due_date_tightness=1.2
    )
    
    # 创建问题实例
    problem = MO_DHFSP_Problem(problem_data)
    
    print(f"📊 测试问题: {problem.n_jobs}作业, {problem.n_factories}工厂")
    print(f"⚡ 紧急度范围: [{min(problem.urgencies):.2f}, {max(problem.urgencies):.2f}]")
    
    try:
        # 创建NSGA-II优化器
        print("\n🔬 开始运行NSGA-II...")
        optimizer = NSGA2_Optimizer(problem, population_size=30, max_generations=20)
        
        # 运行优化
        start_time = time.time()
        pareto_solutions, convergence_data = optimizer.optimize()
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # 输出结果
        print(f"✅ NSGA-II运行成功!")
        print(f"⏱️  运行时间: {execution_time:.2f}秒")
        print(f"📊 帕累托解数量: {len(pareto_solutions)}")
        
        if pareto_solutions:
            best_makespan = min(sol.makespan for sol in pareto_solutions)
            best_tardiness = min(sol.total_tardiness for sol in pareto_solutions)
            print(f"🎯 最优完工时间: {best_makespan:.2f}")
            print(f"📈 最优总拖期: {best_tardiness:.2f}")
        else:
            print("⚠️  未找到帕累托解")
            
    except Exception as e:
        print(f"❌ NSGA-II运行失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_nsga2_fix() 