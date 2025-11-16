#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的MOSA算法测试
"""

from algorithm.mosa import MOSA_Optimizer
from problem.mo_dhfsp import MO_DHFSP_Problem
from utils.data_generator import DataGenerator
import random

def test_mosa():
    """测试MOSA算法"""
    print("开始测试MOSA算法...")
    
    # 设置随机种子
    random.seed(42)
    
    # 生成简单的测试数据
    generator = DataGenerator(seed=42)
    problem_data = generator.generate_problem(
        n_jobs=5,
        n_factories=2,
        n_stages=2,
        machines_per_stage=[1, 1],
        processing_time_range=(1, 10),
        due_date_tightness=1.5
    )
    
    # 创建问题实例
    problem = MO_DHFSP_Problem(problem_data)
    
    # 创建MOSA优化器
    optimizer = MOSA_Optimizer(problem, 
        initial_temperature=100.0,
        max_iterations=10,
        archive_size=10,
        neighborhood_size=2
    )
    
    # 运行优化
    try:
        pareto_solutions, convergence_data = optimizer.optimize()
        print(f'✅ MOSA算法测试成功! 找到 {len(pareto_solutions)} 个帕累托解')
        if pareto_solutions:
            best_sol = min(pareto_solutions, key=lambda x: 0.55*x.makespan + 0.45*x.total_tardiness)
            print(f'   最优解: 完工时间={best_sol.makespan:.2f}, 总拖期={best_sol.total_tardiness:.2f}')
            return True
    except Exception as e:
        print(f'❌ MOSA算法测试失败: {str(e)}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_mosa() 