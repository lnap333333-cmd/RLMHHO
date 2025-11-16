#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QL-ABC完整集成测试脚本 - 包含超体积和IGD指标
只运行小规模实验来验证功能
"""

import os
import sys
import time
import numpy as np
from datetime import datetime

def test_qlabc_full_integration():
    """测试QL-ABC完整集成功能"""
    print("=" * 80)
    print("QL-ABC完整集成测试 - 包含超体积和IGD指标")
    print("=" * 80)
    
    try:
        # 导入必要模块
        print("1. 导入模块...")
        from algorithm.ql_abc import QLABC_Optimizer
        from algorithm.rl_chaotic_hho import RL_ChaoticHHO_Optimizer
        from problem.mo_dhfsp import MO_DHFSP_Problem
        from utils.data_generator import DataGenerator
        print("   ✅ 模块导入成功")
        
        # 生成小规模测试问题
        print("\n2. 生成测试问题...")
        generator = DataGenerator(seed=42)
        problem_data = generator.generate_problem(
            n_jobs=10,
            n_factories=2,
            n_stages=3,
            machines_per_stage=[2, 2, 2],
            processing_time_range=(1, 10),
            due_date_tightness=1.5
        )
        
        # 添加异构机器配置
        problem_data['heterogeneous_machines'] = {
            0: [2, 2, 2],
            1: [2, 2, 2]
        }
        print("   ✅ 测试问题生成成功")
        
        # 创建问题实例
        print("\n3. 创建问题实例...")
        problem = MO_DHFSP_Problem(problem_data)
        print("   ✅ 问题实例创建成功")
        
        # 测试QL-ABC算法
        print("\n4. 测试QL-ABC算法...")
        qlabc_optimizer = QLABC_Optimizer(
            problem,
            population_size=10,
            max_iterations=5,
            learning_rate=0.1,
            epsilon=0.3
        )
        
        start_time = time.time()
        qlabc_solutions, qlabc_convergence = qlabc_optimizer.optimize()
        qlabc_runtime = time.time() - start_time
        
        print(f"   ✅ QL-ABC运行完成，找到{len(qlabc_solutions)}个解，耗时{qlabc_runtime:.2f}s")
        
        # 测试主算法作为对比
        print("\n5. 测试主算法对比...")
        main_optimizer = RL_ChaoticHHO_Optimizer(
            problem,
            population_size=10,
            max_iterations=5
        )
        
        start_time = time.time()
        main_solutions, main_convergence = main_optimizer.optimize()
        main_runtime = time.time() - start_time
        
        print(f"   ✅ 主算法运行完成，找到{len(main_solutions)}个解，耗时{main_runtime:.2f}s")
        
        # 测试指标计算函数
        print("\n6. 测试性能指标计算...")
        
        # 导入指标计算函数
        from table_format_comparison_with_ql_abc_full import (
            calculate_hypervolume, 
            calculate_igd, 
            calculate_combined_pareto_front
        )
        
        # 计算QL-ABC指标
        if qlabc_solutions:
            qlabc_hv = calculate_hypervolume(qlabc_solutions)
            qlabc_igd = calculate_igd(qlabc_solutions)
            print(f"   QL-ABC - 超体积: {qlabc_hv:.0f}, IGD: {qlabc_igd:.2f}")
        
        # 计算主算法指标
        if main_solutions:
            main_hv = calculate_hypervolume(main_solutions)
            main_igd = calculate_igd(main_solutions)
            print(f"   主算法 - 超体积: {main_hv:.0f}, IGD: {main_igd:.2f}")
        
        print("   ✅ 性能指标计算成功")
        
        # 测试联合帕累托前沿计算
        print("\n7. 测试联合帕累托前沿计算...")
        all_results = {
            'QL-ABC': {'pareto_solutions': qlabc_solutions},
            'RL-Chaotic-HHO': {'pareto_solutions': main_solutions}
        }
        
        combined_front = calculate_combined_pareto_front(all_results)
        print(f"   ✅ 联合帕累托前沿包含{len(combined_front)}个解")
        
        # 结果对比
        print("\n8. 结果对比...")
        if qlabc_solutions and main_solutions:
            qlabc_best_makespan = min(sol.makespan for sol in qlabc_solutions)
            qlabc_best_tardiness = min(sol.total_tardiness for sol in qlabc_solutions)
            
            main_best_makespan = min(sol.makespan for sol in main_solutions)
            main_best_tardiness = min(sol.total_tardiness for sol in main_solutions)
            
            print(f"   QL-ABC最佳解: 完工时间={qlabc_best_makespan:.2f}, 拖期={qlabc_best_tardiness:.2f}")
            print(f"   主算法最佳解: 完工时间={main_best_makespan:.2f}, 拖期={main_best_tardiness:.2f}")
            print("   ✅ 结果对比完成")
        
        print("\n" + "=" * 80)
        print("🎉 QL-ABC完整集成测试成功！所有功能正常")
        print("✅ QL-ABC算法运行正常")
        print("✅ 超体积指标计算正常")
        print("✅ IGD指标计算正常")
        print("✅ 联合帕累托前沿计算正常")
        print("✅ 可以运行完整对比实验")
        print("=" * 80)
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_qlabc_full_integration()
    if success:
        print("\n🚀 准备运行完整对比实验...")
        print("请运行: python table_format_comparison_with_ql_abc_full.py")
    else:
        print("\n❌ 请先修复错误再运行完整实验")
        sys.exit(1) 