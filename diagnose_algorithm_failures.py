#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
算法失败诊断脚本
测试各个算法在小规模问题上的执行情况，诊断失败原因
"""

import time
import traceback
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from problem.mo_dhfsp import MO_DHFSP_Problem
from utils.data_generator import DataGenerator

print("开始算法诊断...")

# 生成一个简单的测试问题
def create_test_problem():
    """创建一个简单的测试问题"""
    generator = DataGenerator(seed=42)
    
    problem_data = generator.generate_problem(
        n_jobs=10,  # 小规模：10个作业
        n_factories=2,  # 2个工厂
        n_stages=3,  # 3个阶段
        machines_per_stage=[3, 3, 3],  # 每阶段3台机器
        processing_time_range=(1, 10),
        due_date_tightness=1.5
    )
    
    return MO_DHFSP_Problem(problem_data)

# 测试RL-Chaotic-HHO算法
def test_rl_chaotic_hho():
    """测试RL-Chaotic-HHO算法"""
    print("\n1. 测试 RL-Chaotic-HHO 算法:")
    try:
        from algorithm.rl_chaotic_hho import RL_ChaoticHHO_Optimizer
        
        problem = create_test_problem()
        optimizer = RL_ChaoticHHO_Optimizer(
            problem=problem,
            population_size=20,
            max_iterations=10,
            pareto_size_limit=100
        )
        
        start_time = time.time()
        pareto_solutions, convergence_data = optimizer.optimize()
        end_time = time.time()
        
        print(f"  ✅ 成功执行，耗时: {end_time - start_time:.2f}s")
        print(f"  📊 找到解数量: {len(pareto_solutions)}")
        
        if pareto_solutions:
            makespans = [sol.makespan for sol in pareto_solutions]
            tardiness = [sol.total_tardiness for sol in pareto_solutions]
            print(f"  📈 完工时间范围: {min(makespans):.2f} - {max(makespans):.2f}")
            print(f"  📈 拖期范围: {min(tardiness):.2f} - {max(tardiness):.2f}")
        
        return True, len(pareto_solutions)
        
    except Exception as e:
        print(f"  ❌ 失败: {str(e)}")
        print(f"  🔍 错误详情:\n{traceback.format_exc()}")
        return False, 0

# 测试I-NSGA-II算法
def test_improved_nsga2():
    """测试I-NSGA-II算法"""
    print("\n2. 测试 I-NSGA-II 算法:")
    try:
        from algorithm.improved_nsga2 import ImprovedNSGA2_Optimizer
        
        problem = create_test_problem()
        optimizer = ImprovedNSGA2_Optimizer(
            problem=problem,
            population_size=20,
            max_iterations=10,
            pareto_size_limit=100
        )
        
        start_time = time.time()
        pareto_solutions, convergence_data = optimizer.optimize()
        end_time = time.time()
        
        print(f"  ✅ 成功执行，耗时: {end_time - start_time:.2f}s")
        print(f"  📊 找到解数量: {len(pareto_solutions)}")
        
        if pareto_solutions:
            makespans = [sol.makespan for sol in pareto_solutions]
            tardiness = [sol.total_tardiness for sol in pareto_solutions]
            print(f"  📈 完工时间范围: {min(makespans):.2f} - {max(makespans):.2f}")
            print(f"  📈 拖期范围: {min(tardiness):.2f} - {max(tardiness):.2f}")
        
        return True, len(pareto_solutions)
        
    except Exception as e:
        print(f"  ❌ 失败: {str(e)}")
        print(f"  🔍 错误详情:\n{traceback.format_exc()}")
        return False, 0

# 测试MOPSO算法
def test_mopso():
    """测试MOPSO算法"""
    print("\n3. 测试 MOPSO 算法:")
    try:
        from algorithm.mopso import MOPSO_Optimizer
        
        problem = create_test_problem()
        optimizer = MOPSO_Optimizer(
            problem=problem,
            swarm_size=20,
            max_iterations=10,
            archive_size=100
        )
        
        start_time = time.time()
        pareto_solutions, convergence_data = optimizer.optimize()
        end_time = time.time()
        
        print(f"  ✅ 成功执行，耗时: {end_time - start_time:.2f}s")
        print(f"  📊 找到解数量: {len(pareto_solutions)}")
        
        if pareto_solutions:
            makespans = [sol.makespan for sol in pareto_solutions]
            tardiness = [sol.total_tardiness for sol in pareto_solutions]
            print(f"  📈 完工时间范围: {min(makespans):.2f} - {max(makespans):.2f}")
            print(f"  📈 拖期范围: {min(tardiness):.2f} - {max(tardiness):.2f}")
        
        return True, len(pareto_solutions)
        
    except Exception as e:
        print(f"  ❌ 失败: {str(e)}")
        print(f"  🔍 错误详情:\n{traceback.format_exc()}")
        return False, 0

# 测试MODE算法
def test_mode():
    """测试MODE算法"""
    print("\n4. 测试 MODE 算法:")
    try:
        from algorithm.mode import MODE_Optimizer
        
        problem = create_test_problem()
        optimizer = MODE_Optimizer(
            problem=problem,
            population_size=20,
            max_generations=10
        )
        
        start_time = time.time()
        pareto_solutions, convergence_data = optimizer.optimize()
        end_time = time.time()
        
        print(f"  ✅ 成功执行，耗时: {end_time - start_time:.2f}s")
        print(f"  📊 找到解数量: {len(pareto_solutions)}")
        
        if pareto_solutions:
            makespans = [sol.makespan for sol in pareto_solutions]
            tardiness = [sol.total_tardiness for sol in pareto_solutions]
            print(f"  📈 完工时间范围: {min(makespans):.2f} - {max(makespans):.2f}")
            print(f"  📈 拖期范围: {min(tardiness):.2f} - {max(tardiness):.2f}")
        
        return True, len(pareto_solutions)
        
    except Exception as e:
        print(f"  ❌ 失败: {str(e)}")
        print(f"  🔍 错误详情:\n{traceback.format_exc()}")
        return False, 0

# 测试DQN算法
def test_dqn():
    """测试DQN算法"""
    print("\n5. 测试 DQN 算法:")
    try:
        from algorithm.dqn_algorithm_wrapper import DQNAlgorithmWrapper
        
        problem = create_test_problem()
        optimizer = DQNAlgorithmWrapper(
            problem=problem,
            max_iterations=10,
            memory_size=1000,
            batch_size=32
        )
        
        start_time = time.time()
        pareto_solutions, convergence_data = optimizer.optimize()
        end_time = time.time()
        
        print(f"  ✅ 成功执行，耗时: {end_time - start_time:.2f}s")
        print(f"  📊 找到解数量: {len(pareto_solutions)}")
        
        if pareto_solutions:
            makespans = [sol.makespan for sol in pareto_solutions]
            tardiness = [sol.total_tardiness for sol in pareto_solutions]
            print(f"  📈 完工时间范围: {min(makespans):.2f} - {max(makespans):.2f}")
            print(f"  📈 拖期范围: {min(tardiness):.2f} - {max(tardiness):.2f}")
        
        return True, len(pareto_solutions)
        
    except Exception as e:
        print(f"  ❌ 失败: {str(e)}")
        print(f"  🔍 错误详情:\n{traceback.format_exc()}")
        return False, 0

# 测试QL-ABC算法
def test_qlabc():
    """测试QL-ABC算法"""
    print("\n6. 测试 QL-ABC 算法:")
    try:
        from algorithm.ql_abc import QLABC_Optimizer
        
        problem = create_test_problem()
        optimizer = QLABC_Optimizer(
            problem=problem,
            population_size=20,
            max_iterations=10,
            limit=5
        )
        
        start_time = time.time()
        pareto_solutions, convergence_data = optimizer.optimize()
        end_time = time.time()
        
        print(f"  ✅ 成功执行，耗时: {end_time - start_time:.2f}s")
        print(f"  📊 找到解数量: {len(pareto_solutions)}")
        
        if pareto_solutions:
            makespans = [sol.makespan for sol in pareto_solutions]
            tardiness = [sol.total_tardiness for sol in pareto_solutions]
            print(f"  📈 完工时间范围: {min(makespans):.2f} - {max(makespans):.2f}")
            print(f"  📈 拖期范围: {min(tardiness):.2f} - {max(tardiness):.2f}")
        
        return True, len(pareto_solutions)
        
    except Exception as e:
        print(f"  ❌ 失败: {str(e)}")
        print(f"  🔍 错误详情:\n{traceback.format_exc()}")
        return False, 0

# 主诊断函数
def main():
    """主诊断函数"""
    print("=" * 80)
    print("MO-DHFSP 算法失败诊断报告")
    print("=" * 80)
    print("测试问题规模: 10作业×2工厂×3阶段×3机器/阶段")
    print("参数设置: 种群20, 迭代10代 (小规模快速测试)")
    
    # 测试所有算法
    test_results = {}
    
    test_results['RL-Chaotic-HHO'] = test_rl_chaotic_hho()
    test_results['I-NSGA-II'] = test_improved_nsga2()
    test_results['MOPSO'] = test_mopso()
    test_results['MODE'] = test_mode()
    test_results['DQN'] = test_dqn()
    test_results['QL-ABC'] = test_qlabc()
    
    # 汇总结果
    print("\n" + "=" * 80)
    print("诊断结果汇总")
    print("=" * 80)
    
    success_count = 0
    fail_count = 0
    
    print(f"{'算法名称':<20} {'状态':<10} {'解数量':<10}")
    print("-" * 40)
    
    for alg_name, (success, solution_count) in test_results.items():
        status = "成功 ✅" if success else "失败 ❌"
        print(f"{alg_name:<20} {status:<10} {solution_count:<10}")
        
        if success:
            success_count += 1
        else:
            fail_count += 1
    
    print("-" * 40)
    print(f"成功: {success_count}/6, 失败: {fail_count}/6")
    
    if fail_count > 0:
        print(f"\n❌ 发现 {fail_count} 个算法执行失败！")
        print("💡 建议:")
        print("   1. 检查失败算法的实现代码")
        print("   2. 确认算法参数配置是否合理")
        print("   3. 检查算法依赖是否正确导入")
        print("   4. 确认问题数据格式是否兼容")
        
        # 显示失败的算法
        failed_algorithms = [alg for alg, (success, _) in test_results.items() if not success]
        print(f"\n🔧 失败算法列表: {', '.join(failed_algorithms)}")
    else:
        print("\n✅ 所有算法在小规模测试中均正常运行！")
        print("💡 可能的问题:")
        print("   1. 在大规模问题上算法性能问题")
        print("   2. 参数配置在大规模下不适用")
        print("   3. 内存或时间限制问题")

if __name__ == "__main__":
    main() 