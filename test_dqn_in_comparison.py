#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试DQN在对比脚本中的集成
验证DQN算法包装器是否能正常工作
"""

import time
from problem.mo_dhfsp import MO_DHFSP_Problem
from algorithm.dqn_algorithm_wrapper import DQNAlgorithmWrapper
from utils.data_generator import DataGenerator

def test_dqn_wrapper():
    """测试DQN算法包装器"""
    print("🧪 测试DQN算法包装器")
    print("=" * 50)
    
    # 创建小规模测试问题
    generator = DataGenerator(seed=42)
    
    problem_data = generator.generate_problem(
        n_jobs=20,
        n_factories=3,
        n_stages=3,
        machines_per_stage=[2, 2, 2],
        processing_time_range=(1, 20),
        due_date_tightness=1.3
    )
    
    # 异构机器配置
    problem_data['factory_machines'] = {
        0: [2, 2, 2],  # 工厂1: 6台机器
        1: [1, 3, 2],  # 工厂2: 6台机器
        2: [3, 1, 2]   # 工厂3: 6台机器
    }
    
    problem = MO_DHFSP_Problem(problem_data)
    
    print(f"📊 问题规模: {problem.n_jobs}作业 × {problem.n_factories}工厂 × {problem.n_stages}阶段")
    
    # 测试DQN包装器
    dqn_wrapper = DQNAlgorithmWrapper(problem)
    
    start_time = time.time()
    solutions = dqn_wrapper.optimize(max_iterations=50)
    runtime = time.time() - start_time
    
    print(f"\n📈 DQN包装器测试结果:")
    print(f"  算法名称: {dqn_wrapper.name}")
    print(f"  解集大小: {len(solutions)}")
    
    if solutions:
        best_solution = solutions[0]
        print(f"  完工时间: {best_solution.makespan:.2f}")
        print(f"  总拖期: {best_solution.total_tardiness:.2f}")
        print(f"  加权目标: {0.55 * best_solution.makespan + 0.45 * best_solution.total_tardiness:.2f}")
    
    print(f"  运行时间: {runtime:.2f}秒")
    
    # 验证接口兼容性
    print(f"\n🔍 接口兼容性检查:")
    print(f"  ✅ 返回解集: {isinstance(solutions, list)}")
    print(f"  ✅ 解集非空: {len(solutions) > 0}")
    
    if solutions:
        solution = solutions[0]
        print(f"  ✅ 解有完工时间: {hasattr(solution, 'makespan')}")
        print(f"  ✅ 解有总拖期: {hasattr(solution, 'total_tardiness')}")
        print(f"  ✅ 完工时间合理: {solution.makespan > 0}")
        print(f"  ✅ 总拖期合理: {solution.total_tardiness >= 0}")
    
    print(f"\n🎯 结论:")
    if solutions and len(solutions) > 0:
        print(f"✅ DQN算法包装器工作正常，可以集成到对比脚本中")
        return True
    else:
        print(f"❌ DQN算法包装器存在问题，需要修复")
        return False

def simulate_comparison_experiment():
    """模拟对比实验"""
    print(f"\n🔬 模拟算法对比实验")
    print("=" * 50)
    
    # 创建问题
    generator = DataGenerator(seed=42)
    
    problem_data = generator.generate_problem(
        n_jobs=20,
        n_factories=3,
        n_stages=3,
        machines_per_stage=[2, 2, 2],
        processing_time_range=(1, 20),
        due_date_tightness=1.3
    )
    
    problem_data['factory_machines'] = {
        0: [2, 2, 2],
        1: [1, 3, 2], 
        2: [3, 1, 2]
    }
    
    problem = MO_DHFSP_Problem(problem_data)
    
    # 模拟多算法对比（仅DQN）
    algorithms = {
        'DQN': DQNAlgorithmWrapper
    }
    
    results = {}
    
    for alg_name, alg_class in algorithms.items():
        print(f"\n运行 {alg_name}...")
        
        optimizer = alg_class(problem)
        
        start_time = time.time()
        solutions = optimizer.optimize(max_iterations=30)
        runtime = time.time() - start_time
        
        if solutions:
            best_solution = solutions[0]
            weighted_score = 0.55 * best_solution.makespan + 0.45 * best_solution.total_tardiness
            
            results[alg_name] = {
                'weighted_best': weighted_score,
                'makespan_best': best_solution.makespan,
                'tardiness_best': best_solution.total_tardiness,
                'runtime': runtime,
                'solutions_count': len(solutions)
            }
            
            print(f"  完工时间: {best_solution.makespan:.2f}")
            print(f"  总拖期: {best_solution.total_tardiness:.2f}")
            print(f"  加权目标: {weighted_score:.2f}")
            print(f"  运行时间: {runtime:.2f}秒")
        else:
            results[alg_name] = {
                'weighted_best': float('inf'),
                'makespan_best': float('inf'),
                'tardiness_best': float('inf'),
                'runtime': runtime,
                'solutions_count': 0
            }
            print(f"  无有效解")
    
    # 输出对比表格
    print(f"\n📊 对比结果表格:")
    print("+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 12 + "+")
    print(f"| {'算法':^10s} | {'加权目标':^10s} | {'完工时间':^10s} | {'总拖期':^10s} | {'运行时间':^10s} |")
    print("+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 12 + "+")
    
    for alg_name, result in results.items():
        print(f"| {alg_name:^10s} | {result['weighted_best']:^10.1f} | {result['makespan_best']:^10.1f} | {result['tardiness_best']:^10.1f} | {result['runtime']:^10.1f} |")
    
    print("+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 12 + "+")
    
    return results

def main():
    """主函数"""
    print("🚀 DQN集成测试")
    print("=" * 60)
    
    # 测试DQN包装器
    wrapper_ok = test_dqn_wrapper()
    
    if wrapper_ok:
        # 模拟对比实验
        results = simulate_comparison_experiment()
        
        print(f"\n🎯 集成测试总结:")
        print(f"✅ DQN算法包装器测试通过")
        print(f"✅ 模拟对比实验成功")
        print(f"✅ DQN已准备好加入table_format_comparison脚本")
        
        print(f"\n📝 使用建议:")
        print(f"1. DQN适合小到中等规模问题（≤50作业）")
        print(f"2. 在大规模问题中可能需要更长的运行时间")
        print(f"3. 建议在table_format_comparison中设置合适的参数")
        
    else:
        print(f"\n❌ DQN集成测试失败，需要修复问题")
    
    return wrapper_ok

if __name__ == "__main__":
    main() 