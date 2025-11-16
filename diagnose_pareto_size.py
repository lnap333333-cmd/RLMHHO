#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断主体算法解集数量少的原因
"""

import time
import numpy as np
from algorithm.rl_chaotic_hho import RL_ChaoticHHO_Optimizer
from problem.mo_dhfsp import MO_DHFSP_Problem
from utils.data_generator import DataGenerator

def create_test_problem():
    """创建测试问题"""
    generator = DataGenerator(seed=42)
    
    # 生成小规模测试问题
    problem_data = generator.generate_problem(
        n_jobs=20,
        n_factories=3,
        n_stages=2,
        machines_per_stage=[2, 2],  # 每个阶段2台机器
        processing_time_range=(1, 10),
        due_date_tightness=1.5
    )
    
    # 创建异构机器配置
    machines_config = [
        [2, 2],  # 工厂1: 4台机器
        [2, 2],  # 工厂2: 4台机器  
        [2, 2]   # 工厂3: 4台机器
    ]
    
    # 创建问题实例
    problem_data['n_factories'] = 3
    problem_data['factory_machines'] = {
        0: [2, 2],  # 工厂1: 4台机器
        1: [2, 2],  # 工厂2: 4台机器  
        2: [2, 2]   # 工厂3: 4台机器
    }
    
    problem = MO_DHFSP_Problem(problem_data)
    
    return problem

def diagnose_pareto_size_issue():
    """诊断帕累托解集数量问题"""
    print("🔍 诊断主体算法解集数量问题")
    print("=" * 50)
    
    # 创建测试问题
    problem = create_test_problem()
    
    print(f"测试问题规模:")
    print(f"  作业数: {problem.n_jobs}")
    print(f"  工厂数: {problem.n_factories}")
    print(f"  阶段数: {problem.n_stages}")
    
    # 创建优化器实例
    optimizer = RL_ChaoticHHO_Optimizer(
        problem=problem,
        max_iterations=30,  # 减少迭代次数
        population_size_override=30,  # 减少种群大小
        learning_rate=0.001,
        epsilon_decay=0.995,
        gamma=0.98
    )
    
    print(f"\n算法配置:")
    print(f"  种群大小: {optimizer.population_size}")
    print(f"  最大迭代: {optimizer.max_iterations}")
    print(f"  帕累托前沿限制: 30 (硬编码)")
    
    # 运行优化并监控
    print(f"\n🚀 开始优化监控...")
    start_time = time.time()
    
    try:
        # 初始化种群
        optimizer._initialize_population()
        print(f"✅ 初始化完成，种群大小: {len(optimizer.population)}")
        
        # 更新初始帕累托前沿
        optimizer._update_pareto_front()
        print(f"✅ 初始帕累托解数量: {len(optimizer.pareto_solutions)}")
        
        # 逐代监控
        for iteration in range(optimizer.max_iterations):
            optimizer.current_iteration = iteration
            
            # 执行一代优化
            state = optimizer._get_current_state()
            action = optimizer.rl_coordinator.select_action(state)
            optimizer._execute_strategy(action)
            optimizer._harris_hawks_search()
            
            # 更新帕累托前沿
            previous_size = len(optimizer.pareto_solutions)
            optimizer._update_pareto_front()
            current_size = len(optimizer.pareto_solutions)
            
            # 计算奖励并更新RL
            reward = optimizer._calculate_reward(previous_size, current_size)
            next_state = optimizer._get_current_state()
            optimizer.rl_coordinator.update(state, action, reward, next_state)
            
            # 记录收敛数据
            optimizer._record_convergence_data()
            
            # 详细监控输出
            if iteration % 5 == 0 or iteration == optimizer.max_iterations - 1:
                print(f"代数 {iteration:2d}: 帕累托解={current_size:2d} "
                      f"(变化: {current_size-previous_size:+2d}), "
                      f"种群大小={len(optimizer.population):2d}, "
                      f"动作={action}, 奖励={reward:.4f}")
                
                # 分析种群多样性
                if optimizer.population:
                    makespans = [sol.makespan for sol in optimizer.population]
                    tardiness = [sol.total_tardiness for sol in optimizer.population]
                    print(f"      种群完工时间范围: {min(makespans):.1f} - {max(makespans):.1f}")
                    print(f"      种群拖期范围: {min(tardiness):.1f} - {max(tardiness):.1f}")
                
                # 分析帕累托前沿
                if optimizer.pareto_solutions:
                    p_makespans = [sol.makespan for sol in optimizer.pareto_solutions]
                    p_tardiness = [sol.total_tardiness for sol in optimizer.pareto_solutions]
                    print(f"      帕累托完工时间范围: {min(p_makespans):.1f} - {max(p_makespans):.1f}")
                    print(f"      帕累托拖期范围: {min(p_tardiness):.1f} - {max(p_tardiness):.1f}")
        
        runtime = time.time() - start_time
        final_pareto_size = len(optimizer.pareto_solutions)
        
        print(f"\n📊 诊断结果:")
        print(f"  运行时间: {runtime:.2f}秒")
        print(f"  最终帕累托解数量: {final_pareto_size}")
        print(f"  种群大小: {len(optimizer.population)}")
        
        # 分析可能的原因
        print(f"\n🔍 问题分析:")
        
        # 1. 检查帕累托前沿限制
        print(f"1. 帕累托前沿大小限制:")
        print(f"   硬编码限制: 30个解")
        if final_pareto_size >= 25:
            print(f"   ⚠️  接近限制上限，可能被截断")
        else:
            print(f"   ✅ 未达到限制")
        
        # 2. 检查解的多样性
        if optimizer.pareto_solutions:
            makespans = [sol.makespan for sol in optimizer.pareto_solutions]
            tardiness = [sol.total_tardiness for sol in optimizer.pareto_solutions]
            
            makespan_std = np.std(makespans)
            tardiness_std = np.std(tardiness)
            
            print(f"2. 解的多样性:")
            print(f"   完工时间标准差: {makespan_std:.2f}")
            print(f"   拖期标准差: {tardiness_std:.2f}")
            
            if makespan_std < 10 and tardiness_std < 10:
                print(f"   ⚠️  解集多样性较低，可能收敛过早")
            else:
                print(f"   ✅ 解集多样性良好")
        
        # 3. 检查支配关系
        dominated_count = 0
        if len(optimizer.population) > 1:
            for i, sol1 in enumerate(optimizer.population):
                for j, sol2 in enumerate(optimizer.population):
                    if i != j:
                        if (sol2.makespan <= sol1.makespan and 
                            sol2.total_tardiness <= sol1.total_tardiness and
                            (sol2.makespan < sol1.makespan or sol2.total_tardiness < sol1.total_tardiness)):
                            dominated_count += 1
                            break
        
        non_dominated_ratio = (len(optimizer.population) - dominated_count) / len(optimizer.population) * 100
        print(f"3. 种群非支配解比例:")
        print(f"   非支配解: {len(optimizer.population) - dominated_count}/{len(optimizer.population)} ({non_dominated_ratio:.1f}%)")
        
        if non_dominated_ratio < 20:
            print(f"   ⚠️  非支配解比例过低，种群收敛过度")
        else:
            print(f"   ✅ 非支配解比例合理")
        
        # 4. 检查算法参数影响
        print(f"4. 算法参数分析:")
        print(f"   鹰群分组配置: 探索70% + 开发15% + 平衡10% + 精英5%")
        print(f"   RL学习率: {optimizer.rl_coordinator.learning_rate}")
        print(f"   RL探索衰减: {optimizer.rl_coordinator.epsilon_decay}")
        
        # 5. 建议解决方案
        print(f"\n💡 改进建议:")
        if final_pareto_size < 10:
            print(f"1. 增加帕累托前沿大小限制 (当前30 → 建议50+)")
            print(f"2. 调整多样性选择策略，增强拥挤距离计算")
            print(f"3. 增加种群大小或迭代次数")
            print(f"4. 调整鹰群分组比例，增加探索强度")
            print(f"5. 修改RL奖励函数，更重视多样性")
        
        return {
            'final_pareto_size': final_pareto_size,
            'population_size': len(optimizer.population),
            'non_dominated_ratio': non_dominated_ratio,
            'runtime': runtime
        }
        
    except Exception as e:
        print(f"❌ 诊断过程出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def test_pareto_manager():
    """测试帕累托管理器的行为"""
    print(f"\n🧪 测试帕累托管理器")
    print("=" * 30)
    
    from algorithm.pareto_manager import ParetoManager
    from problem.mo_dhfsp import Solution
    
    manager = ParetoManager()
    
    # 创建一些测试解
    test_solutions = []
    for i in range(20):
        # 创建虚拟解
        solution = Solution(
            factory_assignment=[0] * 10,
            job_sequences=[[j for j in range(10)]]
        )
        # 设置不同的目标函数值
        solution.makespan = 100 + i * 5 + np.random.normal(0, 2)
        solution.total_tardiness = 200 - i * 3 + np.random.normal(0, 5)
        test_solutions.append(solution)
    
    print(f"创建了 {len(test_solutions)} 个测试解")
    
    # 更新帕累托前沿
    pareto_solutions = manager.update_pareto_front(test_solutions)
    print(f"帕累托前沿包含 {len(pareto_solutions)} 个解")
    
    # 测试多样性选择
    diverse_solutions = manager.select_diverse_solutions(pareto_solutions, 10)
    print(f"多样性选择后包含 {len(diverse_solutions)} 个解")
    
    # 分析结果
    if pareto_solutions:
        makespans = [sol.makespan for sol in pareto_solutions]
        tardiness = [sol.total_tardiness for sol in pareto_solutions]
        print(f"帕累托前沿完工时间范围: {min(makespans):.1f} - {max(makespans):.1f}")
        print(f"帕累托前沿拖期范围: {min(tardiness):.1f} - {max(tardiness):.1f}")

def main():
    """主函数"""
    print("🔬 主体算法解集数量诊断")
    print("=" * 60)
    
    # 诊断主要问题
    result = diagnose_pareto_size_issue()
    
    # 测试帕累托管理器
    test_pareto_manager()
    
    if result:
        print(f"\n📋 诊断总结:")
        print(f"  最终帕累托解数量: {result['final_pareto_size']}")
        print(f"  非支配解比例: {result['non_dominated_ratio']:.1f}%")
        print(f"  运行时间: {result['runtime']:.2f}秒")
        
        if result['final_pareto_size'] < 15:
            print(f"  🔴 解集数量偏少，需要优化")
        elif result['final_pareto_size'] < 25:
            print(f"  🟡 解集数量一般，可以改进")
        else:
            print(f"  ✅ 解集数量合理")

if __name__ == "__main__":
    main() 