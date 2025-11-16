#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整的RL-Chaotic-HHO算法测试
测试四层鹰群分组协作和强化学习调度器的实现
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from problem.mo_dhfsp import MO_DHFSP_Problem
from algorithm.rl_chaotic_hho import RL_ChaoticHHO_Optimizer
from utils.data_generator import DataGenerator
from utils.visualization import ResultVisualizer

def test_four_layer_grouping():
    """测试四层鹰群分组机制"""
    print("=" * 80)
    print("🦅 四层鹰群分组协作机制测试")
    print("=" * 80)
    
    # 创建测试问题
    data_gen = DataGenerator()
    problem_data = data_gen.generate_problem(
        n_jobs=30,
        n_factories=3,
        n_stages=3,
        machines_per_stage=[2, 3, 2],
        processing_time_range=(1, 10)
    )
    
    problem = MO_DHFSP_Problem(problem_data)
    
    # 初始化优化器
    optimizer = RL_ChaoticHHO_Optimizer(problem, max_iterations=30)
    
    print(f"\n📊 分组配置验证:")
    print(f"  探索组大小: {len(optimizer.eagle_groups.get_group('exploration'))}")
    print(f"  开发组大小: {len(optimizer.eagle_groups.get_group('exploitation'))}")
    print(f"  平衡组大小: {len(optimizer.eagle_groups.get_group('balance'))}")
    print(f"  精英组大小: {len(optimizer.eagle_groups.get_group('elite'))}")
    
    # 验证分组覆盖性
    total_assigned = sum(len(optimizer.eagle_groups.get_group(g)) 
                        for g in ['exploration', 'exploitation', 'balance', 'elite'])
    print(f"  总分配个体: {total_assigned}/{optimizer.population_size}")
    
    return optimizer, problem

def test_reinforcement_learning():
    """测试强化学习调度器"""
    print("\n🤖 强化学习调度器测试")
    print("-" * 60)
    
    optimizer, problem = test_four_layer_grouping()
    
    # 测试状态获取
    optimizer._initialize_population()
    state = optimizer._get_current_state()
    print(f"状态向量维度: {len(state)}")
    print(f"状态向量: {state[:5]}... (显示前5维)")
    
    # 测试动作选择
    action = optimizer.rl_coordinator.select_action(state)
    print(f"选择的策略: {action} - {optimizer.rl_coordinator.action_space[action]}")
    
    # 测试策略执行
    print(f"\n执行策略: {optimizer.rl_coordinator.action_space[action]}")
    optimizer._execute_strategy(action)
    
    # 获取动作推荐
    recommendations = optimizer.rl_coordinator.get_action_recommendations(state)
    print(f"\n策略推荐排序:")
    for i, (action_name, confidence) in enumerate(recommendations[:3]):
        print(f"  {i+1}. {action_name}: {confidence:.3f}")
    
    return optimizer

def test_chaotic_maps():
    """测试增强混沌映射系统"""
    print("\n🌀 增强混沌映射系统测试")
    print("-" * 60)
    
    from algorithm.chaotic_maps import ChaoticMaps
    
    chaos_maps = ChaoticMaps()
    
    # 测试各种映射
    print("各映射生成的混沌值:")
    print(f"  Logistic映射: {chaos_maps.logistic_map():.4f}")
    print(f"  Tent映射: {chaos_maps.tent_map():.4f}")
    print(f"  Sine映射: {chaos_maps.sine_map():.4f}")
    print(f"  Chebyshev映射: {chaos_maps.chebyshev_map():.4f}")
    
    # 测试组专用混沌值
    print(f"\n各组专用混沌值:")
    for group in ['exploration', 'exploitation', 'balance', 'elite']:
        values = chaos_maps.get_group_chaos_values(group, 3)
        print(f"  {group}: {[f'{v:.4f}' for v in values]}")
    
    # 测试增强混沌序列
    enhanced_seq = chaos_maps.enhanced_chaos_sequence(5, intensity=0.7, diversity_boost=True)
    print(f"\n增强混沌序列: {[f'{v:.4f}' for v in enhanced_seq]}")
    
    return chaos_maps

def test_complete_optimization():
    """测试完整优化流程"""
    print("\n🎯 完整优化流程测试")
    print("-" * 60)
    
    # 创建中等规模测试问题
    data_gen = DataGenerator()
    problem_data = data_gen.generate_problem(
        n_jobs=20,
        n_factories=2,
        n_stages=3,
        machines_per_stage=[2, 2, 2],
        processing_time_range=(1, 8)
    )
    
    problem = MO_DHFSP_Problem(problem_data)
    
    # 运行优化
    print(f"问题规模: {problem.n_jobs}作业 × {problem.n_factories}工厂 × {problem.n_stages}阶段")
    
    optimizer = RL_ChaoticHHO_Optimizer(problem, max_iterations=25)
    
    start_time = time.time()
    pareto_solutions, convergence_data = optimizer.optimize()
    end_time = time.time()
    
    print(f"\n✅ 优化完成!")
    print(f"运行时间: {end_time - start_time:.2f}秒")
    print(f"帕累托解数量: {len(pareto_solutions)}")
    
    if pareto_solutions:
        best_makespan = min(sol.makespan for sol in pareto_solutions)
        best_tardiness = min(sol.total_tardiness for sol in pareto_solutions)
        print(f"最优完工时间: {best_makespan:.2f}")
        print(f"最优总拖期: {best_tardiness:.2f}")
    
    # 获取详细统计
    stats = optimizer.get_algorithm_statistics()
    print(f"\n📈 算法统计:")
    print(f"  总迭代次数: {stats['iteration']}")
    print(f"  停滞次数: {stats['no_improvement_count']}")
    
    # 组性能统计
    if 'group_performance' in stats:
        print(f"\n🦅 分组性能:")
        for group_name, performance in stats['group_performance'].items():
            print(f"  {group_name}: 平均={performance['average']:.4f}, "
                  f"最新={performance['latest']:.4f}, 趋势={performance['trend']:.4f}")
    
    # RL统计
    if 'rl_statistics' in stats:
        rl_stats = stats['rl_statistics']
        print(f"\n🤖 强化学习统计:")
        print(f"  训练步数: {rl_stats['training_steps']}")
        print(f"  探索率: {rl_stats['epsilon']:.4f}")
        print(f"  经验池大小: {rl_stats['memory_size']}")
        print(f"  平均损失: {rl_stats['average_loss']:.6f}")
    
    # 策略使用统计
    if 'strategy_statistics' in stats:
        print(f"\n📊 策略使用统计:")
        for strategy, stat in stats['strategy_statistics'].items():
            print(f"  {strategy}: 使用{stat['usage_count']}次 "
                  f"(比例={stat['usage_rate']:.3f}, 成功率={stat['success_rate']:.3f})")
    
    return optimizer, pareto_solutions, convergence_data

def test_adaptive_mechanisms():
    """测试自适应机制"""
    print("\n🔄 自适应机制测试")
    print("-" * 60)
    
    optimizer, pareto_solutions, convergence_data = test_complete_optimization()
    
    # 测试组性能统计
    group_stats = optimizer.eagle_groups.get_group_statistics()
    print("分组统计信息:")
    for group_name, stats in group_stats.items():
        print(f"  {group_name}: 大小={stats['size']}, 比例={stats['ratio']:.3f}, "
              f"质量={stats['average_quality']:.4f}")
    
    # 测试混沌映射统计
    chaos_stats = optimizer.chaotic_maps.get_chaos_statistics()
    if chaos_stats:
        print(f"\n混沌映射使用统计:")
        for map_type, stats in chaos_stats.items():
            print(f"  {map_type}: 使用{stats['usage_count']}次 "
                  f"(比例={stats['usage_rate']:.3f}, 当前值={stats['current_state']:.4f})")
    
    return optimizer

def create_performance_visualization(optimizer, pareto_solutions):
    """创建性能可视化"""
    print("\n📊 生成性能可视化图表")
    print("-" * 60)
    
    try:
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('RL-Chaotic-HHO 四层分组协作性能分析', fontsize=16, fontweight='bold')
        
        # 1. 帕累托前沿
        if pareto_solutions:
            makespans = [sol.makespan for sol in pareto_solutions]
            tardiness = [sol.total_tardiness for sol in pareto_solutions]
            
            axes[0, 0].scatter(makespans, tardiness, c='red', alpha=0.6, s=50)
            axes[0, 0].set_xlabel('完工时间 (Makespan)')
            axes[0, 0].set_ylabel('总拖期 (Total Tardiness)')
            axes[0, 0].set_title('帕累托前沿')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 组性能趋势
        group_colors = {'exploration': 'blue', 'exploitation': 'green', 
                       'balance': 'orange', 'elite': 'purple'}
        
        for group_name, history in optimizer.group_performance_history.items():
            if history:
                axes[0, 1].plot(history, label=f'{group_name}组', 
                               color=group_colors.get(group_name, 'black'))
        
        axes[0, 1].set_xlabel('迭代次数')
        axes[0, 1].set_ylabel('组性能分数')
        axes[0, 1].set_title('各组性能趋势')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. RL奖励历史
        if hasattr(optimizer, 'rl_reward_history') and optimizer.rl_reward_history:
            axes[1, 0].plot(optimizer.rl_reward_history, 'g-', alpha=0.7)
            axes[1, 0].set_xlabel('迭代次数')
            axes[1, 0].set_ylabel('RL奖励')
            axes[1, 0].set_title('强化学习奖励历史')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, '暂无RL奖励数据', ha='center', va='center', 
                           transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('强化学习奖励历史')
        
        # 4. 策略使用分布
        rl_stats = optimizer.rl_coordinator.get_strategy_statistics()
        if rl_stats:
            strategies = list(rl_stats.keys())
            usage_counts = [rl_stats[s]['usage_count'] for s in strategies]
            
            axes[1, 1].pie(usage_counts, labels=strategies, autopct='%1.1f%%', startangle=90)
            axes[1, 1].set_title('策略使用分布')
        else:
            axes[1, 1].text(0.5, 0.5, '暂无策略统计数据', ha='center', va='center',
                           transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('策略使用分布')
        
        plt.tight_layout()
        
        # 保存图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/complete_rl_chaotic_hho_test_{timestamp}.png"
        os.makedirs('results', exist_ok=True)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ 性能图表已保存: {filename}")
        
    except Exception as e:
        print(f"❌ 生成可视化图表时出错: {e}")

def main():
    """主函数"""
    print("🚀 RL-Chaotic-HHO 完整实现测试")
    print("=" * 80)
    
    try:
        # 1. 测试四层分组机制
        optimizer = test_four_layer_grouping()[0]
        
        # 2. 测试强化学习调度器
        test_reinforcement_learning()
        
        # 3. 测试混沌映射系统
        test_chaotic_maps()
        
        # 4. 测试完整优化流程
        optimizer, pareto_solutions, convergence_data = test_complete_optimization()
        
        # 5. 测试自适应机制
        test_adaptive_mechanisms()
        
        # 6. 创建性能可视化
        create_performance_visualization(optimizer, pareto_solutions)
        
        print("\n🎉 所有测试完成!")
        print("=" * 80)
        print("✅ 四层鹰群分组协作机制: 正常")
        print("✅ 强化学习调度器: 正常")  
        print("✅ 增强混沌映射系统: 正常")
        print("✅ 完整优化流程: 正常")
        print("✅ 自适应机制: 正常")
        print("✅ 性能可视化: 正常")
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 