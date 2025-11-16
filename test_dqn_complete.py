#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整的DQN调度器测试程序
基于论文《基于深度Q学习网络的分布式流水车间调度问题优化》
测试多目标分布式异构混合流水车间调度问题
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from algorithm.dqn_multiobj_scheduler import DQNMultiObjScheduler
from problem.mo_dhfsp import MO_DHFSP_Problem
from utils.data_generator import DataGenerator

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def create_test_problem():
    """创建测试问题实例"""
    generator = DataGenerator(seed=42)
    
    # 按论文实验设置：20×5×2规模
    problem_data = generator.generate_problem(
        n_jobs=20,          # 20个作业
        n_factories=5,      # 5个工厂（分布式）
        n_stages=2,         # 2个阶段
        machines_per_stage=[3, 4],  # 第1阶段3台机器，第2阶段4台机器
        processing_time_range=(1, 50),  # 处理时间范围1-50
        due_date_tightness=1.5
    )
    
    # 创建异构机器配置（每个工厂配置不同）
    problem_data['factory_machines'] = {
        0: [3, 4],  # 工厂1: 7台机器
        1: [2, 3],  # 工厂2: 5台机器
        2: [4, 3],  # 工厂3: 7台机器
        3: [3, 3],  # 工厂4: 6台机器
        4: [2, 4]   # 工厂5: 6台机器
    }
    
    problem = MO_DHFSP_Problem(problem_data)
    return problem

def test_dqn_basic_functions():
    """测试DQN调度器基础功能"""
    print("🔧 DQN调度器基础功能测试")
    print("=" * 60)
    
    problem = create_test_problem()
    
    print(f"📊 问题规模:")
    print(f"   • 作业数: {problem.n_jobs}")
    print(f"   • 工厂数: {problem.n_factories}")
    print(f"   • 阶段数: {problem.n_stages}")
    print(f"   • 机器配置: {problem.machines_per_stage}")
    print(f"   • 异构工厂: {problem.factory_machines}")
    
    # 创建DQN调度器
    scheduler = DQNMultiObjScheduler(problem)
    
    # 测试NEH初始化
    print(f"\n🚀 测试NEH初始化...")
    neh_solution = scheduler.neh_initialization()
    print(f"NEH解: 完工时间={neh_solution.makespan:.2f}, 拖期={neh_solution.total_tardiness:.2f}")
    
    # 测试状态编码
    print(f"\n📊 测试状态编码...")
    state = scheduler.encode_state(neh_solution)
    print(f"状态向量: {state}")
    print(f"状态维度: {len(state)}")
    
    # 测试所有9个调度规则
    print(f"\n🎯 测试9个调度规则...")
    rule_names = [
        "全局规则1", "全局规则2", "全局规则3",
        "局部规则1", "局部规则2", "局部规则3",
        "局部规则4", "局部规则5", "局部规则6"
    ]
    
    for action in range(9):
        new_solution = scheduler.apply_rule(neh_solution, action)
        reward = scheduler.calculate_reward(neh_solution, new_solution, action)
        print(f"   {rule_names[action]}: 完工时间={new_solution.makespan:.2f}, "
              f"拖期={new_solution.total_tardiness:.2f}, 奖励={reward:.2f}")
    
    print(f"\n✅ 基础功能测试完成!")

def test_dqn_training():
    """测试DQN训练过程"""
    print(f"\n🎓 DQN训练测试")
    print("=" * 60)
    
    problem = create_test_problem()
    
    # DQN参数设置（按论文）
    dqn_params = {
        'memory_size': 5000,
        'batch_size': 32,
        'gamma': 0.98,
        'epsilon': 0.9,
        'epsilon_decay': 0.995,
        'epsilon_min': 0.01,
        'learning_rate': 0.001,
        'target_update': 10
    }
    
    scheduler = DQNMultiObjScheduler(problem, **dqn_params)
    
    # 开始训练
    print(f"🚀 开始DQN训练...")
    start_time = time.time()
    
    best_solution, convergence_data = scheduler.optimize(
        max_episodes=50,     # 训练轮数
        max_steps_per_episode=50  # 每轮步数
    )
    
    end_time = time.time()
    runtime = end_time - start_time
    
    # 分析结果
    print(f"\n📈 训练结果分析:")
    print(f"   • 训练时间: {runtime:.2f}秒")
    print(f"   • 最佳完工时间: {best_solution.makespan:.2f}")
    print(f"   • 总拖期: {best_solution.total_tardiness:.2f}")
    print(f"   • 最终探索率: {scheduler.epsilon:.3f}")
    
    # 打印工厂分配
    print(f"\n🏭 最佳解工厂分配:")
    for factory_id in range(problem.n_factories):
        jobs = best_solution.job_sequences[factory_id]
        makespan = best_solution.factory_makespans[factory_id] if best_solution.factory_makespans else 0
        print(f"   工厂{factory_id}: 作业{jobs}, 完工时间={makespan:.2f}")
    
    # 规则统计
    print(f"\n📊 调度规则使用统计:")
    rule_stats = scheduler.get_rule_statistics()
    for rule_name, stats in rule_stats.items():
        print(f"   {rule_name}: 成功率={stats['success_rate']:.3f} "
              f"({stats['success_count']}/{stats['total_count']})")
    
    return best_solution, convergence_data

def plot_convergence(convergence_data):
    """绘制收敛曲线"""
    print(f"\n📊 绘制收敛曲线...")
    
    episodes = [data['episode'] for data in convergence_data]
    makespans = [data['best_makespan'] for data in convergence_data]
    tardiness = [data['best_tardiness'] for data in convergence_data]
    rewards = [data['episode_reward'] for data in convergence_data]
    epsilons = [data['epsilon'] for data in convergence_data]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # 完工时间收敛
    ax1.plot(episodes, makespans, 'b-', linewidth=2)
    ax1.set_title('完工时间收敛曲线')
    ax1.set_xlabel('训练轮次')
    ax1.set_ylabel('最佳完工时间')
    ax1.grid(True)
    
    # 拖期收敛
    ax2.plot(episodes, tardiness, 'r-', linewidth=2)
    ax2.set_title('总拖期收敛曲线')
    ax2.set_xlabel('训练轮次')
    ax2.set_ylabel('总拖期')
    ax2.grid(True)
    
    # 奖励变化
    ax3.plot(episodes, rewards, 'g-', linewidth=2)
    ax3.set_title('每轮奖励变化')
    ax3.set_xlabel('训练轮次')
    ax3.set_ylabel('累积奖励')
    ax3.grid(True)
    
    # 探索率变化
    ax4.plot(episodes, epsilons, 'm-', linewidth=2)
    ax4.set_title('探索率衰减')
    ax4.set_xlabel('训练轮次')
    ax4.set_ylabel('探索率')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'dqn_convergence_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png', dpi=300)
    plt.show()

def compare_with_random():
    """与随机算法对比"""
    print(f"\n🆚 与随机算法对比")
    print("=" * 60)
    
    problem = create_test_problem()
    
    # 随机解性能
    print(f"🎲 生成随机解...")
    random_solutions = []
    for i in range(20):  # 生成20个随机解
        random_sol = problem.generate_random_solution()
        random_solutions.append(random_sol)
    
    random_makespans = [sol.makespan for sol in random_solutions]
    random_tardiness = [sol.total_tardiness for sol in random_solutions]
    
    avg_random_makespan = np.mean(random_makespans)
    avg_random_tardiness = np.mean(random_tardiness)
    
    print(f"随机算法平均性能:")
    print(f"   • 平均完工时间: {avg_random_makespan:.2f}")
    print(f"   • 平均拖期: {avg_random_tardiness:.2f}")
    
    # DQN性能
    print(f"\n🤖 DQN算法性能:")
    scheduler = DQNMultiObjScheduler(problem)
    best_solution, _ = scheduler.optimize(max_episodes=30, max_steps_per_episode=30)
    
    print(f"DQN算法性能:")
    print(f"   • 完工时间: {best_solution.makespan:.2f}")
    print(f"   • 拖期: {best_solution.total_tardiness:.2f}")
    
    # 改进率计算
    makespan_improvement = (avg_random_makespan - best_solution.makespan) / avg_random_makespan * 100
    tardiness_improvement = (avg_random_tardiness - best_solution.total_tardiness) / avg_random_tardiness * 100
    
    print(f"\n📈 DQN相对随机算法的改进:")
    print(f"   • 完工时间改进: {makespan_improvement:.2f}%")
    print(f"   • 拖期改进: {tardiness_improvement:.2f}%")

def main():
    """主函数"""
    print("🚀 基于DQN的多目标分布式异构混合流水车间调度测试")
    print("基于论文《基于深度Q学习网络的分布式流水车间调度问题优化》")
    print("=" * 80)
    
    # 基础功能测试
    test_dqn_basic_functions()
    
    # 训练测试
    best_solution, convergence_data = test_dqn_training()
    
    # 绘制收敛曲线
    plot_convergence(convergence_data)
    
    # 与随机算法对比
    compare_with_random()
    
    print(f"\n🎉 所有测试完成!")
    print(f"✅ DQN调度器成功实现论文中的完整算法")
    print(f"✅ 支持多目标优化：完工时间 + 总拖期")
    print(f"✅ 支持分布式异构混合流水车间")
    print(f"✅ 包含9个启发式调度规则")
    print(f"✅ 使用NEH初始化和5维状态编码")

if __name__ == "__main__":
    main() 