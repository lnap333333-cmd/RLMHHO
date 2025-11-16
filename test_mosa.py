#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试MOSA算法
验证多目标模拟退火算法在MO-DHFSP问题上的性能
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from problem.mo_dhfsp import MO_DHFSP_Problem
from algorithm.mosa import MOSA_Optimizer
from utils.data_generator import DataGenerator
from utils.visualization import ResultVisualizer

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def test_mosa_basic():
    """基础MOSA算法测试"""
    print("🔥 MOSA算法基础功能测试")
    print("=" * 60)
    
    # 生成测试问题
    generator = DataGenerator(seed=42)
    problem_data = generator.generate_problem(
        n_jobs=20,
        n_factories=3,
        n_stages=3,
        machines_per_stage=[2, 3, 2],
        processing_time_range=(1, 20),
        due_date_tightness=1.5
    )
    
    problem = MO_DHFSP_Problem(problem_data)
    
    print(f"📊 测试问题规模:")
    print(f"   • 作业数: {problem.n_jobs}")
    print(f"   • 工厂数: {problem.n_factories}")
    print(f"   • 阶段数: {problem.n_stages}")
    print(f"   • 机器配置: {problem.machines_per_stage}")
    
    # 创建MOSA优化器
    mosa_params = {
        'initial_temperature': 500.0,
        'final_temperature': 0.1,
        'cooling_rate': 0.95,
        'max_iterations': 100,
        'archive_size': 50,
        'neighborhood_size': 5
    }
    
    optimizer = MOSA_Optimizer(problem, **mosa_params)
    
    # 运行优化
    print(f"\n🚀 开始MOSA优化...")
    start_time = time.time()
    
    pareto_solutions, convergence_data = optimizer.optimize()
    
    end_time = time.time()
    runtime = end_time - start_time
    
    # 分析结果
    print(f"\n📈 优化结果分析:")
    print(f"   • 运行时间: {runtime:.2f}秒")
    print(f"   • 帕累托解数量: {len(pareto_solutions)}")
    
    if pareto_solutions:
        makespans = [sol.makespan for sol in pareto_solutions]
        tardiness_values = [sol.total_tardiness for sol in pareto_solutions]
        
        print(f"   • 最优完工时间: {min(makespans):.2f}")
        print(f"   • 最优总拖期: {min(tardiness_values):.2f}")
        print(f"   • 平均完工时间: {np.mean(makespans):.2f}")
        print(f"   • 平均总拖期: {np.mean(tardiness_values):.2f}")
        
        # 计算加权目标函数
        weighted_objectives = [0.55 * sol.makespan + 0.45 * sol.total_tardiness 
                             for sol in pareto_solutions]
        print(f"   • 最优加权目标: {min(weighted_objectives):.2f}")
        
        # 绘制帕累托前沿
        plt.figure(figsize=(10, 6))
        plt.scatter(makespans, tardiness_values, c='red', alpha=0.7, s=50)
        plt.xlabel('完工时间 (Makespan)')
        plt.ylabel('总拖期 (Total Tardiness)')
        plt.title('MOSA算法 - 帕累托前沿')
        plt.grid(True, alpha=0.3)
        
        # 保存图片
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/MOSA_测试_帕累托前沿_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   • 帕累托前沿图已保存: {filename}")
        
        # 绘制收敛曲线
        if convergence_data['convergence_data']:
            iterations = [data['iteration'] for data in convergence_data['convergence_data']]
            best_makespans = [data['best_makespan'] for data in convergence_data['convergence_data']]
            best_tardiness = [data['best_tardiness'] for data in convergence_data['convergence_data']]
            temperatures = [data['temperature'] for data in convergence_data['convergence_data']]
            
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
            
            # 目标函数收敛
            ax1.plot(iterations, best_makespans, 'b-', label='最优完工时间')
            ax1.plot(iterations, best_tardiness, 'r-', label='最优总拖期')
            ax1.set_xlabel('迭代次数')
            ax1.set_ylabel('目标函数值')
            ax1.set_title('MOSA收敛曲线 - 目标函数')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 温度变化
            ax2.plot(iterations, temperatures, 'g-', label='温度')
            ax2.set_xlabel('迭代次数')
            ax2.set_ylabel('温度')
            ax2.set_title('MOSA收敛曲线 - 温度变化')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_yscale('log')
            
            # 档案大小变化
            archive_sizes = [data['archive_size'] for data in convergence_data['convergence_data']]
            ax3.plot(iterations, archive_sizes, 'm-', label='档案大小')
            ax3.set_xlabel('迭代次数')
            ax3.set_ylabel('档案大小')
            ax3.set_title('MOSA收敛曲线 - 档案大小')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            convergence_filename = f"results/MOSA_测试_收敛曲线_{timestamp}.png"
            plt.savefig(convergence_filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   • 收敛曲线图已保存: {convergence_filename}")
    
    print("\n✅ MOSA算法测试完成!")
    return pareto_solutions, convergence_data

def test_mosa_parameters():
    """测试不同参数设置对MOSA性能的影响"""
    print("\n🔧 MOSA参数敏感性测试")
    print("=" * 60)
    
    # 生成测试问题
    generator = DataGenerator(seed=123)
    problem_data = generator.generate_problem(
        n_jobs=15,
        n_factories=2,
        n_stages=3,
        machines_per_stage=[2, 2, 2],
        processing_time_range=(1, 15),
        due_date_tightness=1.3
    )
    
    problem = MO_DHFSP_Problem(problem_data)
    
    # 不同参数配置
    parameter_configs = [
        {
            'name': '高温慢冷',
            'initial_temperature': 1000.0,
            'cooling_rate': 0.98,
            'max_iterations': 50
        },
        {
            'name': '中温中冷',
            'initial_temperature': 500.0,
            'cooling_rate': 0.95,
            'max_iterations': 50
        },
        {
            'name': '低温快冷',
            'initial_temperature': 200.0,
            'cooling_rate': 0.90,
            'max_iterations': 50
        }
    ]
    
    results = {}
    
    for config in parameter_configs:
        print(f"\n🧪 测试配置: {config['name']}")
        
        optimizer = MOSA_Optimizer(problem, **config)
        start_time = time.time()
        pareto_solutions, convergence_data = optimizer.optimize()
        runtime = time.time() - start_time
        
        if pareto_solutions:
            makespans = [sol.makespan for sol in pareto_solutions]
            tardiness_values = [sol.total_tardiness for sol in pareto_solutions]
            weighted_objectives = [0.55 * sol.makespan + 0.45 * sol.total_tardiness 
                                 for sol in pareto_solutions]
            
            results[config['name']] = {
                'runtime': runtime,
                'pareto_size': len(pareto_solutions),
                'best_makespan': min(makespans),
                'best_tardiness': min(tardiness_values),
                'best_weighted': min(weighted_objectives),
                'final_temperature': convergence_data['final_temperature'],
                'acceptance_rate': convergence_data['acceptance_rate']
            }
            
            print(f"   • 运行时间: {runtime:.2f}s")
            print(f"   • 帕累托解数: {len(pareto_solutions)}")
            print(f"   • 最优加权目标: {min(weighted_objectives):.2f}")
            print(f"   • 最终温度: {convergence_data['final_temperature']:.6f}")
            print(f"   • 接受率: {convergence_data['acceptance_rate']*100:.1f}%")
    
    # 生成对比报告
    print(f"\n📊 参数对比总结:")
    print("-" * 80)
    print(f"{'配置':<10} {'运行时间':<8} {'解数':<6} {'最优加权':<10} {'接受率':<8} {'最终温度':<12}")
    print("-" * 80)
    
    for name, result in results.items():
        print(f"{name:<10} {result['runtime']:<8.2f} {result['pareto_size']:<6} "
              f"{result['best_weighted']:<10.2f} {result['acceptance_rate']*100:<8.1f}% "
              f"{result['final_temperature']:<12.6f}")
    
    return results

if __name__ == "__main__":
    # 运行基础测试
    pareto_solutions, convergence_data = test_mosa_basic()
    
    # 运行参数测试
    parameter_results = test_mosa_parameters()
    
    print(f"\n🎉 所有测试完成!") 