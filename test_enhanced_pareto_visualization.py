#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试增强版帕累托图可视化功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_pareto_visualization import EnhancedParetoVisualizer
from problem.mo_dhfsp import MO_DHFSP_Problem
from algorithm.rl_chaotic_hho import RL_ChaoticHHO_Optimizer
from algorithm.improved_nsga2 import ImprovedNSGA2_Optimizer
from algorithm.mopso import MOPSO_Optimizer
from algorithm.mode import MODE_Optimizer
from algorithm.dqn_algorithm_wrapper import DQNAlgorithmWrapper
from algorithm.ql_abc_fixed import QLABC_Optimizer_Fixed
from utils.data_generator import generate_heterogeneous_problem_data
import time

def test_enhanced_visualization():
    """测试增强版可视化功能"""
    print("🧪 测试增强版帕累托图可视化功能...")
    
    # 创建可视化器
    visualizer = EnhancedParetoVisualizer()
    
    # 生成测试数据
    config = {
        'scale': '50J4S3F',
        'n_jobs': 50,
        'n_factories': 3,
        'n_stages': 4,
        'machines_per_stage': [3, 4, 3, 4],
        'urgency_ddt': [0.5, 1.0, 1.5],
        'processing_time_range': (1, 20),
        'heterogeneous_machines': {
            0: [3, 4, 3, 4],
            1: [4, 3, 4, 3],
            2: [3, 3, 4, 4]
        }
    }
    
    print(f"📊 生成测试问题: {config['scale']}")
    problem_data = generate_heterogeneous_problem_data(config)
    
    # 运行算法
    algorithms = {
        'RL-Chaotic-HHO': (RL_ChaoticHHO_Optimizer, {
            'population_size': 50,
            'max_iterations': 30,
            'pareto_size_limit': 100
        }),
        'I-NSGA-II': (ImprovedNSGA2_Optimizer, {
            'population_size': 50,
            'max_generations': 30
        }),
        'MOPSO': (MOPSO_Optimizer, {
            'swarm_size': 50,
            'max_iterations': 30
        }),
        'MODE': (MODE_Optimizer, {
            'population_size': 50,
            'max_generations': 30
        }),
        'DQN': (DQNAlgorithmWrapper, {
            'max_iterations': 20,
            'target_pareto_size': 20
        }),
        'QL-ABC': (QLABC_Optimizer_Fixed, {
            'population_size': 50,
            'max_iterations': 30
        })
    }
    
    results = {}
    
    for alg_name, (alg_class, params) in algorithms.items():
        print(f"\n🔄 运行算法: {alg_name}")
        try:
            start_time = time.time()
            optimizer = alg_class(problem_data, **params)
            result = optimizer.optimize()
            end_time = time.time()
            
            if result and result['pareto_solutions']:
                print(f"   ✅ 成功，解集数量: {len(result['pareto_solutions'])}")
                print(f"   ⏱️  运行时间: {end_time - start_time:.2f}秒")
                results[alg_name] = result
            else:
                print(f"   ❌ 失败，没有生成有效解集")
                results[alg_name] = None
                
        except Exception as e:
            print(f"   ❌ 算法{alg_name}运行失败: {e}")
            results[alg_name] = None
    
    # 测试不同格式的绘图
    print(f"\n🎨 测试不同格式的帕累托图...")
    
    # 1. 标准增强版
    print("\n📊 生成标准增强版帕累托图...")
    files1 = visualizer.plot_enhanced_pareto_comparison(
        results, config['scale'], save_formats=['png', 'pdf', 'svg']
    )
    
    # 2. 发表质量版
    print("\n📊 生成发表质量版帕累托图...")
    files2 = visualizer.create_publication_quality_plot(results, config['scale'])
    
    # 3. 单个算法图
    print("\n📊 生成单个算法帕累托图...")
    for alg_name, result in results.items():
        if result and result['pareto_solutions']:
            files3 = visualizer.plot_single_algorithm_pareto(
                result['pareto_solutions'], alg_name, config['scale']
            )
    
    print(f"\n✅ 测试完成！")
    print(f"📁 文件保存在以下目录:")
    print(f"   • 高分辨率PNG: results/high_res/")
    print(f"   • 矢量图: results/vector/")
    print(f"📊 共生成{len(files1) + len(files2)}个对比图文件")
    
    return results

def test_specific_formats():
    """测试特定格式的生成"""
    print("\n🔧 测试特定格式生成...")
    
    # 创建可视化器
    visualizer = EnhancedParetoVisualizer()
    
    # 生成简单测试数据
    config = {
        'scale': '30J3S2F',
        'n_jobs': 30,
        'n_factories': 2,
        'n_stages': 3,
        'machines_per_stage': [2, 3, 2],
        'urgency_ddt': [0.5, 1.0],
        'processing_time_range': (1, 15),
        'heterogeneous_machines': {
            0: [2, 3, 2],
            1: [3, 2, 3]
        }
    }
    
    problem_data = generate_heterogeneous_problem_data(config)
    
    # 只运行一个算法快速测试
    optimizer = RL_ChaoticHHO_Optimizer(problem_data, 
                                       population_size=30, 
                                       max_iterations=20,
                                       pareto_size_limit=50)
    result = optimizer.optimize()
    
    if result and result['pareto_solutions']:
        # 测试只生成PDF格式
        files = visualizer.plot_single_algorithm_pareto(
            result['pareto_solutions'], 'RL-Chaotic-HHO', config['scale'],
            save_formats=['pdf']
        )
        print(f"✅ 仅PDF格式测试完成，生成文件: {files}")
    else:
        print("❌ 没有生成有效解集")

if __name__ == "__main__":
    print("🚀 开始测试增强版帕累托图可视化功能")
    print("=" * 60)
    
    # 测试主要功能
    results = test_enhanced_visualization()
    
    # 测试特定格式
    test_specific_formats()
    
    print("\n" + "=" * 60)
    print("🎉 所有测试完成！")
    print("\n📋 使用说明:")
    print("1. 高分辨率PNG文件适合屏幕显示和网页使用")
    print("2. PDF文件适合打印和文档插入")
    print("3. SVG文件适合网页和矢量编辑")
    print("4. 发表质量版本适合学术论文和报告") 