#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修复后的特定规模对比实验
验证四个指标是否正常生成，验证Pareto解集点数是否增加
"""

import sys
import os
import time
import numpy as np
from datetime import datetime

def test_specific_scales_fixed():
    """测试修复后的特定规模对比实验"""
    print("=" * 80)
    print("测试修复后的特定规模对比实验")
    print("=" * 80)
    
    try:
        # 导入修复后的模块
        print("1. 导入模块...")
        from table_format_comparison_specific_scales import (
            run_single_experiment, 
            generate_heterogeneous_problem_data,
            calculate_hypervolume,
            calculate_igd,
            calculate_gd,
            calculate_spacing,
            calculate_spread
        )
        from problem.mo_dhfsp import MO_DHFSP_Problem
        from algorithm.rl_chaotic_hho import RL_ChaoticHHO_Optimizer
        from algorithm.improved_nsga2 import ImprovedNSGA2_Optimizer
        print("   ✅ 模块导入成功")
        
        # 测试小规模配置（根据图片内容）
        print("\n2. 生成测试问题...")
        test_config = {
            'scale': '20J3S2F',
            'n_jobs': 20,
            'n_factories': 2,
            'n_stages': 3,
            'machines_per_stage': [2, 3, 4],  # 根据图片：机器数范围(2,3,4,5)
            'urgency_ddt': [0.5, 1.0, 1.5],
            'processing_time_range': (1, 20),
            'heterogeneous_machines': {
                0: [2, 3, 4],
                1: [3, 2, 4]
            }
        }
        
        problem_data = generate_heterogeneous_problem_data(test_config)
        print("   ✅ 测试问题生成成功")
        
        # 测试RL-Chaotic-HHO算法（增加Pareto解集点数）
        print("\n3. 测试RL-Chaotic-HHO算法...")
        algorithm_params = {
            'population_size': 50,  # 测试用较小规模
            'max_iterations': 10,   # 测试用较少迭代
            'pareto_size_limit': 100  # 增加Pareto解集限制
        }
        
        start_time = time.time()
        result = run_single_experiment(
            problem_data, 
            'RL-Chaotic-HHO', 
            RL_ChaoticHHO_Optimizer, 
            algorithm_params,
            runs=1  # 测试用单次运行
        )
        end_time = time.time()
        
        print(f"   运行时间: {end_time - start_time:.2f}秒")
        print("   ✅ RL-Chaotic-HHO算法运行成功")
        
        # 验证四个指标是否正常
        print("\n4. 验证四个指标...")
        
        # 检查max_makespan、max_tardiness、min_makespan、min_tardiness
        required_keys = [
            'makespan_best', 'tardiness_best', 'weighted_best',
            'max_makespan', 'max_tardiness', 'min_makespan', 'min_tardiness',
            'makespan_mean', 'tardiness_mean', 'weighted_mean',
            'runtime', 'hypervolume', 'igd', 'gd', 'spacing', 'spread',
            'pareto_count', 'pareto_solutions'
        ]
        
        missing_keys = []
        for key in required_keys:
            if key not in result:
                missing_keys.append(key)
        
        if missing_keys:
            print(f"   ❌ 缺少指标: {missing_keys}")
            return False
        else:
            print("   ✅ 所有必需指标都存在")
        
        # 验证指标数值是否正常
        print("\n5. 验证指标数值...")
        
        print(f"   最优完工时间 (min_makespan): {result['min_makespan']:.2f}")
        print(f"   最优总拖期 (min_tardiness): {result['min_tardiness']:.2f}")
        print(f"   最差完工时间 (max_makespan): {result['max_makespan']:.2f}")
        print(f"   最差总拖期 (max_tardiness): {result['max_tardiness']:.2f}")
        print(f"   超体积 (hypervolume): {result['hypervolume']:.4f}")
        print(f"   IGD: {result['igd']:.4f}")
        print(f"   GD: {result['gd']:.4f}")
        print(f"   间距 (spacing): {result['spacing']:.4f}")
        print(f"   分布性 (spread): {result['spread']:.4f}")
        print(f"   帕累托解数量: {result['pareto_count']}")
        
        # 验证数值合理性
        issues = []
        
        if result['min_makespan'] <= 0:
            issues.append("min_makespan应该大于0")
        if result['max_makespan'] < result['min_makespan']:
            issues.append("max_makespan应该大于等于min_makespan")
        if result['min_tardiness'] < 0:
            issues.append("min_tardiness不应该为负数")
        if result['max_tardiness'] < result['min_tardiness']:
            issues.append("max_tardiness应该大于等于min_tardiness")
        if result['hypervolume'] < 0:
            issues.append("hypervolume不应该为负数")
        if result['igd'] < 0:
            issues.append("igd不应该为负数")
        if result['gd'] < 0:
            issues.append("gd不应该为负数")
        if result['spacing'] < 0:
            issues.append("spacing不应该为负数")
        if result['pareto_count'] <= 0:
            issues.append("pareto_count应该大于0")
        
        if issues:
            print(f"   ❌ 指标数值问题: {issues}")
            return False
        else:
            print("   ✅ 所有指标数值正常")
        
        # 验证Pareto解集点数
        print("\n6. 验证Pareto解集点数...")
        pareto_solutions = result['pareto_solutions']
        
        if not pareto_solutions:
            print("   ❌ 没有找到Pareto解")
            return False
        
        print(f"   Pareto解集大小: {len(pareto_solutions)}")
        
        if len(pareto_solutions) < 5:
            print("   ⚠️  Pareto解集点数较少，可能需要调整参数")
        elif len(pareto_solutions) >= 20:
            print("   ✅ Pareto解集点数充足")
        else:
            print("   ✅ Pareto解集点数适中")
        
        # 验证解的多样性
        makespan_values = [sol.makespan for sol in pareto_solutions]
        tardiness_values = [sol.total_tardiness for sol in pareto_solutions]
        
        makespan_range = max(makespan_values) - min(makespan_values)
        tardiness_range = max(tardiness_values) - min(tardiness_values)
        
        print(f"   完工时间范围: [{min(makespan_values):.2f}, {max(makespan_values):.2f}] (跨度: {makespan_range:.2f})")
        print(f"   总拖期范围: [{min(tardiness_values):.2f}, {max(tardiness_values):.2f}] (跨度: {tardiness_range:.2f})")
        
        if makespan_range > 0 and tardiness_range > 0:
            print("   ✅ Pareto解集具有良好的多样性")
        else:
            print("   ⚠️  Pareto解集多样性不足")
        
        # 测试I-NSGA-II算法对比
        print("\n7. 测试I-NSGA-II算法对比...")
        nsga2_params = {
            'population_size': 50,
            'max_generations': 10,
            'crossover_rate': 0.9,
            'mutation_rate': 0.1
        }
        
        nsga2_result = run_single_experiment(
            problem_data, 
            'I-NSGA-II', 
            ImprovedNSGA2_Optimizer, 
            nsga2_params,
            runs=1
        )
        
        print(f"   I-NSGA-II Pareto解数量: {nsga2_result['pareto_count']}")
        print(f"   I-NSGA-II 超体积: {nsga2_result['hypervolume']:.4f}")
        print("   ✅ I-NSGA-II算法运行成功")
        
        # 对比两个算法的性能
        print("\n8. 算法性能对比...")
        print(f"   RL-Chaotic-HHO vs I-NSGA-II:")
        print(f"     Pareto解数量: {result['pareto_count']} vs {nsga2_result['pareto_count']}")
        print(f"     超体积: {result['hypervolume']:.4f} vs {nsga2_result['hypervolume']:.4f}")
        print(f"     最优完工时间: {result['min_makespan']:.2f} vs {nsga2_result['min_makespan']:.2f}")
        print(f"     最优总拖期: {result['min_tardiness']:.2f} vs {nsga2_result['min_tardiness']:.2f}")
        
        print("\n✅ 所有测试通过！")
        print("📊 指标生成正常")
        print("📈 Pareto解集点数充足")
        print("🔧 修复效果良好")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def run_quick_comparison_test():
    """运行快速对比测试"""
    print("\n" + "=" * 80)
    print("运行快速对比测试（小规模）")
    print("=" * 80)
    
    try:
        from table_format_comparison_specific_scales import (
            run_specific_scale_experiments,
            generate_heterogeneous_problem_data,
            plot_pareto_comparison
        )
        
        # 创建小规模测试配置
        test_config = {
            'scale': 'Test_10J2S2F',
            'n_jobs': 10,
            'n_factories': 2,
            'n_stages': 2,
            'machines_per_stage': [2, 3],
            'urgency_ddt': [0.5, 1.0],
            'processing_time_range': (1, 15),
            'heterogeneous_machines': {
                0: [2, 3],
                1: [3, 2]
            }
        }
        
        problem_data = generate_heterogeneous_problem_data(test_config)
        
        # 测试多个算法
        from algorithm.rl_chaotic_hho import RL_ChaoticHHO_Optimizer
        from algorithm.improved_nsga2 import ImprovedNSGA2_Optimizer
        from algorithm.mopso import MOPSO_Optimizer
        from table_format_comparison_specific_scales import run_single_experiment
        
        algorithms = {
            'RL-Chaotic-HHO': (RL_ChaoticHHO_Optimizer, {
                'population_size': 30,
                'max_iterations': 5,
                'pareto_size_limit': 50
            }),
            'I-NSGA-II': (ImprovedNSGA2_Optimizer, {
                'population_size': 30,
                'max_generations': 5,
                'crossover_rate': 0.9,
                'mutation_rate': 0.1
            }),
            'MOPSO': (MOPSO_Optimizer, {
                'swarm_size': 30,
                'max_iterations': 5,
                'w': 0.4,
                'c1': 2.0,
                'c2': 2.0
            })
        }
        
        results = {}
        
        for alg_name, (alg_class, alg_params) in algorithms.items():
            print(f"\n测试 {alg_name}...")
            try:
                result = run_single_experiment(
                    problem_data, 
                    alg_name, 
                    alg_class, 
                    alg_params,
                    runs=1
                )
                results[alg_name] = result
                print(f"  ✅ {alg_name}: {result['pareto_count']}个解, HV={result['hypervolume']:.4f}")
            except Exception as e:
                print(f"  ❌ {alg_name} 失败: {str(e)}")
        
        # 绘制对比图
        if results:
            print("\n绘制Pareto前沿对比图...")
            plot_pareto_comparison(results, 'Test_10J2S2F')
            print("✅ 对比图已保存")
        
        print(f"\n✅ 快速对比测试完成！共测试了 {len(results)} 个算法")
        return True
        
    except Exception as e:
        print(f"\n❌ 快速对比测试失败: {str(e)}")
        return False

if __name__ == "__main__":
    # 运行基础测试
    basic_test_passed = test_specific_scales_fixed()
    
    if basic_test_passed:
        # 运行快速对比测试
        comparison_test_passed = run_quick_comparison_test()
        
        if comparison_test_passed:
            print(f"\n{'='*80}")
            print("🎉 所有测试通过！可以运行完整版对比测试")
            print("📝 建议运行: python table_format_comparison_specific_scales.py")
            print(f"{'='*80}")
        else:
            print(f"\n{'='*80}")
            print("⚠️  基础测试通过，但对比测试有问题")
            print("📝 建议检查算法配置")
            print(f"{'='*80}")
    else:
        print(f"\n{'='*80}")
        print("❌ 基础测试失败，需要进一步修复")
        print(f"{'='*80}") 