#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修复后的算法对比功能
验证数据生成和算法运行是否正常
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from table_format_comparison_with_ql_abc_full import run_table_format_experiments, generate_enhanced_table_report

def test_small_scale_comparison():
    """测试小规模对比，验证修复效果"""
    print("🧪 测试修复后的算法对比功能")
    print("=" * 60)
    
    # 临时修改实验配置为小规模测试
    import table_format_comparison_with_ql_abc_full as comparison_module
    
    # 备份原始函数
    original_run_experiments = comparison_module.run_table_format_experiments
    
    def test_run_experiments():
        """运行小规模测试实验"""
        print("小规模算法对比测试")
        print("=" * 40)
        
        # 生成3个测试数据集
        import numpy as np
        experiment_configs = []
        
        for i in range(3):
            dataset_seed = 42 + i * 17
            np.random.seed(dataset_seed)
            
            # 小规模配置
            n_jobs = 20 + i * 10  # 20, 30, 40
            n_factories = 2 + i    # 2, 3, 4
            n_stages = 3           # 固定3个阶段
            
            machines_per_stage = [2 + (i % 3), 3, 2 + ((i+1) % 3)]
            
            # 异构机器配置
            heterogeneous_machines = {}
            for factory_id in range(n_factories):
                factory_machines = []
                for stage in range(n_stages):
                    base_machines = 2 + (factory_id + stage) % 3
                    factory_machines.append(base_machines)
                heterogeneous_machines[factory_id] = factory_machines
            
            time_range = (1 + i, 15 + i * 3)
            urgency_ddt = [0.8 + i*0.1, 1.5 + i*0.1, 2.2 + i*0.1]
            
            scale_name = f"测试数据集{i+1:02d}_{n_jobs}J{n_factories}F{n_stages}S"
            
            config = {
                'scale': scale_name,
                'n_jobs': n_jobs,
                'n_factories': n_factories,
                'n_stages': n_stages,
                'machines_per_stage': machines_per_stage,
                'urgency_ddt': urgency_ddt,
                'processing_time_range': time_range,
                'heterogeneous_machines': heterogeneous_machines,
                'dataset_seed': dataset_seed
            }
            
            experiment_configs.append(config)
        
        print(f"生成了 {len(experiment_configs)} 个测试数据集")
        
        # 简化的算法配置
        algorithm_configs = {
            'RL-Chaotic-HHO': {
                'population_size': 30,
                'max_iterations': 20,
                'pareto_size_limit': 100,
                'diversity_enhancement': True
            },
            'I-NSGA-II': {
                'population_size': 30,
                'max_iterations': 20,
                'pareto_size_limit': 100,
                'crossover_rate': 0.9,
                'mutation_rate': 0.1
            },
            'MOPSO': {
                'swarm_size': 30,
                'max_iterations': 20,
                'archive_size': 100
            }
        }
        
        # 算法类映射
        from algorithm.rl_chaotic_hho import RL_ChaoticHHO_Optimizer
        from algorithm.improved_nsga2 import ImprovedNSGA2_Optimizer
        from algorithm.mopso import MOPSO_Optimizer
        
        algorithm_classes = {
            'RL-Chaotic-HHO': RL_ChaoticHHO_Optimizer,
            'I-NSGA-II': ImprovedNSGA2_Optimizer,
            'MOPSO': MOPSO_Optimizer
        }
        
        # 存储结果
        results = {}
        
        # 为每个数据集运行实验
        for config in experiment_configs:
            scale = config['scale']
            print(f"\n处理 {scale}...")
            
            # 生成问题数据
            problem_data = comparison_module.generate_heterogeneous_problem_data(config)
            
            # 显示数据集信息
            print(f"  作业数: {config['n_jobs']}, 工厂数: {config['n_factories']}")
            print(f"  机器配置: {config['machines_per_stage']}")
            print(f"  处理时间范围: {config['processing_time_range']}")
            
            results[scale] = {}
            
            # 运行每个算法
            for alg_name in ['RL-Chaotic-HHO', 'I-NSGA-II', 'MOPSO']:
                print(f"  运行 {alg_name}...")
                
                try:
                    result = comparison_module.run_single_experiment(
                        problem_data,
                        alg_name,
                        algorithm_classes[alg_name],
                        algorithm_configs[alg_name],
                        runs=1  # 测试时只运行1次
                    )
                    
                    results[scale][alg_name] = result
                    
                    print(f"    ✅ 成功: 加权目标={result['weighted_best']:.2f}, "
                          f"帕累托解数={result['pareto_count']}, HV={result['hypervolume']:.3f}")
                    
                except Exception as e:
                    print(f"    ❌ 失败: {str(e)}")
                    results[scale][alg_name] = {
                        'weighted_best': float('inf'),
                        'makespan_best': float('inf'),
                        'tardiness_best': float('inf'),
                        'pareto_count': 0,
                        'hypervolume': 0.0,
                        'igd': float('inf'),
                        'gd': float('inf'),
                        'spacing': 0.0
                    }
        
        return results, experiment_configs
    
    # 运行测试
    try:
        results, configs = test_run_experiments()
        
        # 验证结果
        print(f"\n📊 测试结果验证")
        print("=" * 40)
        
        all_same_makespan = True
        all_same_tardiness = True
        all_same_hv = True
        
        for scale, scale_results in results.items():
            print(f"\n{scale}:")
            makespans = []
            tardiness = []
            hvs = []
            
            for alg_name, result in scale_results.items():
                makespan = result['makespan_best']
                tard = result['tardiness_best']
                hv = result['hypervolume']
                
                makespans.append(makespan)
                tardiness.append(tard)
                hvs.append(hv)
                
                print(f"  {alg_name}: 完工时间={makespan:.2f}, 拖期={tard:.2f}, HV={hv:.3f}")
            
            # 检查是否所有算法结果相同
            if len(set(f"{m:.1f}" for m in makespans if m != float('inf'))) > 1:
                all_same_makespan = False
            if len(set(f"{t:.1f}" for t in tardiness if t != float('inf'))) > 1:
                all_same_tardiness = False
            if len(set(f"{h:.3f}" for h in hvs if h > 0)) > 1:
                all_same_hv = False
        
        # 输出验证结果
        print(f"\n🔍 异常检测结果:")
        print(f"  完工时间全部相同: {'❌ 异常' if all_same_makespan else '✅ 正常'}")
        print(f"  总拖期全部相同: {'❌ 异常' if all_same_tardiness else '✅ 正常'}")
        print(f"  超体积全部相同: {'❌ 异常' if all_same_hv else '✅ 正常'}")
        
        if not all_same_makespan and not all_same_tardiness and not all_same_hv:
            print(f"\n🎉 修复成功！算法产生了不同的结果")
            
            # 生成测试报告
            print(f"\n📄 生成测试报告...")
            generate_enhanced_table_report(results, configs)
            print(f"✅ 测试报告生成完成")
            
        else:
            print(f"\n⚠️  仍存在异常，需要进一步调查")
            
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_small_scale_comparison() 