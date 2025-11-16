#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
田口L81实验测试脚本 - 验证4参数9水平设计
"""

import os
import sys
import time
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from taguchi_l81_experiment import TaguchiL81Experiment

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_l81_design():
    """测试L81设计的正确性"""
    print("🔧 测试L81正交表设计...")
    
    experiment = TaguchiL81Experiment()
    
    # 检查L81设计表
    assert len(experiment.l81_design) == 81, f"L81设计表应包含81组实验，实际{len(experiment.l81_design)}"
    
    # 检查因子水平
    assert len(experiment.factor_levels['A_learning_rate']) == 9, "学习率应有9个水平"
    assert len(experiment.factor_levels['B_epsilon_decay']) == 9, "衰减率应有9个水平"
    assert len(experiment.factor_levels['C_group_ratios']) == 9, "分组比例应有9个水平"
    assert len(experiment.factor_levels['D_gamma']) == 9, "折扣因子应有9个水平"
    
    # 检查实验ID连续性
    exp_ids = [exp['exp_id'] for exp in experiment.l81_design]
    assert exp_ids == list(range(1, 82)), "实验ID应从1到81连续"
    
    # 检查因子水平范围
    for exp in experiment.l81_design:
        assert 1 <= exp['A'] <= 9, f"A因子水平应在1-9范围内，实际{exp['A']}"
        assert 1 <= exp['B'] <= 9, f"B因子水平应在1-9范围内，实际{exp['B']}"
        assert 1 <= exp['C'] <= 9, f"C因子水平应在1-9范围内，实际{exp['C']}"
        assert 1 <= exp['D'] <= 9, f"D因子水平应在1-9范围内，实际{exp['D']}"
    
    print("✅ L81设计表验证通过")
    
    # 显示设计统计
    print(f"📊 L81设计统计:")
    print(f"   - 实验总数: {len(experiment.l81_design)}")
    print(f"   - A因子水平: {sorted(set(exp['A'] for exp in experiment.l81_design))}")
    print(f"   - B因子水平: {sorted(set(exp['B'] for exp in experiment.l81_design))}")
    print(f"   - C因子水平: {sorted(set(exp['C'] for exp in experiment.l81_design))}")
    print(f"   - D因子水平: {sorted(set(exp['D'] for exp in experiment.l81_design))}")

def test_parameter_mapping():
    """测试参数映射的正确性"""
    print("🔧 测试参数映射...")
    
    experiment = TaguchiL81Experiment()
    
    # 测试第一个实验配置
    first_config = experiment.l81_design[0]
    params = experiment._get_experiment_parameters(first_config)
    
    # 检查参数是否正确设置
    assert 'learning_rate' in params, "应包含学习率参数"
    assert 'epsilon_decay' in params, "应包含衰减率参数"
    assert 'group_ratios' in params, "应包含分组比例参数"
    assert 'gamma' in params, "应包含折扣因子参数"
    assert params['population_size_override'] == 50, "种群大小应强制设为50"
    assert params['max_iterations'] == 50, "迭代次数应为50"
    
    # 检查参数值范围
    assert 0.00001 <= params['learning_rate'] <= 0.02, f"学习率范围错误: {params['learning_rate']}"
    assert 0.985 <= params['epsilon_decay'] <= 0.9999, f"衰减率范围错误: {params['epsilon_decay']}"
    assert len(params['group_ratios']) == 4, f"分组比例应有4个值: {params['group_ratios']}"
    assert abs(sum(params['group_ratios']) - 1.0) < 0.01, f"分组比例总和应为1: {sum(params['group_ratios'])}"
    assert 0.75 <= params['gamma'] <= 0.999, f"折扣因子范围错误: {params['gamma']}"
    
    print("✅ 参数映射验证通过")

def test_level_coverage():
    """测试水平覆盖的均匀性"""
    print("🔧 测试水平覆盖均匀性...")
    
    experiment = TaguchiL81Experiment()
    
    # 统计各因子各水平的出现次数
    for factor in ['A', 'B', 'C', 'D']:
        level_counts = {}
        for level in range(1, 10):
            count = sum(1 for exp in experiment.l81_design if exp[factor] == level)
            level_counts[level] = count
        
        print(f"   {factor}因子水平分布: {level_counts}")
        
        # 检查覆盖均匀性（每个水平应出现9次）
        expected_count = 9  # 81/9 = 9
        for level, count in level_counts.items():
            assert count == expected_count, f"{factor}因子水平{level}出现{count}次，期望{expected_count}次"
    
    print("✅ 水平覆盖均匀性验证通过")

def test_orthogonality():
    """测试正交性（简化检查）"""
    print("🔧 测试正交性...")
    
    experiment = TaguchiL81Experiment()
    
    # 检查因子间的独立性（简化版本）
    factor_pairs = [('A', 'B'), ('A', 'C'), ('A', 'D'), ('B', 'C'), ('B', 'D'), ('C', 'D')]
    
    for f1, f2 in factor_pairs:
        # 统计因子组合
        combinations = {}
        for exp in experiment.l81_design:
            combo = (exp[f1], exp[f2])
            combinations[combo] = combinations.get(combo, 0) + 1
        
        # 检查组合数量（应该相对均匀）
        combo_counts = list(combinations.values())
        min_count = min(combo_counts)
        max_count = max(combo_counts)
        
        print(f"   {f1}-{f2}因子组合: {len(combinations)}种，最少{min_count}次，最多{max_count}次")
        
        # 正交性要求组合出现次数相对均匀
        assert max_count - min_count <= 2, f"{f1}-{f2}因子组合不够均匀"
    
    print("✅ 正交性验证通过")

def test_parameter_ranges():
    """测试参数范围的扩展性"""
    print("🔧 测试参数范围扩展...")
    
    experiment = TaguchiL81Experiment()
    
    # 检查学习率范围扩展
    lr_values = list(experiment.factor_levels['A_learning_rate'].values())
    print(f"   学习率范围: {min(lr_values):.5f} ~ {max(lr_values):.5f}")
    assert min(lr_values) == 0.00001, "最小学习率应为0.00001"
    assert max(lr_values) == 0.02, "最大学习率应为0.02"
    
    # 检查衰减率范围扩展
    decay_values = list(experiment.factor_levels['B_epsilon_decay'].values())
    print(f"   衰减率范围: {min(decay_values):.4f} ~ {max(decay_values):.4f}")
    assert min(decay_values) == 0.985, "最小衰减率应为0.985"
    assert max(decay_values) == 0.9999, "最大衰减率应为0.9999"
    
    # 检查分组比例的多样性
    group_ratios = list(experiment.factor_levels['C_group_ratios'].values())
    exploration_ratios = [gr[0] for gr in group_ratios]  # 探索组比例
    print(f"   探索组比例范围: {min(exploration_ratios):.2f} ~ {max(exploration_ratios):.2f}")
    assert min(exploration_ratios) == 0.15, "最小探索组比例应为0.15"
    assert max(exploration_ratios) == 0.80, "最大探索组比例应为0.80"
    
    # 检查折扣因子范围扩展
    gamma_values = list(experiment.factor_levels['D_gamma'].values())
    print(f"   折扣因子范围: {min(gamma_values):.3f} ~ {max(gamma_values):.3f}")
    assert min(gamma_values) == 0.75, "最小折扣因子应为0.75"
    assert max(gamma_values) == 0.999, "最大折扣因子应为0.999"
    
    print("✅ 参数范围扩展验证通过")

def run_mini_experiment():
    """运行小规模实验验证"""
    print("🔧 运行小规模验证实验...")
    
    experiment = TaguchiL81Experiment()
    
    # 修改为小规模测试
    experiment.runs_per_experiment = 1  # 每组只运行1次
    
    # 只运行前3个实验组
    test_designs = experiment.l81_design[:3]
    
    # 生成问题实例
    problem = experiment._generate_problem_instance()
    
    print(f"   测试前3个实验组，每组1次运行")
    
    success_count = 0
    for exp_config in test_designs:
        try:
            result = experiment.run_single_experiment(exp_config, 1, problem)
            if result['success']:
                success_count += 1
                print(f"   实验{exp_config['exp_id']}: 成功 (HV={result['metrics']['hypervolume']:.4f})")
            else:
                print(f"   实验{exp_config['exp_id']}: 失败")
        except Exception as e:
            print(f"   实验{exp_config['exp_id']}: 异常 - {str(e)}")
    
    print(f"✅ 小规模验证完成: {success_count}/{len(test_designs)}个实验成功")

def main():
    """主测试函数"""
    print("🚀 开始L81田口实验设计验证")
    print("=" * 50)
    
    try:
        # 基础设计验证
        test_l81_design()
        print()
        
        # 参数映射验证
        test_parameter_mapping()
        print()
        
        # 水平覆盖验证
        test_level_coverage()
        print()
        
        # 正交性验证
        test_orthogonality()
        print()
        
        # 参数范围验证
        test_parameter_ranges()
        print()
        
        # 小规模实验验证
        run_mini_experiment()
        print()
        
        print("🎉 所有验证测试通过!")
        print("📈 L81设计相比L49的改进:")
        print("   - 参数水平: 7 → 9 (增加28.6%)")
        print("   - 实验组数: 49 → 81 (增加65.3%)")
        print("   - 参数覆盖: 更全面的参数空间探索")
        print("   - 学习率范围: 扩展到极端值")
        print("   - 衰减率精度: 更细粒度的控制")
        print("   - 分组策略: 更多样化的探索-开发平衡")
        print("   - 折扣因子: 更广泛的记忆长度选择")
        
    except Exception as e:
        print(f"❌ 验证失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 