#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IGD/GD差异化和HV归一化验证脚本
测试修复后的指标计算是否能产生合理且不同的值
"""

import sys
import os
import numpy as np

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from problem.mo_dhfsp import MO_DHFSP_Problem
from utils.data_generator import DataGenerator

# 导入修复后的计算函数
from table_format_comparison_with_ql_abc_full import (
    calculate_hypervolume, calculate_igd, calculate_gd, calculate_spacing
)

def create_test_solutions():
    """创建模拟的帕累托解集用于测试"""
    
    # 解集1：均匀分布的解
    solutions1 = [
        type('obj', (), {'makespan': 80, 'total_tardiness': 120}),
        type('obj', (), {'makespan': 90, 'total_tardiness': 100}),
        type('obj', (), {'makespan': 100, 'total_tardiness': 80}),
        type('obj', (), {'makespan': 110, 'total_tardiness': 60}),
        type('obj', (), {'makespan': 120, 'total_tardiness': 40})
    ]
    
    # 解集2：聚集性分布的解
    solutions2 = [
        type('obj', (), {'makespan': 85, 'total_tardiness': 85}),
        type('obj', (), {'makespan': 87, 'total_tardiness': 83}),
        type('obj', (), {'makespan': 89, 'total_tardiness': 81}),
        type('obj', (), {'makespan': 91, 'total_tardiness': 79})
    ]
    
    # 解集3：边界解（极值解）
    solutions3 = [
        type('obj', (), {'makespan': 75, 'total_tardiness': 150}),
        type('obj', (), {'makespan': 150, 'total_tardiness': 30})
    ]
    
    # 联合帕累托前沿（用作参考）
    reference_front = [
        (75, 30),   # 理想点（实际不可达）
        (80, 40),   # 接近理想的点
        (90, 50),
        (100, 60),
        (110, 80),
        (120, 100)
    ]
    
    return {
        'uniform': solutions1,
        'clustered': solutions2, 
        'boundary': solutions3
    }, reference_front

def test_metrics_calculation():
    """测试指标计算的差异化和归一化效果"""
    print("IGD/GD差异化和HV归一化验证测试")
    print("=" * 60)
    
    # 创建测试解集
    solution_sets, reference_front = create_test_solutions()
    
    print("📊 测试数据集说明:")
    print("  - uniform: 5个均匀分布的解")
    print("  - clustered: 4个聚集分布的解")  
    print("  - boundary: 2个边界极值解")
    print(f"  - reference_front: {len(reference_front)}个参考点")
    
    print(f"\n{'解集类型':<12} {'解数量':<8} {'归一化HV':<12} {'原始HV':<10} {'IGD':<10} {'GD':<10} {'Spacing':<10}")
    print("-" * 80)
    
    results = {}
    
    for set_name, solutions in solution_sets.items():
        # 计算归一化和原始超体积
        hv_normalized = calculate_hypervolume(solutions, normalize=True)
        hv_original = calculate_hypervolume(solutions, normalize=False)
        
        # 计算IGD和GD（使用参考前沿）
        igd = calculate_igd(solutions, reference_front)
        gd = calculate_gd(solutions, reference_front)
        
        # 计算Spacing
        spacing = calculate_spacing(solutions)
        
        results[set_name] = {
            'hv_norm': hv_normalized,
            'hv_orig': hv_original, 
            'igd': igd,
            'gd': gd,
            'spacing': spacing,
            'count': len(solutions)
        }
        
        print(f"{set_name:<12} {len(solutions):<8} {hv_normalized:<12.4f} {hv_original:<10.0f} {igd:<10.3f} {gd:<10.3f} {spacing:<10.3f}")
    
    # 验证修复效果
    print(f"\n✅ 修复验证结果:")
    
    # 1. 检查IGD和GD是否不同
    igd_gd_different = False
    for set_name, result in results.items():
        if abs(result['igd'] - result['gd']) > 0.001:  # 容差0.001
            igd_gd_different = True
            break
    
    if igd_gd_different:
        print("✅ IGD和GD指标修复成功：产生了不同的数值")
        for set_name, result in results.items():
            diff = abs(result['igd'] - result['gd'])
            print(f"   {set_name}: IGD={result['igd']:.3f}, GD={result['gd']:.3f}, 差异={diff:.3f}")
    else:
        print("❌ IGD和GD指标仍有问题：数值相同或过于接近")
    
    # 2. 检查HV归一化是否工作
    hv_normalized_ok = all(0.0 <= result['hv_norm'] <= 1.0 for result in results.values())
    hv_original_large = any(result['hv_orig'] > 10.0 for result in results.values())
    
    if hv_normalized_ok and hv_original_large:
        print("✅ HV归一化修复成功：归一化值在[0,1]区间，原始值较大")
        for set_name, result in results.items():
            ratio = result['hv_norm'] / (result['hv_orig'] / 1000) if result['hv_orig'] > 0 else 0
            print(f"   {set_name}: 归一化={result['hv_norm']:.4f}, 原始={result['hv_orig']:.0f}")
    else:
        print("❌ HV归一化仍有问题")
    
    # 3. 检查不同解集的指标差异
    print(f"\n📈 不同解集的指标差异分析:")
    print(f"  uniform vs clustered:")
    print(f"    HV差异: {abs(results['uniform']['hv_norm'] - results['clustered']['hv_norm']):.4f}")
    print(f"    IGD差异: {abs(results['uniform']['igd'] - results['clustered']['igd']):.3f}")
    print(f"    GD差异: {abs(results['uniform']['gd'] - results['clustered']['gd']):.3f}")
    print(f"    Spacing差异: {abs(results['uniform']['spacing'] - results['clustered']['spacing']):.3f}")
    
    # 4. 解释指标含义
    print(f"\n💡 指标解释:")
    print(f"  - HV (归一化): 解集覆盖的目标空间体积，越大越好，[0,1]区间")
    print(f"  - IGD: 参考前沿到解集的平均距离，越小越好")
    print(f"  - GD: 解集到参考前沿的平均距离，越小越好")
    print(f"  - Spacing: 解集分布的均匀性，越小越好")
    
    return results

if __name__ == "__main__":
    test_metrics_calculation() 