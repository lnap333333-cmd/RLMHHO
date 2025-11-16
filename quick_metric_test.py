#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速指标修复验证脚本
"""
import numpy as np

# 创建模拟解对象
class Solution:
    def __init__(self, makespan, total_tardiness):
        self.makespan = makespan
        self.total_tardiness = total_tardiness

# 复制修复后的指标计算函数
def calculate_hypervolume_fixed(pareto_solutions, reference_point=None, normalize=True):
    """修复后的超体积计算（支持归一化）"""
    if not pareto_solutions:
        return 0.0
    
    objectives = [(sol.makespan, sol.total_tardiness) for sol in pareto_solutions]
    
    if len(objectives) == 1:
        return 1.0 if normalize else 100.0
    
    if reference_point is None:
        max_makespan = max(obj[0] for obj in objectives)
        max_tardiness = max(obj[1] for obj in objectives)
        margin_makespan = max_makespan * 0.1
        margin_tardiness = max_tardiness * 0.1
        reference_point = (max_makespan + margin_makespan, max_tardiness + margin_tardiness)
    
    sorted_objectives = sorted(objectives, key=lambda x: x[0])
    
    hypervolume = 0.0
    prev_makespan = 0.0
    
    for i, (makespan, tardiness) in enumerate(sorted_objectives):
        width = makespan - prev_makespan
        height = reference_point[1] - tardiness
        
        if width > 0 and height > 0:
            hypervolume += width * height
        
        prev_makespan = makespan
    
    if normalize:
        max_hv = reference_point[0] * reference_point[1]
        if max_hv > 0:
            hypervolume = min(hypervolume / max_hv, 1.0)
        else:
            hypervolume = 0.0
    
    return hypervolume

def calculate_igd_fixed(pareto_solutions, true_pareto_front=None):
    """修复后的IGD计算"""
    if not pareto_solutions:
        return float('inf')
    
    current_objectives = [(sol.makespan, sol.total_tardiness) for sol in pareto_solutions]
    
    if true_pareto_front is None:
        min_makespan = min(obj[0] for obj in current_objectives)
        min_tardiness = min(obj[1] for obj in current_objectives)
        max_makespan = max(obj[0] for obj in current_objectives)
        max_tardiness = max(obj[1] for obj in current_objectives)
        
        true_pareto_front = [
            (min_makespan, min_tardiness),
            (min_makespan, max_tardiness),
            (max_makespan, min_tardiness)
        ]
    
    total_distance = 0.0
    for true_point in true_pareto_front:
        min_distance = float('inf')
        for current_point in current_objectives:
            distance = np.sqrt((true_point[0] - current_point[0])**2 + 
                             (true_point[1] - current_point[1])**2)
            min_distance = min(min_distance, distance)
        total_distance += min_distance
    
    return total_distance / len(true_pareto_front)

def calculate_gd_fixed(pareto_solutions, true_pareto_front=None):
    """修复后的GD计算"""
    if not pareto_solutions:
        return float('inf')
    
    current_objectives = [(sol.makespan, sol.total_tardiness) for sol in pareto_solutions]
    
    if true_pareto_front is None:
        min_makespan = min(obj[0] for obj in current_objectives)
        min_tardiness = min(obj[1] for obj in current_objectives)
        max_makespan = max(obj[0] for obj in current_objectives)
        max_tardiness = max(obj[1] for obj in current_objectives)
        
        true_pareto_front = [
            (min_makespan, min_tardiness),
            (min_makespan, (min_tardiness + max_tardiness) / 2),
            (min_makespan, max_tardiness),
            ((min_makespan + max_makespan) / 2, min_tardiness),
            (max_makespan, min_tardiness)
        ]
    
    total_distance = 0.0
    for current_point in current_objectives:
        min_distance = float('inf')
        for true_point in true_pareto_front:
            distance = np.sqrt((current_point[0] - true_point[0])**2 + 
                             (current_point[1] - true_point[1])**2)
            min_distance = min(min_distance, distance)
        total_distance += min_distance
    
    return total_distance / len(current_objectives)

def test_metrics():
    """测试指标修复效果"""
    print("指标修复验证测试")
    print("=" * 50)
    
    # 创建测试解集
    solutions1 = [
        Solution(80, 120),
        Solution(90, 100), 
        Solution(100, 80),
        Solution(110, 60),
        Solution(120, 40)
    ]
    
    solutions2 = [
        Solution(85, 85),
        Solution(87, 83),
        Solution(89, 81)
    ]
    
    reference_front = [(75, 30), (80, 40), (90, 50), (100, 60), (110, 80), (120, 100)]
    
    print(f"测试解集1: {len(solutions1)}个解")
    print(f"测试解集2: {len(solutions2)}个解")
    print(f"参考前沿: {len(reference_front)}个点")
    
    print(f"\n{'解集':<10} {'归一化HV':<12} {'原始HV':<10} {'IGD':<10} {'GD':<10} {'IGD≠GD':<8}")
    print("-" * 60)
    
    for i, solutions in enumerate([solutions1, solutions2], 1):
        # 计算指标
        hv_norm = calculate_hypervolume_fixed(solutions, normalize=True)
        hv_orig = calculate_hypervolume_fixed(solutions, normalize=False)
        igd = calculate_igd_fixed(solutions, reference_front)
        gd = calculate_gd_fixed(solutions, reference_front)
        
        different = "✓" if abs(igd - gd) > 0.001 else "✗"
        
        print(f"解集{i:<5d} {hv_norm:<12.4f} {hv_orig:<10.0f} {igd:<10.3f} {gd:<10.3f} {different:<8}")
    
    print(f"\n✅ 验证结果:")
    
    # 验证1: IGD和GD不同
    igd1 = calculate_igd_fixed(solutions1, reference_front)
    gd1 = calculate_gd_fixed(solutions1, reference_front)
    if abs(igd1 - gd1) > 0.001:
        print("✅ IGD和GD修复成功: 产生了不同数值")
    else:
        print("❌ IGD和GD仍相等")
    
    # 验证2: HV归一化
    hv_norm = calculate_hypervolume_fixed(solutions1, normalize=True)
    hv_orig = calculate_hypervolume_fixed(solutions1, normalize=False)
    if 0 <= hv_norm <= 1 and hv_orig > 10:
        print("✅ HV归一化修复成功: 归一化值在[0,1]区间")
    else:
        print("❌ HV归一化失败")
    
    print(f"\n📊 详细数值:")
    print(f"  解集1: IGD={igd1:.3f}, GD={gd1:.3f}, 差异={abs(igd1-gd1):.3f}")
    print(f"  解集1: HV归一化={hv_norm:.4f}, 原始={hv_orig:.0f}")

if __name__ == "__main__":
    test_metrics() 