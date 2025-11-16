#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
帕累托前沿管理器
用于维护和管理多目标优化的帕累托最优解集
"""

import numpy as np
from typing import List
from problem.mo_dhfsp import Solution

class ParetoManager:
    def enhanced_diversity_selection(self, solutions, max_size):
        """增强的多样性选择策略"""
        if len(solutions) <= max_size:
            return solutions
        
        # 1. 网格化选择
        grid_selected = self._grid_based_selection(solutions, max_size // 2)
        
        # 2. 聚类选择
        cluster_selected = self._cluster_based_selection(solutions, max_size // 3)
        
        # 3. 随机选择保持多样性
        remaining_size = max_size - len(grid_selected) - len(cluster_selected)
        random_selected = self._random_diversity_selection(solutions, remaining_size)
        
        # 合并并手动去重（避免set操作）
        all_selected = grid_selected + cluster_selected + random_selected
        final_selected = []
        
        for sol in all_selected:
            # 检查是否已存在相同的解
            is_duplicate = False
            for existing in final_selected:
                if (abs(sol.makespan - existing.makespan) < 1e-6 and 
                    abs(sol.total_tardiness - existing.total_tardiness) < 1e-6):
                    is_duplicate = True
                    break
            if not is_duplicate:
                final_selected.append(sol)
        
        return final_selected[:max_size]
    
    def _grid_based_selection(self, solutions, target_size):
        """基于网格的选择"""
        if target_size <= 0:
            return []
        
        # 按makespan和tardiness分别排序
        sorted_by_makespan = sorted(solutions, key=lambda x: x.makespan)
        sorted_by_tardiness = sorted(solutions, key=lambda x: x.total_tardiness)
        
        selected = []
        
        # 从makespan维度选择
        step = len(sorted_by_makespan) // (target_size // 2 + 1)
        for i in range(0, len(sorted_by_makespan), max(1, step)):
            if len(selected) < target_size // 2:
                selected.append(sorted_by_makespan[i])
        
        # 从tardiness维度选择
        step = len(sorted_by_tardiness) // (target_size // 2 + 1)
        for i in range(0, len(sorted_by_tardiness), max(1, step)):
            if len(selected) < target_size and sorted_by_tardiness[i] not in selected:
                selected.append(sorted_by_tardiness[i])
        
        return selected
    
    def _cluster_based_selection(self, solutions, target_size):
        """基于聚类的选择"""
        if target_size <= 0 or len(solutions) <= target_size:
            return solutions[:target_size]
        
        import random
        
        # 简单的k-means聚类选择
        selected = []
        
        # 随机选择初始中心
        centers = random.sample(solutions, min(target_size, len(solutions)))
        
        for center in centers:
            if len(selected) < target_size:
                selected.append(center)
        
        return selected
    
    def _random_diversity_selection(self, solutions, target_size):
        """随机多样性选择"""
        if target_size <= 0:
            return []
        
        import random
        return random.sample(solutions, min(target_size, len(solutions)))

    """帕累托前沿管理器"""
    
    def __init__(self):
        """初始化帕累托管理器"""
        pass
    
    def is_dominated(self, sol1: Solution, sol2: Solution) -> bool:
        """
        检查sol1是否被sol2支配
        
        Args:
            sol1: 解1
            sol2: 解2
            
        Returns:
            True如果sol1被sol2支配
        """
        # sol1被sol2支配当且仅当：
        # sol2在所有目标上都不劣于sol1，且至少在一个目标上严格优于sol1
        return (sol2.makespan <= sol1.makespan and 
                sol2.total_tardiness <= sol1.total_tardiness and
                (sol2.makespan < sol1.makespan or sol2.total_tardiness < sol1.total_tardiness))
    
    def update_pareto_front(self, solutions: List[Solution]) -> List[Solution]:
        """
        更新帕累托前沿
        
        Args:
            solutions: 候选解列表
            
        Returns:
            帕累托最优解列表
        """
        if not solutions:
            return []
        
        pareto_solutions = []
        
        for candidate in solutions:
            is_dominated = False
            solutions_to_remove = []
            
            # 检查候选解是否被现有帕累托解支配
            for existing in pareto_solutions:
                if self.is_dominated(candidate, existing):
                    is_dominated = True
                    break
                elif self.is_dominated(existing, candidate):
                    solutions_to_remove.append(existing)
            
            # 如果候选解不被支配，添加到帕累托前沿
            if not is_dominated:
                # 移除被新解支配的解
                for sol in solutions_to_remove:
                    pareto_solutions.remove(sol)
                pareto_solutions.append(candidate)
        
        return pareto_solutions
    
    def select_diverse_solutions(self, solutions: List[Solution], max_size: int) -> List[Solution]:
        """
        从解集中选择多样化的解 - 超级增强版本2.0，极度宽松的多样性要求
        
        Args:
            solutions: 候选解列表
            max_size: 最大选择数量
            
        Returns:
            选择的多样化解列表
        """
        if len(solutions) <= max_size:
            return solutions
        
        # 极度宽松的多样性策略，几乎保留所有解
        selected = []
        remaining = solutions.copy()
        
        # 1. 保留所有关键解（边界解）
        min_makespan_sol = min(remaining, key=lambda x: x.makespan)
        min_tardiness_sol = min(remaining, key=lambda x: x.total_tardiness)
        max_makespan_sol = max(remaining, key=lambda x: x.makespan)
        max_tardiness_sol = max(remaining, key=lambda x: x.total_tardiness)
        
        # 2. 增加更多关键点
        f1_values = [s.makespan for s in remaining]
        f2_values = [s.total_tardiness for s in remaining]
        
        # 分位数解
        q25_f1, q50_f1, q75_f1 = np.percentile(f1_values, [25, 50, 75])
        q25_f2, q50_f2, q75_f2 = np.percentile(f2_values, [25, 50, 75])
        
        key_solutions = [
            min_makespan_sol, min_tardiness_sol, max_makespan_sol, max_tardiness_sol,
            min(remaining, key=lambda x: abs(x.makespan - q25_f1)),
            min(remaining, key=lambda x: abs(x.makespan - q50_f1)),
            min(remaining, key=lambda x: abs(x.makespan - q75_f1)),
            min(remaining, key=lambda x: abs(x.total_tardiness - q25_f2)),
            min(remaining, key=lambda x: abs(x.total_tardiness - q50_f2)),
            min(remaining, key=lambda x: abs(x.total_tardiness - q75_f2))
        ]
        
        # 添加所有关键解
        for sol in key_solutions:
            if sol in remaining and sol not in selected:
                selected.append(sol)
                remaining.remove(sol)
        
        # 3. 极度放宽的网格选择 - 更密集的网格
        if remaining and len(selected) < max_size:
            min_f1 = min(s.makespan for s in solutions)
            max_f1 = max(s.makespan for s in solutions)
            min_f2 = min(s.total_tardiness for s in solutions)
            max_f2 = max(s.total_tardiness for s in solutions)
            
            # 更密集的网格，确保有更多解被选中
            grid_size = min(12, max(8, int(np.sqrt(max_size - len(selected)))))
            
            if grid_size > 0 and max_f1 > min_f1 and max_f2 > min_f2:
                f1_step = (max_f1 - min_f1) / grid_size
                f2_step = (max_f2 - min_f2) / grid_size
                
                for i in range(grid_size + 1):  # 包含边界
                    for j in range(grid_size + 1):
                        if len(selected) >= max_size * 0.9:  # 仅在接近满时停止
                            break
                        
                        target_f1 = min_f1 + i * f1_step
                        target_f2 = min_f2 + j * f2_step
                        
                        # 在更大的邻域内查找解
                        search_radius = max(f1_step, f2_step) * 1.5
                        candidates_in_region = []
                        
                        for sol in remaining:
                            dist = np.sqrt((sol.makespan - target_f1)**2 + 
                                         (sol.total_tardiness - target_f2)**2)
                            if dist <= search_radius:
                                candidates_in_region.append((sol, dist))
                        
                        # 选择该区域内的多个解（不只是1个）
                        candidates_in_region.sort(key=lambda x: x[1])
                        max_per_region = min(3, len(candidates_in_region))  # 每个区域最多3个解
                        
                        for k in range(max_per_region):
                            if k < len(candidates_in_region) and len(selected) < max_size:
                                sol = candidates_in_region[k][0]
                                if sol not in selected:
                                    selected.append(sol)
                                    if sol in remaining:
                                        remaining.remove(sol)
        
        # 4. 分层随机选择 - 极度放宽筛选
        if remaining and len(selected) < max_size:
            # 按目标函数值分层
            sorted_by_f1 = sorted(remaining, key=lambda x: x.makespan)
            sorted_by_f2 = sorted(remaining, key=lambda x: x.total_tardiness)
            
            # 从每层选择解
            layer_size = max(1, len(remaining) // 10)  # 分成10层
            for layer_start in range(0, len(remaining), layer_size):
                if len(selected) >= max_size:
                    break
                
                # 从完工时间排序的层中选择
                layer_end = min(layer_start + layer_size, len(sorted_by_f1))
                for sol in sorted_by_f1[layer_start:layer_end]:
                    if len(selected) >= max_size:
                        break
                    if sol not in selected:
                        selected.append(sol)
                        if sol in remaining:
                            remaining.remove(sol)
                
                # 从拖期排序的层中选择
                layer_end = min(layer_start + layer_size, len(sorted_by_f2))
                for sol in sorted_by_f2[layer_start:layer_end]:
                    if len(selected) >= max_size:
                        break
                    if sol not in selected:
                        selected.append(sol)
                        if sol in remaining:
                            remaining.remove(sol)
        
        # 5. 最终随机补充 - 直接添加所有剩余解直到满额
        while len(selected) < max_size and remaining:
            sol = remaining.pop(0)
            if sol not in selected:
                selected.append(sol)
        
        return selected
    
    def _calculate_crowding_distance(self, solution: Solution, all_solutions: List[Solution]) -> float:
        """
        计算拥挤距离
        
        Args:
            solution: 目标解
            all_solutions: 所有解
            
        Returns:
            拥挤距离
        """
        # 计算在目标空间中与其他解的最小距离
        min_distance = float('inf')
        
        for other in all_solutions:
            if other == solution:
                continue
            
            # 欧几里得距离
            distance = np.sqrt((solution.makespan - other.makespan)**2 + 
                             (solution.total_tardiness - other.total_tardiness)**2)
            min_distance = min(min_distance, distance)
        
        return min_distance if min_distance != float('inf') else 0.0
    
    def _euclidean_distance(self, sol1: Solution, sol2: Solution) -> float:
        """计算两个解之间的欧几里得距离"""
        return np.sqrt((sol1.makespan - sol2.makespan)**2 + 
                      (sol1.total_tardiness - sol2.total_tardiness)**2) 