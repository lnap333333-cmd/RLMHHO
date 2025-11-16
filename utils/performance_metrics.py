#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能评估工具
用于计算各种性能指标
"""

import numpy as np
from typing import List, Dict
from problem.mo_dhfsp import Solution

class PerformanceEvaluator:
    """性能评估器"""
    
    def __init__(self):
        """初始化评估器"""
        pass
    
    def evaluate(self, solutions: List[Solution], convergence_data: Dict) -> Dict:
        """
        综合评估算法性能
        
        Args:
            solutions: 最终解集
            convergence_data: 收敛数据
            
        Returns:
            性能指标字典
        """
        metrics = {}
        
        if solutions:
            # 解质量指标
            metrics.update(self._calculate_solution_quality(solutions))
            
            # 多样性指标
            metrics.update(self._calculate_diversity(solutions))
        
        # 收敛性指标
        metrics.update(self._calculate_convergence_metrics(convergence_data))
        
        return metrics
    
    def _calculate_solution_quality(self, solutions: List[Solution]) -> Dict:
        """计算解质量指标"""
        makespans = [sol.makespan for sol in solutions]
        tardiness = [sol.total_tardiness for sol in solutions]
        
        return {
            'best_makespan': min(makespans),
            'worst_makespan': max(makespans),
            'avg_makespan': np.mean(makespans),
            'std_makespan': np.std(makespans),
            'best_tardiness': min(tardiness),
            'worst_tardiness': max(tardiness),
            'avg_tardiness': np.mean(tardiness),
            'std_tardiness': np.std(tardiness),
            'pareto_size': len(solutions)
        }
    
    def _calculate_diversity(self, solutions: List[Solution]) -> Dict:
        """计算解集多样性指标"""
        if len(solutions) < 2:
            return {'diversity_metric': 0.0, 'spacing_metric': 0.0}
        
        # 计算解之间的平均距离
        distances = []
        for i in range(len(solutions)):
            for j in range(i + 1, len(solutions)):
                dist = np.sqrt(
                    (solutions[i].makespan - solutions[j].makespan)**2 +
                    (solutions[i].total_tardiness - solutions[j].total_tardiness)**2
                )
                distances.append(dist)
        
        diversity_metric = np.mean(distances) if distances else 0.0
        
        # 计算间距指标
        spacing_distances = []
        for i, sol in enumerate(solutions):
            min_dist = float('inf')
            for j, other_sol in enumerate(solutions):
                if i != j:
                    dist = np.sqrt(
                        (sol.makespan - other_sol.makespan)**2 +
                        (sol.total_tardiness - other_sol.total_tardiness)**2
                    )
                    min_dist = min(min_dist, dist)
            spacing_distances.append(min_dist)
        
        mean_spacing = np.mean(spacing_distances)
        spacing_metric = np.sqrt(np.mean([(d - mean_spacing)**2 for d in spacing_distances]))
        
        return {
            'diversity_metric': diversity_metric,
            'spacing_metric': spacing_metric
        }
    
    def _calculate_convergence_metrics(self, convergence_data: Dict) -> Dict:
        """计算收敛性指标"""
        makespan_history = convergence_data.get('makespan_history', [])
        tardiness_history = convergence_data.get('tardiness_history', [])
        
        if not makespan_history:
            return {'convergence_rate': 0.0, 'stability': 0.0}
        
        # 计算收敛速度
        if len(makespan_history) > 1:
            initial_makespan = makespan_history[0]
            final_makespan = makespan_history[-1]
            convergence_rate = (initial_makespan - final_makespan) / max(initial_makespan, 1)
        else:
            convergence_rate = 0.0
        
        # 计算后期稳定性
        if len(makespan_history) >= 20:
            last_20_makespan = makespan_history[-20:]
            stability = 1.0 / (1.0 + np.std(last_20_makespan))
        else:
            stability = 0.0
        
        return {
            'convergence_rate': max(0, convergence_rate),
            'stability': stability,
            'total_iterations': len(makespan_history)
        }
    
    def compare_algorithms(self, results_list: List[Dict], algorithm_names: List[str]) -> Dict:
        """
        比较多个算法的性能
        
        Args:
            results_list: 多个算法的结果列表
            algorithm_names: 算法名称列表
            
        Returns:
            比较结果
        """
        comparison = {}
        
        metrics_to_compare = [
            'best_makespan', 'best_tardiness', 'pareto_size',
            'diversity_metric', 'convergence_rate', 'stability'
        ]
        
        for metric in metrics_to_compare:
            comparison[metric] = {}
            values = []
            
            for i, result in enumerate(results_list):
                if 'metrics' in result and metric in result['metrics']:
                    value = result['metrics'][metric]
                    comparison[metric][algorithm_names[i]] = value
                    values.append(value)
            
            if values:
                comparison[metric]['best'] = min(values) if 'makespan' in metric or 'tardiness' in metric else max(values)
                comparison[metric]['worst'] = max(values) if 'makespan' in metric or 'tardiness' in metric else min(values)
                comparison[metric]['average'] = np.mean(values)
        
        return comparison
    
    def calculate_igd(self, pareto_front: List[Solution], reference_front: List[Solution]) -> float:
        """
        计算反世代距离(Inverted Generational Distance, IGD)
        
        Args:
            pareto_front: 算法得到的帕累托前沿
            reference_front: 参考帕累托前沿（通常是所有算法的合并前沿）
            
        Returns:
            IGD值（越小越好）
        """
        if not pareto_front or not reference_front:
            return float('inf')
        
        # 提取目标函数值
        pf_objectives = [(sol.makespan, sol.total_tardiness) for sol in pareto_front]
        ref_objectives = [(sol.makespan, sol.total_tardiness) for sol in reference_front]
        
        # 对每个参考点，找到最近的帕累托前沿点
        distances = []
        for ref_point in ref_objectives:
            min_distance = float('inf')
            for pf_point in pf_objectives:
                distance = np.sqrt(
                    (ref_point[0] - pf_point[0])**2 + 
                    (ref_point[1] - pf_point[1])**2
                )
                min_distance = min(min_distance, distance)
            distances.append(min_distance)
        
        return np.mean(distances)
    
    def calculate_hypervolume(self, pareto_front: List[Solution], reference_point: List[float]) -> float:
        """
        计算超体积(Hypervolume, HV)
        
        Args:
            pareto_front: 帕累托前沿
            reference_point: 参考点（负理想点）
            
        Returns:
            超体积值（越大越好）
        """
        if not pareto_front:
            return 0.0
        
        # 提取目标函数值并排序
        objectives = []
        for sol in pareto_front:
            objectives.append([sol.makespan, sol.total_tardiness])
        
        # 简化的超体积计算（适用于2D）
        objectives = np.array(objectives)
        
        # 按第一个目标排序
        sorted_indices = np.argsort(objectives[:, 0])
        sorted_objectives = objectives[sorted_indices]
        
        hypervolume = 0.0
        prev_y = reference_point[1]
        
        for i, point in enumerate(sorted_objectives):
            if i == 0:
                width = reference_point[0] - point[0]
            else:
                width = sorted_objectives[i-1][0] - point[0]
            
            height = prev_y - point[1]
            if width > 0 and height > 0:
                hypervolume += width * height
            prev_y = min(prev_y, point[1])
        
        return max(0, hypervolume)
    
    def calculate_spacing(self, pareto_front: List[Solution]) -> float:
        """
        计算间距指标(Spacing)
        
        Args:
            pareto_front: 帕累托前沿
            
        Returns:
            间距值（越小越好，表示分布越均匀）
        """
        if len(pareto_front) < 2:
            return 0.0
        
        # 计算每个解到其最近邻的距离
        distances = []
        for i, sol_i in enumerate(pareto_front):
            min_dist = float('inf')
            for j, sol_j in enumerate(pareto_front):
                if i != j:
                    dist = np.sqrt(
                        (sol_i.makespan - sol_j.makespan)**2 + 
                        (sol_i.total_tardiness - sol_j.total_tardiness)**2
                    )
                    min_dist = min(min_dist, dist)
            distances.append(min_dist)
        
        # 计算间距指标
        mean_distance = np.mean(distances)
        spacing = np.sqrt(np.mean([(d - mean_distance)**2 for d in distances]))
        
        return spacing 