#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
四层鹰群分组协作管理器 - 完整实现
基于强化学习协调的混沌哈里斯鹰优化算法的核心组件
"""

import numpy as np
import random
import copy
from typing import List, Dict, Tuple, Optional
from collections import deque
from dataclasses import dataclass
from problem.mo_dhfsp import Solution
from .chaotic_maps import ChaoticMaps

@dataclass
class GroupPerformance:
    """组性能指标"""
    improvement_count: int = 0           # 改进次数
    average_quality: float = 0.0         # 平均质量
    diversity_score: float = 0.0         # 多样性分数
    convergence_rate: float = 0.0        # 收敛率
    success_rate: float = 0.0            # 成功率
    energy_consumption: float = 0.0      # 能量消耗
    exploration_efficiency: float = 0.0  # 探索效率
    exploitation_efficiency: float = 0.0 # 开发效率

class EagleGroupManager:
    """四层鹰群分组协作管理器"""
    
    def __init__(self, population_size: int, n_jobs: int, n_factories: int):
        """
        初始化鹰群管理器
        
        Args:
            population_size: 种群大小
            n_jobs: 作业数量
            n_factories: 工厂数量
        """
        self.population_size = population_size
        self.n_jobs = n_jobs
        self.n_factories = n_factories
        
        # 混沌映射系统
        self.chaos_maps = ChaoticMaps()
        
        # 四大分组配置 - 调整比例为0.4, 0.35, 0.15, 0.1
        self.group_config = {
            'exploration': {'ratio': 0.40, 'chaos_type': 'logistic'},    # 探索组 40%
            'exploitation': {'ratio': 0.35, 'chaos_type': 'tent'},      # 开发组 35%
            'balance': {'ratio': 0.15, 'chaos_type': 'sine'},           # 平衡组 15%
            'elite': {'ratio': 0.10, 'chaos_type': 'chebyshev'}         # 精英组 10%
        }
        
        # 动态调整参数（必须在初始化分组之前）
        self.adaptation_threshold = 0.1      # 适应阈值
        self.performance_window = 10         # 性能评估窗口
        self.min_group_size = max(2, population_size // 20)  # 最小组大小
        
        # 初始化分组
        self._initialize_groups()
        
        # 性能监控
        self.group_performance = {
            group: GroupPerformance() for group in self.group_config.keys()
        }
        
        # 历史记录
        self.performance_history = {group: deque(maxlen=20) for group in self.group_config.keys()}
        self.adaptation_history = []
        
        print(f"初始化四层鹰群分组管理器:")
        print(f"  探索组: {len(self.groups['exploration'])}只鹰 ({self.group_config['exploration']['ratio']*100:.1f}%)")
        print(f"  开发组: {len(self.groups['exploitation'])}只鹰 ({self.group_config['exploitation']['ratio']*100:.1f}%)")
        print(f"  平衡组: {len(self.groups['balance'])}只鹰 ({self.group_config['balance']['ratio']*100:.1f}%)")
        print(f"  精英组: {len(self.groups['elite'])}只鹰 ({self.group_config['elite']['ratio']*100:.1f}%)")
    
    def _initialize_groups(self):
        """初始化各组分配"""
        indices = list(range(self.population_size))
        random.shuffle(indices)
        
        self.groups = {}
        start_idx = 0
        
        for group_name, config in self.group_config.items():
            group_size = max(self.min_group_size, int(self.population_size * config['ratio']))
            end_idx = min(start_idx + group_size, self.population_size)
            self.groups[group_name] = indices[start_idx:end_idx]
            start_idx = end_idx
        
        # 确保所有个体都被分配
        if start_idx < self.population_size:
            self.groups['exploration'].extend(indices[start_idx:])
    
    def assign_eagles(self, population: List[Solution]):
        """动态分配鹰到各组"""
        self.population = population
        
        # 基于解质量重新分配
        self._quality_based_assignment()
        
        # 更新组性能
        self._update_group_performance()
    
    def _quality_based_assignment(self):
        """基于解质量的动态分配"""
        if not hasattr(self, 'population') or not self.population:
            return
        
        # 计算解的综合质量分数
        quality_scores = []
        for sol in self.population:
            # 归一化目标值
            makespan_norm = sol.makespan / max(s.makespan for s in self.population)
            tardiness_norm = sol.total_tardiness / max(max(s.total_tardiness for s in self.population), 1)
            quality = 1.0 / (1.0 + 0.5 * makespan_norm + 0.5 * tardiness_norm)
            quality_scores.append(quality)
        
        # 按质量排序
        sorted_indices = sorted(range(len(self.population)), 
                              key=lambda i: quality_scores[i], reverse=True)
        
        # 重新分配
        self.groups = {}
        start_idx = 0
        
        # 精英组：最优的10%
        elite_size = max(self.min_group_size, int(self.population_size * self.group_config['elite']['ratio']))
        self.groups['elite'] = sorted_indices[start_idx:start_idx + elite_size]
        start_idx += elite_size
        
        # 开发组：次优的25%
        exploit_size = max(self.min_group_size, int(self.population_size * self.group_config['exploitation']['ratio']))
        self.groups['exploitation'] = sorted_indices[start_idx:start_idx + exploit_size]
        start_idx += exploit_size
        
        # 平衡组：中等的20%
        balance_size = max(self.min_group_size, int(self.population_size * self.group_config['balance']['ratio']))
        self.groups['balance'] = sorted_indices[start_idx:start_idx + balance_size]
        start_idx += balance_size
        
        # 探索组：其余的45%
        self.groups['exploration'] = sorted_indices[start_idx:]
    
    def get_group(self, group_name: str) -> List[int]:
        """获取指定组的鹰索引"""
        return self.groups.get(group_name, [])
    
    def get_group_solutions(self, group_name: str) -> List[Solution]:
        """获取指定组的解"""
        if not hasattr(self, 'population'):
            return []
        
        indices = self.get_group(group_name)
        return [self.population[i] for i in indices if i < len(self.population)]
    
    def get_performance_metrics(self) -> List[float]:
        """获取各组性能指标（20维向量）"""
        metrics = []
        
        for group_name in ['exploration', 'exploitation', 'balance', 'elite']:
            perf = self.group_performance[group_name]
            group_metrics = [
                perf.improvement_count / 100.0,      # 改进次数（归一化）
                min(perf.average_quality, 1.0),      # 平均质量
                min(perf.diversity_score, 1.0),      # 多样性分数
                min(perf.convergence_rate, 1.0),     # 收敛率
                min(perf.success_rate, 1.0)          # 成功率
            ]
            metrics.extend(group_metrics)
        
        return metrics
    
    def _update_group_performance(self):
        """更新各组性能指标"""
        if not hasattr(self, 'population'):
            return
        
        for group_name, indices in self.groups.items():
            if not indices:
                continue
                
            group_solutions = [self.population[i] for i in indices if i < len(self.population)]
            if not group_solutions:
                continue
            
            perf = self.group_performance[group_name]
            
            # 计算平均质量
            makespans = [sol.makespan for sol in group_solutions]
            tardiness = [sol.total_tardiness for sol in group_solutions]
            
            if makespans and tardiness:
                avg_makespan = np.mean(makespans)
                avg_tardiness = np.mean(tardiness)
                perf.average_quality = 1.0 / (1.0 + avg_makespan + avg_tardiness)
                
                # 计算多样性分数
                makespan_std = np.std(makespans) if len(makespans) > 1 else 0
                tardiness_std = np.std(tardiness) if len(tardiness) > 1 else 0
                perf.diversity_score = (makespan_std + tardiness_std) / (avg_makespan + avg_tardiness + 1e-6)
                
                # 更新历史记录
                self.performance_history[group_name].append(perf.average_quality)
                
                # 计算收敛率
                if len(self.performance_history[group_name]) >= 3:
                    recent_qualities = list(self.performance_history[group_name])[-3:]
                    perf.convergence_rate = (recent_qualities[-1] - recent_qualities[0]) / max(recent_qualities[0], 1e-6)
    
    def enhance_exploration(self):
        """强化全局探索"""
        # 增加探索组比例
        self._adjust_group_ratio('exploration', 0.1)
        
        # 提高探索组的混沌强度
        for idx in self.groups['exploration']:
            if hasattr(self, 'population') and idx < len(self.population):
                self._apply_chaotic_perturbation(idx, intensity=0.8)
        
        print("🔍 执行策略：强化全局探索")
    
    def enhance_exploitation(self):
        """强化局部开发"""
        # 增加开发组和精英组比例
        self._adjust_group_ratio('exploitation', 0.08)
        self._adjust_group_ratio('elite', 0.05)
        
        # 对开发组应用精细搜索
        for idx in self.groups['exploitation']:
            if hasattr(self, 'population') and idx < len(self.population):
                self._apply_local_refinement(idx)
        
        print("🎯 执行策略：强化局部开发")
    
    def balance_search(self):
        """平衡搜索"""
        # 调整各组比例趋向平衡
        target_ratios = {'exploration': 0.4, 'exploitation': 0.3, 'balance': 0.2, 'elite': 0.1}
        for group_name, ratio in target_ratios.items():
            self._adjust_group_ratio(group_name, 0.02, target_ratio=ratio)
        
        # 平衡组执行适中强度的搜索
        for idx in self.groups['balance']:
            if hasattr(self, 'population') and idx < len(self.population):
                self._apply_balanced_search(idx)
        
        print("⚖️ 执行策略：平衡搜索")
    
    def diversity_rescue(self):
        """多样性救援策略 - 增强版"""
        print("🎭 执行策略：多样性救援", end="")
        
        # 分析当前种群多样性
        diversity_metrics = self._analyze_diversity()
        
        # 根据多样性情况选择救援策略
        if diversity_metrics['makespan_cv'] < 0.1 and diversity_metrics['tardiness_cv'] < 0.1:
            # 多样性极低，大幅度救援
            affected_groups = ['exploration', 'balance', 'elite']
            rescue_intensity = 0.8  # 80%的个体参与救援
        elif diversity_metrics['makespan_cv'] < 0.2 or diversity_metrics['tardiness_cv'] < 0.2:
            # 多样性较低，中等救援
            affected_groups = ['balance', 'elite']
            rescue_intensity = 0.6  # 60%的个体参与救援
        else:
            # 多样性尚可，轻度救援
            affected_groups = ['elite']
            rescue_intensity = 0.4  # 40%的个体参与救援
        
        print(f" (影响组: {affected_groups}, 强度: {rescue_intensity:.0%})")
        
        # 对选定组进行多样性注入
        for group_name in affected_groups:
            group_indices = self.groups[group_name]
            n_rescue = max(1, int(len(group_indices) * rescue_intensity))
            
            # 随机选择需要救援的个体
            rescue_indices = random.sample(group_indices, min(n_rescue, len(group_indices)))
            
            for idx in rescue_indices:
                if idx < len(self.population):
                    # 生成多样化的新个体
                    self.population[idx] = self._generate_diverse_individual()
        
        # 更新组性能统计
        self._update_group_performance()
    
    def _analyze_diversity(self) -> Dict:
        """分析种群多样性"""
        if not self.population:
            return {'makespan_cv': 0.0, 'tardiness_cv': 0.0}
        
        makespans = [sol.makespan for sol in self.population]
        tardiness = [sol.total_tardiness for sol in self.population]
        
        # 计算变异系数
        makespan_cv = np.std(makespans) / max(np.mean(makespans), 1e-6)
        tardiness_cv = np.std(tardiness) / max(np.mean(tardiness), 1e-6)
        
        return {
            'makespan_cv': makespan_cv,
            'tardiness_cv': tardiness_cv,
            'makespan_range': max(makespans) - min(makespans),
            'tardiness_range': max(tardiness) - min(tardiness)
        }
    
    def _generate_diverse_individual(self) -> 'Solution':
        """生成多样化的个体"""
        # 使用增强的随机生成策略
        from problem.mo_dhfsp import Solution
        import random
        
        # 随机工厂分配（倾向于平衡分配）
        factory_assignment = []
        for job_id in range(self.n_jobs):
            # 选择负载较轻的工厂
            factory_loads = [0] * self.n_factories
            for assigned_job, factory in enumerate(factory_assignment):
                factory_loads[factory] += 1
            
            # 80%概率选择负载最轻的工厂，20%概率随机选择
            if random.random() < 0.8:
                min_load = min(factory_loads)
                lightest_factories = [f for f, load in enumerate(factory_loads) if load == min_load]
                selected_factory = random.choice(lightest_factories)
            else:
                selected_factory = random.randint(0, self.n_factories - 1)
            
            factory_assignment.append(selected_factory)
        
        # 构建作业序列
        job_sequences = [[] for _ in range(self.n_factories)]
        for job_id, factory_id in enumerate(factory_assignment):
            job_sequences[factory_id].append(job_id)
        
        # 随机打乱各工厂内的作业顺序
        for factory_id in range(self.n_factories):
            random.shuffle(job_sequences[factory_id])
        
        return Solution(factory_assignment, job_sequences)
    
    def elite_enhancement(self):
        """精英强化"""
        # 扩大精英组
        self._adjust_group_ratio('elite', 0.05)
        
        # 对精英组应用高强度局部搜索
        for idx in self.groups['elite']:
            if hasattr(self, 'population') and idx < len(self.population):
                self._apply_elite_optimization(idx)
        
        print("👑 执行策略：精英强化")
    
    def redistribute_resources(self):
        """资源重分配"""
        # 基于性能重新分配资源
        performance_scores = {}
        for group_name, perf in self.group_performance.items():
            score = 0.4 * perf.average_quality + 0.3 * perf.success_rate + 0.3 * perf.convergence_rate
            performance_scores[group_name] = score
        
        # 奖励表现好的组，减少表现差的组
        total_score = sum(performance_scores.values())
        if total_score > 0:
            for group_name, score in performance_scores.items():
                ratio_adjustment = (score / total_score - 0.25) * 0.1  # 期望值0.25，调整幅度10%
                self._adjust_group_ratio(group_name, ratio_adjustment)
        
        print("🔄 执行策略：资源重分配")
    
    def _adjust_group_ratio(self, group_name: str, adjustment: float, target_ratio: Optional[float] = None):
        """调整组比例"""
        if group_name not in self.groups:
            return
        
        current_size = len(self.groups[group_name])
        
        if target_ratio is not None:
            target_size = max(self.min_group_size, int(self.population_size * target_ratio))
        else:
            adjustment_size = max(-current_size + self.min_group_size, 
                                int(self.population_size * adjustment))
            target_size = max(self.min_group_size, current_size + adjustment_size)
        
        target_size = min(target_size, self.population_size - 3 * self.min_group_size)  # 确保其他组有空间
        
        if target_size != current_size:
            self._resize_group(group_name, target_size)
    
    def _resize_group(self, group_name: str, target_size: int):
        """调整组大小"""
        current_size = len(self.groups[group_name])
        
        if target_size > current_size:
            # 需要增加成员
            needed = target_size - current_size
            # 从其他组借调成员
            available_indices = []
            for other_group, indices in self.groups.items():
                if other_group != group_name and len(indices) > self.min_group_size:
                    available_indices.extend(indices[:len(indices) - self.min_group_size])
            
            random.shuffle(available_indices)
            to_transfer = available_indices[:needed]
            
            # 执行转移
            for idx in to_transfer:
                for other_group, indices in self.groups.items():
                    if idx in indices:
                        indices.remove(idx)
                        break
                self.groups[group_name].append(idx)
                
        elif target_size < current_size:
            # 需要减少成员
            to_remove = current_size - target_size
            random.shuffle(self.groups[group_name])
            removed_indices = self.groups[group_name][:to_remove]
            self.groups[group_name] = self.groups[group_name][to_remove:]
            
            # 将移除的成员分配到其他组
            other_groups = [g for g in self.groups.keys() if g != group_name]
            for idx in removed_indices:
                target_group = random.choice(other_groups)
                self.groups[target_group].append(idx)
    
    def _apply_chaotic_perturbation(self, eagle_idx: int, intensity: float = 0.5):
        """应用混沌扰动"""
        if not hasattr(self, 'population') or eagle_idx >= len(self.population):
            return
        
        solution = self.population[eagle_idx]
        
        # 获取混沌值
        chaos_values = self.chaos_maps.get_chaos_values(self.n_jobs)
        
        # 随机重分配一些作业的工厂
        n_perturbations = max(1, int(intensity * self.n_jobs * 0.3))
        jobs_to_perturb = random.sample(range(self.n_jobs), min(n_perturbations, self.n_jobs))
        
        for i, job_id in enumerate(jobs_to_perturb):
            if chaos_values[i % len(chaos_values)] > 0.7:
                new_factory = random.randint(0, self.n_factories - 1)
                old_factory = solution.factory_assignment[job_id]
                
                if new_factory != old_factory:
                    # 更新工厂分配
                    solution.factory_assignment[job_id] = new_factory
                    
                    # 更新作业序列
                    if job_id in solution.job_sequences[old_factory]:
                        solution.job_sequences[old_factory].remove(job_id)
                    solution.job_sequences[new_factory].append(job_id)
    
    def _apply_local_refinement(self, eagle_idx: int):
        """应用局部精炼"""
        if not hasattr(self, 'population') or eagle_idx >= len(self.population):
            return
        
        solution = self.population[eagle_idx]
        
        # 尝试作业交换优化
        for _ in range(3):
            factory_id = random.randint(0, self.n_factories - 1)
            jobs = solution.job_sequences[factory_id]
            
            if len(jobs) >= 2:
                i, j = random.sample(range(len(jobs)), 2)
                # 临时交换并评估
                jobs[i], jobs[j] = jobs[j], jobs[i]
    
    def _apply_balanced_search(self, eagle_idx: int):
        """应用平衡搜索"""
        if not hasattr(self, 'population') or eagle_idx >= len(self.population):
            return
        
        # 结合探索和开发的中等强度搜索
        if random.random() < 0.5:
            self._apply_chaotic_perturbation(eagle_idx, intensity=0.3)
        else:
            self._apply_local_refinement(eagle_idx)
    
    def _apply_elite_optimization(self, eagle_idx: int):
        """应用精英优化"""
        if not hasattr(self, 'population') or eagle_idx >= len(self.population):
            return
        
        # 对精英解应用多种局部搜索算子
        for _ in range(5):
            self._apply_local_refinement(eagle_idx)
    
    def get_group_statistics(self) -> Dict:
        """获取分组统计信息"""
        stats = {}
        
        for group_name, indices in self.groups.items():
            perf = self.group_performance[group_name]
            stats[group_name] = {
                'size': len(indices),
                'ratio': len(indices) / self.population_size,
                'average_quality': perf.average_quality,
                'diversity_score': perf.diversity_score,
                'convergence_rate': perf.convergence_rate,
                'success_rate': perf.success_rate
            }
        
        return stats 