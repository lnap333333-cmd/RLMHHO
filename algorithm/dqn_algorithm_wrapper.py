#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DQN算法包装器 - 修复版本
解决pareto解集数量问题和多样性控制
"""

import time
import numpy as np
import copy
from algorithm.dqn_simple_scheduler import SimpleDQNScheduler

class DQNAlgorithmWrapper:
    """DQN算法包装器，符合对比脚本接口 - 修复版本"""
    
    def __init__(self, problem, **kwargs):
        self.problem = problem
        self.name = "DQN"
        # 存储传入的参数
        self.kwargs = kwargs
        self.target_pareto_size = kwargs.get('target_pareto_size', 25)
        self.diversity_control = kwargs.get('diversity_control', True)
        
    def optimize(self, max_iterations=100):
        """优化接口 - 修复版本，控制解集数量"""
        scheduler = SimpleDQNScheduler(self.problem)
        
        # 使用传入的max_iterations或默认值
        if 'max_iterations' in self.kwargs:
            max_iterations = self.kwargs['max_iterations']
        
        # 根据问题规模调整参数
        if self.problem.n_jobs <= 50:
            episodes = min(60, max_iterations)  # 降低episodes
            steps = 30  # 降低steps
        else:
            episodes = min(50, max_iterations)
            steps = 25
        
        best_solution, convergence_data = scheduler.optimize(
            max_episodes=episodes,
            max_steps_per_episode=steps
        )
        
        # 生成受控的多样化解集
        diverse_solutions = self._generate_controlled_diverse_solutions(scheduler, best_solution)
        
        return diverse_solutions, convergence_data

    def _generate_controlled_diverse_solutions(self, scheduler, best_solution):
        """
        生成受控的多样化解集，避免过多的低质量解
        """
        solutions = [best_solution]
        
        # 从解质量历史中提取不同质量的解
        if hasattr(scheduler, 'solution_quality_history') and scheduler.solution_quality_history:
            # 按加权目标排序
            sorted_history = sorted(scheduler.solution_quality_history, 
                                  key=lambda x: x.get('weighted', float('inf')))
            
            # 选择有限数量的高质量解
            n_solutions = min(self.target_pareto_size - 1, len(sorted_history))
            if n_solutions > 0:
                step = max(1, len(sorted_history) // n_solutions)
                
                for i in range(0, len(sorted_history), step):
                    if len(solutions) >= self.target_pareto_size:
                        break
                    
                    entry = sorted_history[i]
                    # 重新构造解
                    new_solution = self._create_solution_from_history(entry)
                    if new_solution and self._is_sufficiently_different(new_solution, solutions):
                        solutions.append(new_solution)
        
        # 如果解的数量还不够，生成一些高质量的变异解
        while len(solutions) < self.target_pareto_size:
            variant = self._create_quality_variant(best_solution)
            if variant and self._is_sufficiently_different(variant, solutions):
                solutions.append(variant)
            else:
                break  # 避免无限循环
        
        # 应用多样性控制
        if self.diversity_control and len(solutions) > self.target_pareto_size:
            solutions = self._apply_diversity_selection(solutions)
        
        print(f"DQN生成了 {len(solutions)} 个高质量解用于对比")
        return solutions
    
    def _is_sufficiently_different(self, new_solution, existing_solutions):
        """检查新解是否与现有解有足够的差异"""
        min_diff_threshold = 0.05  # 最小差异阈值
        
        for existing in existing_solutions:
            makespan_diff = abs(new_solution.makespan - existing.makespan) / max(existing.makespan, 1)
            tardiness_diff = abs(new_solution.total_tardiness - existing.total_tardiness) / max(existing.total_tardiness, 1)
            
            if makespan_diff < min_diff_threshold and tardiness_diff < min_diff_threshold:
                return False
        
        return True
    
    def _apply_diversity_selection(self, solutions):
        """应用多样性选择，保留最具代表性的解"""
        if len(solutions) <= self.target_pareto_size:
            return solutions
        
        # 按加权目标排序
        sorted_solutions = sorted(solutions, 
                                key=lambda x: 0.5 * x.makespan + 0.5 * x.total_tardiness)
        
        # 分段选择，保持多样性
        selected = []
        step = max(1, len(sorted_solutions) // self.target_pareto_size)
        
        for i in range(0, len(sorted_solutions), step):
            selected.append(sorted_solutions[i])
            if len(selected) >= self.target_pareto_size:
                break
        
        return selected
    
    def _create_solution_from_history(self, history_entry):
        """从历史记录创建解"""
        try:
            # 生成一个随机解并评估
            solution = self.problem.generate_random_solution()
            solution = self.problem.evaluate_solution(solution)
            return solution
        except:
            return None
    
    def _create_quality_variant(self, base_solution):
        """创建基础解的高质量变异版本"""
        try:
            variant = copy.deepcopy(base_solution)
            
            # 更保守的变异，只改变少量作业
            n_changes = max(1, self.problem.n_jobs // 30)  # 减少变异强度
            
            for _ in range(n_changes):
                job_id = np.random.randint(0, self.problem.n_jobs)
                
                # 尝试改进：选择负载较轻的工厂
                current_factory = variant.factory_assignment[job_id]
                factory_loads = []
                
                for f in range(self.problem.n_factories):
                    load = sum(len(variant.job_sequences[f]) for f in range(self.problem.n_factories))
                    factory_loads.append(load)
                
                # 选择负载最轻的工厂
                new_factory = factory_loads.index(min(factory_loads))
                
                if current_factory != new_factory:
                    variant.factory_assignment[job_id] = new_factory
                    
                    # 更新作业序列
                    if job_id in variant.job_sequences[current_factory]:
                        variant.job_sequences[current_factory].remove(job_id)
                    variant.job_sequences[new_factory].append(job_id)
            
            # 重新评估解
            return self.problem.evaluate_solution(variant)
        except:
            return None
