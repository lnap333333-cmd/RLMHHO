#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MOEA/D算法实现
Multi-Objective Evolutionary Algorithm based on Decomposition
适配多目标分布式异构混合流水车间调度问题
"""

import numpy as np
import random
import copy
from typing import List, Dict, Tuple
from problem.mo_dhfsp import MO_DHFSP_Problem, Solution

class MOEAD_Optimizer:
    """MOEA/D优化器"""
    
    def __init__(self, problem: MO_DHFSP_Problem, **kwargs):
        """
        初始化MOEA/D优化器
        
        Args:
            problem: 问题实例
            **kwargs: 其他参数
        """
        self.problem = problem
        self.n_jobs = problem.n_jobs
        
        # 算法参数
        self.population_size = kwargs.get('population_size', 100)
        self.max_generations = kwargs.get('max_generations', 100)
        self.crossover_prob = kwargs.get('crossover_prob', 0.9)
        self.mutation_prob = kwargs.get('mutation_prob', 0.1)
        self.neighbor_size = kwargs.get('neighbor_size', 10)  # 邻域大小
        self.delta = kwargs.get('delta', 0.9)  # 从邻域选择父代的概率
        self.nr = kwargs.get('nr', 2)  # 最大替换数量
        
        # 状态跟踪
        self.current_generation = 0
        self.population = []
        self.weight_vectors = []
        self.neighbors = []
        self.ideal_point = None
        self.convergence_data = []
        self.best_makespan_history = []
        self.best_tardiness_history = []
        
        # 初始化权重向量和邻域
        self._initialize_weights()
        self._initialize_neighborhoods()
        
        print(f"初始化MOEA/D: 种群大小={self.population_size}, 最大代数={self.max_generations}")
        print(f"邻域大小={self.neighbor_size}, 权重向量数={len(self.weight_vectors)}")
    
    def _initialize_weights(self):
        """初始化权重向量"""
        # 对于双目标问题，使用均匀分布的权重向量
        self.weight_vectors = []
        
        for i in range(self.population_size):
            w1 = i / (self.population_size - 1)
            w2 = 1.0 - w1
            self.weight_vectors.append([w1, w2])
        
        # 归一化权重向量
        for i in range(len(self.weight_vectors)):
            norm = np.linalg.norm(self.weight_vectors[i])
            if norm > 0:
                self.weight_vectors[i] = [w / norm for w in self.weight_vectors[i]]
    
    def _initialize_neighborhoods(self):
        """初始化邻域结构"""
        self.neighbors = []
        
        for i in range(self.population_size):
            # 计算与其他权重向量的距离
            distances = []
            for j in range(self.population_size):
                if i != j:
                    dist = np.linalg.norm(np.array(self.weight_vectors[i]) - np.array(self.weight_vectors[j]))
                    distances.append((j, dist))
            
            # 选择最近的邻居
            distances.sort(key=lambda x: x[1])
            neighbors = [x[0] for x in distances[:self.neighbor_size]]
            self.neighbors.append(neighbors)
    
    def optimize(self) -> Tuple[List[Solution], Dict]:
        """
        主优化流程
        
        Returns:
            (pareto_solutions, convergence_data): 帕累托最优解集和收敛数据
        """
        print("开始MOEA/D优化...")
        
        # 初始化种群
        self._initialize_population()
        
        # 初始化理想点
        self._initialize_ideal_point()
        
        # 主循环
        for generation in range(self.max_generations):
            self.current_generation = generation
            
            # 对每个子问题进行优化
            for i in range(self.population_size):
                self._optimize_subproblem(i)
            
            # 更新理想点
            self._update_ideal_point()
            
            # 记录收敛数据
            self._record_convergence_data()
            
            # 输出进度
            if generation % 10 == 0 or generation == self.max_generations - 1:
                self._print_progress(generation)
        
        # 提取帕累托前沿
        pareto_solutions = self._extract_pareto_front()
        
        print("MOEA/D优化完成!")
        return pareto_solutions, self._prepare_convergence_data()
    
    def _initialize_population(self):
        """初始化种群"""
        print("初始化MOEA/D种群...")
        
        self.population = []
        for _ in range(self.population_size):
            solution = self.problem.generate_random_solution()
            self.population.append(solution)
        
        print(f"MOEA/D初始化完成，种群大小: {len(self.population)}")
    
    def _initialize_ideal_point(self):
        """初始化理想点"""
        self.ideal_point = [float('inf'), float('inf')]
        
        for solution in self.population:
            if solution.makespan < self.ideal_point[0]:
                self.ideal_point[0] = solution.makespan
            if solution.total_tardiness < self.ideal_point[1]:
                self.ideal_point[1] = solution.total_tardiness
    
    def _update_ideal_point(self):
        """更新理想点"""
        for solution in self.population:
            if solution.makespan < self.ideal_point[0]:
                self.ideal_point[0] = solution.makespan
            if solution.total_tardiness < self.ideal_point[1]:
                self.ideal_point[1] = solution.total_tardiness
    
    def _optimize_subproblem(self, index: int):
        """优化单个子问题"""
        # 选择父代
        if random.random() < self.delta:
            # 从邻域中选择父代
            parent_indices = random.sample(self.neighbors[index], min(2, len(self.neighbors[index])))
        else:
            # 从整个种群中选择父代
            parent_indices = random.sample(range(self.population_size), 2)
        
        if len(parent_indices) < 2:
            parent_indices = random.sample(range(self.population_size), 2)
        
        parent1 = self.population[parent_indices[0]]
        parent2 = self.population[parent_indices[1]]
        
        # 交叉操作
        if random.random() < self.crossover_prob:
            child = self._crossover(parent1, parent2)
        else:
            child = copy.deepcopy(parent1)
        
        # 变异操作
        if random.random() < self.mutation_prob:
            child = self._mutation(child)
        
        # 评估子代
        child = self.problem.evaluate_solution(child)
        
        # 更新邻域解
        self._update_neighborhood(child, index)
    
    def _crossover(self, parent1: Solution, parent2: Solution) -> Solution:
        """交叉操作 - 针对分布式异构混合流水车间调度问题"""
        child = copy.deepcopy(parent1)
        
        # 工厂分配交叉
        if random.random() < 0.5:
            # 单点交叉工厂分配
            crossover_point = random.randint(1, self.n_jobs - 1)
            child.factory_assignment[crossover_point:] = parent2.factory_assignment[crossover_point:]
        
        # 重新构建作业序列
        child.job_sequences = [[] for _ in range(self.problem.n_factories)]
        for job_id in range(self.n_jobs):
            factory_id = child.factory_assignment[job_id]
            child.job_sequences[factory_id].append(job_id)
        
        # 序列交叉 - 对每个工厂的作业序列进行交叉
        for factory_id in range(self.problem.n_factories):
            if (len(child.job_sequences[factory_id]) > 1 and 
                len(parent2.job_sequences[factory_id]) > 1):
                
                # 使用顺序交叉
                child_seq = child.job_sequences[factory_id]
                parent2_seq = parent2.job_sequences[factory_id]
                
                # 找到公共作业
                common_jobs = list(set(child_seq) & set(parent2_seq))
                if len(common_jobs) > 1:
                    # 对公共作业进行顺序交叉
                    new_seq = self._order_crossover_sequence(child_seq, parent2_seq, common_jobs)
                    child.job_sequences[factory_id] = new_seq
        
        return self._validate_solution(child)
    
    def _order_crossover_sequence(self, seq1: List[int], seq2: List[int], common_jobs: List[int]) -> List[int]:
        """对序列中的公共作业进行顺序交叉"""
        if len(common_jobs) <= 1:
            return seq1
        
        # 提取公共作业在两个序列中的顺序
        order1 = [job for job in seq1 if job in common_jobs]
        order2 = [job for job in seq2 if job in common_jobs]
        
        if len(order1) <= 1:
            return seq1
        
        # 选择交叉点
        start = random.randint(0, len(order1) - 1)
        end = random.randint(start, len(order1) - 1)
        
        # 从order1中选择片段
        selected = order1[start:end+1]
        
        # 从order2中按顺序填充剩余位置
        remaining = [job for job in order2 if job not in selected]
        
        # 构造新的公共作业顺序
        new_order = [-1] * len(order1)
        new_order[start:end+1] = selected
        
        # 填充剩余位置
        remaining_idx = 0
        for i in range(len(new_order)):
            if new_order[i] == -1 and remaining_idx < len(remaining):
                new_order[i] = remaining[remaining_idx]
                remaining_idx += 1
        
        # 重新构建完整序列
        new_seq = []
        common_idx = 0
        for job in seq1:
            if job in common_jobs:
                if common_idx < len(new_order):
                    new_seq.append(new_order[common_idx])
                    common_idx += 1
                else:
                    new_seq.append(job)
            else:
                new_seq.append(job)
        
        return new_seq
    
    def _mutation(self, solution: Solution) -> Solution:
        """变异操作 - 针对分布式异构混合流水车间调度问题"""
        mutated = copy.deepcopy(solution)
        
        mutation_type = random.random()
        
        if mutation_type < 0.4:
            # 工厂重分配变异
            job_id = random.randint(0, self.n_jobs - 1)
            old_factory = mutated.factory_assignment[job_id]
            new_factory = random.randint(0, self.problem.n_factories - 1)
            
            if old_factory != new_factory:
                # 安全地移动作业
                if job_id in mutated.job_sequences[old_factory]:
                    mutated.job_sequences[old_factory].remove(job_id)
                mutated.job_sequences[new_factory].append(job_id)
                mutated.factory_assignment[job_id] = new_factory
        
        elif mutation_type < 0.7:
            # 作业交换变异
            factory_id = random.randint(0, self.problem.n_factories - 1)
            jobs = mutated.job_sequences[factory_id]
            if len(jobs) >= 2:
                i, j = random.sample(range(len(jobs)), 2)
                jobs[i], jobs[j] = jobs[j], jobs[i]
        
        else:
            # 作业插入变异
            factory_id = random.randint(0, self.problem.n_factories - 1)
            jobs = mutated.job_sequences[factory_id]
            if len(jobs) >= 2:
                i = random.randint(0, len(jobs) - 1)
                job = jobs.pop(i)
                j = random.randint(0, len(jobs))
                jobs.insert(j, job)
        
        return self._validate_solution(mutated)
    
    def _validate_solution(self, solution: Solution) -> Solution:
        """验证并修复解的一致性"""
        # 重新构建作业序列以确保一致性
        new_job_sequences = [[] for _ in range(self.problem.n_factories)]
        
        for job_id in range(self.n_jobs):
            factory_id = solution.factory_assignment[job_id]
            new_job_sequences[factory_id].append(job_id)
        
        solution.job_sequences = new_job_sequences
        return solution
    
    def _update_neighborhood(self, child: Solution, index: int):
        """更新邻域解"""
        # 确定更新范围
        if random.random() < self.delta:
            update_indices = self.neighbors[index]
        else:
            update_indices = list(range(self.population_size))
        
        # 限制更新数量
        update_count = 0
        random.shuffle(update_indices)
        
        for i in update_indices:
            if update_count >= self.nr:
                break
            
            # 计算聚合函数值
            child_value = self._tchebycheff(child, i)
            current_value = self._tchebycheff(self.population[i], i)
            
            # 如果子代更好，则替换
            if child_value < current_value:
                self.population[i] = copy.deepcopy(child)
                update_count += 1
    
    def _tchebycheff(self, solution: Solution, weight_index: int) -> float:
        """Tchebycheff聚合函数"""
        weight = self.weight_vectors[weight_index]
        
        # 计算目标函数值与理想点的差值
        diff1 = abs(solution.makespan - self.ideal_point[0])
        diff2 = abs(solution.total_tardiness - self.ideal_point[1])
        
        # 加权Tchebycheff函数
        if weight[0] > 0 and weight[1] > 0:
            return max(weight[0] * diff1, weight[1] * diff2)
        elif weight[0] > 0:
            return weight[0] * diff1
        elif weight[1] > 0:
            return weight[1] * diff2
        else:
            return diff1 + diff2
    
    def _extract_pareto_front(self) -> List[Solution]:
        """提取帕累托前沿"""
        pareto_front = []
        
        for solution in self.population:
            is_dominated = False
            
            for other in self.population:
                if self._dominates(other, solution):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_front.append(copy.deepcopy(solution))
        
        return pareto_front
    
    def _dominates(self, sol1: Solution, sol2: Solution) -> bool:
        """判断sol1是否支配sol2"""
        return (sol1.makespan <= sol2.makespan and sol1.total_tardiness <= sol2.total_tardiness and
                (sol1.makespan < sol2.makespan or sol1.total_tardiness < sol2.total_tardiness))
    
    def _record_convergence_data(self):
        """记录收敛数据"""
        pareto_front = self._extract_pareto_front()
        
        if pareto_front:
            best_makespan = min(sol.makespan for sol in pareto_front)
            best_tardiness = min(sol.total_tardiness for sol in pareto_front)
        else:
            best_makespan = float('inf')
            best_tardiness = float('inf')
        
        self.best_makespan_history.append(best_makespan)
        self.best_tardiness_history.append(best_tardiness)
        
        self.convergence_data.append({
            'generation': self.current_generation,
            'best_makespan': best_makespan,
            'best_tardiness': best_tardiness,
            'pareto_size': len(pareto_front),
            'ideal_point': copy.deepcopy(self.ideal_point)
        })
    
    def _print_progress(self, generation: int):
        """打印进度信息"""
        pareto_front = self._extract_pareto_front()
        
        if pareto_front:
            best_makespan = min(sol.makespan for sol in pareto_front)
            best_tardiness = min(sol.total_tardiness for sol in pareto_front)
            print(f"MOEA/D 代数 {generation:3d}: 帕累托解={len(pareto_front):2d}, "
                  f"最优完工时间={best_makespan:.2f}, "
                  f"最优总拖期={best_tardiness:.2f}, "
                  f"理想点=({self.ideal_point[0]:.2f}, {self.ideal_point[1]:.2f})")
        else:
            print(f"MOEA/D 代数 {generation:3d}: 无帕累托解")
    
    def _prepare_convergence_data(self) -> Dict:
        """准备收敛数据"""
        return {
            'generations': list(range(len(self.best_makespan_history))),
            'best_makespan': self.best_makespan_history,
            'best_tardiness': self.best_tardiness_history,
            'convergence_data': self.convergence_data
        } 