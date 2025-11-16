#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved NSGA-II算法实现 - 适配MO-DHFSP问题
基于"Improved Crowding Distance for NSGA-II" (2018年) 论文
核心改进：拥挤距离计算公式优化

改进公式: dis^j = dis^j + (f_{n+1}^k - f_n^k) / (f_max^k - f_min^k)
替代原始: dis^j = dis^j + (f_{n+1}^k - f_{n-1}^k) / (f_max^k - f_min^k)
"""

import numpy as np
import random
import copy
import time
from typing import List, Dict, Tuple
from problem.mo_dhfsp import MO_DHFSP_Problem, Solution

class ImprovedNSGA2_Optimizer:
    """改进NSGA-II优化器 - 适配MO-DHFSP问题"""
    
    def __init__(self, problem: MO_DHFSP_Problem, **kwargs):
        """
        初始化改进NSGA-II优化器
        
        Args:
            problem: MO-DHFSP问题实例
            **kwargs: 其他参数
        """
        self.problem = problem
        self.n_jobs = problem.n_jobs
        self.n_factories = problem.n_factories
        
        # 算法参数
        self.population_size = kwargs.get('population_size', 50)
        self.max_generations = kwargs.get('max_generations', 50)
        self.crossover_prob = kwargs.get('crossover_prob', 0.9)
        self.mutation_prob = kwargs.get('mutation_prob', 0.1)
        
        # 状态跟踪
        self.current_generation = 0
        self.population = []
        self.convergence_data = []
        self.best_makespan_history = []
        self.best_tardiness_history = []
        
        print(f"初始化改进NSGA-II: 种群大小={self.population_size}, 最大代数={self.max_generations}")
        print(f"核心改进: 拥挤距离计算公式优化 (f_{{i+1}} - f_i) / (f_max - f_min)")
    
    def create_individual(self) -> Solution:
        """创建个体"""
        # 随机工厂分配
        factory_assignment = [random.randint(0, self.n_factories - 1) for _ in range(self.n_jobs)]
        
        # 使用问题实例的create_solution方法创建完整解
        solution = self.problem.create_solution(factory_assignment)
        
        return solution
    
    def initialize_population(self) -> List[Solution]:
        """初始化种群"""
        population = []
        
        print("初始化改进NSGA-II种群...")
        for i in range(self.population_size):
            individual = self.create_individual()
            population.append(individual)
        
        print(f"改进NSGA-II初始化完成，种群大小: {len(population)}")
        return population
    
    def dominates(self, a: Solution, b: Solution) -> bool:
        """判断解a是否支配解b（最小化问题）"""
        better_in_any = False
        for i in range(2):  # 两个目标：完工时间和总拖期
            obj_a = a.makespan if i == 0 else a.total_tardiness
            obj_b = b.makespan if i == 0 else b.total_tardiness
            
            if obj_a > obj_b:
                return False
            elif obj_a < obj_b:
                better_in_any = True
        
        return better_in_any
    
    def fast_non_dominated_sort(self, population: List[Solution]) -> List[List[Solution]]:
        """快速非支配排序"""
        fronts = [[]]
        
        # 初始化支配信息
        for individual in population:
            individual.domination_count = 0
            individual.dominated_solutions = []
            individual.rank = -1
        
        # 计算支配关系
        for i, p in enumerate(population):
            for j, q in enumerate(population):
                if i != j:
                    if self.dominates(p, q):
                        p.dominated_solutions.append(q)
                    elif self.dominates(q, p):
                        p.domination_count += 1
            
            if p.domination_count == 0:
                p.rank = 0
                fronts[0].append(p)
        
        # 构建后续前沿
        current_front = 0
        while current_front < len(fronts) and len(fronts[current_front]) > 0:
            next_front = []
            for p in fronts[current_front]:
                for q in p.dominated_solutions:
                    q.domination_count -= 1
                    if q.domination_count == 0:
                        q.rank = current_front + 1
                        next_front.append(q)
            if next_front:
                fronts.append(next_front)
            current_front += 1
        
        return fronts
    
    def calculate_improved_crowding_distance(self, front: List[Solution]):
        """
        计算改进的拥挤距离
        核心改进：f_{n+1}^k - f_n^k 替代 f_{n+1}^k - f_{n-1}^k
        """
        n = len(front)
        if n <= 2:
            for individual in front:
                individual.crowding_distance = float('inf')
            return
        
        # 初始化拥挤距离
        for individual in front:
            individual.crowding_distance = 0.0
        
        # 对每个目标函数计算拥挤距离
        objectives = ['makespan', 'total_tardiness']
        
        for obj_idx, obj_name in enumerate(objectives):
            # 按当前目标函数排序
            front.sort(key=lambda x: getattr(x, obj_name))
            
            # 边界点设为无穷大
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            
            # 计算目标函数范围
            obj_min = getattr(front[0], obj_name)
            obj_max = getattr(front[-1], obj_name)
            
            if obj_max - obj_min == 0:
                continue
            
            # 改进的拥挤距离计算
            for i in range(1, n - 1):
                if front[i].crowding_distance != float('inf'):
                    # 改进版本: (f_{i+1} - f_i) / (f_max - f_min)
                    distance = (getattr(front[i + 1], obj_name) - 
                               getattr(front[i], obj_name)) / (obj_max - obj_min)
                    front[i].crowding_distance += distance
    
    def tournament_selection(self, population: List[Solution]) -> Solution:
        """锦标赛选择"""
        candidate1 = random.choice(population)
        candidate2 = random.choice(population)
        
        # 比较帕累托等级
        if candidate1.rank < candidate2.rank:
            return candidate1
        elif candidate1.rank > candidate2.rank:
            return candidate2
        else:
            # 同等级比较拥挤距离
            if candidate1.crowding_distance > candidate2.crowding_distance:
                return candidate1
            else:
                return candidate2
    
    def crossover(self, parent1: Solution, parent2: Solution) -> Tuple[Solution, Solution]:
        """交叉操作 - 适配MO-DHFSP编码"""
        if random.random() > self.crossover_prob:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
        
        # 工厂分配交叉
        child1_assignment = parent1.factory_assignment.copy()
        child2_assignment = parent2.factory_assignment.copy()
        
        # 单点交叉
        crossover_point = random.randint(1, self.n_jobs - 1)
        
        # 交换交叉点后的基因
        child1_assignment[crossover_point:] = parent2.factory_assignment[crossover_point:]
        child2_assignment[crossover_point:] = parent1.factory_assignment[crossover_point:]
        
        # 创建新解
        child1 = self.problem.create_solution(child1_assignment)
        child2 = self.problem.create_solution(child2_assignment)
        
        return child1, child2
    
    def mutation(self, individual: Solution) -> Solution:
        """变异操作 - 适配MO-DHFSP编码"""
        if random.random() > self.mutation_prob:
            return copy.deepcopy(individual)
        
        # 复制个体
        mutated_assignment = individual.factory_assignment.copy()
        
        # 随机选择变异点
        mutation_point = random.randint(0, self.n_jobs - 1)
        
        # 随机分配新工厂
        new_factory = random.randint(0, self.n_factories - 1)
        mutated_assignment[mutation_point] = new_factory
        
        # 创建变异后的解
        mutated_individual = self.problem.create_solution(mutated_assignment)
        
        return mutated_individual
    
    def environmental_selection(self, combined_population: List[Solution]) -> List[Solution]:
        """环境选择 - 使用改进的拥挤距离"""
        fronts = self.fast_non_dominated_sort(combined_population)
        new_population = []
        
        # 添加完整的前沿
        for front in fronts:
            if len(new_population) + len(front) <= self.population_size:
                # 计算改进的拥挤距离
                self.calculate_improved_crowding_distance(front)
                new_population.extend(front)
            else:
                # 最后一个前沿需要部分选择
                remaining_slots = self.population_size - len(new_population)
                if remaining_slots > 0:
                    self.calculate_improved_crowding_distance(front)
                    # 按改进的拥挤距离降序排序
                    front.sort(key=lambda x: x.crowding_distance, reverse=True)
                    new_population.extend(front[:remaining_slots])
                break
        
        return new_population
    
    def optimize(self) -> Tuple[List[Solution], List[Dict]]:
        """优化主循环"""
        print("🚀 开始改进NSGA-II优化...")
        
        # 初始化种群
        self.population = self.initialize_population()
        
        # 初始化支配信息
        for individual in self.population:
            individual.domination_count = 0
            individual.dominated_solutions = []
            individual.rank = -1
            individual.crowding_distance = 0.0
        
        # 进化循环
        for generation in range(self.max_generations):
            self.current_generation = generation
            
            # 生成子代
            offspring = []
            for _ in range(self.population_size // 2):
                parent1 = self.tournament_selection(self.population)
                parent2 = self.tournament_selection(self.population)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                offspring.extend([child1, child2])
            
            # 合并父代和子代
            combined_population = self.population + offspring
            
            # 环境选择
            self.population = self.environmental_selection(combined_population)
            
            # 记录收敛数据
            fronts = self.fast_non_dominated_sort(self.population)
            if fronts[0]:
                best_makespan = min(sol.makespan for sol in fronts[0])
                best_tardiness = min(sol.total_tardiness for sol in fronts[0])
                self.best_makespan_history.append(best_makespan)
                self.best_tardiness_history.append(best_tardiness)
                
                convergence_info = {
                    'generation': generation,
                    'best_makespan': best_makespan,
                    'best_tardiness': best_tardiness,
                    'pareto_size': len(fronts[0])
                }
                self.convergence_data.append(convergence_info)
            
            # 打印进度
            if generation % 10 == 0:
                pareto_size = len(fronts[0]) if fronts else 0
                print(f"第 {generation} 代: 种群={len(self.population)}, 帕累托解={pareto_size}")
        
        print("✅ 改进NSGA-II优化完成!")
        
        # 返回帕累托前沿
        final_fronts = self.fast_non_dominated_sort(self.population)
        pareto_solutions = final_fronts[0] if final_fronts else []
        
        print(f"🎯 最终帕累托前沿解数量: {len(pareto_solutions)}")
        
        return pareto_solutions, self.convergence_data 