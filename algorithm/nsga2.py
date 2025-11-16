#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NSGA-II算法实现
适配多目标分布式混合流水车间调度问题
Non-dominated Sorting Genetic Algorithm II
"""

import numpy as np
import random
import copy
from typing import List, Dict, Tuple
from problem.mo_dhfsp import MO_DHFSP_Problem, Solution

class NSGA2_Optimizer:
    """NSGA-II优化器"""
    
    def __init__(self, problem: MO_DHFSP_Problem, **kwargs):
        """
        初始化NSGA-II优化器
        
        Args:
            problem: 问题实例
            **kwargs: 其他参数
        """
        self.problem = problem
        self.n_jobs = problem.n_jobs
        
        # 算法参数 - 增加种群规模和迭代次数
        self.population_size = kwargs.get('population_size', 100)
        self.max_generations = kwargs.get('max_generations', 100)
        self.crossover_prob = kwargs.get('crossover_prob', 0.9)
        self.mutation_prob = kwargs.get('mutation_prob', 0.1)
        
        # 状态跟踪
        self.current_generation = 0
        self.population = []
        self.convergence_data = []
        self.best_makespan_history = []
        self.best_tardiness_history = []
        
        print(f"初始化NSGA-II: 种群大小={self.population_size}, 最大代数={self.max_generations}")
    
    def optimize(self) -> Tuple[List[Solution], Dict]:
        """
        主优化流程
        
        Returns:
            (pareto_solutions, convergence_data): 帕累托最优解集和收敛数据
        """
        print("开始NSGA-II优化...")
        
        # 初始化种群
        self._initialize_population()
        
        # 主循环
        for generation in range(self.max_generations):
            self.current_generation = generation
            
            # 生成子代种群
            offspring = self._generate_offspring()
            
            # 合并父代和子代
            combined_population = self.population + offspring
            
            # 环境选择
            self.population = self._environmental_selection(combined_population)
            
            # 记录收敛数据
            self._record_convergence_data()
            
            # 输出进度
            if generation % 10 == 0 or generation == self.max_generations - 1:
                self._print_progress(generation)
        
        # 提取帕累托前沿
        pareto_solutions = self._extract_pareto_front()
        
        print("NSGA-II优化完成!")
        return pareto_solutions, self._prepare_convergence_data()
    
    def _initialize_population(self):
        """初始化种群"""
        print("初始化NSGA-II种群...")
        
        self.population = []
        for _ in range(self.population_size):
            solution = self.problem.generate_random_solution()
            self.population.append(solution)
        
        # 计算适应度信息
        self._calculate_fitness()
        
        print(f"NSGA-II初始化完成，种群大小: {len(self.population)}")
    
    def _generate_offspring(self) -> List[Solution]:
        """生成子代种群"""
        offspring = []
        
        while len(offspring) < self.population_size:
            # 锦标赛选择两个父代
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # 交叉操作
            if random.random() < self.crossover_prob:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
            
            # 验证交叉后的解
            child1 = self._validate_solution(child1)
            child2 = self._validate_solution(child2)
            
            # 变异操作
            if random.random() < self.mutation_prob:
                child1 = self._mutation(child1)
            if random.random() < self.mutation_prob:
                child2 = self._mutation(child2)
            
            # 评估子代
            child1 = self.problem.evaluate_solution(child1)
            child2 = self.problem.evaluate_solution(child2)
            
            offspring.extend([child1, child2])
        
        return offspring[:self.population_size]
    
    def _tournament_selection(self, tournament_size: int = 2) -> Solution:
        """锦标赛选择"""
        tournament = random.sample(self.population, tournament_size)
        
        # 比较支配关系和拥挤距离
        best = tournament[0]
        for individual in tournament[1:]:
            if self._is_better_individual(individual, best):
                best = individual
        
        return best
    
    def _is_better_individual(self, ind1: Solution, ind2: Solution) -> bool:
        """判断个体1是否比个体2更好"""
        # 首先比较支配等级
        if hasattr(ind1, 'rank') and hasattr(ind2, 'rank'):
            if ind1.rank < ind2.rank:
                return True
            elif ind1.rank > ind2.rank:
                return False
        
        # 支配等级相同时比较拥挤距离
        if hasattr(ind1, 'crowding_distance') and hasattr(ind2, 'crowding_distance'):
            return ind1.crowding_distance > ind2.crowding_distance
        
        # 默认比较目标函数值
        return (ind1.makespan + ind1.total_tardiness) < (ind2.makespan + ind2.total_tardiness)
    
    def _crossover(self, parent1: Solution, parent2: Solution) -> Tuple[Solution, Solution]:
        """交叉操作 - 采用基于工厂分配和序列的两点交叉"""
        child1 = self._safe_copy_solution(parent1)
        child2 = self._safe_copy_solution(parent2)
        
        # 工厂分配交叉
        if random.random() < 0.5:
            # 单点交叉工厂分配
            crossover_point = random.randint(1, self.n_jobs - 1)
            
            # 交换交叉点后的工厂分配
            child1.factory_assignment[crossover_point:] = parent2.factory_assignment[crossover_point:]
            child2.factory_assignment[crossover_point:] = parent1.factory_assignment[crossover_point:]
        
        # 重新构建作业序列
        child1.job_sequences = [[] for _ in range(self.problem.n_factories)]
        child2.job_sequences = [[] for _ in range(self.problem.n_factories)]
        
        for job_id in range(self.n_jobs):
            factory1 = child1.factory_assignment[job_id]
            factory2 = child2.factory_assignment[job_id]
            child1.job_sequences[factory1].append(job_id)
            child2.job_sequences[factory2].append(job_id)
        
        # 改进的序列交叉 - 使用顺序交叉
        for factory_id in range(self.problem.n_factories):
            if len(child1.job_sequences[factory_id]) > 1 and len(child2.job_sequences[factory_id]) > 1:
                seq1, seq2 = self._order_crossover(child1.job_sequences[factory_id], 
                                                   child2.job_sequences[factory_id])
                child1.job_sequences[factory_id] = seq1
                child2.job_sequences[factory_id] = seq2
        
        return child1, child2
    
    def _order_crossover(self, seq1: List[int], seq2: List[int]) -> Tuple[List[int], List[int]]:
        """顺序交叉（OX）"""
        if len(seq1) != len(seq2) or len(seq1) <= 1:
            return seq1, seq2
        
        size = len(seq1)
        start, end = sorted(random.sample(range(size), 2))
        
        # 创建子代
        child1 = [-1] * size
        child2 = [-1] * size
        
        # 复制交叉段
        child1[start:end] = seq1[start:end]
        child2[start:end] = seq2[start:end]
        
        # 填充剩余位置
        self._fill_remaining_ox(child1, seq2, start, end)
        self._fill_remaining_ox(child2, seq1, start, end)
        
        return child1, child2
    
    def _fill_remaining_ox(self, child: List[int], parent: List[int], start: int, end: int):
        """顺序交叉的填充操作"""
        # 获取已经选择的元素
        selected = set(child[start:end])
        
        # 获取需要填充的元素（按parent中的顺序）
        remaining_elements = [x for x in parent if x not in selected]
        
        # 填充前半部分
        fill_idx = 0
        for i in range(start):
            if fill_idx < len(remaining_elements):
                child[i] = remaining_elements[fill_idx]
                fill_idx += 1
        
        # 填充后半部分
        for i in range(end, len(child)):
            if fill_idx < len(remaining_elements):
                child[i] = remaining_elements[fill_idx]
                fill_idx += 1
    
    def _mutation(self, solution: Solution) -> Solution:
        """变异操作"""
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
        
        # 验证解的一致性
        mutated = self._validate_solution(mutated)
        return mutated
    
    def _validate_solution(self, solution: Solution) -> Solution:
        """验证并修复解的一致性"""
        # 简化版本：直接重新构建作业序列
        new_job_sequences = [[] for _ in range(self.problem.n_factories)]
        
        for job_id in range(self.n_jobs):
            factory_id = solution.factory_assignment[job_id]
            new_job_sequences[factory_id].append(job_id)
        
        solution.job_sequences = new_job_sequences
        return solution
    
    def _environmental_selection(self, combined_population: List[Solution]) -> List[Solution]:
        """标准NSGA-II环境选择 - 基于非支配排序和拥挤距离"""
        # 快速非支配排序
        fronts = self._fast_non_dominated_sort(combined_population)
        
        # 计算拥挤距离
        for front in fronts:
            self._calculate_crowding_distance(front)
        
        # 选择新种群
        new_population = []
        front_idx = 0
        
        # 按前沿逐层填充
        while front_idx < len(fronts) and len(new_population) + len(fronts[front_idx]) <= self.population_size:
            new_population.extend(fronts[front_idx])
            front_idx += 1
        
        # 如果还需要填充，从下一个前沿中选择拥挤距离大的个体
        if len(new_population) < self.population_size and front_idx < len(fronts):
            remaining_slots = self.population_size - len(new_population)
            current_front = fronts[front_idx]
            
            # 按拥挤距离降序排序
            current_front.sort(key=lambda x: getattr(x, 'crowding_distance', 0), reverse=True)
            new_population.extend(current_front[:remaining_slots])
        
        return new_population
    
    def _fast_non_dominated_sort(self, population: List[Solution]) -> List[List[Solution]]:
        """快速非支配排序"""
        fronts = [[]]
        
        for individual in population:
            individual.domination_count = 0
            individual.dominated_solutions = []
            
            for other in population:
                if self._dominates(individual, other):
                    individual.dominated_solutions.append(other)
                elif self._dominates(other, individual):
                    individual.domination_count += 1
            
            if individual.domination_count == 0:
                individual.rank = 0
                fronts[0].append(individual)
        
        current_front = 0
        max_fronts = len(population)  # 防止无限循环
        
        while (current_front < len(fronts) and 
               len(fronts[current_front]) > 0 and 
               current_front < max_fronts):
            next_front = []
            for individual in fronts[current_front]:
                for dominated in individual.dominated_solutions:
                    dominated.domination_count -= 1
                    if dominated.domination_count == 0:
                        dominated.rank = current_front + 1
                        next_front.append(dominated)
            
            current_front += 1
            fronts.append(next_front)
            
            # 安全检查
            if current_front > max_fronts:
                break
        
        return fronts[:-1]  # 移除最后一个空前沿
    
    def _dominates(self, sol1: Solution, sol2: Solution) -> bool:
        """判断sol1是否支配sol2"""
        return (sol1.makespan <= sol2.makespan and sol1.total_tardiness <= sol2.total_tardiness and
                (sol1.makespan < sol2.makespan or sol1.total_tardiness < sol2.total_tardiness))
    
    def _calculate_crowding_distance(self, front: List[Solution]):
        """计算拥挤距离"""
        if len(front) <= 2:
            for individual in front:
                individual.crowding_distance = float('inf')
            return
        
        for individual in front:
            individual.crowding_distance = 0
        
        # 对每个目标函数计算拥挤距离
        objectives = ['makespan', 'total_tardiness']
        
        for obj in objectives:
            # 按目标函数值排序
            front.sort(key=lambda x: getattr(x, obj))
            
            # 边界个体设为无穷大
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            
            # 计算中间个体的拥挤距离
            obj_range = getattr(front[-1], obj) - getattr(front[0], obj)
            if obj_range == 0:
                continue
            
            for i in range(1, len(front) - 1):
                distance = (getattr(front[i + 1], obj) - getattr(front[i - 1], obj)) / obj_range
                front[i].crowding_distance += distance
    
    def _calculate_fitness(self):
        """计算种群的适应度信息"""
        fronts = self._fast_non_dominated_sort(self.population)
        
        for front in fronts:
            self._calculate_crowding_distance(front)
    
    def _extract_pareto_front(self) -> List[Solution]:
        """提取帕累托前沿"""
        fronts = self._fast_non_dominated_sort(self.population)
        return fronts[0] if fronts else []
    
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
            'pareto_size': len(pareto_front)
        })
    
    def _print_progress(self, generation: int):
        """打印进度信息"""
        pareto_front = self._extract_pareto_front()
        
        if pareto_front:
            best_makespan = min(sol.makespan for sol in pareto_front)
            best_tardiness = min(sol.total_tardiness for sol in pareto_front)
            print(f"NSGA-II 代数 {generation:3d}: 帕累托解={len(pareto_front):2d}, "
                  f"最优完工时间={best_makespan:.2f}, "
                  f"最优拖期={best_tardiness:.2f}")
        else:
            print(f"NSGA-II 代数 {generation:3d}: 还未找到帕累托解")
    
    def _prepare_convergence_data(self) -> Dict:
        """准备收敛数据"""
        return {
            'makespan_history': self.best_makespan_history,
            'tardiness_history': self.best_tardiness_history,
            'detailed_data': self.convergence_data,
            'final_pareto_size': len(self._extract_pareto_front()),
            'total_generations': self.current_generation + 1
        }

    def _safe_copy_solution(self, solution: Solution) -> Solution:
        """安全地复制解，避免deepcopy问题"""
        new_solution = Solution(
            factory_assignment=solution.factory_assignment.copy(),
            job_sequences=[seq.copy() for seq in solution.job_sequences],
            makespan=solution.makespan,
            total_tardiness=solution.total_tardiness,
            completion_times=solution.completion_times.copy() if solution.completion_times else None,
            factory_makespans=solution.factory_makespans.copy() if solution.factory_makespans else None
        )
        
        # 复制可能存在的其他属性
        if hasattr(solution, 'rank'):
            new_solution.rank = solution.rank
        if hasattr(solution, 'crowding_distance'):
            new_solution.crowding_distance = solution.crowding_distance
        if hasattr(solution, 'domination_count'):
            new_solution.domination_count = solution.domination_count
        if hasattr(solution, 'dominated_solutions'):
            new_solution.dominated_solutions = []  # 不复制引用，避免循环引用
            
        return new_solution 