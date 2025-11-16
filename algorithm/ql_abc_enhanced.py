#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QL-ABC算法增强版本 - 解决性能问题
基于论文复现，适配MO-DHFSP问题，并针对性能问题进行优化

主要改进：
1. 增加迭代次数到1000次
2. 改进状态空间定义
3. 优化奖励函数
4. 动态适应度权重
5. 增强探索策略
"""

import numpy as np
import random
import copy
import math
from typing import List, Dict, Tuple
from dataclasses import dataclass
from scipy.special import beta

from problem.mo_dhfsp import MO_DHFSP_Problem, Solution


@dataclass
class QLABCEnhancedParameters:
    """QL-ABC增强版算法参数"""
    population_size: int = 100
    max_iterations: int = 1000  # 增加到1000次
    limit: int = 15  # 增加limit值
    learning_rate: float = 0.3  # 稍微降低学习率
    discount_factor: float = 0.9  # 增加折扣因子
    epsilon: float = 0.2  # 增加探索率
    epsilon_decay: float = 0.995  # 探索率衰减
    mu1: float = 0.4
    mu2: float = 0.3  # 增加多样性权重
    mu3: float = 0.3  # 增加最优适应度权重
    k: int = 8  # 调整动作集参数
    mu: int = 4  # 调整Beta分布参数
    omega: int = 8  # 调整Beta分布参数
    archive_size: int = 200  # 增加档案大小


class QLABC_Optimizer_Enhanced:
    """QL-ABC增强版优化器"""
    
    def __init__(self, problem: MO_DHFSP_Problem, **kwargs):
        """
        初始化QL-ABC增强版优化器
        
        Args:
            problem: MO-DHFSP问题实例
            **kwargs: 算法参数
        """
        self.problem = problem
        self.n_jobs = problem.n_jobs
        self.n_factories = problem.n_factories
        
        # 设置算法参数
        default_params = QLABCEnhancedParameters()
        self.params = QLABCEnhancedParameters(
            population_size=kwargs.get('population_size', default_params.population_size),
            max_iterations=kwargs.get('max_iterations', default_params.max_iterations),
            limit=kwargs.get('limit', default_params.limit),
            learning_rate=kwargs.get('learning_rate', default_params.learning_rate),
            discount_factor=kwargs.get('discount_factor', default_params.discount_factor),
            epsilon=kwargs.get('epsilon', default_params.epsilon),
            epsilon_decay=kwargs.get('epsilon_decay', default_params.epsilon_decay),
            mu1=kwargs.get('mu1', default_params.mu1),
            mu2=kwargs.get('mu2', default_params.mu2),
            mu3=kwargs.get('mu3', default_params.mu3),
            k=kwargs.get('k', default_params.k),
            mu=kwargs.get('mu', default_params.mu),
            omega=kwargs.get('omega', default_params.omega),
            archive_size=kwargs.get('archive_size', default_params.archive_size)
        )
        
        # 算法状态
        self.current_iteration = 0
        self.population = []
        self.trial_counters = []
        self.external_archive = []
        self.convergence_data = []
        self.q_table = {}
        
        # 动态状态空间
        self.state_intervals = []
        self.fitness_history = []
        
        # 动态权重
        self.makespan_weight = 0.55
        self.tardiness_weight = 0.45
        
        print(f"初始化QL-ABC增强版优化器: 种群大小={self.params.population_size}, 最大迭代={self.params.max_iterations}")
        print(f"增强参数: 学习率={self.params.learning_rate}, 折扣因子={self.params.discount_factor}, 探索率={self.params.epsilon}")
    
    def _update_state_intervals(self):
        """动态更新状态空间区间"""
        if len(self.fitness_history) < 10:
            return
        
        # 基于最近的适应度历史动态调整状态空间
        recent_fitness = self.fitness_history[-10:]
        min_fitness = min(recent_fitness)
        max_fitness = max(recent_fitness)
        
        if max_fitness > min_fitness:
            # 创建10个等间距的状态区间
            step = (max_fitness - min_fitness) / 10
            self.state_intervals = [min_fitness + i * step for i in range(1, 10)]
        else:
            # 如果适应度范围太小，使用默认区间
            self.state_intervals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    def _update_weights(self):
        """动态更新适应度权重"""
        if len(self.fitness_history) < 20:
            return
        
        # 基于最近的收敛情况调整权重
        recent_makespans = []
        recent_tardiness = []
        
        for solution in self.population:
            recent_makespans.append(solution.makespan)
            recent_tardiness.append(solution.total_tardiness)
        
        if recent_makespans and recent_tardiness:
            makespan_std = np.std(recent_makespans)
            tardiness_std = np.std(recent_tardiness)
            
            # 如果某个目标的标准差较大，增加其权重
            total_std = makespan_std + tardiness_std
            if total_std > 0:
                self.makespan_weight = makespan_std / total_std
                self.tardiness_weight = tardiness_std / total_std
                
                # 确保权重在合理范围内
                self.makespan_weight = max(0.3, min(0.7, self.makespan_weight))
                self.tardiness_weight = 1.0 - self.makespan_weight
    
    def optimize(self) -> Tuple[List[Solution], List[Dict]]:
        """执行QL-ABC增强版优化"""
        # 初始化
        self._initialize_population()
        self._initialize_q_learning()
        
        # 主循环
        for iteration in range(self.params.max_iterations):
            self.current_iteration = iteration
            
            # 动态更新参数
            self._update_state_intervals()
            self._update_weights()
            
            # 衰减探索率
            self.params.epsilon *= self.params.epsilon_decay
            
            # 三个阶段
            self._employed_bee_phase()
            self._onlooker_bee_phase()
            self._scout_bee_phase()
            
            # 更新档案
            self._update_external_archive()
            self._record_convergence_data()
            
            # 每100轮输出一次进度
            if iteration % 100 == 0:
                print(f"  QL-ABC增强版 迭代 {iteration}/{self.params.max_iterations}, 档案大小: {len(self.external_archive)}")
        
        return self.external_archive, self.convergence_data
    
    def _initialize_population(self):
        """初始化种群 - 增强多样性"""
        self.population = []
        self.trial_counters = []
        
        for i in range(self.params.population_size):
            solution = self._generate_random_solution()
            solution = self.problem.evaluate_solution(solution)
            self.population.append(solution)
            self.trial_counters.append(0)
            
            # 记录适应度历史
            fitness = self._compute_fitness(solution)
            self.fitness_history.append(fitness)
    
    def _initialize_q_learning(self):
        """初始化Q-learning"""
        self.q_table = {}
    
    def _employed_bee_phase(self):
        """引领蜂阶段 - 增强版"""
        for i in range(self.params.population_size):
            current_solution = self.population[i]
            state = self._define_state(current_solution)
            action = self._select_action(state)
            
            # 增强的蜜源更新策略
            neighbor_solution = self._update_honey_source_enhanced(current_solution, action)
            neighbor_solution = self.problem.evaluate_solution(neighbor_solution)
            
            reward = self._compute_reward_enhanced(current_solution, neighbor_solution)
            self._update_q_table(state, action, reward, neighbor_solution)
            
            if self._is_better_solution(neighbor_solution, current_solution):
                self.population[i] = neighbor_solution
                self.trial_counters[i] = 0
                
                # 更新适应度历史
                fitness = self._compute_fitness(neighbor_solution)
                self.fitness_history.append(fitness)
            else:
                self.trial_counters[i] += 1
    
    def _onlooker_bee_phase(self):
        """跟随蜂阶段 - 增强版"""
        fitness_values = [self._compute_fitness(sol) for sol in self.population]
        
        # 使用锦标赛选择替代轮盘赌
        for _ in range(self.params.population_size):
            selected_idx = self._tournament_selection(fitness_values)
            selected_solution = self.population[selected_idx]
            
            state = self._define_state(selected_solution)
            action = self._select_action(state)
            
            neighbor_solution = self._update_honey_source_enhanced(selected_solution, action)
            neighbor_solution = self.problem.evaluate_solution(neighbor_solution)
            
            reward = self._compute_reward_enhanced(selected_solution, neighbor_solution)
            self._update_q_table(state, action, reward, neighbor_solution)
            
            if self._is_better_solution(neighbor_solution, selected_solution):
                self.population[selected_idx] = neighbor_solution
                self.trial_counters[selected_idx] = 0
                
                # 更新适应度历史
                fitness = self._compute_fitness(neighbor_solution)
                self.fitness_history.append(fitness)
    
    def _scout_bee_phase(self):
        """侦察蜂阶段 - 增强版"""
        for i in range(self.params.population_size):
            if self.trial_counters[i] >= self.params.limit:
                # 使用精英解作为基础生成新解
                new_solution = self._generate_elite_based_solution()
                new_solution = self.problem.evaluate_solution(new_solution)
                self.population[i] = new_solution
                self.trial_counters[i] = 0
                
                # 更新适应度历史
                fitness = self._compute_fitness(new_solution)
                self.fitness_history.append(fitness)
    
    def _define_state(self, solution: Solution) -> Tuple:
        """定义状态 - 增强版"""
        # 计算状态特征
        fa = self._compute_average_fitness()
        fv = self._compute_diversity()
        fm = self._compute_optimal_fitness()
        
        # 计算整体状态值
        F = self.params.mu1 * fa + self.params.mu2 * fv + self.params.mu3 * fm
        
        # 将F映射到状态空间
        state_idx = self._map_to_state_space(F)
        return (state_idx,)
    
    def _compute_average_fitness(self) -> float:
        """计算平均适应度"""
        if not self.population:
            return 0.0
        
        fitness_values = [self._compute_fitness(sol) for sol in self.population]
        return np.mean(fitness_values)
    
    def _compute_diversity(self) -> float:
        """计算多样性 - 增强版"""
        if len(self.population) < 2:
            return 0.0
        
        # 计算目标空间的多样性
        makespans = [sol.makespan for sol in self.population]
        tardinesses = [sol.total_tardiness for sol in self.population]
        
        makespan_std = np.std(makespans) if makespans else 0
        tardiness_std = np.std(tardinesses) if tardinesses else 0
        
        return (makespan_std + tardiness_std) / 2.0
    
    def _compute_optimal_fitness(self) -> float:
        """计算最优适应度"""
        if not self.population:
            return 0.0
        
        current_best = max([self._compute_fitness(sol) for sol in self.population])
        
        # 使用档案中的最优解作为参考
        if self.external_archive:
            archive_best = max([self._compute_fitness(sol) for sol in self.external_archive])
            return current_best / archive_best if archive_best > 0 else 0.0
        
        return 1.0
    
    def _map_to_state_space(self, F: float) -> int:
        """将F值映射到状态空间"""
        if not self.state_intervals:
            return 0
        
        for i, threshold in enumerate(self.state_intervals):
            if F <= threshold:
                return i
        return len(self.state_intervals) - 1
    
    def _select_action(self, state: Tuple) -> int:
        """选择动作 - 增强版"""
        H = self.n_jobs
        max_h = max(1, H // self.params.k)
        available_actions = list(range(1, max_h + 1))
        
        if random.random() < self.params.epsilon:
            # 探索：随机选择动作
            return random.choice(available_actions)
        else:
            # 利用：选择Q值最大的动作
            if state in self.q_table and self.q_table[state]:
                return max(self.q_table[state].keys(), key=lambda a: self.q_table[state][a])
            else:
                return random.choice(available_actions)
    
    def _update_honey_source_enhanced(self, solution: Solution, h: int) -> Solution:
        """增强的蜜源更新策略"""
        new_solution = copy.deepcopy(solution)
        
        # 随机选择h个维度进行更新
        dimensions_to_update = random.sample(range(self.n_jobs), min(h, self.n_jobs))
        
        for dim in dimensions_to_update:
            # 使用多种更新策略
            strategy = random.choice(['random', 'elite', 'crossover'])
            
            if strategy == 'random':
                # 随机更新
                new_factory = random.randint(0, self.n_factories - 1)
            elif strategy == 'elite':
                # 使用精英解
                if self.external_archive:
                    elite_solution = random.choice(self.external_archive)
                    new_factory = elite_solution.factory_assignment[dim]
                else:
                    new_factory = random.randint(0, self.n_factories - 1)
            else:  # crossover
                # 交叉更新
                k = random.randint(0, len(self.population) - 1)
                reference_solution = self.population[k]
                rand_val = random.random()
                current_val = solution.factory_assignment[dim]
                reference_val = reference_solution.factory_assignment[dim]
                new_factory = int(current_val + rand_val * (current_val - reference_val))
                new_factory = max(0, min(self.n_factories - 1, new_factory))
            
            new_solution.factory_assignment[dim] = new_factory
        
        # 重新构建作业序列
        new_solution.job_sequences = [[] for _ in range(self.n_factories)]
        for job_id in range(self.n_jobs):
            factory_id = new_solution.factory_assignment[job_id]
            new_solution.job_sequences[factory_id].append(job_id)
        
        # 使用部分映射交叉(PMX)优化作业顺序
        for factory_id in range(self.n_factories):
            if len(new_solution.job_sequences[factory_id]) > 1:
                new_solution.job_sequences[factory_id] = self._pmx_crossover(
                    new_solution.job_sequences[factory_id]
                )
        
        return new_solution
    
    def _pmx_crossover(self, sequence: List[int]) -> List[int]:
        """部分映射交叉"""
        if len(sequence) < 2:
            return sequence
        
        # 随机选择两个交叉点
        points = sorted(random.sample(range(len(sequence)), 2))
        start, end = points[0], points[1]
        
        # 创建新序列
        new_sequence = sequence.copy()
        
        # 交换中间部分
        for i in range(start, end):
            new_sequence[i] = sequence[end - 1 - (i - start)]
        
        return new_sequence
    
    def _compute_reward_enhanced(self, current_solution: Solution, new_solution: Solution) -> float:
        """增强的奖励函数"""
        # 多目标奖励计算
        makespan_improvement = (current_solution.makespan - new_solution.makespan) / max(current_solution.makespan, 1)
        tardiness_improvement = (current_solution.total_tardiness - new_solution.total_tardiness) / max(current_solution.total_tardiness, 1)
        
        # 加权奖励
        weighted_improvement = (self.makespan_weight * makespan_improvement + 
                              self.tardiness_weight * tardiness_improvement)
        
        # 帕累托支配奖励
        pareto_reward = 0.0
        if self._strict_dominates(new_solution, current_solution):
            pareto_reward = 2.0
        elif self._strict_dominates(current_solution, new_solution):
            pareto_reward = -1.0
        else:
            pareto_reward = 0.5  # 非支配解
        
        # 综合奖励
        total_reward = weighted_improvement + pareto_reward
        
        # 限制奖励范围
        return max(-2.0, min(3.0, total_reward))
    
    def _generate_elite_based_solution(self) -> Solution:
        """基于精英解生成新解"""
        if self.external_archive:
            # 从档案中选择一个精英解作为基础
            base_solution = random.choice(self.external_archive)
            new_solution = copy.deepcopy(base_solution)
            
            # 随机变异
            for job_id in range(self.n_jobs):
                if random.random() < 0.1:  # 10%的变异概率
                    new_solution.factory_assignment[job_id] = random.randint(0, self.n_factories - 1)
        else:
            # 如果没有档案，生成随机解
            new_solution = self._generate_random_solution()
        
        # 重新构建作业序列
        new_solution.job_sequences = [[] for _ in range(self.n_factories)]
        for job_id in range(self.n_jobs):
            factory_id = new_solution.factory_assignment[job_id]
            new_solution.job_sequences[factory_id].append(job_id)
        
        # 随机打乱每个工厂的作业顺序
        for factory_id in range(self.n_factories):
            random.shuffle(new_solution.job_sequences[factory_id])
        
        return new_solution
    
    def _tournament_selection(self, fitness_values: List[float]) -> int:
        """锦标赛选择"""
        tournament_size = 3
        tournament_indices = random.sample(range(len(fitness_values)), tournament_size)
        tournament_fitness = [fitness_values[i] for i in tournament_indices]
        
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return winner_idx
    
    def _update_q_table(self, state: Tuple, action: int, reward: float, next_solution: Solution):
        """更新Q表"""
        if state not in self.q_table:
            self.q_table[state] = {}
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0.0
        
        next_state = self._define_state(next_solution)
        max_next_q = 0.0
        if next_state in self.q_table and self.q_table[next_state]:
            max_next_q = max(self.q_table[next_state].values())
        
        old_q = self.q_table[state][action]
        self.q_table[state][action] = old_q + self.params.learning_rate * (
            reward + self.params.discount_factor * max_next_q - old_q
        )
    
    def _compute_fitness(self, solution: Solution) -> float:
        """计算适应度 - 使用动态权重"""
        return self.makespan_weight * solution.makespan + self.tardiness_weight * solution.total_tardiness
    
    def _is_better_solution(self, sol1: Solution, sol2: Solution) -> bool:
        """判断解的优劣 - 帕累托支配"""
        better_makespan = sol1.makespan <= sol2.makespan
        better_tardiness = sol1.total_tardiness <= sol2.total_tardiness
        
        if better_makespan and better_tardiness:
            return sol1.makespan < sol2.makespan or sol1.total_tardiness < sol2.total_tardiness
        
        return False
    
    def _generate_random_solution(self) -> Solution:
        """生成随机解"""
        factory_assignment = [random.randint(0, self.n_factories - 1) for _ in range(self.n_jobs)]
        job_sequences = [[] for _ in range(self.n_factories)]
        
        for job_id in range(self.n_jobs):
            factory_id = factory_assignment[job_id]
            job_sequences[factory_id].append(job_id)
        
        for factory_id in range(self.n_factories):
            random.shuffle(job_sequences[factory_id])
        
        return Solution(factory_assignment, job_sequences)
    
    def _update_external_archive(self):
        """更新外部档案 - 增强版"""
        # 合并当前种群和档案
        all_solutions = self.population + self.external_archive
        
        # 计算帕累托前沿
        pareto_front = self._compute_pareto_front(all_solutions)
        
        # 限制档案大小
        if len(pareto_front) > self.params.archive_size:
            pareto_front = self._select_diverse_archive(pareto_front, self.params.archive_size)
        
        self.external_archive = pareto_front
    
    def _compute_pareto_front(self, solutions: List[Solution]) -> List[Solution]:
        """计算帕累托前沿"""
        pareto_front = []
        
        for solution in solutions:
            is_dominated = False
            
            for other_solution in solutions:
                if solution != other_solution and self._strict_dominates(other_solution, solution):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_front.append(solution)
        
        return pareto_front
    
    def _strict_dominates(self, sol1: Solution, sol2: Solution) -> bool:
        """严格支配关系"""
        better_makespan = sol1.makespan <= sol2.makespan
        better_tardiness = sol1.total_tardiness <= sol2.total_tardiness
        
        return better_makespan and better_tardiness and (sol1.makespan < sol2.makespan or sol1.total_tardiness < sol2.total_tardiness)
    
    def _select_diverse_archive(self, solutions: List[Solution], target_size: int) -> List[Solution]:
        """选择多样化的档案"""
        if len(solutions) <= target_size:
            return solutions
        
        # 计算解之间的距离
        distances = []
        for i, sol1 in enumerate(solutions):
            min_distance = float('inf')
            for j, sol2 in enumerate(solutions):
                if i != j:
                    distance = math.sqrt((sol1.makespan - sol2.makespan)**2 + 
                                       (sol1.total_tardiness - sol2.total_tardiness)**2)
                    min_distance = min(min_distance, distance)
            distances.append(min_distance)
        
        # 选择距离最大的解
        selected_indices = np.argsort(distances)[-target_size:]
        return [solutions[i] for i in selected_indices]
    
    def _record_convergence_data(self):
        """记录收敛数据"""
        if self.external_archive:
            best_makespan = min(sol.makespan for sol in self.external_archive)
            best_tardiness = min(sol.total_tardiness for sol in self.external_archive)
            
            self.convergence_data.append({
                'iteration': self.current_iteration,
                'archive_size': len(self.external_archive),
                'best_makespan': best_makespan,
                'best_tardiness': best_tardiness
            }) 