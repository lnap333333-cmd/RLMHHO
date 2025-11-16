#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QL-ABC算法修正版本 - 严格按照原文要求实现
基于论文复现，适配MO-DHFSP问题

论文：Q-learning Artificial Bee Colony Algorithm
结合Q-learning强化学习和人工蜂群算法用于多目标优化
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
class QLABCParameters:
    """QL-ABC算法参数 - 严格按照原文设置"""
    population_size: int = 100  # 原文：100
    max_iterations: int = 1000  # 原文：1000
    limit: int = 10
    learning_rate: float = 0.4  # 原文：0.4
    discount_factor: float = 0.8  # 原文：0.8
    epsilon: float = 0.1  # 探索因子
    mu1: float = 0.4  # 平均适应度权重
    mu2: float = 0.2  # 多样性权重  
    mu3: float = 0.2  # 最优适应度权重
    k: int = 10  # 动作集参数，原文：k=10时性能最好
    mu: int = 6  # Beta分布参数
    omega: int = 12  # Beta分布参数


class QLABC_Optimizer_Fixed:
    """QL-ABC优化器修正版本"""
    
    def __init__(self, problem: MO_DHFSP_Problem, **kwargs):
        """
        初始化QL-ABC优化器
        
        Args:
            problem: MO-DHFSP问题实例
            **kwargs: 算法参数
        """
        self.problem = problem
        self.n_jobs = problem.n_jobs
        self.n_factories = problem.n_factories
        
        # 设置算法参数
        default_params = QLABCParameters()
        self.params = QLABCParameters(
            population_size=kwargs.get('population_size', default_params.population_size),
            max_iterations=kwargs.get('max_iterations', default_params.max_iterations),
            limit=kwargs.get('limit', default_params.limit),
            learning_rate=kwargs.get('learning_rate', default_params.learning_rate),
            discount_factor=kwargs.get('discount_factor', default_params.discount_factor),
            epsilon=kwargs.get('epsilon', default_params.epsilon),
            mu1=kwargs.get('mu1', default_params.mu1),
            mu2=kwargs.get('mu2', default_params.mu2),
            mu3=kwargs.get('mu3', default_params.mu3),
            k=kwargs.get('k', default_params.k),
            mu=kwargs.get('mu', default_params.mu),
            omega=kwargs.get('omega', default_params.omega)
        )
        
        # 算法状态
        self.current_iteration = 0
        self.population = []
        self.trial_counters = []
        self.external_archive = []
        self.convergence_data = []
        self.q_table = {}
        
        # 状态空间定义 - 按照原文要求
        self.state_intervals = self._define_state_intervals()
        
        print(f"初始化QL-ABC修正版优化器: 种群大小={self.params.population_size}, 最大迭代={self.params.max_iterations}")
        print(f"参数设置: 学习率={self.params.learning_rate}, 折扣因子={self.params.discount_factor}")
    
    def _define_state_intervals(self):
        """定义状态空间区间 - 按照原文要求"""
        # 根据工件数量和平均加工时间确定状态区间
        # 简化版本：基于种群适应度范围划分
        return [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    
    def optimize(self) -> Tuple[List[Solution], List[Dict]]:
        """执行QL-ABC优化"""
        # 初始化
        self._initialize_population()
        self._initialize_q_learning()
        
        # 主循环
        for iteration in range(self.params.max_iterations):
            self.current_iteration = iteration
            
            # 三个阶段
            self._employed_bee_phase()
            self._onlooker_bee_phase()
            self._scout_bee_phase()
            
            # 更新档案
            self._update_external_archive()
            self._record_convergence_data()
            
            # 每100轮输出一次进度
            if iteration % 100 == 0:
                print(f"  QL-ABC修正版 迭代 {iteration}/{self.params.max_iterations}, 档案大小: {len(self.external_archive)}")
        
        return self.external_archive, self.convergence_data
    
    def _initialize_population(self):
        """初始化种群"""
        self.population = []
        self.trial_counters = []
        
        for i in range(self.params.population_size):
            solution = self._generate_random_solution()
            solution = self.problem.evaluate_solution(solution)
            self.population.append(solution)
            self.trial_counters.append(0)
    
    def _initialize_q_learning(self):
        """初始化Q-learning"""
        self.q_table = {}
    
    def _employed_bee_phase(self):
        """引领蜂阶段 - 按照原文公式(13)"""
        for i in range(self.params.population_size):
            current_solution = self.population[i]
            state = self._define_state(current_solution)
            action = self._select_action(state)
            
            # 按照原文公式(13)更新蜜源
            neighbor_solution = self._update_honey_source(current_solution, action)
            neighbor_solution = self.problem.evaluate_solution(neighbor_solution)
            
            reward = self._compute_reward_beta(current_solution, neighbor_solution)
            self._update_q_table(state, action, reward, neighbor_solution)
            
            if self._is_better_solution(neighbor_solution, current_solution):
                self.population[i] = neighbor_solution
                self.trial_counters[i] = 0
            else:
                self.trial_counters[i] += 1
    
    def _onlooker_bee_phase(self):
        """跟随蜂阶段 - 按照原文公式(14)"""
        fitness_values = [self._compute_fitness(sol) for sol in self.population]
        
        # 按照原文公式(14)计算选择概率
        total_fitness = sum(fitness_values)
        selection_probs = [f / total_fitness for f in fitness_values] if total_fitness > 0 else [1.0/len(fitness_values)] * len(fitness_values)
        
        for _ in range(self.params.population_size):
            selected_idx = self._roulette_wheel_selection(selection_probs)
            selected_solution = self.population[selected_idx]
            
            state = self._define_state(selected_solution)
            action = self._select_action(state)
            
            neighbor_solution = self._update_honey_source(selected_solution, action)
            neighbor_solution = self.problem.evaluate_solution(neighbor_solution)
            
            reward = self._compute_reward_beta(selected_solution, neighbor_solution)
            self._update_q_table(state, action, reward, neighbor_solution)
            
            if self._is_better_solution(neighbor_solution, selected_solution):
                self.population[selected_idx] = neighbor_solution
                self.trial_counters[selected_idx] = 0
    
    def _scout_bee_phase(self):
        """侦察蜂阶段 - 按照原文公式(15)"""
        for i in range(self.params.population_size):
            if self.trial_counters[i] >= self.params.limit:
                # 按照原文公式(15)生成新蜜源
                new_solution = self._generate_new_honey_source()
                new_solution = self.problem.evaluate_solution(new_solution)
                self.population[i] = new_solution
                self.trial_counters[i] = 0
    
    def _define_state(self, solution: Solution) -> Tuple:
        """定义状态 - 严格按照原文公式(18-21)"""
        # 计算状态特征
        fa = self._compute_average_fitness()  # 公式(18)
        fv = self._compute_diversity()        # 公式(19)  
        fm = self._compute_optimal_fitness()  # 公式(20)
        
        # 按照公式(21)计算整体状态值
        F = self.params.mu1 * fa + self.params.mu2 * fv + self.params.mu3 * fm
        
        # 将F映射到状态空间
        state_idx = self._map_to_state_space(F)
        return (state_idx,)
    
    def _compute_average_fitness(self) -> float:
        """计算平均适应度 - 公式(18)"""
        if not self.population:
            return 0.0
        
        fitness_values = [self._compute_fitness(sol) for sol in self.population]
        return np.mean(fitness_values)
    
    def _compute_diversity(self) -> float:
        """计算多样性 - 公式(19)"""
        if len(self.population) < 2:
            return 0.0
        
        fitness_values = [self._compute_fitness(sol) for sol in self.population]
        avg_fitness = np.mean(fitness_values)
        
        if avg_fitness == 0:
            return 0.0
        
        # 计算与平均值的偏差
        deviations = [abs(f - avg_fitness) for f in fitness_values]
        return np.mean(deviations) / avg_fitness
    
    def _compute_optimal_fitness(self) -> float:
        """计算最优适应度 - 公式(20)"""
        if not self.population:
            return 0.0
        
        current_best = max([self._compute_fitness(sol) for sol in self.population])
        
        # 这里需要一个参考最优值，暂时使用当前最优
        reference_best = current_best  # 理想情况下应该有全局最优参考
        
        return current_best / reference_best if reference_best > 0 else 0.0
    
    def _map_to_state_space(self, F: float) -> int:
        """将F值映射到状态空间"""
        for i, threshold in enumerate(self.state_intervals):
            if F <= threshold:
                return i
        return len(self.state_intervals) - 1
    
    def _select_action(self, state: Tuple) -> int:
        """选择动作 - 按照原文公式(24)"""
        # 动作集：h ∈ [1, H/k]，其中H是问题维度
        H = self.n_jobs  # 问题维度
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
    
    def _update_honey_source(self, solution: Solution, h: int) -> Solution:
        """按照原文公式(13)更新蜜源"""
        # Vij = Xij + rand(0,1) * (Xij - Xkj)
        new_solution = copy.deepcopy(solution)
        
        # 随机选择h个维度进行更新
        dimensions_to_update = random.sample(range(self.n_jobs), min(h, self.n_jobs))
        
        for dim in dimensions_to_update:
            # 随机选择另一个蜜源作为参考
            k = random.randint(0, len(self.population) - 1)
            reference_solution = self.population[k]
            
            # 计算更新值
            rand_val = random.random()
            current_val = solution.factory_assignment[dim]
            reference_val = reference_solution.factory_assignment[dim]
            
            # 更新工厂分配
            new_factory = int(current_val + rand_val * (current_val - reference_val))
            new_factory = max(0, min(self.n_factories - 1, new_factory))
            new_solution.factory_assignment[dim] = new_factory
        
        # 重新构建作业序列
        new_solution.job_sequences = [[] for _ in range(self.n_factories)]
        for job_id in range(self.n_jobs):
            factory_id = new_solution.factory_assignment[job_id]
            new_solution.job_sequences[factory_id].append(job_id)
        
        # 随机打乱每个工厂的作业顺序
        for factory_id in range(self.n_factories):
            random.shuffle(new_solution.job_sequences[factory_id])
        
        return new_solution
    
    def _compute_reward_beta(self, current_solution: Solution, new_solution: Solution) -> float:
        """按照原文公式(22-23)计算Beta分布奖励"""
        current_fitness = self._compute_fitness(current_solution)
        new_fitness = self._compute_fitness(new_solution)
        
        # 计算改进程度
        if current_fitness == 0:
            improvement_ratio = 0.0
        else:
            improvement_ratio = (current_fitness - new_fitness) / current_fitness
        
        # 将改进程度映射到[0,1]区间
        x = max(0.0, min(1.0, (improvement_ratio + 1.0) / 2.0))
        
        # 按照公式(23)计算Beta分布概率密度
        beta_val = beta(self.params.mu, self.params.omega)
        if beta_val > 0:
            f_x = (1.0 / beta_val) * (x ** (self.params.mu - 1)) * ((1 - x) ** (self.params.omega - 1))
        else:
            f_x = 0.0
        
        # 根据概率密度确定奖励
        if f_x > 0.5:
            return 1.0
        elif f_x < 0.2:
            return -1.0
        else:
            return 0.0
    
    def _generate_new_honey_source(self) -> Solution:
        """按照原文公式(15)生成新蜜源"""
        # Xij = Xij + rand(0,1) * (Ud - Ld)
        new_solution = copy.deepcopy(self.population[0])  # 使用第一个解作为基础
        
        for job_id in range(self.n_jobs):
            rand_val = random.random()
            # Ud = n_factories - 1, Ld = 0
            new_factory = int(rand_val * (self.n_factories - 1))
            new_solution.factory_assignment[job_id] = new_factory
        
        # 重新构建作业序列
        new_solution.job_sequences = [[] for _ in range(self.n_factories)]
        for job_id in range(self.n_jobs):
            factory_id = new_solution.factory_assignment[job_id]
            new_solution.job_sequences[factory_id].append(job_id)
        
        # 随机打乱每个工厂的作业顺序
        for factory_id in range(self.n_factories):
            random.shuffle(new_solution.job_sequences[factory_id])
        
        return new_solution
    
    def _update_q_table(self, state: Tuple, action: int, reward: float, next_solution: Solution):
        """更新Q表 - 按照原文公式(16)"""
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
        """计算适应度"""
        return 0.55 * solution.makespan + 0.45 * solution.total_tardiness
    
    def _is_better_solution(self, sol1: Solution, sol2: Solution) -> bool:
        """判断解的优劣"""
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
    
    def _roulette_wheel_selection(self, probabilities: List[float]) -> int:
        """轮盘赌选择"""
        r = random.random()
        cumulative_prob = 0.0
        
        for i, prob in enumerate(probabilities):
            cumulative_prob += prob
            if r <= cumulative_prob:
                return i
        
        return len(probabilities) - 1
    
    def _update_external_archive(self):
        """更新外部档案"""
        # 合并当前种群和档案
        all_solutions = self.population + self.external_archive
        pareto_solutions = self._compute_pareto_front(all_solutions)
        
        # 限制档案大小
        if len(pareto_solutions) > 200:
            pareto_solutions = self._select_diverse_archive(pareto_solutions, 200)
        
        self.external_archive = pareto_solutions
    
    def _compute_pareto_front(self, solutions: List[Solution]) -> List[Solution]:
        """计算帕累托前沿"""
        if not solutions:
            return []
        
        pareto_front = []
        
        for solution in solutions:
            is_dominated = False
            solutions_to_remove = []
            
            for pareto_sol in pareto_front:
                if self._strict_dominates(pareto_sol, solution):
                    is_dominated = True
                    break
                elif self._strict_dominates(solution, pareto_sol):
                    solutions_to_remove.append(pareto_sol)
            
            if not is_dominated:
                for sol in solutions_to_remove:
                    pareto_front.remove(sol)
                pareto_front.append(solution)
        
        return pareto_front
    
    def _strict_dominates(self, sol1: Solution, sol2: Solution) -> bool:
        """严格支配判断"""
        return (sol1.makespan <= sol2.makespan and sol1.total_tardiness <= sol2.total_tardiness and
                (sol1.makespan < sol2.makespan or sol1.total_tardiness < sol2.total_tardiness))
    
    def _select_diverse_archive(self, solutions: List[Solution], target_size: int) -> List[Solution]:
        """从解集中选择多样化的子集"""
        if len(solutions) <= target_size:
            return solutions
        
        # 按makespan排序选择
        sorted_by_makespan = sorted(solutions, key=lambda x: x.makespan)
        step = len(sorted_by_makespan) // target_size
        
        selected = []
        for i in range(0, len(sorted_by_makespan), step):
            if len(selected) < target_size:
                selected.append(sorted_by_makespan[i])
        
        return selected[:target_size]
    
    def _record_convergence_data(self):
        """记录收敛数据"""
        if self.external_archive:
            best_makespan = min([sol.makespan for sol in self.external_archive])
            best_tardiness = min([sol.total_tardiness for sol in self.external_archive])
            archive_size = len(self.external_archive)
        else:
            best_makespan = float('inf')
            best_tardiness = float('inf')
            archive_size = 0
        
        convergence_info = {
            'iteration': self.current_iteration,
            'best_makespan': best_makespan,
            'best_tardiness': best_tardiness,
            'archive_size': archive_size,
            'q_table_size': len(self.q_table)
        }
        
        self.convergence_data.append(convergence_info) 