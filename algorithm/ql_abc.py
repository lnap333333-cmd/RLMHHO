#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QL-ABC算法实现 - Q-learning人工蜂群算法
基于论文复现，适配MO-DHFSP问题

论文：Q-learning Artificial Bee Colony Algorithm
结合Q-learning强化学习和人工蜂群算法用于多目标优化
"""

import numpy as np
import random
import copy
from typing import List, Dict, Tuple
from dataclasses import dataclass

from problem.mo_dhfsp import MO_DHFSP_Problem, Solution


@dataclass
class QLABCParameters:
    """QL-ABC算法参数"""
    population_size: int = 50
    max_iterations: int = 50
    limit: int = 10
    learning_rate: float = 0.4
    discount_factor: float = 0.2
    epsilon: float = 0.05
    mu1: float = 0.4
    mu2: float = 0.2
    mu3: float = 0.2


class QLABC_Optimizer:
    """QL-ABC优化器"""
    
    def __init__(self, problem: MO_DHFSP_Problem, **kwargs):
        """
        初始化QL-ABC优化器
        
        Args:
            problem: MO_DHFSP问题实例
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
            mu3=kwargs.get('mu3', default_params.mu3)
        )
        
        # 算法状态
        self.current_iteration = 0
        self.population = []
        self.trial_counters = []
        self.external_archive = []
        self.convergence_data = []
        self.q_table = {}
        
        print(f"初始化QL-ABC优化器: 种群大小={self.params.population_size}, 最大迭代={self.params.max_iterations}")
    
    def optimize(self) -> Tuple[List[Solution], List[Dict]]:
        """执行QL-ABC优化"""
        # 初始化
        self._initialize_population()
        self._initialize_q_learning()
        
        # 主循环
        for iteration in range(self.params.max_iterations):
            self.current_iteration = iteration
            
            # 缓存当前种群的平均适应度，避免重复计算
            fitness_values = [self._compute_single_fitness(sol) for sol in self.population]
            self._cached_avg_fitness = np.mean(fitness_values) if fitness_values else 1.0
            
            # 三个阶段
            self._employed_bee_phase()
            self._onlooker_bee_phase()
            self._scout_bee_phase()
            
            # 更新档案
            self._update_external_archive()
            self._record_convergence_data()
            
            # 每10轮输出一次进度
            if iteration % 10 == 0:
                print(f"  QL-ABC 迭代 {iteration}/{self.params.max_iterations}, 档案大小: {len(self.external_archive)}")
        
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
        """引领蜂阶段"""
        for i in range(self.params.population_size):
            current_solution = self.population[i]
            state = self._define_state(current_solution)
            action = self._select_action(state)
            
            neighbor_solution = self._execute_action(current_solution, action)
            neighbor_solution = self.problem.evaluate_solution(neighbor_solution)
            
            reward = self._compute_reward(current_solution, neighbor_solution)
            self._update_q_table(state, action, reward, neighbor_solution)
            
            if self._is_better_solution(neighbor_solution, current_solution):
                self.population[i] = neighbor_solution
                self.trial_counters[i] = 0
            else:
                self.trial_counters[i] += 1
    
    def _onlooker_bee_phase(self):
        """跟随蜂阶段"""
        fitness_values = [self._compute_fitness(sol) for sol in self.population]
        
        if max(fitness_values) > min(fitness_values):
            normalized_fitness = [(f - min(fitness_values)) / (max(fitness_values) - min(fitness_values)) 
                                for f in fitness_values]
        else:
            normalized_fitness = [1.0] * len(fitness_values)
        
        total_fitness = sum(normalized_fitness)
        selection_probs = [f / total_fitness for f in normalized_fitness] if total_fitness > 0 else [1.0/len(normalized_fitness)] * len(normalized_fitness)
        
        for _ in range(self.params.population_size):
            selected_idx = self._roulette_wheel_selection(selection_probs)
            selected_solution = self.population[selected_idx]
            
            state = self._define_state(selected_solution)
            action = self._select_action(state)
            
            neighbor_solution = self._execute_action(selected_solution, action)
            neighbor_solution = self.problem.evaluate_solution(neighbor_solution)
            
            reward = self._compute_reward(selected_solution, neighbor_solution)
            self._update_q_table(state, action, reward, neighbor_solution)
            
            if self._is_better_solution(neighbor_solution, selected_solution):
                self.population[selected_idx] = neighbor_solution
                self.trial_counters[selected_idx] = 0
    
    def _scout_bee_phase(self):
        """侦察蜂阶段"""
        for i in range(self.params.population_size):
            if self.trial_counters[i] >= self.params.limit:
                new_solution = self._generate_random_solution()
                new_solution = self.problem.evaluate_solution(new_solution)
                self.population[i] = new_solution
                self.trial_counters[i] = 0
    
    def _define_state(self, solution: Solution) -> Tuple:
        """定义状态 - 简化版本避免重复计算"""
        current_fitness = self._compute_single_fitness(solution)
        
        # 简化状态定义，避免重复计算整个种群
        if hasattr(self, '_cached_avg_fitness'):
            avg_fitness = self._cached_avg_fitness
        else:
            fitness_values = [self._compute_single_fitness(sol) for sol in self.population]
            if not fitness_values:
                return (0,)
            avg_fitness = np.mean(fitness_values)
            self._cached_avg_fitness = avg_fitness
        
        if avg_fitness == 0:
            return (0,)
            
        state_value = current_fitness / avg_fitness
        return (1,) if state_value <= 1.1 else (0,)
    
    def _select_action(self, state: Tuple) -> str:
        """选择动作"""
        available_actions = ['swap_jobs', 'insert_job', 'swap_factories', 'reorder_factory']
        
        if random.random() < self.params.epsilon:
            return random.choice(available_actions)
        else:
            if state in self.q_table and self.q_table[state]:
                return max(self.q_table[state].keys(), key=lambda a: self.q_table[state][a])
            else:
                return random.choice(available_actions)
    
    def _execute_action(self, solution: Solution, action: str) -> Solution:
        """执行动作"""
        new_solution = copy.deepcopy(solution)
        
        if action == 'swap_jobs':
            self._swap_jobs_in_solution(new_solution)
        elif action == 'insert_job':
            self._insert_job_in_solution(new_solution)
        elif action == 'swap_factories':
            self._swap_factory_assignment(new_solution)
        elif action == 'reorder_factory':
            self._reorder_factory_jobs(new_solution)
        
        return new_solution
    
    def _compute_reward(self, current_solution: Solution, new_solution: Solution) -> float:
        """计算奖励"""
        current_obj = 0.55 * current_solution.makespan + 0.45 * current_solution.total_tardiness
        new_obj = 0.55 * new_solution.makespan + 0.45 * new_solution.total_tardiness
        
        improvement = current_obj - new_obj
        return 1.0 if improvement > 0 else (-1.0 if improvement < 0 else 0.0)
    
    def _update_q_table(self, state: Tuple, action: str, reward: float, next_solution: Solution):
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
        """计算适应度 - 简化版本"""
        current_fitness = self._compute_single_fitness(solution)
        
        # 使用缓存的平均适应度
        if hasattr(self, '_cached_avg_fitness') and self._cached_avg_fitness > 0:
            return current_fitness / self._cached_avg_fitness
        else:
            return 1.0
    
    def _compute_single_fitness(self, solution: Solution) -> float:
        """计算单个适应度"""
        return 0.55 * solution.makespan + 0.45 * solution.total_tardiness
    
    def _compute_diversity(self) -> float:
        """计算多样性"""
        if len(self.population) < 2:
            return 0.0
        
        fitness_values = [self._compute_single_fitness(sol) for sol in self.population]
        avg_fitness = np.mean(fitness_values)
        
        if avg_fitness == 0:
            return 0.0
            
        diversity = np.sqrt(np.mean([(f - avg_fitness) ** 2 for f in fitness_values]))
        return diversity / avg_fitness
    
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
    
    def _swap_jobs_in_solution(self, solution: Solution):
        """交换作业"""
        non_empty_factories = [i for i in range(self.n_factories) if len(solution.job_sequences[i]) > 1]
        
        if non_empty_factories:
            factory_id = random.choice(non_empty_factories)
            jobs = solution.job_sequences[factory_id]
            if len(jobs) >= 2:
                i, j = random.sample(range(len(jobs)), 2)
                jobs[i], jobs[j] = jobs[j], jobs[i]
    
    def _insert_job_in_solution(self, solution: Solution):
        """插入作业"""
        non_empty_factories = [i for i in range(self.n_factories) if len(solution.job_sequences[i]) > 0]
        
        if non_empty_factories:
            factory_id = random.choice(non_empty_factories)
            jobs = solution.job_sequences[factory_id]
            if len(jobs) >= 2:
                job_idx = random.randint(0, len(jobs) - 1)
                job = jobs.pop(job_idx)
                new_pos = random.randint(0, len(jobs))
                jobs.insert(new_pos, job)
    
    def _swap_factory_assignment(self, solution: Solution):
        """交换工厂分配"""
        if self.n_jobs >= 2:
            job1, job2 = random.sample(range(self.n_jobs), 2)
            factory1, factory2 = solution.factory_assignment[job1], solution.factory_assignment[job2]
            
            solution.factory_assignment[job1] = factory2
            solution.factory_assignment[job2] = factory1
            
            solution.job_sequences[factory1].remove(job1)
            solution.job_sequences[factory1].append(job2)
            solution.job_sequences[factory2].remove(job2)
            solution.job_sequences[factory2].append(job1)
    
    def _reorder_factory_jobs(self, solution: Solution):
        """重排工厂作业"""
        factory_id = random.randint(0, self.n_factories - 1)
        if len(solution.job_sequences[factory_id]) > 1:
            random.shuffle(solution.job_sequences[factory_id])
    
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
        """更新外部档案 - 限制大小避免性能问题"""
        # 限制档案最大大小
        max_archive_size = 200
        
        # 合并当前种群和档案
        all_solutions = self.population + self.external_archive
        
        # 如果解的数量过多，先进行快速筛选
        if len(all_solutions) > max_archive_size * 2:
            # 随机采样保持多样性
            import random
            all_solutions = random.sample(all_solutions, max_archive_size * 2)
        
        pareto_solutions = self._compute_pareto_front(all_solutions)
        
        # 如果Pareto前沿仍然过大，进行进一步筛选
        if len(pareto_solutions) > max_archive_size:
            pareto_solutions = self._select_diverse_archive(pareto_solutions, max_archive_size)
        
        self.external_archive = pareto_solutions
    
    def _compute_pareto_front(self, solutions: List[Solution]) -> List[Solution]:
        """计算帕累托前沿 - 极度宽松版本，保留更多解"""
        if not solutions:
            return []
        
        pareto_front = []
        
        for solution in solutions:
            is_dominated = False
            solutions_to_remove = []
            
            # 极度宽松的支配判断 - 只有明显优势时才认为支配
            for pareto_sol in pareto_front:
                if self._strict_dominates(pareto_sol, solution):
                    is_dominated = True
                    break
                elif self._strict_dominates(solution, pareto_sol):
                    solutions_to_remove.append(pareto_sol)
            
            if not is_dominated:
                # 移除被新解严格支配的解
                for sol in solutions_to_remove:
                    pareto_front.remove(sol)
                pareto_front.append(solution)
        
        return pareto_front
    
    def _strict_dominates(self, sol1: Solution, sol2: Solution) -> bool:
        """严格支配判断 - 需要非常显著的优势才认为支配"""
        # 计算相对优势
        makespan_advantage = (sol2.makespan - sol1.makespan) / max(sol2.makespan, 1)
        tardiness_advantage = (sol2.total_tardiness - sol1.total_tardiness) / max(sol2.total_tardiness, 1)
        
        # 只有在至少一个目标有非常显著优势（>5%），且另一个目标不劣的情况下才认为支配
        significant_threshold = 0.05  # 5%的显著优势阈值（大幅放宽）
        
        makespan_better = makespan_advantage > significant_threshold
        tardiness_better = tardiness_advantage > significant_threshold
        makespan_not_worse = sol1.makespan <= sol2.makespan * (1 + significant_threshold)
        tardiness_not_worse = sol1.total_tardiness <= sol2.total_tardiness * (1 + significant_threshold)
        
        return ((makespan_better and tardiness_not_worse) or 
                (tardiness_better and makespan_not_worse))
    
    def _select_diverse_archive(self, solutions: List[Solution], target_size: int) -> List[Solution]:
        """从解集中选择多样化的子集"""
        if len(solutions) <= target_size:
            return solutions
        
        # 简单的网格化选择策略
        import random
        
        # 按makespan排序
        sorted_by_makespan = sorted(solutions, key=lambda x: x.makespan)
        
        # 分段选择
        # 多维度多样性选择
        selected = []
        
        # 1. 按makespan选择1/3
        makespan_step = len(sorted_by_makespan) // (target_size // 3 + 1)
        for i in range(0, len(sorted_by_makespan), max(1, makespan_step)):
            if len(selected) < target_size // 3:
                selected.append(sorted_by_makespan[i])
        
        # 2. 按tardiness选择1/3
        sorted_by_tardiness = sorted(solutions, key=lambda x: x.total_tardiness)
        tardiness_step = len(sorted_by_tardiness) // (target_size // 3 + 1)
        for i in range(0, len(sorted_by_tardiness), max(1, tardiness_step)):
            if len(selected) < target_size * 2 // 3 and sorted_by_tardiness[i] not in selected:
                selected.append(sorted_by_tardiness[i])
        
        # 3. 随机选择剩余的
        remaining = [sol for sol in solutions if sol not in selected]
        remaining_needed = target_size - len(selected)
        if remaining and remaining_needed > 0:
            selected.extend(random.sample(remaining, min(remaining_needed, len(remaining))))
        
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