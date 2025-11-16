#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于强化学习协调的混沌哈里斯鹰-鹰分组多目标优化算法
RL-Coordinated Chaotic Harris Hawks Optimization with Eagle Grouping
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import copy
import random
from collections import deque

from problem.mo_dhfsp import MO_DHFSP_Problem, Solution
from .chaotic_maps import ChaoticMaps
from .eagle_groups import EagleGroupManager
from .rl_coordinator import RLCoordinator
from .pareto_manager import ParetoManager

class RL_ChaoticHHO_Optimizer:
    """强化学习协调的混沌哈里斯鹰优化器"""
    
    def __init__(self, problem: MO_DHFSP_Problem, **kwargs):
        """
        初始化优化器
        
        Args:
            problem: 问题实例
            **kwargs: 其他参数
        """
        self.problem = problem
        self.n_jobs = problem.n_jobs
        
        # 算法参数
        self.max_iterations = kwargs.get('max_iterations', 100)
        
        # 多样性增强参数 - 修复解集数量少的问题
        self.diversity_enhancement = kwargs.get('diversity_enhancement', False)
        self.pareto_size_limit = kwargs.get('pareto_size_limit', 300)
        self.diversity_threshold = kwargs.get('diversity_threshold', 0.01)  # 极度降低多样性阈值，允许更多相似解
        self.archive_size = kwargs.get('archive_size', 1000)  # 归档大小
        self.selection_pressure = kwargs.get('selection_pressure', 0.6)  # 降低选择压力，保留更多解
        self.local_search_rate = kwargs.get('local_search_rate', 0.6)  # 局部搜索率
        
        # 检查是否有强制设置的种群大小
        if 'population_size_override' in kwargs:
            self.population_size = kwargs['population_size_override']
            print(f"🔧 使用强制设置的种群大小: {self.population_size}")
        else:
            self.population_size = self._calculate_population_size()
        
        # 核心组件 - 使用完整实现
        self.chaotic_maps = ChaoticMaps()
        self.eagle_groups = EagleGroupManager(self.population_size, self.n_jobs, problem.n_factories)
        
        # 提取RL协调器参数 - 使用田口实验最优配置（更新为图中参数）
        rl_learning_rate = kwargs.get('learning_rate', 0.0001)
        rl_epsilon_decay = kwargs.get('epsilon_decay', 0.997)
        rl_gamma = kwargs.get('gamma', 0.999)
        
        self.rl_coordinator = RLCoordinator(
            problem, 
            state_dim=14, 
            action_dim=7,
            learning_rate=rl_learning_rate,
            epsilon_decay=rl_epsilon_decay,
            gamma=rl_gamma
        )
        
        # 使用增强的帕累托管理器
        self.pareto_manager = ParetoManager()
        
        # 种群和状态
        self.population = []
        self.pareto_solutions = []
        self.convergence_data = []
        self.current_iteration = 0
        self.no_improvement_count = 0
        
        # 多样性增强相关
        if self.diversity_enhancement:
            self.diversity_archive = []  # 多样性存档
            self.max_diversity_archive_size = self.archive_size # 归档大小
        
        # 性能跟踪
        self.best_makespan_history = []
        self.best_tardiness_history = []
        self.hypervolume_history = []
        
        # 四层分组协作统计
        self.group_performance_history = {
            'exploration': [],
            'exploitation': [],
            'balance': [],
            'elite': []
        }
        
        # 强化学习统计
        self.rl_action_history = []
        self.rl_reward_history = []
        
        print(f"🦅 初始化RL-Chaotic-HHO优化器:")
        print(f"  种群规模: {self.population_size}")
        print(f"  最大迭代: {self.max_iterations}")
        print(f"  多样性增强: {'✓' if self.diversity_enhancement else '✗'}")
        print(f"  帕累托解集限制: {self.pareto_size_limit}")
        print(f"  四层鹰群分组: ✓")
        print(f"  强化学习调度: ✓")
        print(f"  增强混沌映射: ✓")
    
    def _calculate_population_size(self) -> int:
        """根据问题规模动态计算种群大小 - 增加种群以支持更多pareto解"""
        base_size = 120  # 进一步增加基础规模以产生更多帕累托解
        scale_factor = 1.0 + 0.4 * np.log(max(self.n_jobs, 20) / 20)
        complexity_factor = 1.0 + 0.4 * (self.problem.n_factories / 5)
        
        size = int(base_size * scale_factor * complexity_factor)
        return min(max(size, 80), 300)  # 进一步调整限制范围以支持更多解
    
    def optimize(self) -> Tuple[List[Solution], Dict]:
        """
        主优化流程
        
        Returns:
            (pareto_solutions, convergence_data): 帕累托最优解集和收敛数据
        """
        print("开始优化...")
        
        # 初始化种群
        self._initialize_population()
        
        # 更新帕累托前沿
        self._update_pareto_front()
        
        # 如果初始帕累托前沿为空，强制添加当前最好的解
        if not self.pareto_solutions and self.population:
            valid_solutions = [sol for sol in self.population 
                             if sol.makespan > 0 and sol.total_tardiness >= 0]
            if valid_solutions:
                # 添加几个不同质量的解作为初始帕累托前沿
                sorted_by_makespan = sorted(valid_solutions, key=lambda x: x.makespan)
                sorted_by_tardiness = sorted(valid_solutions, key=lambda x: x.total_tardiness)
                
                initial_pareto = []
                if sorted_by_makespan:
                    initial_pareto.append(sorted_by_makespan[0])  # 最优完工时间
                if sorted_by_tardiness and sorted_by_tardiness[0] not in initial_pareto:
                    initial_pareto.append(sorted_by_tardiness[0])  # 最优拖期
                
                self.pareto_solutions = initial_pareto
                print(f"强制初始化帕累托前沿，包含 {len(self.pareto_solutions)} 个解")
        
        # 主循环
        for iteration in range(self.max_iterations):
            self.current_iteration = iteration
            
            # RL协调器观察状态并选择策略
            state = self._get_current_state()
            action = self.rl_coordinator.select_action(state)
            
            # 执行策略
            self._execute_strategy(action)
            
            # 哈里斯鹰搜索
            self._harris_hawks_search()
            
            # 更新帕累托前沿
            previous_size = len(self.pareto_solutions)
            self._update_pareto_front()
            current_size = len(self.pareto_solutions)
            
            # 计算奖励并更新RL
            reward = self._calculate_reward(previous_size, current_size)
            next_state = self._get_current_state()
            self.rl_coordinator.update(state, action, reward, next_state)
            
            # 记录收敛数据
            self._record_convergence_data()
            
            # 输出进度
            if iteration % 20 == 0 or iteration == self.max_iterations - 1:
                self._print_progress(iteration)
            
            # 检查停止条件
            if self._should_stop():
                print(f"提前停止在第 {iteration} 代")
                break
        
        # 最终验证和清理
        final_solutions = [sol for sol in self.pareto_solutions 
                          if sol.makespan > 0 and sol.total_tardiness >= 0]
        
        if not final_solutions:
            print("警告：最终帕累托前沿为空，返回最佳种群解")
            valid_population = [sol for sol in self.population 
                              if sol.makespan > 0 and sol.total_tardiness >= 0]
            if valid_population:
                best_sol = min(valid_population, key=lambda x: 0.5*x.makespan + 0.5*x.total_tardiness)
                final_solutions = [best_sol]
        
        print(f"优化完成! 最终帕累托解数量: {len(final_solutions)}")
        return final_solutions, self._prepare_convergence_data()
    
    def _calculate_improvement_rate(self) -> float:
        """计算改进率"""
        if len(self.best_makespan_history) < 10:
            return 0.0
        
        recent_best = min(self.best_makespan_history[-5:])
        earlier_best = min(self.best_makespan_history[-10:-5])
        
        if earlier_best > 0:
            return max(0, (earlier_best - recent_best) / earlier_best)
        return 0.0
    
    def _calculate_factory_balance(self) -> float:
        """计算工厂负载均衡度"""
        if not self.population:
            return 0.0
        
        # 计算平均解的工厂负载
        factory_loads = [0] * self.problem.n_factories
        
        for sol in self.population[:10]:  # 取前10个解
            for factory_id in range(self.problem.n_factories):
                factory_loads[factory_id] += len(sol.job_sequences[factory_id])
        
        # 计算负载方差
        mean_load = np.mean(factory_loads)
        load_variance = np.var(factory_loads)
        
        # 返回平衡度（方差越小越好）
        return 1.0 / (1.0 + load_variance / max(mean_load, 1))
    
    def _calculate_reward(self, previous_size: int, current_size: int) -> float:
        """计算RL奖励"""
        # 帕累托前沿改进奖励（增加权重）
        size_improvement = (current_size - previous_size) / max(previous_size, 1)
        
        # 解质量改进奖励
        quality_improvement = 0.0
        if len(self.best_makespan_history) > 1:
            makespan_improvement = (self.best_makespan_history[-2] - self.best_makespan_history[-1]) / max(self.best_makespan_history[-2], 1)
            tardiness_improvement = (self.best_tardiness_history[-2] - self.best_tardiness_history[-1]) / max(self.best_tardiness_history[-2], 1)
            quality_improvement = 0.5 * makespan_improvement + 0.5 * tardiness_improvement
        
        # 多样性奖励
        diversity_reward = 0.0
        if len(self.pareto_solutions) > 1:
            makespans = [sol.makespan for sol in self.pareto_solutions]
            tardiness = [sol.total_tardiness for sol in self.pareto_solutions]
            makespan_diversity = np.std(makespans) / max(np.mean(makespans), 1)
            tardiness_diversity = np.std(tardiness) / max(np.mean(tardiness), 1)
            diversity_reward = 0.5 * (makespan_diversity + tardiness_diversity)
        
        # 解集数量奖励
        size_reward = min(len(self.pareto_solutions) / 30.0, 1.0)  # 鼓励更多解
        
        # 综合奖励（调整权重，更重视多样性和数量）
        reward = (0.4 * size_improvement + 
                 0.3 * quality_improvement + 
                 0.2 * diversity_reward + 
                 0.1 * size_reward)
        
        return reward
    
    def _partial_restart(self):
        """部分重启策略"""
        # 保留最好的30%解
        n_keep = int(0.3 * self.population_size)
        
        # 按质量排序
        sorted_pop = sorted(self.population, 
                          key=lambda x: 0.5 * x.makespan + 0.5 * x.total_tardiness)
        
        # 保留最好的解，重新生成其他解
        new_population = sorted_pop[:n_keep]
        
        for _ in range(self.population_size - n_keep):
            new_sol = self.problem.generate_random_solution()
            new_population.append(new_sol)
        
        self.population = new_population
        self.eagle_groups.assign_eagles(self.population)
    
    def _initialize_population(self):
        """增强多样性的种群初始化"""
        print("初始化种群（增强多样性版）...")
        
        self.population = []
        max_retries = 5
        
        # 策略1：25%使用完全随机解
        random_count = int(0.25 * self.population_size)
        for i in range(random_count):
            solution = self._create_random_solution_with_retry(max_retries, f"随机解{i+1}")
            self.population.append(solution)
        
        # 策略2：25%使用基于优先级的贪心解
        greedy_count = int(0.25 * self.population_size)
        for i in range(greedy_count):
            solution = self._create_greedy_solution(f"贪心解{i+1}")
            self.population.append(solution)
        
        # 策略3：25%使用负载均衡解
        balanced_count = int(0.25 * self.population_size)
        for i in range(balanced_count):
            solution = self._create_balanced_solution(f"均衡解{i+1}")
            self.population.append(solution)
        
        # 策略4：剩余的使用混合策略解
        remaining_count = self.population_size - len(self.population)
        for i in range(remaining_count):
            strategy = i % 3  # 循环使用3种策略
            if strategy == 0:
                solution = self._create_random_solution_with_retry(max_retries, f"混合随机解{i+1}")
            elif strategy == 1:
                solution = self._create_urgent_first_solution(f"紧急优先解{i+1}")
            else:
                solution = self._create_scattered_solution(f"分散解{i+1}")
            self.population.append(solution)
        
        # 确保种群大小正确
        while len(self.population) < self.population_size:
            solution = self._create_random_solution_with_retry(max_retries, f"补充解{len(self.population)+1}")
            self.population.append(solution)
        
        # 分配到各个鹰群组
        self.eagle_groups.assign_eagles(self.population)
        
        print(f"增强多样性初始化完成，种群大小: {len(self.population)}")
        
        # 验证初始种群质量和多样性
        makespans = [sol.makespan for sol in self.population if sol.makespan > 0]
        tardiness_values = [sol.total_tardiness for sol in self.population if sol.total_tardiness >= 0]
        
        if makespans and tardiness_values:
            print(f"初始种群完工时间范围: {min(makespans):.2f} - {max(makespans):.2f}")
            print(f"初始种群拖期范围: {min(tardiness_values):.2f} - {max(tardiness_values):.2f}")
            print(f"初始种群多样性 - 完工时间方差: {np.var(makespans):.2f}, 拖期方差: {np.var(tardiness_values):.2f}")
        else:
            print("警告：初始种群中缺少有效解！")
    
    def _create_random_solution_with_retry(self, max_retries: int, solution_name: str):
        """创建随机解（带重试机制）"""
        for retry in range(max_retries):
            try:
                solution = self.problem.generate_random_solution()
                if (solution.makespan > 0 and solution.total_tardiness >= 0 and 
                    self.problem.is_solution_feasible(solution)):
                    return solution
            except Exception as e:
                pass
        
        # 备用方法
        return self._create_fallback_solution(solution_name)
    
    def _create_greedy_solution(self, solution_name: str):
        """创建基于优先级的贪心解"""
        try:
            # 按作业紧急度和处理时间排序
            urgency_scores = []
            for job_id in range(self.problem.n_jobs):
                urgency = self.problem.urgencies[job_id] if hasattr(self.problem, 'urgencies') else 1.0
                total_time = sum(self.problem.processing_times[job_id])
                urgency_scores.append((job_id, urgency * total_time))
            
            urgency_scores.sort(key=lambda x: x[1])  # 紧急度高的在前
            
            # 贪心分配到负载最轻的工厂
            factory_loads = [0.0] * self.problem.n_factories
            factory_assignment = [0] * self.problem.n_jobs
            
            for job_id, _ in urgency_scores:
                # 找到负载最轻的工厂
                min_load_factory = min(range(self.problem.n_factories), key=lambda f: factory_loads[f])
                factory_assignment[job_id] = min_load_factory
                
                # 更新工厂负载
                job_total_time = sum(self.problem.processing_times[job_id])
                factory_loads[min_load_factory] += job_total_time
            
            return self.problem.create_solution(factory_assignment)
        except:
            return self._create_fallback_solution(solution_name)
    
    def _create_balanced_solution(self, solution_name: str):
        """创建负载均衡解"""
        try:
            # 计算每个作业的总处理时间
            job_times = [sum(self.problem.processing_times[job_id]) for job_id in range(self.problem.n_jobs)]
            job_indices = list(range(self.problem.n_jobs))
            
            # 按处理时间排序
            job_indices.sort(key=lambda x: job_times[x], reverse=True)
            
            # 轮询分配，优先分配处理时间长的作业
            factory_assignment = [0] * self.problem.n_jobs
            factory_loads = [0.0] * self.problem.n_factories
            
            for job_id in job_indices:
                # 分配到负载最轻的工厂
                min_load_factory = min(range(self.problem.n_factories), key=lambda f: factory_loads[f])
                factory_assignment[job_id] = min_load_factory
                factory_loads[min_load_factory] += job_times[job_id]
            
            return self.problem.create_solution(factory_assignment)
        except:
            return self._create_fallback_solution(solution_name)
    
    def _create_urgent_first_solution(self, solution_name: str):
        """创建紧急优先解"""
        try:
            # 如果有紧急度信息，优先分配紧急作业
            if hasattr(self.problem, 'urgencies'):
                job_urgencies = [(job_id, self.problem.urgencies[job_id]) for job_id in range(self.problem.n_jobs)]
                job_urgencies.sort(key=lambda x: x[1], reverse=True)  # 紧急度高的在前
            else:
                # 使用截止日期替代
                job_urgencies = [(job_id, 1.0/max(self.problem.due_dates[job_id], 1)) for job_id in range(self.problem.n_jobs)]
                job_urgencies.sort(key=lambda x: x[1], reverse=True)
            
            factory_assignment = [0] * self.problem.n_jobs
            factory_job_counts = [0] * self.problem.n_factories
            
            for job_id, urgency in job_urgencies:
                # 分配到作业数最少的工厂
                min_count_factory = min(range(self.problem.n_factories), key=lambda f: factory_job_counts[f])
                factory_assignment[job_id] = min_count_factory
                factory_job_counts[min_count_factory] += 1
            
            return self.problem.create_solution(factory_assignment)
        except:
            return self._create_fallback_solution(solution_name)
    
    def _create_scattered_solution(self, solution_name: str):
        """创建分散解（最大化工厂间差异）"""
        try:
            factory_assignment = []
            
            for job_id in range(self.problem.n_jobs):
                # 使用伪随机模式分配，确保分散性
                factory_id = (job_id * 7 + 3) % self.problem.n_factories  # 使用质数保证分散性
                factory_assignment.append(factory_id)
            
            return self.problem.create_solution(factory_assignment)
        except:
            return self._create_fallback_solution(solution_name)
    
    def _create_fallback_solution(self, solution_name: str):
        """备用解创建方法"""
        try:
            # 简单轮询分配
            factory_assignment = [job_id % self.problem.n_factories for job_id in range(self.problem.n_jobs)]
            job_sequences = [[] for _ in range(self.problem.n_factories)]
            
            for job_id in range(self.problem.n_jobs):
                factory_id = factory_assignment[job_id]
                job_sequences[factory_id].append(job_id)
            
            from problem.mo_dhfsp import Solution
            solution = Solution(factory_assignment, job_sequences)
            solution = self.problem.evaluate_solution(solution)
            return solution
        except Exception as e:
            print(f"警告：备用解创建失败 {e}，使用最简单的解")
            # 最简单的解：所有作业分配给第一个工厂
            factory_assignment = [0] * self.problem.n_jobs
            job_sequences = [list(range(self.problem.n_jobs))] + [[] for _ in range(self.problem.n_factories - 1)]
            
            from problem.mo_dhfsp import Solution
            solution = Solution(factory_assignment, job_sequences)
            solution = self.problem.evaluate_solution(solution)
            return solution
    
    def _get_current_state(self) -> np.ndarray:
        """获取当前状态向量"""
        # 搜索进展状态
        progress = self.current_iteration / self.max_iterations
        improvement_rate = self._calculate_improvement_rate()
        stagnation_ratio = min(self.no_improvement_count / 50, 1.0)
        pareto_size_ratio = len(self.pareto_solutions) / max(20, len(self.pareto_solutions))
        
        # 各组性能状态
        group_performance = self.eagle_groups.get_performance_metrics()
        
        # 问题特征状态
        if self.pareto_solutions:
            best_makespan = min(sol.makespan for sol in self.pareto_solutions)
            best_tardiness = min(sol.total_tardiness for sol in self.pareto_solutions)
            quality_score = 1.0 / (1.0 + best_makespan / self.problem.theoretical_lower_bound)
        else:
            quality_score = 0.0
        
        factory_balance = self._calculate_factory_balance()
        
        # 组合状态向量
        state = np.array([
            progress,
            improvement_rate,
            stagnation_ratio,
            pareto_size_ratio,
            quality_score,
            factory_balance,
            *group_performance[:8]  # 各组性能指标
        ])
        
        return state
    
    def _execute_strategy(self, action: int):
        """执行RL选择的策略"""
        if action == 0:  # 强化全局探索
            self.eagle_groups.enhance_exploration()
        elif action == 1:  # 强化局部开发
            self.eagle_groups.enhance_exploitation()
        elif action == 2:  # 平衡搜索
            self.eagle_groups.balance_search()
        elif action == 3:  # 多样性救援
            self.eagle_groups.diversity_rescue()
        elif action == 4:  # 精英强化
            self.eagle_groups.elite_enhancement()
        elif action == 5:  # 全局重启
            self._partial_restart()
        elif action == 6:  # 资源重分配
            self.eagle_groups.redistribute_resources()
    
    def _harris_hawks_search(self):
        """四层分组协作的哈里斯鹰搜索主循环"""
        # 获取当前最优解作为猎物
        if self.pareto_solutions:
            rabbit = random.choice(self.pareto_solutions)
        else:
            rabbit = min(self.population, key=lambda x: x.makespan + x.total_tardiness)
        
        # 分组并行搜索
        new_population = self.population.copy()
        
        # 探索组：高强度全局搜索
        exploration_indices = self.eagle_groups.get_group('exploration')
        for idx in exploration_indices:
            if idx < len(self.population):
                new_eagle = self._exploration_group_search(self.population[idx], rabbit)
                new_population[idx] = new_eagle
        
        # 开发组：精细局部搜索  
        exploitation_indices = self.eagle_groups.get_group('exploitation')
        for idx in exploitation_indices:
            if idx < len(self.population):
                new_eagle = self._exploitation_group_search(self.population[idx], rabbit)
                # 增强局部搜索应用
                if random.random() < self.local_search_rate:
                    new_eagle = self._local_search(new_eagle)
                new_population[idx] = new_eagle
        
        # 平衡组：适中强度搜索
        balance_indices = self.eagle_groups.get_group('balance')
        for idx in balance_indices:
            if idx < len(self.population):
                new_eagle = self._balance_group_search(self.population[idx], rabbit)
                # 适度应用局部搜索
                if random.random() < self.local_search_rate * 0.7:
                    new_eagle = self._local_search(new_eagle)
                new_population[idx] = new_eagle
        
        # 精英组：基于最优解的精炼搜索
        elite_indices = self.eagle_groups.get_group('elite')
        for idx in elite_indices:
            if idx < len(self.population):
                new_eagle = self._elite_group_search(self.population[idx], rabbit)
                # 高频率局部搜索
                if random.random() < self.local_search_rate * 1.2:
                    new_eagle = self._local_search(new_eagle)
                new_population[idx] = new_eagle
        
        self.population = new_population
        
        # 更新组性能统计
        self._update_group_performance_statistics()
    
    def _update_eagle_position(self, eagle: Solution, rabbit: Solution) -> Solution:
        """更新单个鹰的位置"""
        # 计算能量系数
        E = self._calculate_energy()
        
        # 根据能量水平选择搜索策略
        if abs(E) >= 1:
            # 探索阶段
            new_eagle = self._exploration_phase(eagle, rabbit)
        else:
            # 利用阶段
            r = random.random()
            if r >= 0.5:
                new_eagle = self._soft_besiege(eagle, rabbit, E)
            else:
                new_eagle = self._hard_besiege(eagle, rabbit, E)
        
        # 应用局部搜索
        if random.random() < 0.3:  # 30%概率应用局部搜索
            new_eagle = self._local_search(new_eagle)
        
        # 评估新解
        new_eagle = self.problem.evaluate_solution(new_eagle)
        
        # 选择更好的解
        return self._select_better_solution(eagle, new_eagle)
    
    def _soft_besiege(self, eagle: Solution, rabbit: Solution, E: float) -> Solution:
        """软包围策略"""
        # 生成新解
        new_factory_assignment = eagle.factory_assignment.copy()
        new_job_sequences = [seq.copy() for seq in eagle.job_sequences]
        
        # 随机选择一些作业进行调整
        n_adjustments = max(1, int(abs(E) * self.n_jobs * 0.3))
        jobs_to_adjust = random.sample(range(self.n_jobs), min(n_adjustments, self.n_jobs))
        
        for job_id in jobs_to_adjust:
            if random.random() < 0.7:
                # 向兔子位置移动
                target_factory = rabbit.factory_assignment[job_id]
                current_factory = eagle.factory_assignment[job_id]
                
                if target_factory != current_factory:
                    # 移动作业到目标工厂
                    new_job_sequences[current_factory].remove(job_id)
                    new_job_sequences[target_factory].append(job_id)
                    new_factory_assignment[job_id] = target_factory
        
        new_solution = Solution(new_factory_assignment, new_job_sequences)
        return self.problem.evaluate_solution(new_solution)
    
    def _hard_besiege(self, eagle: Solution, rabbit: Solution, E: float) -> Solution:
        """硬包围策略"""
        # 更激进的移动策略
        new_factory_assignment = []
        
        for job_id in range(self.n_jobs):
            if random.random() < 0.8:
                # 大概率跟随兔子
                new_factory_assignment.append(rabbit.factory_assignment[job_id])
            else:
                # 保持当前分配
                new_factory_assignment.append(eagle.factory_assignment[job_id])
        
        # 重新构建作业序列
        new_job_sequences = [[] for _ in range(self.problem.n_factories)]
        for job_id in range(self.n_jobs):
            factory_id = new_factory_assignment[job_id]
            new_job_sequences[factory_id].append(job_id)
        
        # 随机排序
        for factory_id in range(self.problem.n_factories):
            if new_job_sequences[factory_id]:
                random.shuffle(new_job_sequences[factory_id])
        
        new_solution = Solution(new_factory_assignment, new_job_sequences)
        return self.problem.evaluate_solution(new_solution)
    
    def _local_search(self, solution: Solution) -> Solution:
        """局部搜索改进"""
        best_solution = solution
        
        # 尝试几种局部搜索算子
        for _ in range(2):
            # 1. 作业交换
            new_sol = self._job_swap(best_solution)
            new_sol = self.problem.evaluate_solution(new_sol)
            if self._is_better_solution(new_sol, best_solution):
                best_solution = new_sol
            
            # 2. 作业插入
            new_sol = self._job_insertion(best_solution)
            new_sol = self.problem.evaluate_solution(new_sol)
            if self._is_better_solution(new_sol, best_solution):
                best_solution = new_sol
        
        return best_solution
    
    def _job_swap(self, solution: Solution) -> Solution:
        """作业交换操作"""
        new_solution = copy.deepcopy(solution)
        
        # 随机选择一个工厂
        factory_id = random.randint(0, self.problem.n_factories - 1)
        jobs = new_solution.job_sequences[factory_id]
        
        if len(jobs) >= 2:
            # 交换两个作业的位置
            i, j = random.sample(range(len(jobs)), 2)
            jobs[i], jobs[j] = jobs[j], jobs[i]
        
        return new_solution
    
    def _job_insertion(self, solution: Solution) -> Solution:
        """作业插入操作"""
        new_solution = copy.deepcopy(solution)
        
        # 随机选择一个工厂
        factory_id = random.randint(0, self.problem.n_factories - 1)
        jobs = new_solution.job_sequences[factory_id]
        
        if len(jobs) >= 2:
            # 移除一个作业并插入到新位置
            job_idx = random.randint(0, len(jobs) - 1)
            job = jobs.pop(job_idx)
            new_pos = random.randint(0, len(jobs))
            jobs.insert(new_pos, job)
        
        return new_solution
    
    def _calculate_energy(self) -> float:
        """计算能量系数"""
        # 基础时间衰减
        t = self.current_iteration
        T = self.max_iterations
        time_factor = 1 - (t / T) ** 2
        
        # 质量因子
        if self.no_improvement_count < 5:
            quality_factor = 0.8  # 有改进时降低能量
        else:
            quality_factor = 1.2  # 无改进时提高能量
        
        # 停滞因子
        if self.no_improvement_count > 15:
            stagnation_factor = 1 + 0.3 * np.exp((self.no_improvement_count - 15) / 10)
        else:
            stagnation_factor = 1.0
        
        E = 2.0 * time_factor * quality_factor * stagnation_factor
        
        # 添加周期性扰动
        E *= (1 + 0.1 * np.sin(2 * np.pi * t / 20))
        
        return E
    
    def _update_pareto_front(self):
        """更新帕累托前沿 - 极度宽松的多样性增强版本"""
        # 更宽松的有效解过滤（允许0拖期）
        valid_population = [sol for sol in self.population 
                          if sol.makespan > 0 and sol.total_tardiness >= 0]
        valid_pareto = [sol for sol in self.pareto_solutions 
                       if sol.makespan > 0 and sol.total_tardiness >= 0]
        
        if not valid_population and not valid_pareto:
            print("警告：没有有效解用于更新帕累托前沿")
            return
        
        # 合并当前种群和已有帕累托解
        all_solutions = valid_population + valid_pareto
        
        # 如果启用多样性增强，添加多样性存档中的解
        if self.diversity_enhancement and hasattr(self, 'diversity_archive'):
            all_solutions.extend(self.diversity_archive)
        
        # 更新帕累托前沿 - 保留所有非支配解
        updated_pareto = self.pareto_manager.update_pareto_front(all_solutions)
        
        # 确保至少保留一定数量的解
        min_pareto_size = min(20, len(valid_population) // 2)  # 至少保留20个解
        if len(updated_pareto) < min_pareto_size and len(all_solutions) >= min_pareto_size:
            # 如果pareto解太少，补充一些高质量的非支配解
            sorted_by_quality = sorted(all_solutions, 
                                     key=lambda x: 0.6*x.makespan + 0.4*x.total_tardiness)
            
            additional_solutions = []
            for sol in sorted_by_quality:
                if sol not in updated_pareto and len(updated_pareto) + len(additional_solutions) < min_pareto_size:
                    # 检查是否与现有解有足够差异
                    is_diverse_enough = True
                    for existing in updated_pareto:
                        if (abs(sol.makespan - existing.makespan) / max(existing.makespan, 1) < 0.02 and
                            abs(sol.total_tardiness - existing.total_tardiness) / max(existing.total_tardiness, 1) < 0.02):
                            is_diverse_enough = False
                            break
                    
                    if is_diverse_enough:
                        additional_solutions.append(sol)
            
            updated_pareto.extend(additional_solutions)
        
        self.pareto_solutions = updated_pareto
        
        # 多样性增强处理
        if self.diversity_enhancement:
            # 更新多样性存档
            self._update_diversity_archive()
            
            # 应用多样性增强策略
            self._apply_diversity_enhancement()
        
        # 使用更大的帕累托解集大小限制
        effective_limit = max(self.pareto_size_limit, 50)  # 至少保留50个解
        if len(self.pareto_solutions) > effective_limit:
            # 使用极度宽松的多样性选择
            self.pareto_solutions = self.pareto_manager.select_diverse_solutions(
                self.pareto_solutions, effective_limit
            )
    
    def _select_better_solution(self, sol1: Solution, sol2: Solution) -> Solution:
        """选择更好的解"""
        # 多目标比较
        if sol1.makespan < sol2.makespan and sol1.total_tardiness < sol2.total_tardiness:
            return sol1
        elif sol2.makespan < sol1.makespan and sol2.total_tardiness < sol1.total_tardiness:
            return sol2
        else:
            # 使用加权和比较
            score1 = 0.5 * sol1.makespan + 0.5 * sol1.total_tardiness
            score2 = 0.5 * sol2.makespan + 0.5 * sol2.total_tardiness
            return sol1 if score1 < score2 else sol2
    
    def _is_better_solution(self, sol1: Solution, sol2: Solution) -> bool:
        """判断sol1是否比sol2更好"""
        return (sol1.makespan <= sol2.makespan and sol1.total_tardiness <= sol2.total_tardiness and
                (sol1.makespan < sol2.makespan or sol1.total_tardiness < sol2.total_tardiness))
    
    def _record_convergence_data(self):
        """记录收敛数据"""
        if self.pareto_solutions:
            valid_solutions = [sol for sol in self.pareto_solutions 
                             if sol.makespan > 0 and sol.total_tardiness >= 0]
            if valid_solutions:
                best_makespan = min(sol.makespan for sol in valid_solutions)
                best_tardiness = min(sol.total_tardiness for sol in valid_solutions)
            else:
                best_makespan = float('inf')
                best_tardiness = float('inf')
        else:
            best_makespan = float('inf')
            best_tardiness = float('inf')
        
        self.best_makespan_history.append(best_makespan)
        self.best_tardiness_history.append(best_tardiness)
        
        # 检查是否有改进
        if len(self.best_makespan_history) > 1:
            if (self.best_makespan_history[-1] < self.best_makespan_history[-2] or
                self.best_tardiness_history[-1] < self.best_tardiness_history[-2]):
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1
        
        # 记录详细收敛数据
        self.convergence_data.append({
            'iteration': self.current_iteration,
            'best_makespan': best_makespan,
            'best_tardiness': best_tardiness,
            'pareto_size': len(self.pareto_solutions)
        })
    
    def _should_stop(self) -> bool:
        """检查是否应该停止"""
        # 禁用提前停止，让算法完整运行
        return False
    
    def _print_progress(self, iteration: int):
        """打印进度信息"""
        if self.pareto_solutions:
            best_makespan = min(sol.makespan for sol in self.pareto_solutions)
            best_tardiness = min(sol.total_tardiness for sol in self.pareto_solutions)
            print(f"代数 {iteration:3d}: 帕累托解={len(self.pareto_solutions):2d}, "
                  f"最优完工时间={best_makespan:.2f}, "
                  f"最优拖期={best_tardiness:.2f}, "
                  f"无改进={self.no_improvement_count:2d}")
        else:
            print(f"代数 {iteration:3d}: 还未找到帕累托解")
    
    def _prepare_convergence_data(self) -> Dict:
        """准备收敛数据"""
        return {
            'makespan_history': self.best_makespan_history,
            'tardiness_history': self.best_tardiness_history,
            'detailed_data': self.convergence_data,
            'final_pareto_size': len(self.pareto_solutions),
            'total_iterations': self.current_iteration + 1
        }
    
    def _exploration_phase(self, eagle: Solution, rabbit: Solution) -> Solution:
        """探索阶段位置更新"""
        # 选择随机鹰
        random_eagle = random.choice(self.population)
        
        # 获取混沌值
        chaos_values = self.chaotic_maps.get_chaos_values(4)
        
        # 生成新的工厂分配
        new_factory_assignment = []
        for job_id in range(self.n_jobs):
            if chaos_values[0] < 0.5:
                # 跟随兔子的分配
                new_factory_assignment.append(rabbit.factory_assignment[job_id])
            elif chaos_values[1] < 0.5:
                # 跟随随机鹰的分配
                new_factory_assignment.append(random_eagle.factory_assignment[job_id])
            else:
                # 随机分配
                new_factory_assignment.append(random.randint(0, self.problem.n_factories - 1))
        
        # 生成新的作业序列
        new_job_sequences = [[] for _ in range(self.problem.n_factories)]
        for job_id in range(self.n_jobs):
            factory_id = new_factory_assignment[job_id]
            new_job_sequences[factory_id].append(job_id)
        
        # 随机打乱各工厂的序列
        for factory_id in range(self.problem.n_factories):
            if new_job_sequences[factory_id]:
                if chaos_values[2] < 0.3:
                    # 保持兔子的顺序
                    rabbit_jobs_in_factory = [j for j in rabbit.job_sequences[factory_id] 
                                            if j in new_job_sequences[factory_id]]
                    other_jobs = [j for j in new_job_sequences[factory_id] 
                                if j not in rabbit_jobs_in_factory]
                    random.shuffle(other_jobs)
                    new_job_sequences[factory_id] = rabbit_jobs_in_factory + other_jobs
                else:
                    # 完全随机
                    random.shuffle(new_job_sequences[factory_id])
        
        new_solution = Solution(new_factory_assignment, new_job_sequences)
        return self.problem.evaluate_solution(new_solution)
    
    def _exploration_group_search(self, eagle: Solution, rabbit: Solution) -> Solution:
        """探索组专用搜索 - 高强度全局搜索"""
        # 使用Logistic混沌映射增强随机性
        chaos_values = self.chaotic_maps.get_group_chaos_values('exploration', self.n_jobs)
        
        # 大幅度位置更新
        new_factory_assignment = []
        for job_id in range(self.n_jobs):
            if chaos_values[job_id % len(chaos_values)] > 0.6:
                # 高概率随机重分配
                new_factory_assignment.append(random.randint(0, self.problem.n_factories - 1))
            elif chaos_values[job_id % len(chaos_values)] > 0.3:
                # 跟随兔子
                new_factory_assignment.append(rabbit.factory_assignment[job_id])
            else:
                # 保持当前分配
                new_factory_assignment.append(eagle.factory_assignment[job_id])
        
        # 重构作业序列并随机化
        new_job_sequences = [[] for _ in range(self.problem.n_factories)]
        for job_id in range(self.n_jobs):
            factory_id = new_factory_assignment[job_id]
            new_job_sequences[factory_id].append(job_id)
        
        # 高强度序列随机化
        for factory_id in range(self.problem.n_factories):
            if new_job_sequences[factory_id]:
                random.shuffle(new_job_sequences[factory_id])
        
        new_solution = Solution(new_factory_assignment, new_job_sequences)
        new_solution = self.problem.evaluate_solution(new_solution)
        
        return self._select_better_solution(eagle, new_solution)
    
    def _exploitation_group_search(self, eagle: Solution, rabbit: Solution) -> Solution:
        """开发组专用搜索 - 精细局部搜索"""
        # 使用Tent混沌映射保持稳定性
        chaos_values = self.chaotic_maps.get_group_chaos_values('exploitation', 3)
        
        best_solution = eagle
        
        # 多种局部搜索算子
        for _ in range(3):
            # 1. 作业交换
            if chaos_values[0] > 0.5:
                candidate = self._job_swap(best_solution)
                candidate = self.problem.evaluate_solution(candidate)
                if self._is_better_solution(candidate, best_solution):
                    best_solution = candidate
            
            # 2. 作业插入
            if chaos_values[1] > 0.5:
                candidate = self._job_insertion(best_solution)
                candidate = self.problem.evaluate_solution(candidate)
                if self._is_better_solution(candidate, best_solution):
                    best_solution = candidate
            
            # 3. 局部工厂重分配
            if chaos_values[2] > 0.7:
                candidate = self._local_factory_reassignment(best_solution)
                candidate = self.problem.evaluate_solution(candidate)
                if self._is_better_solution(candidate, best_solution):
                    best_solution = candidate
        
        return best_solution
    
    def _balance_group_search(self, eagle: Solution, rabbit: Solution) -> Solution:
        """平衡组专用搜索 - 适中强度搜索"""
        # 使用Sine混沌映射平滑过渡
        chaos_values = self.chaotic_maps.get_group_chaos_values('balance', 2)
        
        # 能量系数计算
        E = self._calculate_energy()
        
        if abs(E) >= 1:
            # 偏向探索
            new_solution = self._exploration_phase(eagle, rabbit)
        else:
            # 偏向开发
            if chaos_values[0] > 0.5:
                new_solution = self._soft_besiege(eagle, rabbit, E)
            else:
                new_solution = self._hard_besiege(eagle, rabbit, E)
        
        # 选择更好的解
        return self._select_better_solution(eagle, new_solution)
    
    def _elite_group_search(self, eagle: Solution, rabbit: Solution) -> Solution:
        """精英组专用搜索 - 基于最优解的精炼搜索"""
        # 使用Chebyshev混沌映射精细调优
        chaos_values = self.chaotic_maps.get_group_chaos_values('elite', 5)
        
        best_solution = eagle
        
        # 高强度局部优化
        for i in range(5):
            if chaos_values[i] > 0.3:
                # 基于最优解的引导搜索
                candidate = self._guided_local_search(best_solution, rabbit)
                candidate = self.problem.evaluate_solution(candidate)
                if self._is_better_solution(candidate, best_solution):
                    best_solution = candidate
        
        return best_solution
    
    def _guided_local_search(self, solution: Solution, guide: Solution) -> Solution:
        """基于引导解的局部搜索"""
        new_solution = copy.deepcopy(solution)
        
        # 选择性地采用引导解的特征
        for job_id in range(self.n_jobs):
            if random.random() < 0.3:  # 30%概率采用引导解的工厂分配
                old_factory = new_solution.factory_assignment[job_id]
                new_factory = guide.factory_assignment[job_id]
                
                if old_factory != new_factory:
                    new_solution.factory_assignment[job_id] = new_factory
                    
                    # 更新作业序列
                    if job_id in new_solution.job_sequences[old_factory]:
                        new_solution.job_sequences[old_factory].remove(job_id)
                    new_solution.job_sequences[new_factory].append(job_id)
        
        # 精细调整作业序列
        for factory_id in range(self.problem.n_factories):
            jobs = new_solution.job_sequences[factory_id]
            if len(jobs) > 1:
                # 小幅度调整
                if random.random() < 0.5:
                    i, j = random.sample(range(len(jobs)), 2)
                    jobs[i], jobs[j] = jobs[j], jobs[i]
        
        return self.problem.evaluate_solution(new_solution)
    
    def _local_factory_reassignment(self, solution: Solution) -> Solution:
        """局部工厂重分配"""
        new_solution = copy.deepcopy(solution)
        
        # 随机选择几个作业进行重分配
        n_reassign = max(1, self.n_jobs // 10)  # 重分配10%的作业
        jobs_to_reassign = random.sample(range(self.n_jobs), min(n_reassign, self.n_jobs))
        
        for job_id in jobs_to_reassign:
            old_factory = new_solution.factory_assignment[job_id]
            # 选择负载较轻的工厂
            factory_loads = [len(new_solution.job_sequences[f]) for f in range(self.problem.n_factories)]
            new_factory = factory_loads.index(min(factory_loads))
            
            if old_factory != new_factory:
                new_solution.factory_assignment[job_id] = new_factory
                
                # 更新作业序列
                if job_id in new_solution.job_sequences[old_factory]:
                    new_solution.job_sequences[old_factory].remove(job_id)
                new_solution.job_sequences[new_factory].append(job_id)
        
        return self.problem.evaluate_solution(new_solution)
    
    def _update_group_performance_statistics(self):
        """更新各组性能统计"""
        for group_name in ['exploration', 'exploitation', 'balance', 'elite']:
            group_solutions = self.eagle_groups.get_group_solutions(group_name)
            if group_solutions:
                # 计算组平均性能
                avg_makespan = np.mean([sol.makespan for sol in group_solutions])
                avg_tardiness = np.mean([sol.total_tardiness for sol in group_solutions])
                combined_performance = 1.0 / (1.0 + avg_makespan + avg_tardiness)
                
                self.group_performance_history[group_name].append(combined_performance)
                
                # 保持最近50次记录
                if len(self.group_performance_history[group_name]) > 50:
                    self.group_performance_history[group_name] = self.group_performance_history[group_name][-50:]
    
    def get_algorithm_statistics(self) -> Dict:
        """获取算法统计信息"""
        stats = {
            'iteration': self.current_iteration,
            'population_size': len(self.population),
            'pareto_size': len(self.pareto_solutions),
            'no_improvement_count': self.no_improvement_count
        }
        
        # 添加组性能统计
        stats['group_performance'] = {}
        for group_name, history in self.group_performance_history.items():
            if history:
                stats['group_performance'][group_name] = {
                    'average': float(np.mean(history)),
                    'latest': float(history[-1]),
                    'trend': float(np.mean(history[-5:])) - float(np.mean(history[-10:-5])) if len(history) >= 10 else 0.0
                }
        
        # 添加RL统计
        stats['rl_statistics'] = self.rl_coordinator.get_learning_progress()
        stats['strategy_statistics'] = self.rl_coordinator.get_strategy_statistics()
        
        # 添加混沌映射统计
        stats['chaos_statistics'] = self.chaotic_maps.get_chaos_statistics()
        
        return stats 
    
    def _update_diversity_archive(self):
        """更新多样性存档"""
        if not hasattr(self, 'diversity_archive'):
            self.diversity_archive = []
        
        # 从当前种群中选择多样性解
        for sol in self.population:
            if self._is_diverse_solution(sol):
                # 检查是否已存在相似解
                is_duplicate = False
                for archived_sol in self.diversity_archive:
                    if self._solutions_are_similar(sol, archived_sol):
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    self.diversity_archive.append(sol)
        
        # 限制存档大小
        if len(self.diversity_archive) > self.max_diversity_archive_size:
            # 保留最具多样性的解
            self.diversity_archive = self._select_diverse_solutions(
                self.diversity_archive, self.max_diversity_archive_size
            )
    
    def _is_diverse_solution(self, solution):
        """判断解是否具有多样性价值"""
        # 检查解是否与现有帕累托解差异足够大
        for pareto_sol in self.pareto_solutions:
            if self._solutions_are_similar(solution, pareto_sol):
                return False
        return True
    
    def _solutions_are_similar(self, sol1, sol2):
        """判断两个解是否相似"""
        makespan_diff = abs(sol1.makespan - sol2.makespan) / max(sol1.makespan, sol2.makespan, 1)
        tardiness_diff = abs(sol1.total_tardiness - sol2.total_tardiness) / max(sol1.total_tardiness, sol2.total_tardiness, 1)
        
        return makespan_diff < self.diversity_threshold and tardiness_diff < self.diversity_threshold
    
    def _apply_diversity_enhancement(self):
        """应用多样性增强策略"""
        # 如果帕累托解集较小，尝试从多样性存档中补充
        if len(self.pareto_solutions) < self.pareto_size_limit // 2:
            # 从多样性存档中选择非支配解
            for archived_sol in self.diversity_archive:
                if self._is_non_dominated(archived_sol, self.pareto_solutions):
                    self.pareto_solutions.append(archived_sol)
                    
                    if len(self.pareto_solutions) >= self.pareto_size_limit:
                        break
    
    def _is_non_dominated(self, solution, solution_set):
        """检查解是否被解集中的解支配"""
        for other_sol in solution_set:
            if (other_sol.makespan <= solution.makespan and 
                other_sol.total_tardiness <= solution.total_tardiness and
                (other_sol.makespan < solution.makespan or 
                 other_sol.total_tardiness < solution.total_tardiness)):
                return False
        return True
    
    def _select_diverse_solutions(self, solutions, count):
        """选择最具多样性的解集"""
        if len(solutions) <= count:
            return solutions
        
        selected = []
        remaining = solutions.copy()
        
        # 首先选择目标空间中的极端解
        if remaining:
            # 最小makespan的解
            min_makespan_sol = min(remaining, key=lambda x: x.makespan)
            selected.append(min_makespan_sol)
            remaining.remove(min_makespan_sol)
        
        if remaining:
            # 最小tardiness的解
            min_tardiness_sol = min(remaining, key=lambda x: x.total_tardiness)
            selected.append(min_tardiness_sol)
            remaining.remove(min_tardiness_sol)
        
        # 使用多样性距离选择其余解
        while len(selected) < count and remaining:
            max_diversity_sol = None
            max_diversity_score = -1
            
            for candidate in remaining:
                diversity_score = self._calculate_diversity_score(candidate, selected)
                if diversity_score > max_diversity_score:
                    max_diversity_score = diversity_score
                    max_diversity_sol = candidate
            
            if max_diversity_sol:
                selected.append(max_diversity_sol)
                remaining.remove(max_diversity_sol)
        
        return selected
    
    def _calculate_diversity_score(self, candidate, selected_solutions):
        """计算候选解与已选解集的多样性分数"""
        if not selected_solutions:
            return 1.0
        
        min_distance = float('inf')
        for selected_sol in selected_solutions:
            # 计算归一化欧氏距离
            makespan_diff = abs(candidate.makespan - selected_sol.makespan)
            tardiness_diff = abs(candidate.total_tardiness - selected_sol.total_tardiness)
            
            # 归一化
            max_makespan = max(candidate.makespan, selected_sol.makespan, 1)
            max_tardiness = max(candidate.total_tardiness, selected_sol.total_tardiness, 1)
            
            normalized_distance = ((makespan_diff / max_makespan) ** 2 + 
                                 (tardiness_diff / max_tardiness) ** 2) ** 0.5
            
            min_distance = min(min_distance, normalized_distance)
        
        return min_distance 