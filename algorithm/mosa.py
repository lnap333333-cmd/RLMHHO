#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多目标模拟退火算法 (MOSA - Multi-Objective Simulated Annealing)
用于求解多目标分布式混合流水车间调度问题
"""

import numpy as np
import random
import time
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import copy

from problem.mo_dhfsp import Solution  # 使用项目标准的Solution类


@dataclass
class MOSAParameters:
    """MOSA算法参数"""
    initial_temperature: float = 1000.0      # 初始温度
    final_temperature: float = 0.1           # 终止温度
    cooling_rate: float = 0.95               # 冷却率
    max_iterations: int = 1000               # 最大迭代次数
    archive_size: int = 100                  # 外部档案大小
    neighborhood_size: int = 5               # 邻域操作次数
    acceptance_probability_threshold: float = 0.01  # 接受概率阈值


class MOSA_Optimizer:
    """多目标模拟退火优化器"""
    
    def __init__(self, problem, **kwargs):
        """
        初始化MOSA优化器
        
        Args:
            problem: MO_DHFSP_Problem实例
            **kwargs: 算法参数
        """
        self.problem = problem
        
        # 设置算法参数
        default_params = MOSAParameters()
        self.params = MOSAParameters(
            initial_temperature=kwargs.get('initial_temperature', default_params.initial_temperature),
            final_temperature=kwargs.get('final_temperature', default_params.final_temperature),
            cooling_rate=kwargs.get('cooling_rate', default_params.cooling_rate),
            max_iterations=kwargs.get('max_iterations', default_params.max_iterations),
            archive_size=kwargs.get('archive_size', default_params.archive_size),
            neighborhood_size=kwargs.get('neighborhood_size', default_params.neighborhood_size),
            acceptance_probability_threshold=kwargs.get('acceptance_probability_threshold', 
                                                       default_params.acceptance_probability_threshold)
        )
        
        # 初始化算法状态
        self.current_temperature = self.params.initial_temperature
        self.external_archive = []  # 外部档案存储非支配解
        self.convergence_data = []
        
        # 统计信息
        self.iteration_count = 0
        self.accepted_solutions = 0
        self.rejected_solutions = 0
        
    def optimize(self) -> Tuple[List[Solution], Dict[str, Any]]:
        """
        执行MOSA优化
        
        Returns:
            Tuple[List[Solution], Dict]: (帕累托解集, 收敛数据)
        """
        print(f"🔥 开始MOSA优化 (T₀={self.params.initial_temperature}, 冷却率={self.params.cooling_rate})")
        
        start_time = time.time()
        
        # 1. 生成初始解
        current_solution = self._generate_initial_solution()
        current_solution = self.problem.evaluate_solution(current_solution)
        
        # 2. 初始化外部档案
        self.external_archive = [copy.deepcopy(current_solution)]
        
        # 3. 主循环
        self.iteration_count = 0
        self.current_temperature = self.params.initial_temperature
        
        while (self.current_temperature > self.params.final_temperature and 
               self.iteration_count < self.params.max_iterations):
            
            # 在当前温度下进行多次邻域搜索
            for _ in range(self.params.neighborhood_size):
                # 生成邻域解
                neighbor_solution = self._generate_neighbor(current_solution)
                neighbor_solution = self.problem.evaluate_solution(neighbor_solution)
                
                # 决定是否接受邻域解
                if self._accept_solution(current_solution, neighbor_solution):
                    current_solution = copy.deepcopy(neighbor_solution)
                    self.accepted_solutions += 1
                else:
                    self.rejected_solutions += 1
                
                # 更新外部档案
                self._update_external_archive(neighbor_solution)
            
            # 冷却
            self.current_temperature *= self.params.cooling_rate
            self.iteration_count += 1
            
            # 记录收敛数据
            if self.iteration_count % 10 == 0:
                self._record_convergence_data()
        
        end_time = time.time()
        
        # 最终的档案维护
        self._maintain_archive_size()
        
        print(f"✅ MOSA优化完成:")
        print(f"   • 迭代次数: {self.iteration_count}")
        print(f"   • 最终温度: {self.current_temperature:.6f}")
        print(f"   • 接受率: {self.accepted_solutions/(self.accepted_solutions + self.rejected_solutions)*100:.1f}%")
        print(f"   • 帕累托解数量: {len(self.external_archive)}")
        print(f"   • 运行时间: {end_time - start_time:.2f}秒")
        
        return self.external_archive, {
            'convergence_data': self.convergence_data,
            'iterations': self.iteration_count,
            'final_temperature': self.current_temperature,
            'acceptance_rate': self.accepted_solutions/(self.accepted_solutions + self.rejected_solutions),
            'runtime': end_time - start_time
        }
    
    def _generate_initial_solution(self) -> Solution:
        """生成初始解 - 使用与其他算法相同的方式"""
        # 使用问题类的标准随机解生成方法
        return self.problem.generate_random_solution()
    
    def _generate_neighbor(self, solution: Solution) -> Solution:
        """生成邻域解"""
        neighbor = copy.deepcopy(solution)
        
        # 随机选择邻域操作
        operation = random.choice(['swap_jobs', 'insert_job', 'factory_change', 'sequence_swap'])
        
        if operation == 'swap_jobs':
            # 交换两个作业的工厂分配
            if len(neighbor.factory_assignment) >= 2:
                i, j = random.sample(range(len(neighbor.factory_assignment)), 2)
                neighbor.factory_assignment[i], neighbor.factory_assignment[j] = \
                    neighbor.factory_assignment[j], neighbor.factory_assignment[i]
                
                # 重新构建作业序列
                neighbor.job_sequences = self._rebuild_job_sequences(neighbor.factory_assignment)
        
        elif operation == 'insert_job':
            # 将一个作业插入到同一工厂的不同位置
            non_empty_factories = [f for f in range(self.problem.n_factories) if neighbor.job_sequences[f]]
            if non_empty_factories:
                factory = random.choice(non_empty_factories)
                if len(neighbor.job_sequences[factory]) >= 2:
                    i = random.randint(0, len(neighbor.job_sequences[factory]) - 1)
                    j = random.randint(0, len(neighbor.job_sequences[factory]) - 1)
                    job = neighbor.job_sequences[factory].pop(i)
                    neighbor.job_sequences[factory].insert(j, job)
        
        elif operation == 'factory_change':
            # 改变一个作业的工厂分配
            if neighbor.factory_assignment:
                job_idx = random.randint(0, len(neighbor.factory_assignment) - 1)
                old_factory = neighbor.factory_assignment[job_idx]
                new_factory = random.randint(0, self.problem.n_factories - 1)
                
                if old_factory != new_factory:
                    neighbor.factory_assignment[job_idx] = new_factory
                    
                    # 从旧工厂移除作业（注意：这里移除的是作业ID，不是索引）
                    if job_idx in neighbor.job_sequences[old_factory]:
                        neighbor.job_sequences[old_factory].remove(job_idx)
                    
                    # 添加到新工厂
                    neighbor.job_sequences[new_factory].append(job_idx)
        
        elif operation == 'sequence_swap':
            # 在同一工厂内交换两个作业的顺序
            non_empty_factories = [f for f in range(self.problem.n_factories) if len(neighbor.job_sequences[f]) >= 2]
            if non_empty_factories:
                factory = random.choice(non_empty_factories)
                i, j = random.sample(range(len(neighbor.job_sequences[factory])), 2)
                neighbor.job_sequences[factory][i], neighbor.job_sequences[factory][j] = \
                    neighbor.job_sequences[factory][j], neighbor.job_sequences[factory][i]
        
        return neighbor
    
    def _rebuild_job_sequences(self, factory_assignment: List[int]) -> List[List[int]]:
        """根据工厂分配重建作业序列"""
        job_sequences = [[] for _ in range(self.problem.n_factories)]
        for job_id, factory_id in enumerate(factory_assignment):
            job_sequences[factory_id].append(job_id)
        return job_sequences
    
    def _accept_solution(self, current: Solution, neighbor: Solution) -> bool:
        """决定是否接受邻域解"""
        # 多目标接受准则
        
        # 1. 如果邻域解支配当前解，直接接受
        if self._dominates(neighbor, current):
            return True
        
        # 2. 如果当前解支配邻域解，计算接受概率
        if self._dominates(current, neighbor):
            # 计算目标函数差值
            delta_makespan = neighbor.makespan - current.makespan
            delta_tardiness = neighbor.total_tardiness - current.total_tardiness
            
            # 使用加权和计算总差值
            delta = 0.55 * delta_makespan + 0.45 * delta_tardiness
            
            # 计算接受概率
            if delta < 0:
                return True
            else:
                acceptance_prob = np.exp(-delta / self.current_temperature)
                return random.random() < acceptance_prob
        
        # 3. 如果两解互不支配，使用概率接受
        # 基于拥挤距离和多样性考虑
        diversity_factor = self._calculate_diversity_factor(neighbor)
        base_prob = 0.5 * diversity_factor
        
        # 温度调节
        temp_factor = self.current_temperature / self.params.initial_temperature
        final_prob = base_prob * temp_factor
        
        return random.random() < final_prob
    
    def _dominates(self, sol1: Solution, sol2: Solution) -> bool:
        """判断sol1是否支配sol2"""
        better_in_one = False
        
        # 检查完工时间
        if sol1.makespan < sol2.makespan:
            better_in_one = True
        elif sol1.makespan > sol2.makespan:
            return False
        
        # 检查总拖期
        if sol1.total_tardiness < sol2.total_tardiness:
            better_in_one = True
        elif sol1.total_tardiness > sol2.total_tardiness:
            return False
        
        return better_in_one
    
    def _calculate_diversity_factor(self, solution: Solution) -> float:
        """计算解的多样性因子"""
        if not self.external_archive:
            return 1.0
        
        # 计算与档案中解的最小距离
        min_distance = float('inf')
        for archived_sol in self.external_archive:
            distance = np.sqrt(
                (solution.makespan - archived_sol.makespan) ** 2 +
                (solution.total_tardiness - archived_sol.total_tardiness) ** 2
            )
            min_distance = min(min_distance, distance)
        
        # 距离越大，多样性因子越大
        return min(1.0, min_distance / 100.0)
    
    def _update_external_archive(self, new_solution: Solution):
        """更新外部档案 - 极度宽松版本，保留更多解"""
        # 极度宽松的支配检查 - 只有明显优势时才认为被支配
        dominated_by_archive = False
        for archived_sol in self.external_archive:
            if self._strict_dominates(archived_sol, new_solution):
                dominated_by_archive = True
                break
        
        if not dominated_by_archive:
            # 新解不被严格支配，加入档案
            # 同时移除被新解严格支配的解
            self.external_archive = [
                sol for sol in self.external_archive 
                if not self._strict_dominates(new_solution, sol)
            ]
            self.external_archive.append(copy.deepcopy(new_solution))
    
    def _strict_dominates(self, sol1: Solution, sol2: Solution) -> bool:
        """严格支配判断 - 需要非常显著的优势才认为支配"""
        # 计算相对优势
        if sol2.makespan == 0 or sol2.total_tardiness == 0:
            return False  # 避免除零
        
        makespan_advantage = (sol2.makespan - sol1.makespan) / sol2.makespan
        tardiness_advantage = (sol2.total_tardiness - sol1.total_tardiness) / sol2.total_tardiness
        
        # 只有在至少一个目标有非常显著优势（>5%），且另一个目标不劣的情况下才认为支配
        significant_threshold = 0.05  # 5%的显著优势阈值（大幅放宽）
        
        makespan_better = makespan_advantage > significant_threshold
        tardiness_better = tardiness_advantage > significant_threshold
        makespan_not_worse = sol1.makespan <= sol2.makespan * (1 + significant_threshold)
        tardiness_not_worse = sol1.total_tardiness <= sol2.total_tardiness * (1 + significant_threshold)
        
        return ((makespan_better and tardiness_not_worse) or 
                (tardiness_better and makespan_not_worse))
    
    def _maintain_archive_size(self):
        """维护档案大小 - 极度宽松的多样性保护策略"""
        if len(self.external_archive) <= self.params.archive_size:
            return
        
        # 使用极度宽松的拥挤距离选择
        crowding_distances = self._calculate_crowding_distances()
        
        # 1. 保护边界解
        makespan_values = [sol.makespan for sol in self.external_archive]
        tardiness_values = [sol.total_tardiness for sol in self.external_archive]
        
        min_makespan_idx = makespan_values.index(min(makespan_values))
        min_tardiness_idx = tardiness_values.index(min(tardiness_values))
        max_makespan_idx = makespan_values.index(max(makespan_values))
        max_tardiness_idx = tardiness_values.index(max(tardiness_values))
        
        protected_indices = {min_makespan_idx, min_tardiness_idx, max_makespan_idx, max_tardiness_idx}
        
        # 2. 按拥挤距离排序，但保留更多解
        indexed_distances = [(i, dist) for i, dist in enumerate(crowding_distances)]
        indexed_distances.sort(key=lambda x: x[1], reverse=True)
        
        # 3. 选择策略：保护边界解 + 高拥挤距离解 + 随机补充
        selected_indices = list(protected_indices)
        
        # 添加高拥挤距离解
        for i, _ in indexed_distances:
            if len(selected_indices) >= self.params.archive_size:
                break
            if i not in selected_indices:
                selected_indices.append(i)
        
        # 极度宽松：如果还有空间，随机保留更多解
        if len(selected_indices) < self.params.archive_size:
            remaining_indices = [i for i in range(len(self.external_archive)) if i not in selected_indices]
            import random
            additional_count = min(len(remaining_indices), self.params.archive_size - len(selected_indices))
            if additional_count > 0:
                additional_indices = random.sample(remaining_indices, additional_count)
                selected_indices.extend(additional_indices)
        
        # 更新档案
        self.external_archive = [self.external_archive[i] for i in selected_indices]
    
    def _calculate_crowding_distances(self) -> List[float]:
        """计算拥挤距离"""
        n_solutions = len(self.external_archive)
        if n_solutions <= 2:
            return [float('inf')] * n_solutions
        
        distances = [0.0] * n_solutions
        
        # 对每个目标函数计算拥挤距离
        objectives = ['makespan', 'total_tardiness']
        
        for obj in objectives:
            # 获取目标值并排序
            obj_values = [getattr(sol, obj) for sol in self.external_archive]
            sorted_indices = sorted(range(n_solutions), key=lambda i: obj_values[i])
            
            # 边界解设为无穷大
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')
            
            # 计算中间解的拥挤距离
            obj_range = obj_values[sorted_indices[-1]] - obj_values[sorted_indices[0]]
            if obj_range > 0:
                for i in range(1, n_solutions - 1):
                    idx = sorted_indices[i]
                    prev_idx = sorted_indices[i - 1]
                    next_idx = sorted_indices[i + 1]
                    
                    if distances[idx] != float('inf'):
                        distances[idx] += (obj_values[next_idx] - obj_values[prev_idx]) / obj_range
        
        return distances
    
    def _record_convergence_data(self):
        """记录收敛数据"""
        if not self.external_archive:
            return
        
        # 计算档案中解的统计信息
        makespans = [sol.makespan for sol in self.external_archive]
        tardiness_values = [sol.total_tardiness for sol in self.external_archive]
        
        convergence_info = {
            'iteration': self.iteration_count,
            'temperature': self.current_temperature,
            'archive_size': len(self.external_archive),
            'best_makespan': min(makespans),
            'best_tardiness': min(tardiness_values),
            'avg_makespan': np.mean(makespans),
            'avg_tardiness': np.mean(tardiness_values),
            'acceptance_rate': self.accepted_solutions / (self.accepted_solutions + self.rejected_solutions) if (self.accepted_solutions + self.rejected_solutions) > 0 else 0
        }
        
        self.convergence_data.append(convergence_info)


# 为了保持与其他算法的一致性，提供简化的接口
def create_mosa_optimizer(problem, **kwargs):
    """创建MOSA优化器的工厂函数"""
    return MOSA_Optimizer(problem, **kwargs)