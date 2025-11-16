#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多目标粒子群优化算法 (MOPSO)
Multi-Objective Particle Swarm Optimization

用于求解多目标分布式异构混合流水车间调度问题
优化目标：完工时间 (Makespan) 和总拖期 (Total Tardiness)
"""

import numpy as np
import random
import copy
import time
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

from problem.mo_dhfsp import MO_DHFSP_Problem, Solution


@dataclass
class Particle:
    """粒子类"""
    position: List[int]  # 当前位置（解）
    velocity: List[float]  # 速度
    fitness: Tuple[float, float]  # 适应度值 (makespan, total_tardiness)
    pbest_position: List[int]  # 个体最优位置
    pbest_fitness: Tuple[float, float]  # 个体最优适应度
    
    def __post_init__(self):
        if self.pbest_position is None:
            self.pbest_position = copy.deepcopy(self.position)
        if self.pbest_fitness is None:
            self.pbest_fitness = self.fitness


class ExternalArchive:
    """外部存档类，用于存储非支配解"""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.solutions: List[Solution] = []
        self.fitness_values: List[Tuple[float, float]] = []
    
    def add_solution(self, solution: Solution):
        """添加解到存档 - 极度宽松版本"""
        new_fitness = (solution.makespan, solution.total_tardiness)
        
        # 极度宽松的支配检查 - 只有明显优势时才认为被支配
        dominated = False
        for existing_fitness in self.fitness_values:
            if self._strict_dominates(existing_fitness, new_fitness):
                dominated = True
                break
        
        if not dominated:
            # 极度宽松的清理策略 - 只移除明显被支配的解
            to_remove = []
            for i, existing_fitness in enumerate(self.fitness_values):
                if self._strict_dominates(new_fitness, existing_fitness):
                    to_remove.append(i)
            
            # 从后往前删除，避免索引问题
            for i in sorted(to_remove, reverse=True):
                del self.solutions[i]
                del self.fitness_values[i]
            
            # 添加新解
            self.solutions.append(copy.deepcopy(solution))
            self.fitness_values.append(new_fitness)
            
            # 如果存档超过最大容量，使用拥挤距离选择
            if len(self.solutions) > self.max_size:
                self._maintain_archive_size()
    
    def _strict_dominates(self, fitness1: Tuple[float, float], fitness2: Tuple[float, float]) -> bool:
        """严格支配判断 - 需要非常显著的优势才认为支配"""
        # 计算相对优势
        if fitness2[0] == 0 or fitness2[1] == 0:
            return False  # 避免除零
        
        makespan_advantage = (fitness2[0] - fitness1[0]) / fitness2[0]
        tardiness_advantage = (fitness2[1] - fitness1[1]) / fitness2[1]
        
        # 只有在至少一个目标有非常显著优势（>5%），且另一个目标不劣的情况下才认为支配
        significant_threshold = 0.05  # 5%的显著优势阈值（大幅放宽）
        
        makespan_better = makespan_advantage > significant_threshold
        tardiness_better = tardiness_advantage > significant_threshold
        makespan_not_worse = fitness1[0] <= fitness2[0] * (1 + significant_threshold)
        tardiness_not_worse = fitness1[1] <= fitness2[1] * (1 + significant_threshold)
        
        return ((makespan_better and tardiness_not_worse) or 
                (tardiness_better and makespan_not_worse))
    
    def _maintain_archive_size(self):
        """维持存档大小，使用极度宽松的拥挤距离选择"""
        if len(self.solutions) <= self.max_size:
            return
        
        # 计算拥挤距离
        crowding_distances = self._calculate_crowding_distance()
        
        # 极度宽松的选择策略 - 优先保留边界解和多样化解
        
        # 1. 保留所有边界解
        min_f1_idx = min(range(len(self.fitness_values)), key=lambda i: self.fitness_values[i][0])
        min_f2_idx = min(range(len(self.fitness_values)), key=lambda i: self.fitness_values[i][1])
        max_f1_idx = max(range(len(self.fitness_values)), key=lambda i: self.fitness_values[i][0])
        max_f2_idx = max(range(len(self.fitness_values)), key=lambda i: self.fitness_values[i][1])
        
        protected_indices = {min_f1_idx, min_f2_idx, max_f1_idx, max_f2_idx}
        
        # 2. 保留拥挤距离最大的解
        sorted_indices = sorted(range(len(crowding_distances)), 
                              key=lambda i: crowding_distances[i], reverse=True)
        
        # 3. 组合选择策略：保护边界解 + 高拥挤距离解
        selected_indices = list(protected_indices)
        
        # 添加剩余的高拥挤距离解
        for idx in sorted_indices:
            if len(selected_indices) >= self.max_size:
                break
            if idx not in selected_indices:
                selected_indices.append(idx)
        
        # 如果还有空间，随机添加更多解（极度宽松）
        if len(selected_indices) < self.max_size:
            remaining_indices = [i for i in range(len(self.solutions)) if i not in selected_indices]
            import random
            additional_count = min(len(remaining_indices), self.max_size - len(selected_indices))
            additional_indices = random.sample(remaining_indices, additional_count)
            selected_indices.extend(additional_indices)
        
        # 更新存档
        self.solutions = [self.solutions[i] for i in selected_indices]
        self.fitness_values = [self.fitness_values[i] for i in selected_indices]
    
    def _calculate_crowding_distance(self) -> List[float]:
        """计算拥挤距离"""
        n = len(self.fitness_values)
        if n <= 2:
            return [float('inf')] * n
        
        distances = [0.0] * n
        
        # 对每个目标函数计算拥挤距离
        for obj_idx in range(2):  # 两个目标函数
            # 按当前目标函数值排序
            sorted_indices = sorted(range(n), key=lambda i: self.fitness_values[i][obj_idx])
            
            # 边界解设为无穷大
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')
            
            # 计算目标函数值的范围
            obj_min = self.fitness_values[sorted_indices[0]][obj_idx]
            obj_max = self.fitness_values[sorted_indices[-1]][obj_idx]
            obj_range = obj_max - obj_min
            
            if obj_range > 0:
                # 计算中间解的拥挤距离
                for i in range(1, n - 1):
                    if distances[sorted_indices[i]] != float('inf'):
                        distance = (self.fitness_values[sorted_indices[i + 1]][obj_idx] - 
                                  self.fitness_values[sorted_indices[i - 1]][obj_idx]) / obj_range
                        distances[sorted_indices[i]] += distance
        
        return distances
    
    def get_random_leader(self) -> Optional[Solution]:
        """随机选择一个领导者"""
        if not self.solutions:
            return None
        return random.choice(self.solutions)
    
    def get_best_leader(self) -> Optional[Solution]:
        """选择最好的领导者（基于拥挤距离）"""
        if not self.solutions:
            return None
        
        crowding_distances = self._calculate_crowding_distance()
        best_idx = max(range(len(crowding_distances)), key=lambda i: crowding_distances[i])
        return self.solutions[best_idx]


class MOPSO_Optimizer:
    """多目标粒子群优化算法"""
    
    def __init__(self, problem: MO_DHFSP_Problem, 
                 swarm_size: int = 100,
                 max_iterations: int = 100,
                 w: float = 0.9,
                 c1: float = 2.0,
                 c2: float = 2.0,
                 archive_size: int = 100,
                 mutation_prob: float = 0.1):
        """
        初始化MOPSO优化器
        
        Args:
            problem: 问题实例
            swarm_size: 群体大小
            max_iterations: 最大迭代次数
            w: 惯性权重
            c1: 个体学习因子
            c2: 社会学习因子
            archive_size: 外部存档大小
            mutation_prob: 变异概率
        """
        self.problem = problem
        self.swarm_size = swarm_size
        self.max_iterations = max_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.archive_size = archive_size
        self.mutation_prob = mutation_prob
        
        # 初始化组件
        self.swarm: List[Particle] = []
        self.archive = ExternalArchive(archive_size)
        
        # 统计信息
        self.convergence_data = []
        self.best_makespan_history = []
        self.best_tardiness_history = []
        
        print(f"MOPSO优化器初始化完成:")
        print(f"  群体大小: {swarm_size}")
        print(f"  最大迭代次数: {max_iterations}")
        print(f"  惯性权重: {w}")
        print(f"  学习因子: c1={c1}, c2={c2}")
        print(f"  外部存档大小: {archive_size}")
    
    def optimize(self) -> Tuple[List[Solution], List[Dict]]:
        """
        执行MOPSO优化
        
        Returns:
            Tuple[帕累托最优解集, 收敛数据]
        """
        print("开始MOPSO优化...")
        start_time = time.time()
        
        # 初始化群体
        self._initialize_swarm()
        
        # 主优化循环
        for iteration in range(self.max_iterations):
            iteration_start = time.time()
            
            # 更新粒子
            self._update_swarm()
            
            # 更新外部存档
            self._update_archive()
            
            # 记录收敛数据
            self._record_convergence_data(iteration)
            
            iteration_time = time.time() - iteration_start
            
            if (iteration + 1) % 10 == 0:
                best_makespan = min(f[0] for f in self.archive.fitness_values) if self.archive.fitness_values else float('inf')
                best_tardiness = min(f[1] for f in self.archive.fitness_values) if self.archive.fitness_values else float('inf')
                print(f"  迭代 {iteration + 1}/{self.max_iterations}: "
                      f"存档大小={len(self.archive.solutions)}, "
                      f"最优完工时间={best_makespan:.2f}, "
                      f"最优拖期={best_tardiness:.2f}, "
                      f"用时={iteration_time:.2f}s")
        
        total_time = time.time() - start_time
        print(f"MOPSO优化完成! 总用时: {total_time:.2f}s")
        print(f"最终帕累托解数量: {len(self.archive.solutions)}")
        
        return self.archive.solutions, self.convergence_data
    
    def _initialize_swarm(self):
        """初始化粒子群"""
        print("初始化粒子群...")
        
        self.swarm = []
        
        for i in range(self.swarm_size):
            # 生成随机解
            solution = self.problem.generate_random_solution()
            
            # 创建粒子
            particle = Particle(
                position=solution.factory_assignment.copy(),
                velocity=[0.0] * len(solution.factory_assignment),
                fitness=(solution.makespan, solution.total_tardiness),
                pbest_position=solution.factory_assignment.copy(),
                pbest_fitness=(solution.makespan, solution.total_tardiness)
            )
            
            self.swarm.append(particle)
            
            # 添加到外部存档
            self.archive.add_solution(solution)
        
        print(f"粒子群初始化完成，初始存档大小: {len(self.archive.solutions)}")
    
    def _update_swarm(self):
        """更新粒子群"""
        for particle in self.swarm:
            # 选择全局最优位置（从存档中选择）
            gbest_solution = self.archive.get_random_leader()
            if gbest_solution is None:
                continue
            
            gbest_position = gbest_solution.factory_assignment
            
            # 更新速度
            self._update_velocity(particle, gbest_position)
            
            # 更新位置
            self._update_position(particle)
            
            # 应用变异
            if random.random() < self.mutation_prob:
                self._mutate_particle(particle)
            
            # 评估新位置
            new_solution = self.problem.create_solution(particle.position)
            particle.fitness = (new_solution.makespan, new_solution.total_tardiness)
            
            # 更新个体最优
            if self._dominates(particle.fitness, particle.pbest_fitness):
                particle.pbest_position = particle.position.copy()
                particle.pbest_fitness = particle.fitness
    
    def _update_velocity(self, particle: Particle, gbest_position: List[int]):
        """更新粒子速度"""
        for i in range(len(particle.velocity)):
            r1 = random.random()
            r2 = random.random()
            
            # 计算速度更新
            cognitive_component = self.c1 * r1 * (particle.pbest_position[i] - particle.position[i])
            social_component = self.c2 * r2 * (gbest_position[i] - particle.position[i])
            
            particle.velocity[i] = (self.w * particle.velocity[i] + 
                                  cognitive_component + social_component)
            
            # 限制速度范围
            max_velocity = self.problem.n_factories
            particle.velocity[i] = max(-max_velocity, min(max_velocity, particle.velocity[i]))
    
    def _update_position(self, particle: Particle):
        """更新粒子位置"""
        for i in range(len(particle.position)):
            # 位置更新
            new_pos = particle.position[i] + particle.velocity[i]
            
            # 应用sigmoid函数转换为概率
            prob = 1.0 / (1.0 + np.exp(-new_pos))
            
            # 根据概率决定工厂分配
            if random.random() < prob:
                # 随机选择一个不同的工厂
                available_factories = [f for f in range(self.problem.n_factories) if f != particle.position[i]]
                if available_factories:
                    particle.position[i] = random.choice(available_factories)
            
            # 确保工厂编号在有效范围内
            particle.position[i] = max(0, min(self.problem.n_factories - 1, particle.position[i]))
    
    def _mutate_particle(self, particle: Particle):
        """粒子变异操作"""
        mutation_type = random.choice(['factory_reassign', 'swap', 'insert'])
        
        if mutation_type == 'factory_reassign':
            # 工厂重分配变异
            job_idx = random.randint(0, len(particle.position) - 1)
            new_factory = random.randint(0, self.problem.n_factories - 1)
            particle.position[job_idx] = new_factory
            
        elif mutation_type == 'swap':
            # 交换变异
            if len(particle.position) >= 2:
                idx1, idx2 = random.sample(range(len(particle.position)), 2)
                particle.position[idx1], particle.position[idx2] = particle.position[idx2], particle.position[idx1]
                
        elif mutation_type == 'insert':
            # 插入变异
            if len(particle.position) >= 2:
                from_idx = random.randint(0, len(particle.position) - 1)
                to_idx = random.randint(0, len(particle.position) - 1)
                
                if from_idx != to_idx:
                    job_factory = particle.position.pop(from_idx)
                    particle.position.insert(to_idx, job_factory)
    
    def _update_archive(self):
        """更新外部存档"""
        for particle in self.swarm:
            solution = self.problem.create_solution(particle.position)
            self.archive.add_solution(solution)
    
    def _dominates(self, fitness1: Tuple[float, float], fitness2: Tuple[float, float]) -> bool:
        """判断fitness1是否支配fitness2"""
        return (fitness1[0] <= fitness2[0] and fitness1[1] <= fitness2[1] and 
                (fitness1[0] < fitness2[0] or fitness1[1] < fitness2[1]))
    
    def _record_convergence_data(self, iteration: int):
        """记录收敛数据"""
        if not self.archive.fitness_values:
            return
        
        # 计算当前代的统计信息
        makespans = [f[0] for f in self.archive.fitness_values]
        tardiness = [f[1] for f in self.archive.fitness_values]
        
        best_makespan = min(makespans)
        best_tardiness = min(tardiness)
        avg_makespan = np.mean(makespans)
        avg_tardiness = np.mean(tardiness)
        
        self.best_makespan_history.append(best_makespan)
        self.best_tardiness_history.append(best_tardiness)
        
        # 记录收敛数据
        convergence_info = {
            'iteration': iteration,
            'archive_size': len(self.archive.solutions),
            'best_makespan': best_makespan,
            'best_tardiness': best_tardiness,
            'avg_makespan': avg_makespan,
            'avg_tardiness': avg_tardiness,
            'hypervolume': self._calculate_hypervolume(),
            'spacing': self._calculate_spacing()
        }
        
        self.convergence_data.append(convergence_info)
    
    def _calculate_hypervolume(self) -> float:
        """计算超体积指标"""
        if not self.archive.fitness_values:
            return 0.0
        
        try:
            # 使用简化的超体积计算
            # 参考点设为所有目标函数的最大值
            ref_point = [
                max(f[0] for f in self.archive.fitness_values) * 1.1,
                max(f[1] for f in self.archive.fitness_values) * 1.1
            ]
            
            # 计算每个解的贡献
            total_volume = 0.0
            for fitness in self.archive.fitness_values:
                if fitness[0] < ref_point[0] and fitness[1] < ref_point[1]:
                    volume = (ref_point[0] - fitness[0]) * (ref_point[1] - fitness[1])
                    total_volume += volume
            
            return total_volume
            
        except Exception as e:
            print(f"超体积计算错误: {e}")
            return 0.0
    
    def _calculate_spacing(self) -> float:
        """计算间距指标"""
        if len(self.archive.fitness_values) < 2:
            return 0.0
        
        try:
            distances = []
            
            for i, fitness1 in enumerate(self.archive.fitness_values):
                min_dist = float('inf')
                for j, fitness2 in enumerate(self.archive.fitness_values):
                    if i != j:
                        # 计算欧几里得距离
                        dist = np.sqrt((fitness1[0] - fitness2[0])**2 + (fitness1[1] - fitness2[1])**2)
                        min_dist = min(min_dist, dist)
                distances.append(min_dist)
            
            # 计算间距
            mean_dist = np.mean(distances)
            spacing = np.sqrt(np.mean([(d - mean_dist)**2 for d in distances]))
            
            return spacing
            
        except Exception as e:
            print(f"间距计算错误: {e}")
            return 0.0
    
    def get_pareto_solutions(self) -> List[Solution]:
        """获取帕累托最优解"""
        return self.archive.solutions.copy()
    
    def get_best_solutions(self) -> Dict[str, Solution]:
        """获取各目标的最优解"""
        if not self.archive.solutions:
            return {}
        
        best_makespan_idx = min(range(len(self.archive.fitness_values)), 
                               key=lambda i: self.archive.fitness_values[i][0])
        best_tardiness_idx = min(range(len(self.archive.fitness_values)), 
                                key=lambda i: self.archive.fitness_values[i][1])
        
        return {
            'best_makespan': self.archive.solutions[best_makespan_idx],
            'best_tardiness': self.archive.solutions[best_tardiness_idx]
        }
    
    def get_statistics(self) -> Dict:
        """获取优化统计信息"""
        if not self.archive.fitness_values:
            return {}
        
        makespans = [f[0] for f in self.archive.fitness_values]
        tardiness = [f[1] for f in self.archive.fitness_values]
        
        return {
            'pareto_solutions_count': len(self.archive.solutions),
            'best_makespan': min(makespans),
            'best_tardiness': min(tardiness),
            'avg_makespan': np.mean(makespans),
            'avg_tardiness': np.mean(tardiness),
            'makespan_std': np.std(makespans),
            'tardiness_std': np.std(tardiness),
            'final_hypervolume': self._calculate_hypervolume(),
            'final_spacing': self._calculate_spacing()
        } 