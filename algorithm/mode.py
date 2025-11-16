#!/usr/bin/env python3
"""
MODE (Multi-Objective Differential Evolution) 算法实现
用于多目标分布式异构混合流水车间调度问题

基于差分进化的多目标优化算法
"""

import numpy as np
import random
import copy
import time
from typing import List, Tuple, Dict
from dataclasses import dataclass

from problem.mo_dhfsp import MO_DHFSP_Problem, Solution


@dataclass
class Individual:
    """个体类"""
    factory_assignment: List[int]  # 工厂分配
    job_sequence: List[int]        # 作业序列
    makespan: float = 0.0          # 完工时间
    total_tardiness: float = 0.0   # 总拖期
    rank: int = 0                  # 支配等级
    crowding_distance: float = 0.0 # 拥挤距离
    
    def __post_init__(self):
        """初始化后处理"""
        if not hasattr(self, 'objectives'):
            self.objectives = [self.makespan, self.total_tardiness]


class MODE_Optimizer:
    """MODE优化器"""
    
    def __init__(self, problem: MO_DHFSP_Problem, 
                 population_size: int = 100,
                 max_generations: int = 100,
                 F: float = 0.5,           # 缩放因子
                 CR: float = 0.9,          # 交叉概率
                 mutation_prob: float = 0.1):
        """
        初始化MODE优化器
        
        Args:
            problem: 问题实例
            population_size: 种群大小
            max_generations: 最大代数
            F: 差分进化缩放因子
            CR: 交叉概率
            mutation_prob: 变异概率
        """
        self.problem = problem
        self.population_size = population_size
        self.max_generations = max_generations
        self.F = F
        self.CR = CR
        self.mutation_prob = mutation_prob
        
        # 问题参数
        self.n_jobs = problem.n_jobs
        self.n_factories = problem.n_factories
        self.n_stages = problem.n_stages
        
        # 统计信息
        self.generation = 0
        self.convergence_data = []
        
        print(f"初始化MODE: 种群大小={population_size}, 最大代数={max_generations}")
        print(f"参数: F={F}, CR={CR}, 变异概率={mutation_prob}")
    
    def create_individual(self) -> Individual:
        """创建个体"""
        # 随机工厂分配
        factory_assignment = [random.randint(0, self.n_factories - 1) for _ in range(self.n_jobs)]
        
        # 随机作业序列
        job_sequence = list(range(self.n_jobs))
        random.shuffle(job_sequence)
        
        individual = Individual(
            factory_assignment=factory_assignment,
            job_sequence=job_sequence
        )
        
        # 评估个体
        self.evaluate_individual(individual)
        
        return individual
    
    def evaluate_individual(self, individual: Individual):
        """评估个体"""
        # 生成各工厂的作业序列
        job_sequences = [[] for _ in range(self.n_factories)]
        for job_id in range(self.n_jobs):
            factory_id = individual.factory_assignment[job_id]
            job_sequences[factory_id].append(job_id)
        
        # 根据个体的作业序列重新排序各工厂内的作业
        for factory_id in range(self.n_factories):
            if job_sequences[factory_id]:
                # 按照个体的作业序列排序
                factory_jobs = job_sequences[factory_id]
                factory_jobs.sort(key=lambda x: individual.job_sequence.index(x))
                job_sequences[factory_id] = factory_jobs
        
        # 创建解决方案对象
        solution = Solution(
            factory_assignment=individual.factory_assignment.copy(),
            job_sequences=job_sequences
        )
        
        # 计算目标函数
        solution = self.problem.evaluate_solution(solution)
        
        # 更新个体
        individual.makespan = solution.makespan
        individual.total_tardiness = solution.total_tardiness
        individual.objectives = [solution.makespan, solution.total_tardiness]
    
    def initialize_population(self) -> List[Individual]:
        """初始化种群"""
        population = []
        
        print("初始化MODE种群...")
        for i in range(self.population_size):
            individual = self.create_individual()
            population.append(individual)
        
        print(f"MODE初始化完成，种群大小: {len(population)}")
        return population
    
    def differential_evolution_crossover(self, target: Individual, population: List[Individual]) -> Individual:
        """差分进化交叉操作"""
        # 随机选择三个不同的个体
        candidates = [ind for ind in population if ind != target]
        if len(candidates) < 3:
            return copy.deepcopy(target)
        
        r1, r2, r3 = random.sample(candidates, 3)
        
        # 创建试验个体
        trial = Individual(
            factory_assignment=[0] * self.n_jobs,
            job_sequence=[0] * self.n_jobs
        )
        
        # 工厂分配的差分进化
        for i in range(self.n_jobs):
            if random.random() < self.CR:
                # 差分向量
                diff = r2.factory_assignment[i] - r3.factory_assignment[i]
                new_factory = int(r1.factory_assignment[i] + self.F * diff)
                # 边界处理
                trial.factory_assignment[i] = max(0, min(self.n_factories - 1, new_factory))
            else:
                trial.factory_assignment[i] = target.factory_assignment[i]
        
        # 作业序列的交叉（使用顺序交叉）
        if random.random() < self.CR:
            trial.job_sequence = self.order_crossover(
                target.job_sequence, 
                r1.job_sequence
            )
        else:
            trial.job_sequence = target.job_sequence.copy()
        
        return trial
    
    def order_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """顺序交叉"""
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        
        # 从parent1复制片段
        child = [-1] * size
        child[start:end] = parent1[start:end]
        
        # 从parent2按顺序填充剩余位置
        pointer = end
        for job in parent2[end:] + parent2[:end]:
            if job not in child:
                if pointer >= size:
                    pointer = 0
                while child[pointer] != -1:
                    pointer += 1
                child[pointer] = job
                pointer += 1
        
        return child
    
    def mutate_individual(self, individual: Individual) -> Individual:
        """变异操作"""
        mutated = copy.deepcopy(individual)
        
        if random.random() < self.mutation_prob:
            # 工厂重分配变异
            job_idx = random.randint(0, self.n_jobs - 1)
            new_factory = random.randint(0, self.n_factories - 1)
            mutated.factory_assignment[job_idx] = new_factory
        
        if random.random() < self.mutation_prob:
            # 作业序列变异（交换两个作业）
            if self.n_jobs >= 2:
                i, j = random.sample(range(self.n_jobs), 2)
                mutated.job_sequence[i], mutated.job_sequence[j] = \
                    mutated.job_sequence[j], mutated.job_sequence[i]
        
        return mutated
    
    def dominates(self, ind1: Individual, ind2: Individual) -> bool:
        """检查ind1是否支配ind2"""
        better_in_any = False
        for i in range(len(ind1.objectives)):
            if ind1.objectives[i] > ind2.objectives[i]:
                return False
            elif ind1.objectives[i] < ind2.objectives[i]:
                better_in_any = True
        return better_in_any
    
    def fast_non_dominated_sort(self, population: List[Individual]) -> List[List[Individual]]:
        """快速非支配排序"""
        fronts = []
        
        for individual in population:
            individual.domination_count = 0
            individual.dominated_solutions = []
        
        # 计算支配关系
        for i, ind1 in enumerate(population):
            for j, ind2 in enumerate(population):
                if i != j:
                    if self.dominates(ind1, ind2):
                        ind1.dominated_solutions.append(ind2)
                    elif self.dominates(ind2, ind1):
                        ind1.domination_count += 1
        
        # 第一层前沿
        current_front = []
        for individual in population:
            if individual.domination_count == 0:
                individual.rank = 0
                current_front.append(individual)
        
        fronts.append(current_front)
        
        # 后续层前沿
        front_index = 0
        while front_index < len(fronts) and len(fronts[front_index]) > 0:
            next_front = []
            for individual in fronts[front_index]:
                for dominated in individual.dominated_solutions:
                    dominated.domination_count -= 1
                    if dominated.domination_count == 0:
                        dominated.rank = front_index + 1
                        next_front.append(dominated)
            
            if next_front:
                fronts.append(next_front)
            else:
                break
            front_index += 1
        
        return fronts
    
    def calculate_crowding_distance(self, front: List[Individual]):
        """计算拥挤距离"""
        if len(front) <= 2:
            for individual in front:
                individual.crowding_distance = float('inf')
            return
        
        # 初始化拥挤距离
        for individual in front:
            individual.crowding_distance = 0
        
        # 对每个目标计算拥挤距离
        for obj_index in range(len(front[0].objectives)):
            # 按目标函数值排序
            front.sort(key=lambda x: x.objectives[obj_index])
            
            # 边界个体设为无穷大
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            
            # 计算目标函数范围
            obj_range = front[-1].objectives[obj_index] - front[0].objectives[obj_index]
            
            if obj_range > 0:
                # 计算中间个体的拥挤距离
                for i in range(1, len(front) - 1):
                    distance = (front[i + 1].objectives[obj_index] - 
                              front[i - 1].objectives[obj_index]) / obj_range
                    front[i].crowding_distance += distance
    
    def environmental_selection(self, population: List[Individual]) -> List[Individual]:
        """环境选择"""
        # 非支配排序
        fronts = self.fast_non_dominated_sort(population)
        
        # 计算拥挤距离
        for front in fronts:
            self.calculate_crowding_distance(front)
        
        # 选择下一代种群
        new_population = []
        for front in fronts:
            if len(new_population) + len(front) <= self.population_size:
                new_population.extend(front)
            else:
                # 按拥挤距离排序，选择距离大的个体
                remaining = self.population_size - len(new_population)
                front.sort(key=lambda x: x.crowding_distance, reverse=True)
                new_population.extend(front[:remaining])
                break
        
        return new_population
    
    def get_pareto_front(self, population: List[Individual]) -> List[Solution]:
        """获取帕累托前沿"""
        fronts = self.fast_non_dominated_sort(population)
        if not fronts:
            return []
        
        pareto_solutions = []
        for individual in fronts[0]:
            # 生成各工厂的作业序列
            job_sequences = [[] for _ in range(self.n_factories)]
            for job_id in range(self.n_jobs):
                factory_id = individual.factory_assignment[job_id]
                job_sequences[factory_id].append(job_id)
            
            # 根据个体的作业序列重新排序各工厂内的作业
            for factory_id in range(self.n_factories):
                if job_sequences[factory_id]:
                    factory_jobs = job_sequences[factory_id]
                    factory_jobs.sort(key=lambda x: individual.job_sequence.index(x))
                    job_sequences[factory_id] = factory_jobs
            
            solution = Solution(
                factory_assignment=individual.factory_assignment.copy(),
                job_sequences=job_sequences
            )
            solution.makespan = individual.makespan
            solution.total_tardiness = individual.total_tardiness
            pareto_solutions.append(solution)
        
        return pareto_solutions
    
    def optimize(self) -> Tuple[List[Solution], List[Dict]]:
        """运行MODE优化"""
        print("开始MODE优化...")
        
        # 初始化种群
        population = self.initialize_population()
        
        # 记录初始状态
        pareto_front = self.get_pareto_front(population)
        best_makespan = min([sol.makespan for sol in pareto_front]) if pareto_front else float('inf')
        best_tardiness = min([sol.total_tardiness for sol in pareto_front]) if pareto_front else float('inf')
        
        self.convergence_data.append({
            'generation': 0,
            'pareto_count': len(pareto_front),
            'best_makespan': best_makespan,
            'best_tardiness': best_tardiness
        })
        
        print(f"MODE 代数   0: 帕累托解={len(pareto_front):2d}, 最优完工时间={best_makespan:.2f}, 最优总拖期={best_tardiness:.2f}")
        
        # 主循环
        for generation in range(1, self.max_generations + 1):
            self.generation = generation
            
            # 创建新种群
            offspring = []
            
            for individual in population:
                # 差分进化交叉
                trial = self.differential_evolution_crossover(individual, population)
                
                # 变异
                trial = self.mutate_individual(trial)
                
                # 评估试验个体
                self.evaluate_individual(trial)
                
                offspring.append(trial)
            
            # 合并父代和子代
            combined_population = population + offspring
            
            # 环境选择
            population = self.environmental_selection(combined_population)
            
            # 记录收敛数据
            if generation % 10 == 0 or generation == self.max_generations:
                pareto_front = self.get_pareto_front(population)
                best_makespan = min([sol.makespan for sol in pareto_front]) if pareto_front else float('inf')
                best_tardiness = min([sol.total_tardiness for sol in pareto_front]) if pareto_front else float('inf')
                
                self.convergence_data.append({
                    'generation': generation,
                    'pareto_count': len(pareto_front),
                    'best_makespan': best_makespan,
                    'best_tardiness': best_tardiness
                })
                
                print(f"MODE 代数 {generation:3d}: 帕累托解={len(pareto_front):2d}, 最优完工时间={best_makespan:.2f}, 最优总拖期={best_tardiness:.2f}")
        
        # 获取最终帕累托前沿
        final_pareto_front = self.get_pareto_front(population)
        
        print("MODE优化完成!")
        
        return final_pareto_front, self.convergence_data


def test_mode():
    """测试MODE算法"""
    from utils.data_generator import DataGenerator
    
    print("MODE算法测试")
    print("=" * 50)
    
    # 生成测试问题
    generator = DataGenerator(seed=42)
    problem_data = generator.generate_problem(
        n_jobs=10,
        n_factories=2,
        n_stages=2,
        machines_per_stage=[2, 2],
        processing_time_range=(1, 10),
        due_date_tightness=1.5
    )
    
    problem = MO_DHFSP_Problem(problem_data)
    
    # 创建优化器
    optimizer = MODE_Optimizer(
        problem=problem,
        population_size=30,
        max_generations=20,
        F=0.5,
        CR=0.9,
        mutation_prob=0.1
    )
    
    # 运行优化
    start_time = time.time()
    pareto_solutions, convergence_data = optimizer.optimize()
    end_time = time.time()
    
    # 输出结果
    print(f"\n优化结果:")
    print(f"帕累托解数量: {len(pareto_solutions)}")
    print(f"运行时间: {end_time - start_time:.2f}秒")
    
    if pareto_solutions:
        makespans = [sol.makespan for sol in pareto_solutions]
        tardiness = [sol.total_tardiness for sol in pareto_solutions]
        
        print(f"完工时间范围: [{min(makespans):.2f}, {max(makespans):.2f}]")
        print(f"总拖期范围: [{min(tardiness):.2f}, {max(tardiness):.2f}]")
        
        print(f"\n前5个帕累托解:")
        for i, sol in enumerate(pareto_solutions[:5]):
            print(f"  解 {i+1}: 完工时间={sol.makespan:.2f}, 总拖期={sol.total_tardiness:.2f}")


if __name__ == "__main__":
    test_mode() 