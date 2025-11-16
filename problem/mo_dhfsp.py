#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多目标分布式混合流水车间调度问题定义
Multi-Objective Distributed Hybrid Flow Shop Scheduling Problem
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Solution:
    """解的数据结构"""
    factory_assignment: List[int]  # 作业到工厂的分配
    job_sequences: List[List[int]]  # 各工厂的作业序列
    makespan: float = 0.0  # 最大完工时间
    total_tardiness: float = 0.0  # 总拖期
    completion_times: List[float] = None  # 各作业完工时间
    factory_makespans: List[float] = None  # 各工厂完工时间
    
    def __post_init__(self):
        if self.completion_times is None:
            self.completion_times = []
        if self.factory_makespans is None:
            self.factory_makespans = []

class MO_DHFSP_Problem:
    """多目标分布式混合流水车间调度问题类"""
    
    def __init__(self, problem_data: Dict):
        """
        初始化问题实例
        
        Args:
            problem_data: 问题数据字典，包含：
                - n_jobs: 作业数量
                - n_factories: 工厂数量
                - n_stages: 阶段数量
                - machines_per_stage: 各阶段机器数量列表（平均配置）
                - heterogeneous_machines: 异构机器配置字典（可选）
                - processing_times: 处理时间矩阵 [job][stage]
                - due_dates: 交货期列表
        """
        self.n_jobs = problem_data['n_jobs']
        self.n_factories = problem_data['n_factories']
        self.n_stages = problem_data['n_stages']
        self.machines_per_stage = problem_data['machines_per_stage']
        self.processing_times = np.array(problem_data['processing_times'])
        self.due_dates = np.array(problem_data['due_dates'])
        self.urgencies = np.array(problem_data.get('urgencies', [1.0] * self.n_jobs))
        
        # 处理异构机器配置
        self.heterogeneous_machines = problem_data.get('heterogeneous_machines', None)
        self.factory_machines = problem_data.get('factory_machines', None)
        
        if self.factory_machines:
            # 使用factory_machines配置（优先级最高）
            pass  # 已经设置了
        elif self.heterogeneous_machines:
            # 使用异构配置
            self.factory_machines = self.heterogeneous_machines
        else:
            # 使用统一配置
            self.factory_machines = {
                factory_id: self.machines_per_stage 
                for factory_id in range(self.n_factories)
            }
        
        # 验证数据一致性
        self._validate_data()
        
        # 计算一些有用的统计信息
        self._compute_statistics()
    
    def _validate_data(self):
        """验证输入数据的一致性"""
        assert len(self.machines_per_stage) == self.n_stages, "机器数量与阶段数不匹配"
        assert self.processing_times.shape == (self.n_jobs, self.n_stages), "处理时间矩阵维度错误"
        assert len(self.due_dates) == self.n_jobs, "交货期数量与作业数不匹配"
        assert all(m > 0 for m in self.machines_per_stage), "机器数量必须大于0"
        assert all(self.due_dates > 0), "交货期必须大于0"
    
    def _compute_statistics(self):
        """计算问题统计信息"""
        self.total_processing_time = np.sum(self.processing_times)
        self.avg_processing_time = np.mean(self.processing_times)
        self.max_processing_time = np.max(self.processing_times)
        self.min_processing_time = np.min(self.processing_times)
        
        # 计算每个作业的总处理时间
        self.job_total_times = np.sum(self.processing_times, axis=1)
        
        # 计算理论下界
        self.theoretical_lower_bound = max(
            np.max(self.job_total_times),  # 最长作业时间
            self.total_processing_time / (self.n_factories * np.mean(self.machines_per_stage))  # 平均负载
        )
    
    def evaluate_solution(self, solution: Solution) -> Solution:
        """
        评估解的目标函数值
        
        Args:
            solution: 待评估的解
            
        Returns:
            更新了目标函数值的解
        """
        # 解码解并计算调度
        completion_times, factory_makespans = self._decode_solution(solution)
        
        # 计算目标函数值
        makespan = max(factory_makespans)
        total_tardiness = sum(max(0, completion_times[i] - self.due_dates[i]) 
                            for i in range(self.n_jobs))
        
        # 更新解的信息
        solution.makespan = makespan
        solution.total_tardiness = total_tardiness
        solution.completion_times = completion_times
        solution.factory_makespans = factory_makespans
        
        return solution
    
    def _decode_solution(self, solution: Solution) -> Tuple[List[float], List[float]]:
        """
        解码解，计算各作业完工时间和各工厂完工时间
        支持异构机器配置
        
        Args:
            solution: 待解码的解
            
        Returns:
            (completion_times, factory_makespans): 作业完工时间和工厂完工时间
        """
        completion_times = [0.0] * self.n_jobs
        factory_makespans = [0.0] * self.n_factories
        
        # 为每个工厂计算调度
        for factory_id in range(self.n_factories):
            factory_jobs = solution.job_sequences[factory_id]
            if not factory_jobs:
                continue
                
            # 获取该工厂的机器配置
            factory_machine_config = self.factory_machines[factory_id]
            
            # 各阶段各机器的完工时间
            machine_completion_times = [
                [0.0] * factory_machine_config[stage] 
                for stage in range(self.n_stages)
            ]
            
            # 各作业在各阶段的完工时间
            job_stage_completion = {}
            
            # 按顺序处理该工厂的每个作业
            for job_id in factory_jobs:
                job_stage_completion[job_id] = [0.0] * self.n_stages
                
                # 按阶段顺序处理作业
                for stage in range(self.n_stages):
                    processing_time = self.processing_times[job_id][stage]
                    
                    # 获取该工厂该阶段的机器数量
                    n_machines_in_stage = factory_machine_config[stage]
                    
                    if n_machines_in_stage == 0:
                        # 如果某个阶段没有机器，跳过该阶段
                        job_stage_completion[job_id][stage] = job_stage_completion[job_id][stage-1] if stage > 0 else 0
                        continue
                    
                    # 找到最早可用的机器
                    earliest_machine = 0
                    earliest_time = machine_completion_times[stage][0]
                    
                    for machine in range(1, n_machines_in_stage):
                        if machine_completion_times[stage][machine] < earliest_time:
                            earliest_machine = machine
                            earliest_time = machine_completion_times[stage][machine]
                    
                    # 计算作业在该阶段的开始时间
                    start_time = earliest_time
                    if stage > 0:
                        # 必须等待上一阶段完成
                        start_time = max(start_time, job_stage_completion[job_id][stage-1])
                    
                    # 计算完工时间
                    completion_time = start_time + processing_time
                    job_stage_completion[job_id][stage] = completion_time
                    machine_completion_times[stage][earliest_machine] = completion_time
                
                # 作业的最终完工时间是最后一个阶段的完工时间
                completion_times[job_id] = job_stage_completion[job_id][-1]
            
            # 计算该工厂的完工时间
            if factory_jobs:
                factory_makespans[factory_id] = max(
                    max(machine_completion_times[stage]) if len(machine_completion_times[stage]) > 0 else 0
                    for stage in range(self.n_stages)
                )
        
        return completion_times, factory_makespans
    
    def generate_random_solution(self) -> Solution:
        """生成随机解"""
        # 随机工厂分配
        factory_assignment = np.random.randint(0, self.n_factories, self.n_jobs).tolist()
        
        # 生成各工厂的作业序列
        job_sequences = [[] for _ in range(self.n_factories)]
        for job_id in range(self.n_jobs):
            factory_id = factory_assignment[job_id]
            job_sequences[factory_id].append(job_id)
        
        # 随机打乱各工厂内的作业顺序
        for factory_id in range(self.n_factories):
            np.random.shuffle(job_sequences[factory_id])
        
        solution = Solution(factory_assignment, job_sequences)
        return self.evaluate_solution(solution)
    
    def create_solution(self, factory_assignment: List[int]) -> Solution:
        """
        根据工厂分配创建解
        
        Args:
            factory_assignment: 作业到工厂的分配列表
            
        Returns:
            完整的解对象
        """
        # 生成各工厂的作业序列
        job_sequences = [[] for _ in range(self.n_factories)]
        for job_id in range(self.n_jobs):
            factory_id = factory_assignment[job_id]
            job_sequences[factory_id].append(job_id)
        
        # 按紧急度排序各工厂内的作业（紧急度越小越优先）
        for factory_id in range(self.n_factories):
            if job_sequences[factory_id]:
                job_sequences[factory_id].sort(key=lambda job: self.urgencies[job])
        
        solution = Solution(factory_assignment, job_sequences)
        return self.evaluate_solution(solution)
    
    def is_solution_feasible(self, solution: Solution) -> bool:
        """检查解的可行性"""
        # 检查所有作业都被分配
        assigned_jobs = set()
        for factory_jobs in solution.job_sequences:
            assigned_jobs.update(factory_jobs)
        
        if len(assigned_jobs) != self.n_jobs:
            return False
        
        if assigned_jobs != set(range(self.n_jobs)):
            return False
        
        # 检查工厂分配一致性
        for job_id in range(self.n_jobs):
            factory_id = solution.factory_assignment[job_id]
            if job_id not in solution.job_sequences[factory_id]:
                return False
        
        return True
    
    def get_problem_info(self) -> Dict:
        """获取问题信息"""
        return {
            'n_jobs': self.n_jobs,
            'n_factories': self.n_factories,
            'n_stages': self.n_stages,
            'machines_per_stage': self.machines_per_stage,
            'total_machines': sum(self.machines_per_stage),
            'theoretical_lower_bound': self.theoretical_lower_bound,
            'avg_processing_time': self.avg_processing_time,
            'max_processing_time': self.max_processing_time,
            'min_processing_time': self.min_processing_time
        } 