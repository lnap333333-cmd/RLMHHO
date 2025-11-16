#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试数据生成器
用于生成不同规模的MO-DHFSP测试实例
"""

import numpy as np
from typing import Dict, List

class DataGenerator:
    """测试数据生成器"""
    
    def __init__(self, seed: int = None):
        """
        初始化数据生成器
        
        Args:
            seed: 随机种子，用于结果复现
        """
        if seed is not None:
            np.random.seed(seed)
    
    def generate_problem(self, 
                        n_jobs: int, 
                        n_factories: int, 
                        n_stages: int,
                        machines_per_stage: List[int],
                        processing_time_range: tuple = (1, 20),
                        due_date_tightness: float = 1.5) -> Dict:
        """
        生成MO-DHFSP问题实例
        
        Args:
            n_jobs: 作业数量
            n_factories: 工厂数量
            n_stages: 阶段数量
            machines_per_stage: 各阶段机器数量列表
            processing_time_range: 处理时间范围
            due_date_tightness: 交货期紧张程度（越小越紧张）
            
        Returns:
            问题数据字典
        """
        # 生成处理时间矩阵
        processing_times = self._generate_processing_times(
            n_jobs, n_stages, processing_time_range
        )
        
        # 生成基于紧急度的交货期
        due_dates = self._generate_due_dates(
            processing_times, due_date_tightness
        )
        
        # 生成多样化的紧急度
        urgencies = self._generate_urgencies(n_jobs)
        
        problem_data = {
            'n_jobs': n_jobs,
            'n_factories': n_factories,
            'n_stages': n_stages,
            'machines_per_stage': machines_per_stage,
            'processing_times': processing_times.tolist(),
            'due_dates': due_dates.tolist(),
            'urgencies': urgencies.tolist()
        }
        
        return problem_data
    
    def _generate_processing_times(self, 
                                 n_jobs: int, 
                                 n_stages: int,
                                 time_range: tuple) -> np.ndarray:
        """
        生成处理时间矩阵
        
        Args:
            n_jobs: 作业数量
            n_stages: 阶段数量
            time_range: 时间范围
            
        Returns:
            处理时间矩阵 [n_jobs, n_stages]
        """
        min_time, max_time = time_range
        
        # 基础随机处理时间
        processing_times = np.random.uniform(
            min_time, max_time, (n_jobs, n_stages)
        )
        
        # 添加一些结构化特征
        # 1. 某些作业在某些阶段处理时间较长（模拟复杂作业）
        complex_jobs = np.random.choice(n_jobs, n_jobs // 4, replace=False)
        complex_stages = np.random.choice(n_stages, n_stages // 2, replace=False)
        
        for job in complex_jobs:
            for stage in complex_stages:
                processing_times[job, stage] *= 1.5
        
        # 2. 瓶颈阶段（某个阶段整体处理时间较长）
        if n_stages > 2:
            bottleneck_stage = np.random.randint(0, n_stages)
            processing_times[:, bottleneck_stage] *= 1.3
        
        return processing_times
    
    def _generate_due_dates(self, 
                          processing_times: np.ndarray,
                          tightness: float) -> np.ndarray:
        """
        基于紧急度动态生成交货期
        
        Args:
            processing_times: 处理时间矩阵
            tightness: 紧张程度
            
        Returns:
            交货期数组
        """
        n_jobs, n_stages = processing_times.shape
        
        # 计算每个作业的总处理时间
        job_total_times = np.sum(processing_times, axis=1)
        
        # 计算作业复杂度（考虑处理时间分布）
        job_complexity = np.std(processing_times, axis=1) / np.mean(processing_times, axis=1)
        
        # 定义三种紧急度类型
        n_urgent = int(n_jobs * 0.3)      # 30%紧急订单
        n_normal = int(n_jobs * 0.5)      # 50%标准订单
        n_loose = n_jobs - n_urgent - n_normal  # 20%宽松订单
        
        # 随机分配紧急度类型
        urgency_types = ['urgent'] * n_urgent + ['normal'] * n_normal + ['loose'] * n_loose
        np.random.shuffle(urgency_types)
        
        due_dates = np.zeros(n_jobs)
        
        for i in range(n_jobs):
            base_time = job_total_times[i]
            complexity_factor = 1.0 + 0.2 * job_complexity[i]
            
            if urgency_types[i] == 'urgent':
                # 紧急订单：交货期 = 总处理时间 × (0.8-1.2)
                due_dates[i] = base_time * complexity_factor * np.random.uniform(0.8, 1.2)
            elif urgency_types[i] == 'normal':
                # 标准订单：交货期 = 总处理时间 × (1.2-1.8) × tightness
                due_dates[i] = base_time * complexity_factor * tightness * np.random.uniform(1.2, 1.8)
            else:  # loose
                # 宽松订单：交货期 = 总处理时间 × (1.8-2.5) × tightness
                due_dates[i] = base_time * complexity_factor * tightness * np.random.uniform(1.8, 2.5)
        
        return due_dates
    
    def _generate_urgencies(self, n_jobs: int) -> np.ndarray:
        """
        生成多样化的紧急度系数
        
        Args:
            n_jobs: 作业数量
            
        Returns:
            紧急度数组
        """
        urgencies = np.ones(n_jobs)
        
        # 定义三种紧急度类型
        n_high = int(n_jobs * 0.2)     # 20%高紧急度 (1.5-2.0)
        n_medium = int(n_jobs * 0.6)   # 60%中等紧急度 (1.0-1.5)  
        n_low = n_jobs - n_high - n_medium  # 20%低紧急度 (0.5-1.0)
        
        # 随机分配紧急度类型
        urgency_types = ['high'] * n_high + ['medium'] * n_medium + ['low'] * n_low
        np.random.shuffle(urgency_types)
        
        for i in range(n_jobs):
            if urgency_types[i] == 'high':
                urgencies[i] = np.random.uniform(1.5, 2.0)
            elif urgency_types[i] == 'medium':
                urgencies[i] = np.random.uniform(1.0, 1.5)
            else:  # low
                urgencies[i] = np.random.uniform(0.5, 1.0)
        
        return urgencies
    
    def generate_benchmark_suite(self) -> List[Dict]:
        """
        生成标准测试套件
        
        Returns:
            测试问题列表
        """
        test_suite = []
        
        # 小规模问题集
        small_configs = [
            (10, 2, 3, [2, 2, 2]),
            (15, 2, 3, [2, 3, 2]),
            (20, 3, 4, [2, 3, 2, 3]),
        ]
        
        # 中规模问题集
        medium_configs = [
            (30, 3, 4, [3, 4, 3, 2]),
            (50, 4, 5, [3, 4, 3, 4, 2]),
            (80, 4, 5, [4, 5, 3, 4, 3]),
        ]
        
        # 大规模问题集
        large_configs = [
            (100, 5, 6, [4, 5, 4, 5, 3, 4]),
            (150, 6, 6, [5, 6, 4, 5, 4, 3]),
            (200, 6, 7, [5, 6, 5, 6, 4, 5, 3]),
        ]
        
        # 超大规模问题集
        xlarge_configs = [
            (300, 8, 8, [6, 7, 5, 6, 5, 6, 4, 5]),
            (500, 10, 10, [7, 8, 6, 7, 6, 7, 5, 6, 5, 6]),
        ]
        
        all_configs = [
            ('small', small_configs),
            ('medium', medium_configs), 
            ('large', large_configs),
            ('xlarge', xlarge_configs)
        ]
        
        for scale_name, configs in all_configs:
            for i, (n_jobs, n_factories, n_stages, machines_per_stage) in enumerate(configs):
                problem_data = self.generate_problem(
                    n_jobs=n_jobs,
                    n_factories=n_factories,
                    n_stages=n_stages,
                    machines_per_stage=machines_per_stage,
                    processing_time_range=(1, 20),
                    due_date_tightness=1.5
                )
                
                problem_data['name'] = f"{scale_name}_{i+1}"
                problem_data['scale'] = scale_name
                test_suite.append(problem_data)
        
        return test_suite
    
    def save_problem(self, problem_data: Dict, filename: str):
        """
        保存问题到文件
        
        Args:
            problem_data: 问题数据
            filename: 文件名
        """
        import json
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(problem_data, f, indent=2, ensure_ascii=False)
    
    def load_problem(self, filename: str) -> Dict:
        """
        从文件加载问题
        
        Args:
            filename: 文件名
            
        Returns:
            问题数据
        """
        import json
        
        with open(filename, 'r', encoding='utf-8') as f:
            problem_data = json.load(f)
        
        return problem_data 