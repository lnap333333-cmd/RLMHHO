#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强混沌映射系统 - 完整实现
支持四层鹰群分组的不同混沌映射需求
用于增强哈里斯鹰优化算法的随机性和多样性
"""

import numpy as np
from typing import List, Dict, Optional
import random

class ChaoticMaps:
    """增强混沌映射类 - 支持四种映射"""
    
    def __init__(self, seed: Optional[int] = None):
        """
        初始化混沌映射
        
        Args:
            seed: 随机种子，用于可重复实验
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # 初始化各映射的状态值
        self.x_logistic = np.random.uniform(0.01, 0.99)  # Logistic映射状态
        self.x_tent = np.random.uniform(0.01, 0.99)      # Tent映射状态  
        self.x_sine = np.random.uniform(0.01, 0.99)      # Sine映射状态
        self.x_chebyshev = np.random.uniform(-1, 1)      # Chebyshev映射状态
        
        # 映射参数
        self.logistic_r = 4.0           # Logistic映射参数
        self.tent_a = 2.0               # Tent映射参数
        self.sine_a = 1.0               # Sine映射参数
        self.chebyshev_n = 4            # Chebyshev映射阶数
        
        # 各组专用的映射类型
        self.group_chaos_mapping = {
            'exploration': 'logistic',    # 探索组使用Logistic映射
            'exploitation': 'tent',       # 开发组使用Tent映射
            'balance': 'sine',            # 平衡组使用Sine映射
            'elite': 'chebyshev'          # 精英组使用Chebyshev映射
        }
        
        # 映射质量统计
        self.map_usage_count = {
            'logistic': 0,
            'tent': 0,
            'sine': 0,
            'chebyshev': 0
        }
        
        # 映射性能历史
        self.map_performance = {
            'logistic': [],
            'tent': [],
            'sine': [],
            'chebyshev': []
        }
        
        print(f"初始化增强混沌映射系统:")
        print(f"  Logistic映射 (探索组): r={self.logistic_r}")
        print(f"  Tent映射 (开发组): a={self.tent_a}")
        print(f"  Sine映射 (平衡组): a={self.sine_a}")
        print(f"  Chebyshev映射 (精英组): n={self.chebyshev_n}")
    
    def logistic_map(self, r: float = None) -> float:
        """
        Logistic映射 - 适合探索阶段
        x_{n+1} = r * x_n * (1 - x_n)
        
        Args:
            r: 映射参数，默认使用4.0（混沌状态）
            
        Returns:
            映射值
        """
        if r is None:
            r = self.logistic_r
        
        self.x_logistic = r * self.x_logistic * (1 - self.x_logistic)
        
        # 防止陷入固定点
        if self.x_logistic < 1e-10 or self.x_logistic > 1 - 1e-10:
            self.x_logistic = np.random.uniform(0.01, 0.99)
        
        self.map_usage_count['logistic'] += 1
        return self.x_logistic
    
    def tent_map(self, a: float = None) -> float:
        """
        Tent映射 - 适合开发阶段
        x_{n+1} = a * x_n if x_n < 0.5, else a * (1 - x_n)
        
        Args:
            a: 映射参数，默认使用2.0
            
        Returns:
            映射值
        """
        if a is None:
            a = self.tent_a
        
        if self.x_tent < 0.5:
            self.x_tent = a * self.x_tent
        else:
            self.x_tent = a * (1 - self.x_tent)
        
        # 确保值在有效范围内
        self.x_tent = max(0.001, min(0.999, self.x_tent))
        
        self.map_usage_count['tent'] += 1
        return self.x_tent
    
    def sine_map(self, a: float = None) -> float:
        """
        Sine映射 - 适合平衡搜索
        x_{n+1} = a * sin(π * x_n)
        
        Args:
            a: 映射参数，默认使用1.0
            
        Returns:
            映射值
        """
        if a is None:
            a = self.sine_a
        
        self.x_sine = a * np.sin(np.pi * self.x_sine)
        self.x_sine = abs(self.x_sine)  # 取绝对值确保为正
        
        # 防止值过小
        if self.x_sine < 1e-10:
            self.x_sine = np.random.uniform(0.01, 0.99)
        
        self.map_usage_count['sine'] += 1
        return self.x_sine
    
    def chebyshev_map(self, n: int = None) -> float:
        """
        Chebyshev映射 - 适合精英优化
        x_{n+1} = cos(n * arccos(x_n))
        
        Args:
            n: 映射阶数，默认使用4
            
        Returns:
            映射值（范围[0,1]）
        """
        if n is None:
            n = self.chebyshev_n
        
        # 确保输入在有效范围内
        self.x_chebyshev = max(-0.999, min(0.999, self.x_chebyshev))
        
        # Chebyshev映射
        self.x_chebyshev = np.cos(n * np.arccos(self.x_chebyshev))
        
        # 转换到[0,1]范围
        normalized_value = (self.x_chebyshev + 1) / 2
        
        self.map_usage_count['chebyshev'] += 1
        return normalized_value
    
    def get_chaos_values(self, count: int, map_type: Optional[str] = None) -> List[float]:
        """
        获取混沌值序列
        
        Args:
            count: 需要的值数量
            map_type: 指定映射类型，可选：'logistic', 'tent', 'sine', 'chebyshev'
            
        Returns:
            混沌值列表
        """
        values = []
        
        if map_type is not None:
            # 使用指定的映射类型
            for _ in range(count):
                if map_type == 'logistic':
                    values.append(self.logistic_map())
                elif map_type == 'tent':
                    values.append(self.tent_map())
                elif map_type == 'sine':
                    values.append(self.sine_map())
                elif map_type == 'chebyshev':
                    values.append(self.chebyshev_map())
                else:
                    # 默认使用logistic
                    values.append(self.logistic_map())
        else:
            # 轮流使用不同的混沌映射
            for i in range(count):
                map_index = i % 4
                if map_index == 0:
                    values.append(self.logistic_map())
                elif map_index == 1:
                    values.append(self.tent_map())
                elif map_index == 2:
                    values.append(self.sine_map())
                else:
                    values.append(self.chebyshev_map())
        
        return values
    
    def get_group_chaos_values(self, group_name: str, count: int) -> List[float]:
        """
        为特定鹰群组获取专用的混沌值序列
        
        Args:
            group_name: 组名称 ('exploration', 'exploitation', 'balance', 'elite')
            count: 需要的值数量
            
        Returns:
            该组专用的混沌值列表
        """
        map_type = self.group_chaos_mapping.get(group_name, 'logistic')
        return self.get_chaos_values(count, map_type)
    
    def adaptive_chaos_selection(self, performance_scores: Dict[str, float]) -> str:
        """
        基于性能自适应选择混沌映射
        
        Args:
            performance_scores: 各映射的性能分数
            
        Returns:
            选择的映射类型
        """
        if not performance_scores:
            return 'logistic'
        
        # 计算选择概率（基于性能的softmax）
        scores = np.array(list(performance_scores.values()))
        if np.std(scores) < 1e-6:  # 性能相近时随机选择
            return random.choice(list(performance_scores.keys()))
        
        # Softmax概率
        exp_scores = np.exp(scores - np.max(scores))
        probabilities = exp_scores / np.sum(exp_scores)
        
        # 根据概率选择
        map_types = list(performance_scores.keys())
        selected_idx = np.random.choice(len(map_types), p=probabilities)
        
        return map_types[selected_idx]
    
    def enhanced_chaos_sequence(self, count: int, intensity: float = 0.5, 
                               diversity_boost: bool = False) -> List[float]:
        """
        生成增强的混沌序列
        
        Args:
            count: 序列长度
            intensity: 混沌强度 [0,1]
            diversity_boost: 是否启用多样性增强
            
        Returns:
            增强混沌序列
        """
        values = []
        
        # 根据强度选择映射组合
        if intensity < 0.3:
            # 低强度：主要使用tent映射（稳定）
            primary_maps = ['tent'] * 3 + ['sine']
        elif intensity < 0.7:
            # 中强度：平衡使用
            primary_maps = ['logistic', 'tent', 'sine', 'chebyshev']
        else:
            # 高强度：主要使用logistic映射（混沌）
            primary_maps = ['logistic'] * 2 + ['chebyshev'] * 2
        
        for i in range(count):
            if diversity_boost and i % 5 == 0:
                # 每5个值插入一个高多样性值
                chaos_value = self.get_diverse_chaos_value()
            else:
                # 正常混沌值
                map_type = primary_maps[i % len(primary_maps)]
                chaos_value = self.get_chaos_values(1, map_type)[0]
            
            values.append(chaos_value)
        
        return values
    
    def get_diverse_chaos_value(self) -> float:
        """获取高多样性的混沌值"""
        # 组合多个映射的结果
        logistic_val = self.logistic_map()
        tent_val = self.tent_map()
        sine_val = self.sine_map()
        chebyshev_val = self.chebyshev_map()
        
        # 加权组合
        combined = 0.3 * logistic_val + 0.2 * tent_val + 0.2 * sine_val + 0.3 * chebyshev_val
        
        return min(max(combined, 0.001), 0.999)
    
    def reset_chaos_states(self):
        """重置所有混沌映射的状态"""
        self.x_logistic = np.random.uniform(0.01, 0.99)
        self.x_tent = np.random.uniform(0.01, 0.99)
        self.x_sine = np.random.uniform(0.01, 0.99)
        self.x_chebyshev = np.random.uniform(-1, 1)
        
        print("🔄 重置所有混沌映射状态")
    
    def get_chaos_statistics(self) -> Dict:
        """获取混沌映射使用统计"""
        total_usage = sum(self.map_usage_count.values())
        
        if total_usage == 0:
            return {}
        
        stats = {}
        for map_type, count in self.map_usage_count.items():
            stats[map_type] = {
                'usage_count': count,
                'usage_rate': count / total_usage,
                'current_state': self._get_current_state(map_type)
            }
        
        return stats
    
    def _get_current_state(self, map_type: str) -> float:
        """获取指定映射的当前状态"""
        if map_type == 'logistic':
            return self.x_logistic
        elif map_type == 'tent':
            return self.x_tent
        elif map_type == 'sine':
            return self.x_sine
        elif map_type == 'chebyshev':
            return (self.x_chebyshev + 1) / 2  # 归一化到[0,1]
        else:
            return 0.0
    
    def update_performance(self, map_type: str, performance_score: float):
        """更新映射性能记录"""
        if map_type in self.map_performance:
            self.map_performance[map_type].append(performance_score)
            # 保持最近50次记录
            if len(self.map_performance[map_type]) > 50:
                self.map_performance[map_type] = self.map_performance[map_type][-50:]
    
    def get_best_performing_map(self) -> str:
        """获取性能最好的映射类型"""
        avg_performances = {}
        
        for map_type, scores in self.map_performance.items():
            if scores:
                avg_performances[map_type] = np.mean(scores)
        
        if not avg_performances:
            return 'logistic'  # 默认返回
        
        return max(avg_performances, key=avg_performances.get)
    
    def chaos_parameter_adaptation(self, improvement_rate: float):
        """基于改进率自适应调整混沌参数"""
        if improvement_rate < 0.1:
            # 改进率低，增加混沌强度
            self.logistic_r = min(4.0, self.logistic_r + 0.1)
            self.tent_a = min(2.0, self.tent_a + 0.05)
            self.chebyshev_n = min(6, self.chebyshev_n + 1)
        elif improvement_rate > 0.5:
            # 改进率高，减少混沌强度
            self.logistic_r = max(3.5, self.logistic_r - 0.05)
            self.tent_a = max(1.8, self.tent_a - 0.02)
            self.chebyshev_n = max(2, self.chebyshev_n - 1)
    
    def generate_chaos_matrix(self, rows: int, cols: int, map_type: str = 'mixed') -> np.ndarray:
        """
        生成混沌矩阵
        
        Args:
            rows: 矩阵行数
            cols: 矩阵列数  
            map_type: 映射类型或'mixed'
            
        Returns:
            混沌值矩阵
        """
        matrix = np.zeros((rows, cols))
        
        if map_type == 'mixed':
            # 混合使用所有映射
            for i in range(rows):
                for j in range(cols):
                    map_idx = (i * cols + j) % 4
                    if map_idx == 0:
                        matrix[i, j] = self.logistic_map()
                    elif map_idx == 1:
                        matrix[i, j] = self.tent_map()
                    elif map_idx == 2:
                        matrix[i, j] = self.sine_map()
                    else:
                        matrix[i, j] = self.chebyshev_map()
        else:
            # 使用指定映射
            for i in range(rows):
                for j in range(cols):
                    matrix[i, j] = self.get_chaos_values(1, map_type)[0]
        
        return matrix 