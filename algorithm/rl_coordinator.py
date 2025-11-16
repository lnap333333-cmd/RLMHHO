#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
强化学习协调器 - 完整实现
基于深度Q网络(DQN)的智能策略选择和适应性调度系统
"""

import numpy as np
import random
import copy
from typing import List, Tuple, Dict, Optional
from collections import deque, namedtuple
import pickle
import os

from problem.mo_dhfsp import MO_DHFSP_Problem

# 经验元组定义
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class PrioritizedReplayBuffer:
    """优先级经验回放缓冲区"""
    
    def __init__(self, capacity: int = 10000, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha  # 优先级指数
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.beta = 0.4  # 重要性采样指数
        self.beta_increment = 0.001
        
    def push(self, experience: Experience, error: float = None):
        """添加经验"""
        if error is None:
            error = max([p for p in self.priorities] + [1.0])
        
        self.buffer.append(experience)
        self.priorities.append((abs(error) + 1e-5) ** self.alpha)
    
    def sample(self, batch_size: int) -> Tuple[List[Experience], List[int], List[float]]:
        """采样经验批次"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        # 计算采样概率
        priorities = np.array(self.priorities)
        probabilities = priorities / priorities.sum()
        
        # 采样索引
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # 计算重要性权重
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights = weights / weights.max()
        
        # 获取经验
        experiences = [self.buffer[i] for i in indices]
        
        # 更新beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return experiences, indices, weights
    
    def update_priorities(self, indices: List[int], errors: List[float]):
        """更新优先级"""
        for idx, error in zip(indices, errors):
            if idx < len(self.priorities):
                self.priorities[idx] = (abs(error) + 1e-5) ** self.alpha
    
    def __len__(self):
        return len(self.buffer)

class DQNNetwork:
    """深度Q网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [128, 64]):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        
        # 简化的神经网络参数（线性近似）
        self.weights = {}
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        # 输入层到第一个隐藏层
        self.weights['W1'] = np.random.normal(0, 0.1, (self.state_dim, self.hidden_dims[0]))
        self.weights['b1'] = np.zeros(self.hidden_dims[0])
        
        # 隐藏层
        for i in range(len(self.hidden_dims) - 1):
            self.weights[f'W{i+2}'] = np.random.normal(0, 0.1, 
                                                      (self.hidden_dims[i], self.hidden_dims[i+1]))
            self.weights[f'b{i+2}'] = np.zeros(self.hidden_dims[i+1])
        
        # 输出层
        last_hidden = self.hidden_dims[-1]
        self.weights['W_out'] = np.random.normal(0, 0.1, (last_hidden, self.action_dim))
        self.weights['b_out'] = np.zeros(self.action_dim)
    
    def forward(self, state: np.ndarray) -> np.ndarray:
        """前向传播"""
        x = state.flatten() if len(state.shape) > 1 else state
        
        # 第一层
        x = np.dot(x, self.weights['W1']) + self.weights['b1']
        x = np.maximum(0, x)  # ReLU激活
        
        # 隐藏层
        for i in range(len(self.hidden_dims) - 1):
            x = np.dot(x, self.weights[f'W{i+2}']) + self.weights[f'b{i+2}']
            x = np.maximum(0, x)  # ReLU激活
        
        # 输出层
        q_values = np.dot(x, self.weights['W_out']) + self.weights['b_out']
        
        return q_values
    
    def update_weights(self, gradients: Dict, learning_rate: float = 0.001):
        """更新权重"""
        for key, grad in gradients.items():
            if key in self.weights:
                self.weights[key] -= learning_rate * grad
    
    def copy_weights_from(self, other_network):
        """从另一个网络复制权重"""
        self.weights = copy.deepcopy(other_network.weights)

class RLCoordinator:
    """强化学习协调器 - 完整实现"""
    
    def __init__(self, problem: MO_DHFSP_Problem, 
                 state_dim: int = 14,
                 action_dim: int = 7,
                 learning_rate: float = 0.001,
                 epsilon: float = 0.9,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 gamma: float = 0.98,
                 batch_size: int = 32,
                 target_update_freq: int = 100,
                 memory_size: int = 10000):
        """
        初始化强化学习协调器
        
        Args:
            problem: 问题实例
            state_dim: 状态维度
            action_dim: 动作维度
            learning_rate: 学习率
            epsilon: 探索率
            epsilon_decay: 探索衰减率
            epsilon_min: 最小探索率
            gamma: 折扣因子
            batch_size: 批次大小
            target_update_freq: 目标网络更新频率
            memory_size: 经验缓冲区大小
        """
        self.problem = problem
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # 深度Q网络
        self.q_network = DQNNetwork(state_dim, action_dim)
        self.target_network = DQNNetwork(state_dim, action_dim)
        self.target_network.copy_weights_from(self.q_network)
        
        # 经验回放缓冲区
        self.memory = PrioritizedReplayBuffer(memory_size)
        
        # 训练统计
        self.training_step = 0
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.loss_history = deque(maxlen=1000)
        
        # 策略统计
        self.action_counts = np.zeros(action_dim)
        self.action_rewards = np.zeros(action_dim)
        self.action_success_rates = np.zeros(action_dim)
        
        # 状态-动作值历史
        self.q_value_history = deque(maxlen=500)
        
        # 动作空间定义
        self.action_space = {
            0: "强化全局探索",
            1: "强化局部开发", 
            2: "平衡搜索",
            3: "多样性救援",
            4: "精英强化",
            5: "全局重启",
            6: "资源重分配"
        }
        
        print(f"初始化强化学习协调器:")
        print(f"  状态维度: {state_dim}")
        print(f"  动作空间: {action_dim}种策略")
        print(f"  学习率: {learning_rate}")
        print(f"  初始探索率: {epsilon}")
        print(f"  经验缓冲区: {memory_size}")
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        选择动作
        
        Args:
            state: 当前状态
            training: 是否处于训练模式
            
        Returns:
            选择的动作
        """
        # 确保状态维度正确
        if len(state) != self.state_dim:
            # 截断或填充状态向量
            if len(state) > self.state_dim:
                state = state[:self.state_dim]
            else:
                state = np.pad(state, (0, self.state_dim - len(state)), 'constant')
        
        # epsilon-贪婪策略
        if training and random.random() < self.epsilon:
            action = random.randint(0, self.action_dim - 1)
        else:
            # 使用Q网络选择动作
            q_values = self.q_network.forward(state)
            action = np.argmax(q_values)
            
            # 记录Q值历史
            self.q_value_history.append(q_values.copy())
        
        # 更新动作统计
        self.action_counts[action] += 1
        
        return int(action)
    
    def store_experience(self, state: np.ndarray, action: int, reward: float, 
                        next_state: np.ndarray, done: bool = False):
        """存储经验"""
        # 确保状态维度一致
        if len(state) != self.state_dim:
            if len(state) > self.state_dim:
                state = state[:self.state_dim]
            else:
                state = np.pad(state, (0, self.state_dim - len(state)), 'constant')
        
        if len(next_state) != self.state_dim:
            if len(next_state) > self.state_dim:
                next_state = next_state[:self.state_dim]
            else:
                next_state = np.pad(next_state, (0, self.state_dim - len(next_state)), 'constant')
        
        experience = Experience(state, action, reward, next_state, done)
        
        # 计算TD误差作为优先级
        current_q = self.q_network.forward(state)[action]
        if done:
            target_q = reward
        else:
            next_q_values = self.target_network.forward(next_state)
            target_q = reward + self.gamma * np.max(next_q_values)
        
        td_error = abs(target_q - current_q)
        
        self.memory.push(experience, td_error)
    
    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray):
        """更新Q网络"""
        # 存储经验
        self.store_experience(state, action, reward, next_state)
        
        # 更新动作奖励统计
        self.action_rewards[action] += reward
        if reward > 0:
            self.action_success_rates[action] += 1
        
        # 训练网络
        if len(self.memory) >= self.batch_size:
            self._train_network()
        
        # 更新目标网络
        if self.training_step % self.target_update_freq == 0:
            self.target_network.copy_weights_from(self.q_network)
            print(f"🎯 更新目标网络 (步骤: {self.training_step})")
        
        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.training_step += 1
    
    def _train_network(self):
        """训练神经网络"""
        # 采样经验批次
        experiences, indices, weights = self.memory.sample(self.batch_size)
        
        states = np.array([exp.state for exp in experiences])
        actions = np.array([exp.action for exp in experiences])
        rewards = np.array([exp.reward for exp in experiences])
        next_states = np.array([exp.next_state for exp in experiences])
        dones = np.array([exp.done for exp in experiences])
        
        # 计算目标Q值
        current_q_values = np.array([self.q_network.forward(state) for state in states])
        next_q_values = np.array([self.target_network.forward(state) for state in next_states])
        
        target_q_values = current_q_values.copy()
        
        for i in range(len(experiences)):
            if dones[i]:
                target_q_values[i][actions[i]] = rewards[i]
            else:
                target_q_values[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        # 计算损失和梯度
        td_errors = []
        total_loss = 0
        
        for i in range(len(experiences)):
            state = states[i]
            action = actions[i]
            target = target_q_values[i][action]
            current = current_q_values[i][action]
            
            td_error = target - current
            td_errors.append(abs(td_error))
            total_loss += weights[i] * (td_error ** 2)
            
            # 简化的梯度计算和更新
            self._simple_gradient_update(state, action, td_error, weights[i])
        
        # 更新优先级
        self.memory.update_priorities(indices, td_errors)
        
        # 记录损失
        self.loss_history.append(total_loss / len(experiences))
    
    def _simple_gradient_update(self, state: np.ndarray, action: int, td_error: float, weight: float):
        """简化的梯度更新"""
        # 这是一个简化的更新方法，实际应该使用反向传播
        q_values = self.q_network.forward(state)
        
        # 直接调整对应动作的Q值
        adjustment = self.learning_rate * weight * td_error
        
        # 更新输出层权重（简化）
        state_features = state.flatten() if len(state.shape) > 1 else state
        
        # 简单的权重调整
        if hasattr(self.q_network, 'weights'):
            for key in self.q_network.weights:
                if 'W_out' in key:
                    self.q_network.weights[key][action] += adjustment * 0.01 * np.sign(state_features).mean()
    
    def get_strategy_statistics(self) -> Dict:
        """获取策略统计信息"""
        total_actions = np.sum(self.action_counts)
        if total_actions == 0:
            return {}
        
        stats = {}
        for action_id in range(self.action_dim):
            action_name = self.action_space[action_id]
            count = self.action_counts[action_id]
            
            stats[action_name] = {
                'usage_count': int(count),
                'usage_rate': float(count / total_actions),
                'average_reward': float(self.action_rewards[action_id] / max(count, 1)),
                'success_rate': float(self.action_success_rates[action_id] / max(count, 1))
            }
        
        return stats
    
    def get_learning_progress(self) -> Dict:
        """获取学习进度"""
        return {
            'training_steps': self.training_step,
            'epsilon': self.epsilon,
            'memory_size': len(self.memory),
            'average_loss': float(np.mean(self.loss_history)) if self.loss_history else 0.0,
            'average_q_value': float(np.mean([np.mean(q) for q in self.q_value_history])) if self.q_value_history else 0.0
        }
    
    def save_model(self, filepath: str):
        """保存模型"""
        model_data = {
            'q_network_weights': self.q_network.weights,
            'target_network_weights': self.target_network.weights,
            'training_step': self.training_step,
            'epsilon': self.epsilon,
            'action_counts': self.action_counts,
            'action_rewards': self.action_rewards,
            'action_success_rates': self.action_success_rates
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 模型已保存到: {filepath}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        if not os.path.exists(filepath):
            print(f"⚠️ 模型文件不存在: {filepath}")
            return False
        
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.q_network.weights = model_data['q_network_weights']
            self.target_network.weights = model_data['target_network_weights']
            self.training_step = model_data['training_step']
            self.epsilon = model_data['epsilon']
            self.action_counts = model_data['action_counts']
            self.action_rewards = model_data['action_rewards']
            self.action_success_rates = model_data['action_success_rates']
            
            print(f"📖 模型已从 {filepath} 加载")
            return True
            
        except Exception as e:
            print(f"❌ 加载模型失败: {e}")
            return False
    
    def get_action_recommendations(self, state: np.ndarray) -> List[Tuple[str, float]]:
        """获取动作推荐及其置信度"""
        if len(state) != self.state_dim:
            if len(state) > self.state_dim:
                state = state[:self.state_dim]
            else:
                state = np.pad(state, (0, self.state_dim - len(state)), 'constant')
        
        q_values = self.q_network.forward(state)
        
        # 计算softmax概率作为置信度
        exp_q = np.exp(q_values - np.max(q_values))
        probabilities = exp_q / np.sum(exp_q)
        
        # 按置信度排序
        sorted_indices = np.argsort(probabilities)[::-1]
        
        recommendations = []
        for idx in sorted_indices:
            action_name = self.action_space[idx]
            confidence = float(probabilities[idx])
            recommendations.append((action_name, confidence))
        
        return recommendations 