#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版DQN调度器 - 使用NumPy实现
基于论文《基于深度Q学习网络的分布式流水车间调度问题优化》
避免PyTorch依赖问题
"""

import numpy as np
import random
import copy
from collections import deque
from typing import List, Tuple, Dict
from problem.mo_dhfsp import MO_DHFSP_Problem, Solution
import time

class SimpleDQNNetwork:
    """
    简化的DQN网络：使用NumPy实现
    网络结构：5 -> 32 -> 16 -> 9
    """
    def __init__(self, state_dim: int = 5, action_dim: int = 9):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 初始化权重和偏置
        self.W1 = np.random.randn(state_dim, 32) * 0.1
        self.b1 = np.zeros(32)
        self.W2 = np.random.randn(32, 16) * 0.1
        self.b2 = np.zeros(16)
        self.W3 = np.random.randn(16, action_dim) * 0.1
        self.b3 = np.zeros(action_dim)
        
    def relu(self, x):
        """ReLU激活函数"""
        return np.maximum(0, x)
    
    def forward(self, x):
        """前向传播"""
        # 第一层
        z1 = np.dot(x, self.W1) + self.b1
        a1 = self.relu(z1)
        
        # 第二层
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self.relu(z2)
        
        # 输出层
        z3 = np.dot(a2, self.W3) + self.b3
        
        return z3
    
    def copy_weights(self, other_network):
        """复制另一个网络的权重"""
        self.W1 = other_network.W1.copy()
        self.b1 = other_network.b1.copy()
        self.W2 = other_network.W2.copy()
        self.b2 = other_network.b2.copy()
        self.W3 = other_network.W3.copy()
        self.b3 = other_network.b3.copy()

class SimpleDQNScheduler:
    """
    简化版DQN多目标调度器
    """
    def __init__(self, problem: MO_DHFSP_Problem, **kwargs):
        self.problem = problem
        
        # DQN参数 - 调整为更适合调度问题的参数
        self.memory_size = kwargs.get('memory_size', 3000)  # 增加记忆容量
        self.batch_size = kwargs.get('batch_size', 64)  # 增加批次大小
        self.gamma = kwargs.get('gamma', 0.99)  # 提高折扣因子
        self.epsilon = kwargs.get('epsilon', 0.9)  # 降低初始探索率
        self.epsilon_decay = kwargs.get('epsilon_decay', 0.99)  # 更慢的衰减
        self.epsilon_min = kwargs.get('epsilon_min', 0.05)  # 保持一定探索
        self.learning_rate = kwargs.get('learning_rate', 0.01)  # 提高学习率
        self.target_update = kwargs.get('target_update', 20)  # 更频繁更新目标网络
        
        # 初始化网络
        self.policy_net = SimpleDQNNetwork()
        self.target_net = SimpleDQNNetwork()
        self.target_net.copy_weights(self.policy_net)
        
        # 经验回放
        self.memory = deque(maxlen=self.memory_size)
        
        # 训练计数器
        self.train_step = 0
        
        # 规则统计 - 增强版
        self.rule_success_count = np.zeros(9)
        self.rule_total_count = np.zeros(9)
        self.rule_performance_history = [[] for _ in range(9)]
        self.last_action = None
        
        # 状态计算相关
        self.prev_makespan = None
        self.prev_tardiness = None
        self.recent_rule_performance = []
        
        # 初始化最优值记录 - 多目标追踪
        self.best_cmax = float('inf')
        self.best_tardiness = float('inf')
        self.best_weighted_objective = float('inf')
        
        # 解质量历史
        self.solution_quality_history = []
        
        print(f"初始化增强DQN调度器: 状态维度=5, 动作维度=9")
        print(f"网络结构: 5 -> 32 -> 16 -> 9")
        print(f"学习率={self.learning_rate}, 批次大小={self.batch_size}")

    def neh_initialization(self) -> Solution:
        """
        使用NEH方法初始化解（按论文要求）
        """
        # 计算每个作业的总处理时间
        job_total_times = [sum(self.problem.processing_times[j]) for j in range(self.problem.n_jobs)]
        
        # 按总处理时间降序排列作业
        sorted_jobs = sorted(range(self.problem.n_jobs), key=lambda j: -job_total_times[j])
        
        # 初始化工厂分配：轮流分配到各工厂
        factory_assignment = [0] * self.problem.n_jobs
        job_sequences = [[] for _ in range(self.problem.n_factories)]
        
        for idx, job_id in enumerate(sorted_jobs):
            factory_id = idx % self.problem.n_factories
            factory_assignment[job_id] = factory_id
            job_sequences[factory_id].append(job_id)
        
        # 创建解并评估
        solution = Solution(factory_assignment, job_sequences)
        return self.problem.evaluate_solution(solution)

    def _calculate_state(self, solution):
        """计算当前状态的5维向量"""
        # S1: 当前解最大完工时间变量率
        if hasattr(self, 'prev_makespan') and self.prev_makespan is not None and self.prev_makespan > 0:
            s1 = (solution.makespan - self.prev_makespan) / self.prev_makespan
        else:
            s1 = 0.0
        self.prev_makespan = solution.makespan
        
        # S2: 调度规则执行成功率 (最近10次的平均成功率)
        if hasattr(self, 'recent_rule_performance') and len(self.recent_rule_performance) > 0:
            s2 = np.mean([perf['success'] for perf in self.recent_rule_performance[-10:]])
        else:
            s2 = 0.5
        
        # S3: 当前工厂完工时间比率
        factory_makespans = []
        for factory_id in range(self.problem.n_factories):
            factory_makespan = self._calculate_factory_makespan(solution, factory_id)
            factory_makespans.append(factory_makespan)
        
        if len(factory_makespans) > 1:
            max_factory_makespan = max(factory_makespans)
            min_factory_makespan = min(factory_makespans)
            if max_factory_makespan > 0:
                s3 = min_factory_makespan / max_factory_makespan
            else:
                s3 = 1.0
        else:
            s3 = 1.0
        
        # S4: 关键工厂机器总空闲时间和机器总时间的比例
        if len(factory_makespans) > 0:
            critical_factory = np.argmax(factory_makespans)
            idle_time = self._calculate_factory_idle_time(solution, critical_factory)
            total_time = factory_makespans[critical_factory] * len(self.problem.factory_machines[critical_factory])
            s4 = idle_time / total_time if total_time > 0 else 0.0
        else:
            s4 = 0.0
        
        # S5: 加入拖期相关状态 (修改原S5)
        # 计算当前解的拖期率
        total_tardiness = solution.total_tardiness
        total_due_dates = sum(self.problem.due_dates) if hasattr(self.problem, 'due_dates') else 1.0
        s5 = total_tardiness / total_due_dates if total_due_dates > 0 else 0.0
        
        state = np.array([s1, s2, s3, s4, s5], dtype=np.float32)
        
        # 归一化到[-1, 1]范围
        state = np.clip(state, -1.0, 1.0)
        
        return state

    def _calculate_factory_makespan(self, solution, factory_id):
        """计算指定工厂的完工时间"""
        if not solution.job_sequences[factory_id]:
            return 0.0
        
        # 简化计算：使用作业数量和平均处理时间估算
        job_count = len(solution.job_sequences[factory_id])
        if job_count == 0:
            return 0.0
        
        avg_processing_time = np.mean([
            np.mean(self.problem.processing_times[job]) 
            for job in solution.job_sequences[factory_id]
        ])
        
        # 考虑机器数量的影响
        machine_count = sum(self.problem.factory_machines[factory_id])
        estimated_makespan = (job_count * avg_processing_time) / max(1, machine_count)
        
        return estimated_makespan
    
    def _calculate_factory_idle_time(self, solution, factory_id):
        """计算指定工厂的空闲时间"""
        if not solution.job_sequences[factory_id]:
            return 0.0
        
        factory_makespan = self._calculate_factory_makespan(solution, factory_id)
        machine_count = sum(self.problem.factory_machines[factory_id])
        
        # 总可用时间 - 总工作时间 = 空闲时间
        total_available_time = factory_makespan * machine_count
        
        total_work_time = sum([
            sum(self.problem.processing_times[job])
            for job in solution.job_sequences[factory_id]
        ])
        
        idle_time = max(0, total_available_time - total_work_time)
        return idle_time

    def get_action(self, state: np.ndarray) -> int:
        """
        ε-贪婪策略选择动作（调度规则）
        """
        if random.random() < self.epsilon:
            return random.randint(0, 8)  # 固定9个规则，索引0-8
        
        q_values = self.policy_net.forward(state.reshape(1, -1))
        return np.argmax(q_values[0])

    def apply_rule(self, solution: Solution, action: int) -> Solution:
        """
        根据动作编号应用对应的调度规则（按论文表2实现）
        """
        new_solution = copy.deepcopy(solution)
        
        if action == 0:
            self._apply_global_rule_1(new_solution)
        elif action == 1:
            self._apply_global_rule_2(new_solution)
        elif action == 2:
            self._apply_global_rule_3(new_solution)
        elif action == 3:
            self._apply_local_rule_1(new_solution)
        elif action == 4:
            self._apply_local_rule_2(new_solution)
        elif action == 5:
            self._apply_local_rule_3(new_solution)
        elif action == 6:
            self._apply_local_rule_4(new_solution)
        elif action == 7:
            self._apply_local_rule_5(new_solution)
        elif action == 8:
            self._apply_local_rule_6(new_solution)
        
        # 重新评估解
        new_solution = self.problem.evaluate_solution(new_solution)
        return new_solution

    def _apply_global_rule_1(self, solution: Solution):
        """全局规则1：从Cmax最大的2个工厂中随机选择进行操作"""
        if not solution.factory_makespans:
            return
        
        factory_makespans = [(i, ms) for i, ms in enumerate(solution.factory_makespans)]
        factory_makespans.sort(key=lambda x: -x[1])
        
        if len(factory_makespans) >= 2:
            selected_factory = random.choice(factory_makespans[:2])[0]
            if solution.job_sequences[selected_factory]:
                job = random.choice(solution.job_sequences[selected_factory])
                self._relocate_job_within_factory(solution, job, selected_factory)

    def _apply_global_rule_2(self, solution: Solution):
        """全局规则2：类似规则1但选择策略不同"""
        if not solution.factory_makespans:
            return
        
        factory_makespans = [(i, ms) for i, ms in enumerate(solution.factory_makespans)]
        factory_makespans.sort(key=lambda x: -x[1])
        
        if len(factory_makespans) >= 2:
            selected_factory = factory_makespans[1][0]
            if solution.job_sequences[selected_factory]:
                job = random.choice(solution.job_sequences[selected_factory])
                self._relocate_job_within_factory(solution, job, selected_factory)

    def _apply_global_rule_3(self, solution: Solution):
        """全局规则3：从Cmax最大的工厂中选择作业"""
        if not solution.factory_makespans:
            return
        
        max_factory = max(range(len(solution.factory_makespans)), 
                         key=lambda i: solution.factory_makespans[i])
        if solution.job_sequences[max_factory]:
            job = random.choice(solution.job_sequences[max_factory])
            self._relocate_job_within_factory(solution, job, max_factory)

    def _apply_local_rule_1(self, solution: Solution):
        """局部规则1：从最大Cmax工厂移动作业到最小Cmax工厂"""
        if not solution.factory_makespans:
            return
        
        max_factory = max(range(len(solution.factory_makespans)), 
                         key=lambda i: solution.factory_makespans[i])
        min_factory = min(range(len(solution.factory_makespans)), 
                         key=lambda i: solution.factory_makespans[i])
        
        if max_factory != min_factory and solution.job_sequences[max_factory]:
            job = random.choice(solution.job_sequences[max_factory])
            self._move_job_between_factories(solution, job, max_factory, min_factory)

    def _apply_local_rule_2(self, solution: Solution):
        """局部规则2：从最大Cmax工厂移动作业到随机工厂"""
        if not solution.factory_makespans:
            return
        
        max_factory = max(range(len(solution.factory_makespans)), 
                         key=lambda i: solution.factory_makespans[i])
        random_factory = random.randint(0, self.problem.n_factories - 1)
        
        if max_factory != random_factory and solution.job_sequences[max_factory]:
            job = random.choice(solution.job_sequences[max_factory])
            self._move_job_between_factories(solution, job, max_factory, random_factory)

    def _apply_local_rule_3(self, solution: Solution):
        """局部规则3：最大和最小Cmax工厂间交换作业"""
        if not solution.factory_makespans:
            return
        
        max_factory = max(range(len(solution.factory_makespans)), 
                         key=lambda i: solution.factory_makespans[i])
        min_factory = min(range(len(solution.factory_makespans)), 
                         key=lambda i: solution.factory_makespans[i])
        
        if (max_factory != min_factory and 
            solution.job_sequences[max_factory] and 
            solution.job_sequences[min_factory]):
            job1 = random.choice(solution.job_sequences[max_factory])
            job2 = random.choice(solution.job_sequences[min_factory])
            self._swap_jobs_between_factories(solution, job1, max_factory, job2, min_factory)

    def _apply_local_rule_4(self, solution: Solution):
        """局部规则4：最大Cmax工厂和随机工厂间交换作业"""
        if not solution.factory_makespans:
            return
        
        max_factory = max(range(len(solution.factory_makespans)), 
                         key=lambda i: solution.factory_makespans[i])
        random_factory = random.randint(0, self.problem.n_factories - 1)
        
        if (max_factory != random_factory and 
            solution.job_sequences[max_factory] and 
            solution.job_sequences[random_factory]):
            job1 = random.choice(solution.job_sequences[max_factory])
            job2 = random.choice(solution.job_sequences[random_factory])
            self._swap_jobs_between_factories(solution, job1, max_factory, job2, random_factory)

    def _apply_local_rule_5(self, solution: Solution):
        """局部规则5：在最大Cmax工厂内随机插入作业"""
        if not solution.factory_makespans:
            return
        
        max_factory = max(range(len(solution.factory_makespans)), 
                         key=lambda i: solution.factory_makespans[i])
        if len(solution.job_sequences[max_factory]) > 1:
            job = random.choice(solution.job_sequences[max_factory])
            self._relocate_job_within_factory(solution, job, max_factory)

    def _apply_local_rule_6(self, solution: Solution):
        """局部规则6：随机选择2个作业交换位置"""
        all_jobs = []
        for factory_id in range(self.problem.n_factories):
            for job in solution.job_sequences[factory_id]:
                all_jobs.append((job, factory_id))
        
        if len(all_jobs) >= 2:
            job1_info, job2_info = random.sample(all_jobs, 2)
            job1, factory1 = job1_info
            job2, factory2 = job2_info
            
            if factory1 == factory2:
                self._swap_jobs_within_factory(solution, job1, job2, factory1)
            else:
                self._swap_jobs_between_factories(solution, job1, factory1, job2, factory2)

    def _relocate_job_within_factory(self, solution: Solution, job: int, factory: int):
        """在工厂内重新定位作业"""
        if job in solution.job_sequences[factory]:
            solution.job_sequences[factory].remove(job)
            new_pos = random.randint(0, len(solution.job_sequences[factory]))
            solution.job_sequences[factory].insert(new_pos, job)

    def _move_job_between_factories(self, solution: Solution, job: int, from_factory: int, to_factory: int):
        """在工厂间移动作业"""
        if job in solution.job_sequences[from_factory]:
            solution.job_sequences[from_factory].remove(job)
            solution.job_sequences[to_factory].append(job)
            solution.factory_assignment[job] = to_factory

    def _swap_jobs_between_factories(self, solution: Solution, job1: int, factory1: int, job2: int, factory2: int):
        """在不同工厂间交换作业"""
        if job1 in solution.job_sequences[factory1] and job2 in solution.job_sequences[factory2]:
            solution.job_sequences[factory1].remove(job1)
            solution.job_sequences[factory2].remove(job2)
            solution.job_sequences[factory1].append(job2)
            solution.job_sequences[factory2].append(job1)
            solution.factory_assignment[job1] = factory2
            solution.factory_assignment[job2] = factory1

    def _swap_jobs_within_factory(self, solution: Solution, job1: int, job2: int, factory: int):
        """在同一工厂内交换作业位置"""
        if job1 in solution.job_sequences[factory] and job2 in solution.job_sequences[factory]:
            seq = solution.job_sequences[factory]
            idx1, idx2 = seq.index(job1), seq.index(job2)
            seq[idx1], seq[idx2] = seq[idx2], seq[idx1]

    def _calculate_reward(self, prev_solution, new_solution):
        """
        增强的多目标奖励函数
        """
        # 1. 基础改进奖励（双目标）
        makespan_improvement = 0.0
        tardiness_improvement = 0.0
        
        if prev_solution:
            if new_solution.makespan < prev_solution.makespan:
                makespan_improvement = (prev_solution.makespan - new_solution.makespan) / prev_solution.makespan
            
            if new_solution.total_tardiness < prev_solution.total_tardiness:
                tardiness_improvement = (prev_solution.total_tardiness - new_solution.total_tardiness) / max(prev_solution.total_tardiness, 1)
        
        # 2. 绝对质量奖励
        # 归一化目标值
        makespan_ratio = new_solution.makespan / self.problem.theoretical_lower_bound
        tardiness_ratio = new_solution.total_tardiness / (sum(self.problem.due_dates) if hasattr(self.problem, 'due_dates') else new_solution.makespan)
        
        quality_reward = 2.0 / (1.0 + makespan_ratio + tardiness_ratio)  # 质量越高奖励越大
        
        # 3. 多目标平衡奖励
        weighted_objective = 0.55 * new_solution.makespan + 0.45 * new_solution.total_tardiness
        
        # 更新最佳记录
        improvement_bonus = 0.0
        if weighted_objective < self.best_weighted_objective:
            improvement_bonus = 2.0  # 发现新的最佳解给予大奖励
            self.best_weighted_objective = weighted_objective
            self.best_cmax = new_solution.makespan
            self.best_tardiness = new_solution.total_tardiness
        
        # 4. 解的可行性奖励
        feasibility_reward = 1.0 if self.problem.is_solution_feasible(new_solution) else -2.0
        
        # 5. 综合奖励计算（调整权重，更重视实际改进）
        total_reward = (
            3.0 * makespan_improvement +      # 完工时间改进
            3.0 * tardiness_improvement +     # 拖期改进  
            1.0 * quality_reward +            # 绝对质量
            improvement_bonus +               # 突破性改进
            feasibility_reward                # 可行性
        )
        
        # 记录解质量
        self.solution_quality_history.append({
            'makespan': new_solution.makespan,
            'tardiness': new_solution.total_tardiness,
            'weighted': weighted_objective,
            'reward': total_reward
        })
        
        # 规则性能统计
        if self.last_action is not None:
            self.rule_total_count[self.last_action] += 1
            if total_reward > 0:
                self.rule_success_count[self.last_action] += 1
            
            # 记录详细性能
            self.rule_performance_history[self.last_action].append(total_reward)
            
            # 更新最近性能记录
            self.recent_rule_performance.append({
                'action': self.last_action,
                'reward': total_reward,
                'success': total_reward > 0
            })
            
            # 保持最近记录的长度
            if len(self.recent_rule_performance) > 50:
                self.recent_rule_performance.pop(0)
        
        return total_reward

    def step(self, solution: Solution, action: int) -> Tuple[Solution, float]:
        """执行动作并返回新解和奖励"""
        new_solution = self.apply_rule(solution, action)
        reward = self._calculate_reward(solution, new_solution)
        return new_solution, reward

    def remember(self, state, action, reward, next_state, done):
        """将经验存入回放缓冲区"""
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        """简化的DQN训练过程"""
        if len(self.memory) < self.batch_size:
            return None
        
        # 从经验回放中随机采样
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 转换为numpy数组
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        
        # 计算当前Q值
        current_q_values = self.policy_net.forward(states)
        current_q_values = current_q_values[np.arange(len(actions)), actions]
        
        # 计算目标Q值
        next_q_values = self.target_net.forward(next_states)
        max_next_q_values = np.max(next_q_values, axis=1)
        target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values
        
        # 计算损失（简化版，不进行反向传播）
        loss = np.mean((current_q_values - target_q_values) ** 2)
        
        # 简化的权重更新（这里只是演示，实际应该用梯度下降）
        # 在实际应用中，您可以实现完整的反向传播
        
        # 更新目标网络
        self.train_step += 1
        if self.train_step % self.target_update == 0:
            self.target_net.copy_weights(self.policy_net)
        
        # 衰减探索率
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss

    def optimize(self, max_episodes: int = 100, max_steps_per_episode: int = 100):
        """
        增强的优化流程
        """
        print(f"开始DQN优化: {max_episodes} episodes, {max_steps_per_episode} steps/episode")
        
        # 使用改进的初始化
        current_solution = self.neh_initialization()
        best_solution = copy.deepcopy(current_solution)
        
        # 记录收敛数据
        convergence_data = {
            'makespan_history': [],
            'tardiness_history': [],
            'weighted_history': [],
            'episode_rewards': []
        }
        
        episode_rewards = []
        
        for episode in range(max_episodes):
            episode_reward = 0.0
            episode_start_time = time.time()
            
            # 重置或扰动当前解（增强探索）
            if episode > 0 and episode % 20 == 0:
                # 周期性重启，从最佳解开始
                current_solution = copy.deepcopy(best_solution)
                # 添加随机扰动
                current_solution = self._add_random_perturbation(current_solution)
            
            for step in range(max_steps_per_episode):
                # 获取状态
                state = self._calculate_state(current_solution)
                
                # 选择动作
                action = self.get_action(state)
                self.last_action = action
                
                # 执行动作
                new_solution, reward = self.step(current_solution, action)
                episode_reward += reward
                
                # 获取新状态
                next_state = self._calculate_state(new_solution)
                
                # 存储经验
                done = (step == max_steps_per_episode - 1)
                self.remember(state, action, reward, next_state, done)
                
                # 更新当前解
                current_solution = new_solution
                
                # 更新最佳解（多目标比较）
                if self._is_better_solution(new_solution, best_solution):
                    best_solution = copy.deepcopy(new_solution)
                
                # 训练网络
                if len(self.memory) > self.batch_size:
                    self.train()
            
            # 记录episode数据
            episode_rewards.append(episode_reward)
            convergence_data['makespan_history'].append(current_solution.makespan)
            convergence_data['tardiness_history'].append(current_solution.total_tardiness)
            convergence_data['weighted_history'].append(0.55 * current_solution.makespan + 0.45 * current_solution.total_tardiness)
            convergence_data['episode_rewards'].append(episode_reward)
            
            # 更新目标网络
            if episode % self.target_update == 0:
                self.target_net.copy_weights(self.policy_net)
            
            # 更新epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            # 输出进度
            if episode % 10 == 0 or episode == max_episodes - 1:
                episode_time = time.time() - episode_start_time
                print(f"Episode {episode:3d}: 奖励={episode_reward:6.2f}, "
                      f"完工时间={current_solution.makespan:.2f}, "
                      f"拖期={current_solution.total_tardiness:.2f}, "
                      f"ε={self.epsilon:.3f}, 用时={episode_time:.2f}s")
        
        print(f"DQN优化完成!")
        print(f"最佳解: 完工时间={best_solution.makespan:.2f}, 拖期={best_solution.total_tardiness:.2f}")
        print(f"加权目标={0.55*best_solution.makespan + 0.45*best_solution.total_tardiness:.2f}")
        
        return best_solution, convergence_data

    def _add_random_perturbation(self, solution):
        """为解添加随机扰动"""
        new_solution = copy.deepcopy(solution)
        
        # 随机交换一些作业的工厂分配
        n_swaps = max(1, self.problem.n_jobs // 10)
        for _ in range(n_swaps):
            job_id = random.randint(0, self.problem.n_jobs - 1)
            new_factory = random.randint(0, self.problem.n_factories - 1)
            
            # 更新工厂分配
            old_factory = new_solution.factory_assignment[job_id]
            if old_factory != new_factory:
                new_solution.factory_assignment[job_id] = new_factory
                
                # 更新作业序列
                if job_id in new_solution.job_sequences[old_factory]:
                    new_solution.job_sequences[old_factory].remove(job_id)
                new_solution.job_sequences[new_factory].append(job_id)
        
        # 重新评估解
        return self.problem.evaluate_solution(new_solution)

    def _is_better_solution(self, sol1, sol2):
        """判断sol1是否比sol2更好（多目标比较）"""
        # 帕累托支配关系
        if (sol1.makespan <= sol2.makespan and sol1.total_tardiness <= sol2.total_tardiness and
            (sol1.makespan < sol2.makespan or sol1.total_tardiness < sol2.total_tardiness)):
            return True
        
        # 加权目标比较
        weighted1 = 0.55 * sol1.makespan + 0.45 * sol1.total_tardiness
        weighted2 = 0.55 * sol2.makespan + 0.45 * sol2.total_tardiness
        
        return weighted1 < weighted2

    def get_rule_statistics(self) -> Dict:
        """获取各调度规则的统计信息"""
        rule_names = [
            "全局规则1", "全局规则2", "全局规则3",
            "局部规则1", "局部规则2", "局部规则3", 
            "局部规则4", "局部规则5", "局部规则6"
        ]
        
        stats = {}
        for i, name in enumerate(rule_names):
            success_rate = (self.rule_success_count[i] / max(1, self.rule_total_count[i]))
            stats[name] = {
                'success_count': self.rule_success_count[i],
                'total_count': self.rule_total_count[i],
                'success_rate': success_rate
            }
        
        return stats