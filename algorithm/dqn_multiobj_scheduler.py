import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from collections import deque
from typing import List, Tuple, Dict
from problem.mo_dhfsp import MO_DHFSP_Problem, Solution

class DQNNetwork(nn.Module):
    """
    DQN网络结构：按论文要求设计
    输入层5个节点，隐藏层32和16个节点，输出层9个节点
    """
    def __init__(self, state_dim: int = 5, action_dim: int = 9):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 32)  # 按论文：第一层32个节点
        self.fc2 = nn.Linear(32, 16)         # 按论文：第二层16个节点
        self.fc3 = nn.Linear(16, action_dim) # 输出层9个动作
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class DQNMultiObjScheduler:
    """
    基于DQN的多目标分布式异构混合流水车间调度器
    完全按照论文《基于深度Q学习网络的分布式流水车间调度问题优化》实现
    目标：最小化完工时间（makespan）和总拖期（total tardiness）
    """
    def __init__(self, problem: MO_DHFSP_Problem, **kwargs):
        self.problem = problem
        self.state_dim = 5   # 按论文：5维状态空间
        self.action_dim = 9  # 按论文：9个调度规则
        
        # DQN网络
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQNNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_net = DQNNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # 优化器
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # 经验回放
        self.memory = deque(maxlen=kwargs.get('memory_size', 10000))
        self.batch_size = kwargs.get('batch_size', 32)
        
        # 训练参数
        self.gamma = kwargs.get('gamma', 0.98)  # 折扣因子
        self.epsilon = kwargs.get('epsilon', 0.9)  # 探索率
        self.epsilon_decay = kwargs.get('epsilon_decay', 0.995)
        self.epsilon_min = kwargs.get('epsilon_min', 0.01)
        self.target_update = kwargs.get('target_update', 10)  # 目标网络更新频率
        
        # 训练记录
        self.train_step = 0
        self.best_cmax = float('inf')  # 历史最优完工时间
        self.rule_success_count = [0] * self.action_dim  # 各规则成功次数
        self.rule_total_count = [0] * self.action_dim    # 各规则总次数
        
        print(f"初始化DQN调度器: 状态维度={self.state_dim}, 动作维度={self.action_dim}")
        print(f"网络结构: {self.state_dim} -> 32 -> 16 -> {self.action_dim}")

    def neh_initialization(self) -> Solution:
        """
        使用NEH方法初始化解（按论文要求）
        Nawaz-Encore-Ham启发式算法
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

    def encode_state(self, solution: Solution) -> np.ndarray:
        """
        按论文公式（7）~（11）编码5维状态向量
        """
        # 确保解已被评估
        if solution.makespan == 0:
            solution = self.problem.evaluate_solution(solution)
        
        # S1: 当前解最大完工时间变量率（公式7）
        cmax_before = self.best_cmax if hasattr(self, 'best_cmax') else solution.makespan
        cmax_after = solution.makespan
        if cmax_before > 0:
            S1 = cmax_after / cmax_before
        else:
            S1 = 1.0
        
        # 更新历史最优
        self.best_cmax = min(self.best_cmax, solution.makespan)
        
        # S2: 调度规则执行成功率（公式8）
        if hasattr(self, 'last_action') and self.last_action is not None:
            success_rate = (self.rule_success_count[self.last_action] / 
                          max(1, self.rule_total_count[self.last_action]))
            S2 = success_rate
        else:
            S2 = 1.0
        
        # S3: 当前工厂完工时间比率（公式9）
        if solution.factory_makespans and len(solution.factory_makespans) > 0:
            current_factory_cmax = max(solution.factory_makespans)
            total_cmax = sum(solution.factory_makespans)
            S3 = current_factory_cmax / (total_cmax + 1e-6)
        else:
            S3 = 1.0
        
        # S4: 关键工厂机器总空闲时间和机器总时间的比例（公式10）
        # 简化计算：使用工厂负载不均衡度
        if solution.factory_makespans and len(solution.factory_makespans) > 1:
            max_makespan = max(solution.factory_makespans)
            avg_makespan = np.mean(solution.factory_makespans)
            S4 = max_makespan / (avg_makespan + 1e-6)
        else:
            S4 = 1.0
        
        # S5: 关键工厂机器总空闲时间和所有工厂机器总空闲时间的比例（公式11）
        # 简化计算：使用处理时间分布特征
        if hasattr(self.problem, 'processing_times'):
            factory_loads = [len(seq) for seq in solution.job_sequences]
            if sum(factory_loads) > 0:
                max_load = max(factory_loads)
                total_load = sum(factory_loads)
                S5 = max_load / total_load
            else:
                S5 = 1.0
        else:
            S5 = 1.0
        
        state = np.array([S1, S2, S3, S4, S5], dtype=np.float32)
        return state

    def get_action(self, state: np.ndarray) -> int:
        """
        ε-贪婪策略选择动作（调度规则）
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def apply_rule(self, solution: Solution, action: int) -> Solution:
        """
        根据动作编号应用对应的调度规则（按论文表2实现）
        """
        new_solution = copy.deepcopy(solution)
        n_jobs = self.problem.n_jobs
        n_factories = self.problem.n_factories
        
        if action == 0:
            # 规则1：从Cmax最大的2个工厂中随机选择1个工厂，然后依次尝试从不同位置置换
            self._apply_global_rule_1(new_solution)
        elif action == 1:
            # 规则2：全局规则2 - 从Cmax最大的2个工厂中随机选择1个工厂，但是从最小随机变化的，然后依次尝试从不同位置置换
            self._apply_global_rule_2(new_solution)
        elif action == 2:
            # 规则3：从Cmax最大的工厂中随机选择1个工作，然后依次尝试从不同位置置换
            self._apply_global_rule_3(new_solution)
        elif action == 3:
            # 局部规则1：从Cmax最大的工厂中随机选择1个工作，尝试插入到Cmax最小的工厂中
            self._apply_local_rule_1(new_solution)
        elif action == 4:
            # 局部规则2：从Cmax最大的工厂中随机选择1个工作，随机选择1个工厂插入
            self._apply_local_rule_2(new_solution)
        elif action == 5:
            # 局部规则3：从Cmax最大的工厂中和Cmax最小的工厂中各随机选择1个工作交换
            self._apply_local_rule_3(new_solution)
        elif action == 6:
            # 局部规则4：从Cmax最大的工厂中和随机选择的工厂中各随机选择1个工作交换
            self._apply_local_rule_4(new_solution)
        elif action == 7:
            # 局部规则5：从Cmax最大的工厂中随机选择1个工作，随机选择1个位置插入
            self._apply_local_rule_5(new_solution)
        elif action == 8:
            # 局部规则6：随机选择2个工作，交换位置
            self._apply_local_rule_6(new_solution)
        
        # 重新评估解
        new_solution = self.problem.evaluate_solution(new_solution)
        return new_solution

    def _apply_global_rule_1(self, solution: Solution):
        """全局规则1：从Cmax最大的2个工厂中随机选择进行操作"""
        if not solution.factory_makespans:
            return
        
        # 找到Cmax最大的2个工厂
        factory_makespans = [(i, ms) for i, ms in enumerate(solution.factory_makespans)]
        factory_makespans.sort(key=lambda x: -x[1])  # 按makespan降序
        
        if len(factory_makespans) >= 2:
            # 从前2个中随机选择1个
            selected_factory = random.choice(factory_makespans[:2])[0]
            if solution.job_sequences[selected_factory]:
                # 随机选择一个作业进行位置变换
                job = random.choice(solution.job_sequences[selected_factory])
                self._relocate_job_within_factory(solution, job, selected_factory)

    def _apply_global_rule_2(self, solution: Solution):
        """全局规则2：类似规则1但选择策略不同"""
        if not solution.factory_makespans:
            return
        
        factory_makespans = [(i, ms) for i, ms in enumerate(solution.factory_makespans)]
        factory_makespans.sort(key=lambda x: -x[1])
        
        if len(factory_makespans) >= 2:
            # 选择makespan较小的那个（在最大的2个中）
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
                # 同工厂内交换位置
                self._swap_jobs_within_factory(solution, job1, job2, factory1)
            else:
                # 不同工厂间交换
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
            # 移除作业
            solution.job_sequences[factory1].remove(job1)
            solution.job_sequences[factory2].remove(job2)
            
            # 交换分配
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

    def calculate_reward(self, old_solution: Solution, new_solution: Solution, action: int) -> float:
        """
        按论文公式（6）计算奖励函数
        """
        old_cmax = old_solution.makespan
        new_cmax = new_solution.makespan
        
        # 按论文公式计算奖励
        if new_cmax < old_cmax:
            reward = 1.0  # 改进了
            self.rule_success_count[action] += 1
        elif new_cmax == old_cmax:
            reward = 1.0  # 保持不变
        else:
            reward = 0.0  # 恶化了
        
        self.rule_total_count[action] += 1
        return reward

    def step(self, solution: Solution, action: int) -> Tuple[Solution, float]:
        """
        执行动作并返回新解和奖励
        """
        new_solution = self.apply_rule(solution, action)
        reward = self.calculate_reward(solution, new_solution, action)
        return new_solution, reward

    def remember(self, state, action, reward, next_state, done):
        """将经验存入回放缓冲区"""
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        """DQN训练过程"""
        if len(self.memory) < self.batch_size:
            return None
        
        # 从经验回放中随机采样
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 转换为tensor
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 计算当前Q值
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # 计算损失
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新目标网络
        self.train_step += 1
        if self.train_step % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # 衰减探索率
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()

    def optimize(self, max_episodes: int = 100, max_steps_per_episode: int = 100):
        """
        运行DQN优化过程
        
        Args:
            max_episodes: 最大训练轮数
            max_steps_per_episode: 每轮最大步数
            
        Returns:
            最佳解和收敛数据
        """
        best_solution = None
        best_makespan = float('inf')
        convergence_data = []
        
        print(f"开始DQN训练: {max_episodes}轮, 每轮{max_steps_per_episode}步")
        
        for episode in range(max_episodes):
            # 使用NEH初始化
            current_solution = self.neh_initialization()
            current_state = self.encode_state(current_solution)
            
            episode_reward = 0
            episode_loss = 0
            loss_count = 0
            
            for step in range(max_steps_per_episode):
                # 选择动作
                action = self.get_action(current_state)
                self.last_action = action  # 记录当前动作用于状态编码
                
                # 执行动作
                next_solution, reward = self.step(current_solution, action)
                next_state = self.encode_state(next_solution)
                
                # 存储经验
                done = (step == max_steps_per_episode - 1)
                self.remember(current_state, action, reward, next_state, done)
                
                # 训练
                loss = self.train()
                if loss is not None:
                    episode_loss += loss
                    loss_count += 1
                
                # 更新状态
                current_state = next_state
                current_solution = next_solution
                episode_reward += reward
                
                # 更新最佳解
                if current_solution.makespan < best_makespan:
                    best_makespan = current_solution.makespan
                    best_solution = copy.deepcopy(current_solution)
            
            # 记录收敛数据
            avg_loss = episode_loss / max(1, loss_count)
            convergence_data.append({
                'episode': episode,
                'best_makespan': best_makespan,
                'best_tardiness': best_solution.total_tardiness if best_solution else 0,
                'episode_reward': episode_reward,
                'epsilon': self.epsilon,
                'avg_loss': avg_loss
            })
            
            # 打印进度
            if episode % 10 == 0:
                print(f"轮次 {episode}: 最佳完工时间={best_makespan:.2f}, "
                      f"拖期={best_solution.total_tardiness:.2f}, "
                      f"探索率={self.epsilon:.3f}, 损失={avg_loss:.4f}")
        
        print(f"DQN训练完成! 最终结果:")
        print(f"最佳完工时间: {best_solution.makespan:.2f}")
        print(f"总拖期: {best_solution.total_tardiness:.2f}")
        
        return best_solution, convergence_data

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