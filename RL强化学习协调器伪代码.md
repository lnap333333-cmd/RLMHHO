# RL强化学习协调器伪代码

## 1. 算法概述

基于深度Q网络（DQN）的强化学习协调器，用于RL-Chaotic-HHO算法的智能策略选择和自适应调度。协调器将多目标优化过程建模为马尔可夫决策过程，通过感知当前搜索状态，智能选择最优的策略组合。

## 2. 核心数据结构

### 2.1 经验元组定义
```
Experience = (state, action, reward, next_state, done)
```

### 2.2 DQN网络结构
```
Network Architecture:
  Input Layer: 14 neurons (状态维度)
  Hidden Layer 1: 128 neurons + ReLU
  Hidden Layer 2: 64 neurons + ReLU  
  Output Layer: 7 neurons (动作Q值)
```

### 2.3 动作空间定义
```
ActionSpace = {
  0: "强化全局探索",
  1: "强化局部开发", 
  2: "平衡搜索",
  3: "多样性救援",
  4: "精英强化",
  5: "全局重启",
  6: "资源重分配"
}
```

## 3. 主要算法流程

### 算法1: RL协调器初始化
```
Algorithm 1: RLCoordinator_Initialize
Input: problem P, state_dim=14, action_dim=7, learning_rate=0.001
Output: 初始化的RL协调器

Begin
  // 网络初始化
  q_network ← DQNNetwork(state_dim, action_dim)
  target_network ← DQNNetwork(state_dim, action_dim)
  target_network.copy_weights_from(q_network)
  
  // 经验回放缓冲区初始化
  memory ← PrioritizedReplayBuffer(capacity=10000, alpha=0.6)
  
  // 超参数设置
  epsilon ← 0.9                    // 初始探索率
  epsilon_decay ← 0.995            // 探索率衰减
  epsilon_min ← 0.01               // 最小探索率
  gamma ← 0.98                     // 折扣因子
  batch_size ← 32                  // 批次大小
  target_update_freq ← 100         // 目标网络更新频率
  
  // 统计变量初始化
  training_step ← 0
  action_counts ← zeros(action_dim)
  action_rewards ← zeros(action_dim)
  action_success_rates ← zeros(action_dim)
  
  Return RL协调器实例
End
```

### 算法2: 状态构建
```
Algorithm 2: ConstructState
Input: population Pop, pareto_solutions P, current_iteration t, max_iterations T_max
Output: 14维状态向量 state

Begin
  state ← vector(14)
  
  // 1. 搜索进展特征 (维度0-3)
  progress ← t / T_max
  improvement_rate ← CalculateImprovementRate()
  stagnation_ratio ← min(no_improvement_count / 50, 1.0)
  pareto_size_ratio ← |P| / max(20, |P|)
  
  // 2. 解集质量特征 (维度4-5)
  If |P| > 0 Then
    best_makespan ← min{sol.makespan : sol ∈ P}
    best_tardiness ← min{sol.total_tardiness : sol ∈ P}
    quality_score ← 1.0 / (1.0 + best_makespan / theoretical_lower_bound)
  Else
    quality_score ← 0.0
  EndIf
  
  factory_balance ← CalculateFactoryBalance()
  
  // 3. 各组性能特征 (维度6-13)
  group_performance ← GetGroupPerformanceMetrics()
  
  // 组合状态向量
  state[0] ← progress
  state[1] ← improvement_rate  
  state[2] ← stagnation_ratio
  state[3] ← pareto_size_ratio
  state[4] ← quality_score
  state[5] ← factory_balance
  state[6:14] ← group_performance[0:8]
  
  Return state
End
```

### 算法3: 状态构建（图中公式版本）
```
Algorithm 3: ConstructStateVector (基于图中公式23)
Input: 前沿解集 P_t, 种群 Pop_t, 精英解集 Elite_t
Output: 状态向量 S_t = [D_t, ΔHV_t, S_t, Γ_t, ρ_t]

Begin
  // D_t: 前非支配解集的平均拥挤度
  D_t ← CalculateAverageCrowdingDistance(P_t)
  
  // ΔHV_t: 超体积增量（反映解集整体扩展能力）
  HV_current ← CalculateHypervolume(P_t)
  HV_previous ← GetPreviousHypervolume()
  ΔHV_t ← (HV_current - HV_previous) / max(HV_previous, 1e-10)
  
  // S_t: Spacing指标（衡量解集分布的均匀性）
  S_t ← CalculateSpacingMetric(P_t)
  
  // Γ_t: 精英解中新引入解所占比例
  new_solutions_count ← CountNewSolutionsInElite(Elite_t)
  Γ_t ← new_solutions_count / |Elite_t| if |Elite_t| > 0 else 0
  
  // ρ_t: 成功率（启动跳动所带来的有效解比例）
  ρ_t ← CalculateSuccessRate()
  
  S_t ← [D_t, ΔHV_t, S_t, Γ_t, ρ_t]
  Return S_t
End
```

### 算法4: 动作选择
```
Algorithm 4: SelectAction
Input: state s, training模式标志
Output: 选择的动作 action

Begin
  // 确保状态维度正确
  If len(state) ≠ state_dim Then
    If len(state) > state_dim Then
      state ← state[0:state_dim]
    Else
      state ← pad(state, (0, state_dim - len(state)), 'constant')
    EndIf
  EndIf
  
  // ε-贪婪策略
  If training AND random() < epsilon Then
    action ← randint(0, action_dim - 1)  // 探索
  Else
    q_values ← q_network.forward(state)  // 利用
    action ← argmax(q_values)
    
    // 记录Q值历史
    q_value_history.append(q_values)
  EndIf
  
  // 更新动作统计
  action_counts[action] ← action_counts[action] + 1
  
  Return action
End
```

### 算法5: 奖励计算
```
Algorithm 5: CalculateReward (基于图中公式26)
Input: 状态 s_t, 动作 a_t, 下一状态 s_{t+1}
Output: 奖励值 r_t

Begin
  // 提取状态特征
  D_t ← s_t[0]         // 拥挤度
  ΔHV_t ← s_{t+1}[1]   // 超体积改进
  S_t ← s_{t+1}[2]     // Spacing指标
  Γ_t ← s_{t+1}[3]     // 精英解新引入比例
  
  // 权重系数
  α ← 0.4              // 超体积改进权重
  β ← 0.3              // 分布均匀性权重
  γ ← 0.2              // 精英更新权重
  λ ← 0.1              // 拥挤度惩罚权重
  
  // 综合奖励计算（公式26）
  r_t ← α × ΔHV_t + β × (1 - S_t) + γ × Γ_t - λ × D_t
  
  Return r_t
End
```

### 算法6: 经验存储
```
Algorithm 6: StoreExperience
Input: state, action, reward, next_state, done=False
Output: 存储经验到缓冲区

Begin
  // 确保状态维度一致
  state ← EnsureStateDimension(state, state_dim)
  next_state ← EnsureStateDimension(next_state, state_dim)
  
  experience ← Experience(state, action, reward, next_state, done)
  
  // 计算TD误差作为优先级
  current_q ← q_network.forward(state)[action]
  If done Then
    target_q ← reward
  Else
    next_q_values ← target_network.forward(next_state)
    target_q ← reward + gamma × max(next_q_values)
  EndIf
  
  td_error ← |target_q - current_q|
  
  // 存储到优先级经验回放缓冲区
  memory.push(experience, td_error)
End
```

### 算法7: 网络训练
```
Algorithm 7: TrainNetwork
Input: 无
Output: 更新网络权重

Begin
  If len(memory) < batch_size Then
    Return  // 经验不足，跳过训练
  EndIf
  
  // 采样经验批次
  experiences, indices, weights ← memory.sample(batch_size)
  
  // 提取批次数据
  states ← [exp.state for exp in experiences]
  actions ← [exp.action for exp in experiences]
  rewards ← [exp.reward for exp in experiences]
  next_states ← [exp.next_state for exp in experiences]
  dones ← [exp.done for exp in experiences]
  
  // 计算目标Q值
  current_q_values ← [q_network.forward(state) for state in states]
  next_q_values ← [target_network.forward(state) for state in next_states]
  
  target_q_values ← copy(current_q_values)
  
  For i = 0 to len(experiences) - 1 Do
    If dones[i] Then
      target_q_values[i][actions[i]] ← rewards[i]
    Else
      target_q_values[i][actions[i]] ← rewards[i] + gamma × max(next_q_values[i])
    EndIf
  EndFor
  
  // 计算损失和更新网络
  td_errors ← []
  total_loss ← 0
  
  For i = 0 to len(experiences) - 1 Do
    state ← states[i]
    action ← actions[i]
    target ← target_q_values[i][action]
    current ← current_q_values[i][action]
    
    td_error ← target - current
    td_errors.append(|td_error|)
    total_loss ← total_loss + weights[i] × (td_error)²
    
    // 梯度更新
    SimpleGradientUpdate(state, action, td_error, weights[i])
  EndFor
  
  // 更新优先级
  memory.update_priorities(indices, td_errors)
  
  // 记录损失
  loss_history.append(total_loss / len(experiences))
End
```

### 算法8: 主更新流程
```
Algorithm 8: Update
Input: state, action, reward, next_state
Output: 更新协调器

Begin
  // 存储经验
  StoreExperience(state, action, reward, next_state)
  
  // 更新动作奖励统计
  action_rewards[action] ← action_rewards[action] + reward
  If reward > 0 Then
    action_success_rates[action] ← action_success_rates[action] + 1
  EndIf
  
  // 训练网络
  If len(memory) ≥ batch_size Then
    TrainNetwork()
  EndIf
  
  // 更新目标网络
  If training_step mod target_update_freq = 0 Then
    target_network.copy_weights_from(q_network)
    Print("🎯 更新目标网络 (步骤: " + training_step + ")")
  EndIf
  
  // 衰减探索率
  If epsilon > epsilon_min Then
    epsilon ← epsilon × epsilon_decay
  EndIf
  
  training_step ← training_step + 1
End
```

### 算法9: 策略统计
```
Algorithm 9: GetStrategyStatistics
Input: 无
Output: 策略使用统计信息

Begin
  total_actions ← sum(action_counts)
  If total_actions = 0 Then
    Return {}
  EndIf
  
  stats ← {}
  For action_id = 0 to action_dim - 1 Do
    action_name ← action_space[action_id]
    count ← action_counts[action_id]
    
    stats[action_name] ← {
      'usage_count': count,
      'usage_rate': count / total_actions,
      'average_reward': action_rewards[action_id] / max(count, 1),
      'success_rate': action_success_rates[action_id] / max(count, 1)
    }
  EndFor
  
  Return stats
End
```

## 4. 关键特性说明

### 4.1 状态空间设计
- **14维状态向量**：全面反映搜索过程的关键信息
- **多层次特征**：包含搜索进展、解集质量、种群特征、组性能等
- **归一化处理**：所有特征映射到[0,1]区间，提高学习效率

### 4.2 动作空间设计  
- **7种策略**：涵盖探索、开发、平衡、多样性、精英化等核心需求
- **离散动作**：便于Q学习的价值评估和策略选择
- **策略互补**：各动作针对不同搜索阶段和问题状态

### 4.3 奖励函数设计
- **多目标综合**：结合超体积改进、分布均匀性、精英更新等指标
- **自适应权重**：根据优化阶段动态调整各项权重
- **即时反馈**：提供及时的策略选择指导信号

### 4.4 学习机制
- **DQN架构**：深度Q网络处理高维连续状态空间
- **经验回放**：优先级经验回放提高学习效率
- **目标网络**：双网络结构增强训练稳定性
- **ε-贪婪策略**：平衡探索与利用

## 5. 算法复杂度分析

- **时间复杂度**: O(batch_size × network_forward_time)
- **空间复杂度**: O(memory_size + network_parameters)
- **收敛性**: 在满足一定条件下，DQN算法具有收敛保证

## 6. 实际应用要点

1. **状态特征选择**: 确保状态特征具有马尔可夫性
2. **奖励函数调优**: 根据具体问题调整权重系数
3. **超参数设置**: 学习率、探索率等需要细致调参
4. **网络架构**: 根据问题复杂度调整隐藏层大小
5. **训练策略**: 采用课程学习或预训练提高效果 