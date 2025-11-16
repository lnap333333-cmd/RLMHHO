# RL-Chaotic-HHO算法规范伪代码

---

## 算法1: RL-Chaotic-HHO主优化算法 (RL-Chaotic-HHO Main Algorithm)

**Input:** 问题实例P, 种群大小N, 最大迭代数MaxIter  
**Output:** 帕累托最优解集ParetoSet

1   **Begin**
2       params ← {N, MaxIter, ε₀=0.9, α=0.001, γ=0.99};
3       ChaoticMaps ← InitChaoticMaps();
4       EagleGroups ← InitEagleGroups(N);
5       RLCoordinator ← InitRLCoordinator();
6       **For** i ← 1 **to** N **do**
7           X[i] ← GenerateRandomSolution(P);
8           fitness[i] ← Evaluate(X[i]);
9       **EndFor**
10      ParetoSet ← UpdateParetoFront(X);
11      t ← 1;
12      **While** t ≤ MaxIter **do**
13          state ← ConstructStateVector(X, ParetoSet, t);
14          action ← RLCoordinator.SelectAction(state);
15          X_new ← ExecuteStrategy(action, X, EagleGroups, ChaoticMaps);
16          **For** i ← 1 **to** N **do**
17              fitness_new[i] ← Evaluate(X_new[i]);
18          **EndFor**
19          ParetoSet ← UpdateParetoFront(ParetoSet ∪ X_new);
20          reward ← CalculateReward(ParetoSet, X_new);
21          RLCoordinator.UpdateNetwork(state, action, reward);
22          X ← SelectSurvivors(X, X_new);
23          t ← t + 1;
24      **EndWhile**
25      **Return** ParetoSet;
26  **End**

---

## 算法2: 强化学习策略选择器 (RL Strategy Selector)

**Input:** 当前状态向量state  
**Output:** 选定的策略动作action

1   **Begin**
2       **Function** SelectAction(state)
3           **If** random() < ε **then**
4               action ← RandomChoice([0, 1, 2, 3, 4, 5, 6]);
5           **Else**
6               Q_values ← DQN.Forward(state);
7               action ← ArgMax(Q_values);
8           **EndIf**
9           **Return** action;
10      **EndFunction**
11      
12      **Function** UpdateNetwork(state, action, reward)
13          next_state ← ConstructNextState();
14          experience ← (state, action, reward, next_state);
15          ReplayBuffer.Store(experience);
16          **If** ReplayBuffer.Size() ≥ batch_size **then**
17              batch ← ReplayBuffer.Sample(batch_size);
18              **For** each (s, a, r, s') ∈ batch **do**
19                  target ← r + γ × max(TargetDQN.Forward(s'));
20                  loss += (DQN.Forward(s)[a] - target)²;
21              **EndFor**
22              DQN.BackwardUpdate(loss);
23          **EndIf**
24          **If** t % target_update_freq = 0 **then**
25              TargetDQN ← DQN;
26          **EndIf**
27          ε ← max(ε_min, ε × ε_decay);
28      **EndFunction**
29  **End**

---

## 算法3: 四层鹰群协同搜索 (Four-Layer Eagle Cooperative Search)

**Input:** 种群X, 选定策略strategy  
**Output:** 更新后的种群X_new

1   **Begin**
2       **Function** InitEagleGroups(N)
3           exploration_size ← ⌊N × 0.70⌋;
4           exploitation_size ← ⌊N × 0.15⌋;
5           balance_size ← ⌊N × 0.10⌋;
6           elite_size ← N - exploration_size - exploitation_size - balance_size;
7           **Return** {exploration_size, exploitation_size, balance_size, elite_size};
8       **EndFunction**
9       
10      **Function** ExecuteStrategy(strategy, X, Groups, ChaoticMaps)
11          X_new ← X;
12          **Switch** strategy **do**
13              **Case** 0: // 强化全局探索
14                  **For** each x ∈ Groups.exploration **do**
15                      x_new ← GlobalExplorationUpdate(x, X);
16                      x_new ← ChaoticPerturbation(x_new, "logistic", 0.1);
17                  **EndFor**
18              **Case** 1: // 强化局部开发
19                  **For** each x ∈ Groups.exploitation **do**
20                      x_new ← LocalExploitationUpdate(x, Groups.elite);
21                      x_new ← ChaoticPerturbation(x_new, "tent", 0.05);
22                  **EndFor**
23              **Case** 2: // 平衡搜索
24                  **For** each x ∈ Groups.balance **do**
25                      x_new ← BalanceSearchUpdate(x, X);
26                      x_new ← ChaoticPerturbation(x_new, "sine", 0.08);
27                  **EndFor**
28              **Case** 3: // 精英强化
29                  **For** each x ∈ Groups.elite **do**
30                      x_new ← EliteEnhancementUpdate(x);
31                      x_new ← ChaoticPerturbation(x_new, "chebyshev", 0.03);
32                  **EndFor**
33              **Case** 4: // 多样性救援
34                  **For** each x ∈ X **do**
35                      **If** diversity(x) < threshold **then**
36                          x_new ← DiversityRescueUpdate(x);
37                      **EndIf**
38                  **EndFor**
39              **Case** 5: // 资源重分配
40                  X_new ← ResourceReallocation(X, Groups);
41              **Case** 6: // 自适应调整
42                  X_new ← AdaptiveAdjustment(X, t, MaxIter);
43          **EndSwitch**
44          **Return** X_new;
45      **EndFunction**
46  **End**

---

## 算法4: 策略反馈下的结构资源调度机制 (Strategy Feedback-based Structural Resource Scheduling)

**Input:** 当前种群X, 目标结构比例向量R^*(t+1)  
**Output:** 调整后的种群结构X_adjusted

1   **Begin**
2       **Function** StructuralResourceScheduling(X, R_target)
3           // 计算目标结构比例向量
4           R_target^(t+1) ← [r_E^*, r_O^*, r_B^*, r_EL^*];
5           **If** ∑r_i^* ≠ 1 **then**
6               R_target^(t+1) ← NormalizeProportions(R_target^(t+1));
7           **EndIf**
8           
9           // 计算各子群体的目标个体数变化
10          **For** i ← 1 **to** 4 **do**
11              ΔN_i ← (r_i^* - r_i^(t)) × N;
12          **EndFor**
13          
14          // 结构重组过程
15          **For** group ← 1 **to** 4 **do**
16              **If** ΔN_group > 0 **then** // 需要增加个体
17                  **For** k ← 1 **to** ⌊ΔN_group⌋ **do**
18                      **Switch** group **do**
19                          **Case** exploration:
20                              new_individual ← CreateDiverseIndividual(X);
21                          **Case** exploitation:
22                              new_individual ← CreateEliteBasedIndividual(X);
23                          **Case** balance:
24                              new_individual ← MigrateFromOtherGroups(X);
25                          **Case** elite:
26                              new_individual ← ChaoticPerturbation(BestIndividual(X));
27                      **EndSwitch**
28                      Groups[group].Add(new_individual);
29                  **EndFor**
30              **ElseIf** ΔN_group < 0 **then** // 需要减少个体
31                  **For** k ← 1 **to** ⌊|ΔN_group|⌋ **do**
32                      worst_individual ← SelectWorstIndividual(Groups[group]);
33                      Groups[group].Remove(worst_individual);
34                  **EndFor**
35              **EndIf**
36          **EndFor**
37          
38          // 个体迁移机制
39          **For** each individual ∈ X **do**
40              **If** RequiresMigration(individual) **then**
41                  target_group ← SelectOptimalTargetGroup(individual);
42                  MigrateIndividual(individual, target_group);
43              **EndIf**
44          **EndFor**
45          
46          X_adjusted ← ReconstructPopulation(Groups);
47          **Return** X_adjusted;
48      **EndFunction**
49  **End**

---

## 算法5: 增强混沌映射扰动 (Enhanced Chaotic Perturbation)

**Input:** 当前位置x, 混沌类型mapType, 扰动强度intensity  
**Output:** 扰动后位置x_new

1   **Begin**
2       **Function** ChaoticPerturbation(x, mapType, intensity)
3           x_new ← x;
4           chaotic_value ← random(0, 1);
5           **For** i ← 1 **to** length(x) **do**
6               **Switch** mapType **do**
7                   **Case** "logistic":
8                       chaotic_value ← 4.0 × chaotic_value × (1 - chaotic_value);
9                   **Case** "tent":
10                      **If** chaotic_value < 0.5 **then**
11                          chaotic_value ← 2.0 × chaotic_value;
12                      **Else**
13                          chaotic_value ← 2.0 × (1 - chaotic_value);
14                      **EndIf**
15                  **Case** "sine":
16                      chaotic_value ← 1.0 × sin(π × chaotic_value);
17                  **Case** "chebyshev":
18                      chaotic_value ← cos(4 × arccos(chaotic_value));
19              **EndSwitch**
20              perturbation ← intensity × (chaotic_value - 0.5);
21              x_new[i] ← x[i] + perturbation;
22              **If** x_new[i] < lower_bound[i] **then**
23                  x_new[i] ← lower_bound[i];
24              **ElseIf** x_new[i] > upper_bound[i] **then**
25                  x_new[i] ← upper_bound[i];
26              **EndIf**
27          **EndFor**
28          **Return** x_new;
29      **EndFunction**
30  **End**

---

## 算法6: 哈里斯鹰位置更新 (Harris Hawks Position Update)

**Input:** 鹰群位置X, 猎物位置X_rabbit, 迭代次数t  
**Output:** 更新后鹰群位置X_new

1   **Begin**
2       **Function** HarrisHawksUpdate(X, X_rabbit, t)
3           **For** i ← 1 **to** N **do**
4               E₀ ← 2 × random() - 1;
5               E ← 2 × E₀ × (1 - t/MaxIter);
6               **If** |E| ≥ 1 **then** // 探索阶段
7                   **If** random() < 0.5 **then**
8                       r₁ ← random(); r₂ ← random();
9                       X_rand ← X[RandomIndex()];
10                      X_new[i] ← X_rand - r₁ × |X_rand - 2 × r₂ × X[i]|;
11                  **Else**
12                      r₃ ← random(); r₄ ← random();
13                      X_mean ← Mean(X);
14                      LB ← lower_bound; UB ← upper_bound;
15                      X_new[i] ← (X_rabbit - X_mean) - r₃ × (LB + r₄ × (UB - LB));
16                  **EndIf**
17              **Else** // 开发阶段
18                  r ← random();
19                  **If** r ≥ 0.5 **and** |E| ≥ 0.5 **then** // 软围攻
20                      **If** random() < 0.5 **then** // 无跳跃
21                          ΔX ← X_rabbit - X[i];
22                          X_new[i] ← ΔX - E × |J × X_rabbit - X[i]|;
23                      **Else** // 有跳跃
24                          S ← RandomLevyFlight();
25                          LF ← LevyFlight(dimension);
26                          X_new[i] ← X_rabbit - E × |J × X_rabbit - X[i]| + S × LF;
27                      **EndIf**
28                  **Else** // 硬围攻
29                      **If** random() < 0.5 **then** // 无跳跃
30                          X_new[i] ← X_rabbit - E × |ΔX|;
31                      **Else** // 有跳跃
32                          S ← RandomLevyFlight();
33                          LF ← LevyFlight(dimension);
34                          X_new[i] ← X_rabbit - E × |ΔX| + S × LF;
35                      **EndIf**
36                  **EndIf**
37              **EndIf**
38              X_new[i] ← BoundaryCheck(X_new[i]);
39          **EndFor**
40          **Return** X_new;
41      **EndFunction**
42  **End**

---

## 算法7: 双策略编码解码 (Dual-Strategy Encoding-Decoding)

**Input:** 作业集合Jobs, 工厂集合Factories  
**Output:** 调度方案Schedule

1   **Begin**
2       **Function** DualStrategyEncoding(Jobs, Factories)
3           strategy ← SelectStrategy(); // random 或 urgency
4           // 工厂分配向量编码
5           **For** j ← 1 **to** n_jobs **do**
6               **If** strategy = "random" **then**
7                   XF[j] ← RandomInt(0, n_factories-1);
8               **Else** // urgency策略
9                   min_urgency ← ∞; selected_factory ← 0;
10                  **For** f ← 0 **to** n_factories-1 **do**
11                      **If** urgency[j] < min_urgency **then**
12                          min_urgency ← urgency[j];
13                          selected_factory ← f;
14                      **EndIf**
15                  **EndFor**
16                  XF[j] ← selected_factory;
17              **EndIf**
18          **EndFor**
19          
20          // 工序排序集合编码
21          **For** f ← 0 **to** n_factories-1 **do**
22              job_list ← GetJobsAssignedToFactory(f, XF);
23              **If** strategy = "random" **then**
24                  Shuffle(job_list);
25              **Else** // urgency排序
26                  **For** i ← 1 **to** length(job_list)-1 **do**
27                      **For** k ← i+1 **to** length(job_list) **do**
28                          **If** urgency[job_list[i]] > urgency[job_list[k]] **then**
29                              Swap(job_list[i], job_list[k]);
30                          **EndIf**
31                      **EndFor**
32                  **EndFor**
33              **EndIf**
34              S[f] ← job_list;
35          **EndFor**
36          
37          // 机器分配矩阵编码
38          **For** f ← 0 **to** n_factories-1 **do**
39              **For** j ∈ S[f] **do**
40                  **For** stage ← 0 **to** n_stages-1 **do**
41                      available_machines ← GetAvailableMachines(f, stage);
42                      M[f][j][stage] ← SelectOptimalMachine(available_machines, j, stage);
43                  **EndFor**
44              **EndFor**
45          **EndFor**
46          **Return** {XF, S, M};
47      **EndFunction**
48      
49      **Function** ScheduleDecoding(XF, S, M)
50          Schedule ← InitializeSchedule();
51          **For** f ← 0 **to** n_factories-1 **do**
52              **For** j ∈ S[f] **do**
53                  **For** stage ← 0 **to** n_stages-1 **do**
54                      machine ← M[f][j][stage];
55                      // 计算最早开始时间
56                      **If** stage = 0 **then**
57                          start_time ← GetMachineAvailableTime(machine);
58                      **Else**
59                          prev_end_time ← GetJobStageEndTime(j, stage-1);
60                          machine_available ← GetMachineAvailableTime(machine);
61                          start_time ← max(prev_end_time, machine_available);
62                      **EndIf**
63                      end_time ← start_time + processing_time[j][stage];
64                      Schedule.AddOperation(j, stage, machine, start_time, end_time);
65                      UpdateMachineAvailableTime(machine, end_time);
66                  **EndFor**
67              **EndFor**
68          **EndFor**
69          **Return** Schedule;
70      **EndFunction**
71  **End**

---

## 算法8: 帕累托前沿更新管理 (Pareto Front Update Management)

**Input:** 当前解集X, 新解集X_new  
**Output:** 更新后的帕累托前沿ParetoSet

1   **Begin**
2       **Function** UpdateParetoFront(X, X_new)
3           CombinedSet ← X ∪ X_new;
4           ParetoSet ← ∅;
5           **For** i ← 1 **to** |CombinedSet| **do**
6               is_dominated ← false;
7               **For** j ← 1 **to** |CombinedSet| **do**
8                   **If** i ≠ j **then**
9                       **If** Dominates(CombinedSet[j], CombinedSet[i]) **then**
10                          is_dominated ← true;
11                          **Break**;
12                      **EndIf**
13                  **EndIf**
14              **EndFor**
15              **If** ¬is_dominated **then**
16                  ParetoSet ← ParetoSet ∪ {CombinedSet[i]};
17              **EndIf**
18          **EndFor**
19          **Return** ParetoSet;
20      **EndFunction**
21      
22      **Function** Dominates(x₁, x₂)
23          better_count ← 0; equal_count ← 0;
24          **For** obj ← 1 **to** n_objectives **do**
25              **If** x₁.objectives[obj] < x₂.objectives[obj] **then**
26                  better_count ← better_count + 1;
27              **ElseIf** x₁.objectives[obj] = x₂.objectives[obj] **then**
28                  equal_count ← equal_count + 1;
29              **Else**
30                  **Return** false; // x₁在某个目标上更差
31              **EndIf**
32          **EndFor**
33          **Return** (better_count > 0) ∧ (better_count + equal_count = n_objectives);
34      **EndFunction**
35  **End**

---

## 算法9: 性能指标计算器 (Performance Metrics Calculator)

**Input:** 帕累托解集ParetoSet, 参考前沿RefFront  
**Output:** 性能指标集合Metrics

1   **Begin**
2       **Function** CalculateHypervolume(ParetoSet, ref_point)
3           **If** |ParetoSet| = 0 **then** **Return** 0; **EndIf**
4           SortedSet ← SortByFirstObjective(ParetoSet);
5           hv ← 0; prev_x ← 0;
6           **For** i ← 1 **to** |SortedSet| **do**
7               curr_x ← SortedSet[i].makespan;
8               curr_y ← SortedSet[i].tardiness;
9               **If** curr_x < ref_point.x ∧ curr_y < ref_point.y **then**
10                  width ← curr_x - prev_x;
11                  height ← ref_point.y - curr_y;
12                  **If** width > 0 ∧ height > 0 **then**
13                      hv ← hv + width × height;
14                  **EndIf**
15                  prev_x ← curr_x;
16              **EndIf**
17          **EndFor**
18          // 添加最后一段的面积
19          **If** |SortedSet| > 0 **then**
20              last_x ← SortedSet[|SortedSet|].makespan;
21              min_y ← min(s.tardiness for s in SortedSet);
22              **If** last_x < ref_point.x **then**
23                  width ← ref_point.x - last_x;
24                  height ← ref_point.y - min_y;
25                  **If** width > 0 ∧ height > 0 **then**
26                      hv ← hv + width × height;
27                  **EndIf**
28              **EndIf**
29          **EndIf**
30          **Return** hv / (ref_point.x × ref_point.y);
31      **EndFunction**
32      
33      **Function** CalculateIGD(ParetoSet, RefFront)
34          **If** |RefFront| = 0 **then** **Return** ∞; **EndIf**
35          total_distance ← 0;
36          **For** each ref_point ∈ RefFront **do**
37              min_distance ← ∞;
38              **For** each solution ∈ ParetoSet **do**
39                  dx ← ref_point.makespan - solution.makespan;
40                  dy ← ref_point.tardiness - solution.tardiness;
41                  distance ← √(dx² + dy²);
42                  **If** distance < min_distance **then**
43                      min_distance ← distance;
44                  **EndIf**
45              **EndFor**
46              total_distance ← total_distance + min_distance;
47          **EndFor**
48          **Return** total_distance / |RefFront|;
49      **EndFunction**
50  **End** 