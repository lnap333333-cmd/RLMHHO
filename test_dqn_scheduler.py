import numpy as np
from algorithm.dqn_multiobj_scheduler import DQNMultiObjScheduler
from problem.mo_dhfsp import MO_DHFSP_Problem

def test_dqn_scheduler():
    """
    测试DQN调度器的功能
    """
    # 创建问题实例
    n_jobs = 20  # 作业数
    n_factories = 4  # 工厂数
    n_stages = 3  # 工序数
    
    # 随机生成加工时间和截止时间
    processing_times = np.random.randint(1, 50, size=(n_jobs, n_stages))
    due_dates = np.random.randint(50, 200, size=n_jobs)
    
    # 创建问题对象
    problem = MO_DHFSP_Problem(
        n_jobs=n_jobs,
        n_factories=n_factories,
        n_stages=n_stages,
        processing_times=processing_times,
        due_dates=due_dates
    )
    
    # 创建DQN调度器
    state_dim = 5  # 状态维度
    action_dim = 9  # 动作维度（9个调度规则）
    scheduler = DQNMultiObjScheduler(
        problem=problem,
        state_dim=state_dim,
        action_dim=action_dim,
        memory_size=10000,
        batch_size=32,
        gamma=0.98,
        epsilon=0.9,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        learning_rate=0.001,
        target_update=10
    )
    
    # 运行调度器
    max_steps = 1000
    final_solution, total_reward = scheduler.run(max_steps=max_steps)
    
    # 打印结果
    print(f"最终解的目标值：")
    print(f"- 最大完工时间：{final_solution.makespan:.2f}")
    print(f"- 总拖期：{final_solution.total_tardiness:.2f}")
    print(f"总奖励：{total_reward:.2f}")
    
    # 打印每个工厂的作业分配
    print("\n各工厂作业分配：")
    for f in range(n_factories):
        print(f"工厂 {f}: {final_solution.job_sequences[f]}")

if __name__ == "__main__":
    test_dqn_scheduler()