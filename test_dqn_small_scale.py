#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DQN小规模测试
验证DQN算法在合理规模下的性能表现
"""

import time
import numpy as np
from algorithm.dqn_simple_scheduler import SimpleDQNScheduler
from problem.mo_dhfsp import MO_DHFSP_Problem
from utils.data_generator import DataGenerator

def test_dqn_small_scale():
    """测试DQN在小规模问题上的性能"""
    print("🧪 DQN小规模性能测试")
    print("规模：30作业 × 5工厂 × 3阶段")
    print("=" * 50)
    
    # 创建小规模问题
    generator = DataGenerator(seed=42)
    
    problem_data = generator.generate_problem(
        n_jobs=30,
        n_factories=5,
        n_stages=3,
        machines_per_stage=[2, 2, 2],
        processing_time_range=(1, 20),
        due_date_tightness=1.3
    )
    
    # 异构机器配置
    problem_data['factory_machines'] = {
        0: [2, 2, 2],  # 工厂1: 6台机器
        1: [1, 3, 2],  # 工厂2: 6台机器
        2: [3, 1, 2],  # 工厂3: 6台机器
        3: [2, 2, 2],  # 工厂4: 6台机器
        4: [2, 1, 3]   # 工厂5: 6台机器
    }
    
    problem = MO_DHFSP_Problem(problem_data)
    
    # 测试DQN
    print(f"📊 问题规模: {problem.n_jobs}作业 × {problem.n_factories}工厂 × {problem.n_stages}阶段")
    
    scheduler = SimpleDQNScheduler(problem)
    
    start_time = time.time()
    best_solution, convergence_data = scheduler.optimize(
        max_episodes=50,
        max_steps_per_episode=30
    )
    runtime = time.time() - start_time
    
    print(f"\n📈 DQN测试结果:")
    print(f"  完工时间: {best_solution.makespan:.2f}")
    print(f"  总拖期: {best_solution.total_tardiness:.2f}")
    print(f"  运行时间: {runtime:.2f}秒")
    
    # 规则统计
    rule_stats = scheduler.get_rule_statistics()
    print(f"\n📊 最有效的调度规则:")
    sorted_rules = sorted(rule_stats.items(), key=lambda x: x[1]['success_rate'], reverse=True)
    for rule_name, stats in sorted_rules[:3]:
        print(f"  {rule_name}: 成功率={stats['success_rate']:.3f}, 使用次数={stats['total_count']}")
    
    # 性能评估
    print(f"\n🎯 性能评估:")
    if best_solution.makespan < 100 and best_solution.total_tardiness < 500:
        print(f"✅ DQN在小规模问题上表现良好")
        performance_ok = True
    else:
        print(f"⚠️ DQN性能有待改进")
        performance_ok = False
    
    return best_solution, runtime, performance_ok

def create_adjusted_dqn_for_comparison():
    """创建适合对比的DQN版本"""
    print(f"\n🔧 创建适合对比的DQN版本")
    print("=" * 50)
    
    # 由于DQN在大规模问题上表现不佳，我们将创建一个简化版本
    # 专门用于与其他算法对比
    
    dqn_wrapper_code = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DQN算法包装器 - 用于算法对比
基于夏保丽、马永忠论文的DQN实现
"""

import time
import numpy as np
from algorithm.dqn_simple_scheduler import SimpleDQNScheduler

class DQNAlgorithmWrapper:
    """DQN算法包装器，符合对比脚本接口"""
    
    def __init__(self, problem):
        self.problem = problem
        self.name = "DQN"
        
    def optimize(self, max_iterations=100):
        """优化接口"""
        scheduler = SimpleDQNScheduler(self.problem)
        
        # 根据问题规模调整参数
        if self.problem.n_jobs <= 50:
            episodes = 50
            steps = 30
        else:
            episodes = 30
            steps = 20
        
        best_solution, convergence_data = scheduler.optimize(
            max_episodes=episodes,
            max_steps_per_episode=steps
        )
        
        # 返回单个解（转换为列表以符合接口）
        return [best_solution]
'''
    
    # 写入文件
    with open('algorithm/dqn_algorithm_wrapper.py', 'w', encoding='utf-8') as f:
        f.write(dqn_wrapper_code)
    
    print(f"✅ 已创建 algorithm/dqn_algorithm_wrapper.py")
    print(f"📝 该文件可用于table_format_comparison对比脚本")

def main():
    """主函数"""
    print("🚀 DQN算法验证与准备")
    print("=" * 60)
    
    # 小规模测试
    solution, runtime, performance_ok = test_dqn_small_scale()
    
    # 创建对比版本
    create_adjusted_dqn_for_comparison()
    
    print(f"\n🎯 总结:")
    print(f"=" * 50)
    
    if performance_ok:
        print(f"✅ DQN算法验证通过")
        print(f"✅ 已创建DQN包装器用于算法对比")
        print(f"📋 建议：可以将DQN加入对比脚本，但需要调整参数以适应不同规模")
        
        print(f"\n📝 使用说明:")
        print(f"1. DQN适合中小规模问题（≤50作业）")
        print(f"2. 大规模问题建议减少训练轮数")
        print(f"3. 可在table_format_comparison中导入DQNAlgorithmWrapper")
    else:
        print(f"⚠️ DQN算法需要进一步优化")
        print(f"📋 建议：暂时不加入大规模对比，专注于中小规模测试")
    
    return performance_ok

if __name__ == "__main__":
    main() 