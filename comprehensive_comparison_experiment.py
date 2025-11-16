#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整的多目标算法对比实验方案
RL-Chaotic-HHO vs 其他主流多目标算法在完全异构MO-DHFSP问题上的性能对比
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Any
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from problem.mo_dhfsp import MO_DHFSP_Problem
from algorithm.rl_chaotic_hho import RL_ChaoticHHO_Optimizer
from algorithm.nsga2 import NSGA2_Optimizer
from algorithm.moead import MOEAD_Optimizer
from algorithm.mopso import MOPSO_Optimizer
from algorithm.mode import MODE_Optimizer
from algorithm.mosa import MOSA_Optimizer
from utils.data_generator import DataGenerator
from utils.performance_metrics import PerformanceEvaluator

class ComprehensiveComparisonExperiment:
    """完整算法对比实验类"""
    
    def __init__(self):
        self.results_dir = "results/comprehensive_comparison"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 完全异构测试问题集
        self.test_problems = self._generate_comprehensive_test_suite()
        
        # 算法配置
        self.algorithms = self._setup_algorithm_configurations()
        
    def _generate_comprehensive_test_suite(self) -> List[Dict]:
        """生成全面的完全异构测试问题集"""
        problems = []
        
        # 小规模问题集 (20作业)
        small_problems = [
            {
                'name': '小规模20×3×3',
                'n_jobs': 20, 'n_factories': 3, 'n_stages': 3,
                'heterogeneous_machines': {0: [2, 2, 2], 1: [2, 3, 3], 2: [2, 3, 4]},
                'complexity': 'low'
            },
            {
                'name': '小规模20×3×4',
                'n_jobs': 20, 'n_factories': 3, 'n_stages': 4,
                'heterogeneous_machines': {0: [1, 2, 2, 2], 1: [2, 3, 3, 2], 2: [3, 4, 4, 2]},
                'complexity': 'low'
            }
        ]
        
        # 中规模问题集 (50作业)
        medium_problems = [
            {
                'name': '中规模50×4×3',
                'n_jobs': 50, 'n_factories': 4, 'n_stages': 3,
                'heterogeneous_machines': {0: [2, 3, 2], 1: [3, 4, 3], 2: [3, 5, 3], 3: [4, 4, 4]},
                'complexity': 'medium'
            },
            {
                'name': '中规模50×4×4',
                'n_jobs': 50, 'n_factories': 4, 'n_stages': 4,
                'heterogeneous_machines': {0: [2, 2, 3, 2], 1: [3, 3, 4, 3], 2: [3, 4, 4, 3], 3: [3, 3, 4, 3]},
                'complexity': 'medium'
            }
        ]
        
        # 大规模问题集 (100作业)
        large_problems = [
            {
                'name': '大规模100×5×3',
                'n_jobs': 100, 'n_factories': 5, 'n_stages': 3,
                'heterogeneous_machines': {0: [2, 2, 3], 1: [3, 3, 4], 2: [3, 4, 4], 3: [4, 3, 5], 4: [3, 3, 4]},
                'complexity': 'high'
            },
            {
                'name': '大规模100×5×4',
                'n_jobs': 100, 'n_factories': 5, 'n_stages': 4,
                'heterogeneous_machines': {0: [1, 2, 2, 1], 1: [2, 3, 3, 2], 2: [2, 3, 4, 2], 3: [3, 4, 3, 2], 4: [2, 3, 4, 2]},
                'complexity': 'high'
            }
        ]
        
        # 超大规模问题集 (200作业)
        extra_large_problems = [
            {
                'name': '超大规模200×6×3',
                'n_jobs': 200, 'n_factories': 6, 'n_stages': 3,
                'heterogeneous_machines': {0: [3, 3, 4], 1: [4, 4, 5], 2: [4, 5, 5], 3: [5, 4, 6], 4: [4, 4, 5], 5: [3, 4, 5]},
                'complexity': 'very_high'
            }
        ]
        
        problems.extend(small_problems)
        problems.extend(medium_problems) 
        problems.extend(large_problems)
        problems.extend(extra_large_problems)
        
        return problems
    
    def _setup_algorithm_configurations(self) -> Dict:
        """设置算法配置"""
        return {
            'RL-Chaotic-HHO': {
                'class': RL_ChaoticHHO_Optimizer,
                'name': 'RL-Chaotic-HHO',
                'description': '基于强化学习协调的混沌哈里斯鹰优化算法',
                'params': {
                    'small': {'max_iterations': 80},
                    'medium': {'max_iterations': 100}, 
                    'large': {'max_iterations': 120},
                    'very_large': {'max_iterations': 150}
                }
            },
            'NSGA-II': {
                'class': NSGA2_Optimizer,
                'name': 'NSGA-II',
                'description': '非支配排序遗传算法II',
                'params': {
                    'small': {'population_size': 60, 'max_generations': 80, 'crossover_prob': 0.9, 'mutation_prob': 0.1},
                    'medium': {'population_size': 80, 'max_generations': 100, 'crossover_prob': 0.9, 'mutation_prob': 0.1},
                    'large': {'population_size': 100, 'max_generations': 120, 'crossover_prob': 0.9, 'mutation_prob': 0.1},
                    'very_large': {'population_size': 120, 'max_generations': 150, 'crossover_prob': 0.9, 'mutation_prob': 0.1}
                }
            },
            'MOEA/D': {
                'class': MOEAD_Optimizer,
                'name': 'MOEA/D',
                'description': '基于分解的多目标进化算法',
                'params': {
                    'small': {'population_size': 60, 'max_generations': 80, 'neighbor_size': 10},
                    'medium': {'population_size': 80, 'max_generations': 100, 'neighbor_size': 15},
                    'large': {'population_size': 100, 'max_generations': 120, 'neighbor_size': 20},
                    'very_large': {'population_size': 120, 'max_generations': 150, 'neighbor_size': 25}
                }
            },
            'MOPSO': {
                'class': MOPSO_Optimizer,
                'name': 'MOPSO',
                'description': '多目标粒子群优化算法',
                'params': {
                    'small': {'swarm_size': 60, 'max_iterations': 80, 'archive_size': 90},
                    'medium': {'swarm_size': 80, 'max_iterations': 100, 'archive_size': 120},
                    'large': {'swarm_size': 100, 'max_iterations': 120, 'archive_size': 150},
                    'very_large': {'swarm_size': 120, 'max_iterations': 150, 'archive_size': 180}
                }
            },
            'MODE': {
                'class': MODE_Optimizer,
                'name': 'MODE',
                'description': '多目标差分进化算法',
                'params': {
                    'small': {'population_size': 60, 'max_generations': 80, 'F': 0.5, 'CR': 0.9},
                    'medium': {'population_size': 80, 'max_generations': 100, 'F': 0.5, 'CR': 0.9},
                    'large': {'population_size': 100, 'max_generations': 120, 'F': 0.5, 'CR': 0.9},
                    'very_large': {'population_size': 120, 'max_generations': 150, 'F': 0.5, 'CR': 0.9}
                }
            },
            'MOSA': {
                'class': MOSA_Optimizer,
                'name': 'MOSA',
                'description': '多目标模拟退火算法',
                'params': {
                    'small': {'max_iterations': 800, 'initial_temperature': 500, 'cooling_rate': 0.98, 'neighborhood_size': 10},
                    'medium': {'max_iterations': 1000, 'initial_temperature': 800, 'cooling_rate': 0.98, 'neighborhood_size': 12},
                    'large': {'max_iterations': 1200, 'initial_temperature': 1000, 'cooling_rate': 0.98, 'neighborhood_size': 15},
                    'very_large': {'max_iterations': 1500, 'initial_temperature': 1200, 'cooling_rate': 0.98, 'neighborhood_size': 18}
                }
            }
        }
    
    def run_comprehensive_comparison(self):
        """运行完整的算法对比实验"""
        print("🚀 开始完整的多目标算法对比实验")
        print("=" * 80)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 总体实验结果
        all_results = {}
        
        # 对每个测试问题运行对比实验
        for problem_config in self.test_problems:
            problem_name = problem_config['name']
            complexity = problem_config['complexity']
            
            print(f"\n🧪 测试问题: {problem_name} (复杂度: {complexity})")
            print("-" * 60)
            
            # 生成问题数据
            problem_data = self._generate_problem_data(problem_config)
            
            # 运行所有算法
            problem_results = {}
            for alg_name, alg_config in self.algorithms.items():
                print(f"  运行算法: {alg_name}")
                
                # 获取对应复杂度的参数
                scale_key = self._get_scale_key(complexity)
                params = alg_config['params'][scale_key]
                
                # 运行算法
                result = self._run_algorithm_experiment(
                    problem_data, 
                    alg_config['class'], 
                    params,
                    runs=5  # 每个算法运行5次
                )
                
                problem_results[alg_name] = result
                
                print(f"    最优加权目标: {result['best_weighted']:.2f}")
                print(f"    平均运行时间: {result['avg_runtime']:.2f}s")
            
            all_results[problem_name] = problem_results
            
            # 绘制该问题的帕累托前沿对比
            self._plot_pareto_comparison(problem_results, problem_name, timestamp)
        
        # 生成综合报告
        self._generate_comprehensive_report(all_results, timestamp)
        
        # 绘制综合性能图表
        self._plot_comprehensive_performance(all_results, timestamp)
        
        print(f"\n🎉 完整对比实验完成！结果保存在: {self.results_dir}/")
        
        return all_results

def main():
    """主函数"""
    print("🔬 启动完整的多目标算法对比实验")
    
    # 创建实验实例
    experiment = ComprehensiveComparisonExperiment()
    
    # 运行完整对比实验
    results = experiment.run_comprehensive_comparison()
    
    print("\n✅ 所有实验完成！")

if __name__ == "__main__":
    main() 
# -*- coding: utf-8 -*-
"""
完整的多目标算法对比实验方案
RL-Chaotic-HHO vs 其他主流多目标算法在完全异构MO-DHFSP问题上的性能对比
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Any
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from problem.mo_dhfsp import MO_DHFSP_Problem
from algorithm.rl_chaotic_hho import RL_ChaoticHHO_Optimizer
from algorithm.nsga2 import NSGA2_Optimizer
from algorithm.moead import MOEAD_Optimizer
from algorithm.mopso import MOPSO_Optimizer
from algorithm.mode import MODE_Optimizer
from algorithm.mosa import MOSA_Optimizer
from utils.data_generator import DataGenerator
from utils.performance_metrics import PerformanceEvaluator

class ComprehensiveComparisonExperiment:
    """完整算法对比实验类"""
    
    def __init__(self):
        self.results_dir = "results/comprehensive_comparison"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 完全异构测试问题集
        self.test_problems = self._generate_comprehensive_test_suite()
        
        # 算法配置
        self.algorithms = self._setup_algorithm_configurations()
        
    def _generate_comprehensive_test_suite(self) -> List[Dict]:
        """生成全面的完全异构测试问题集"""
        problems = []
        
        # 小规模问题集 (20作业)
        small_problems = [
            {
                'name': '小规模20×3×3',
                'n_jobs': 20, 'n_factories': 3, 'n_stages': 3,
                'heterogeneous_machines': {0: [2, 2, 2], 1: [2, 3, 3], 2: [2, 3, 4]},
                'complexity': 'low'
            },
            {
                'name': '小规模20×3×4',
                'n_jobs': 20, 'n_factories': 3, 'n_stages': 4,
                'heterogeneous_machines': {0: [1, 2, 2, 2], 1: [2, 3, 3, 2], 2: [3, 4, 4, 2]},
                'complexity': 'low'
            }
        ]
        
        # 中规模问题集 (50作业)
        medium_problems = [
            {
                'name': '中规模50×4×3',
                'n_jobs': 50, 'n_factories': 4, 'n_stages': 3,
                'heterogeneous_machines': {0: [2, 3, 2], 1: [3, 4, 3], 2: [3, 5, 3], 3: [4, 4, 4]},
                'complexity': 'medium'
            },
            {
                'name': '中规模50×4×4',
                'n_jobs': 50, 'n_factories': 4, 'n_stages': 4,
                'heterogeneous_machines': {0: [2, 2, 3, 2], 1: [3, 3, 4, 3], 2: [3, 4, 4, 3], 3: [3, 3, 4, 3]},
                'complexity': 'medium'
            }
        ]
        
        # 大规模问题集 (100作业)
        large_problems = [
            {
                'name': '大规模100×5×3',
                'n_jobs': 100, 'n_factories': 5, 'n_stages': 3,
                'heterogeneous_machines': {0: [2, 2, 3], 1: [3, 3, 4], 2: [3, 4, 4], 3: [4, 3, 5], 4: [3, 3, 4]},
                'complexity': 'high'
            },
            {
                'name': '大规模100×5×4',
                'n_jobs': 100, 'n_factories': 5, 'n_stages': 4,
                'heterogeneous_machines': {0: [1, 2, 2, 1], 1: [2, 3, 3, 2], 2: [2, 3, 4, 2], 3: [3, 4, 3, 2], 4: [2, 3, 4, 2]},
                'complexity': 'high'
            }
        ]
        
        # 超大规模问题集 (200作业)
        extra_large_problems = [
            {
                'name': '超大规模200×6×3',
                'n_jobs': 200, 'n_factories': 6, 'n_stages': 3,
                'heterogeneous_machines': {0: [3, 3, 4], 1: [4, 4, 5], 2: [4, 5, 5], 3: [5, 4, 6], 4: [4, 4, 5], 5: [3, 4, 5]},
                'complexity': 'very_high'
            }
        ]
        
        problems.extend(small_problems)
        problems.extend(medium_problems) 
        problems.extend(large_problems)
        problems.extend(extra_large_problems)
        
        return problems
    
    def _setup_algorithm_configurations(self) -> Dict:
        """设置算法配置"""
        return {
            'RL-Chaotic-HHO': {
                'class': RL_ChaoticHHO_Optimizer,
                'name': 'RL-Chaotic-HHO',
                'description': '基于强化学习协调的混沌哈里斯鹰优化算法',
                'params': {
                    'small': {'max_iterations': 80},
                    'medium': {'max_iterations': 100}, 
                    'large': {'max_iterations': 120},
                    'very_large': {'max_iterations': 150}
                }
            },
            'NSGA-II': {
                'class': NSGA2_Optimizer,
                'name': 'NSGA-II',
                'description': '非支配排序遗传算法II',
                'params': {
                    'small': {'population_size': 60, 'max_generations': 80, 'crossover_prob': 0.9, 'mutation_prob': 0.1},
                    'medium': {'population_size': 80, 'max_generations': 100, 'crossover_prob': 0.9, 'mutation_prob': 0.1},
                    'large': {'population_size': 100, 'max_generations': 120, 'crossover_prob': 0.9, 'mutation_prob': 0.1},
                    'very_large': {'population_size': 120, 'max_generations': 150, 'crossover_prob': 0.9, 'mutation_prob': 0.1}
                }
            },
            'MOEA/D': {
                'class': MOEAD_Optimizer,
                'name': 'MOEA/D',
                'description': '基于分解的多目标进化算法',
                'params': {
                    'small': {'population_size': 60, 'max_generations': 80, 'neighbor_size': 10},
                    'medium': {'population_size': 80, 'max_generations': 100, 'neighbor_size': 15},
                    'large': {'population_size': 100, 'max_generations': 120, 'neighbor_size': 20},
                    'very_large': {'population_size': 120, 'max_generations': 150, 'neighbor_size': 25}
                }
            },
            'MOPSO': {
                'class': MOPSO_Optimizer,
                'name': 'MOPSO',
                'description': '多目标粒子群优化算法',
                'params': {
                    'small': {'swarm_size': 60, 'max_iterations': 80, 'archive_size': 90},
                    'medium': {'swarm_size': 80, 'max_iterations': 100, 'archive_size': 120},
                    'large': {'swarm_size': 100, 'max_iterations': 120, 'archive_size': 150},
                    'very_large': {'swarm_size': 120, 'max_iterations': 150, 'archive_size': 180}
                }
            },
            'MODE': {
                'class': MODE_Optimizer,
                'name': 'MODE',
                'description': '多目标差分进化算法',
                'params': {
                    'small': {'population_size': 60, 'max_generations': 80, 'F': 0.5, 'CR': 0.9},
                    'medium': {'population_size': 80, 'max_generations': 100, 'F': 0.5, 'CR': 0.9},
                    'large': {'population_size': 100, 'max_generations': 120, 'F': 0.5, 'CR': 0.9},
                    'very_large': {'population_size': 120, 'max_generations': 150, 'F': 0.5, 'CR': 0.9}
                }
            },
            'MOSA': {
                'class': MOSA_Optimizer,
                'name': 'MOSA',
                'description': '多目标模拟退火算法',
                'params': {
                    'small': {'max_iterations': 800, 'initial_temperature': 500, 'cooling_rate': 0.98, 'neighborhood_size': 10},
                    'medium': {'max_iterations': 1000, 'initial_temperature': 800, 'cooling_rate': 0.98, 'neighborhood_size': 12},
                    'large': {'max_iterations': 1200, 'initial_temperature': 1000, 'cooling_rate': 0.98, 'neighborhood_size': 15},
                    'very_large': {'max_iterations': 1500, 'initial_temperature': 1200, 'cooling_rate': 0.98, 'neighborhood_size': 18}
                }
            }
        }
    
    def run_comprehensive_comparison(self):
        """运行完整的算法对比实验"""
        print("🚀 开始完整的多目标算法对比实验")
        print("=" * 80)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 总体实验结果
        all_results = {}
        
        # 对每个测试问题运行对比实验
        for problem_config in self.test_problems:
            problem_name = problem_config['name']
            complexity = problem_config['complexity']
            
            print(f"\n🧪 测试问题: {problem_name} (复杂度: {complexity})")
            print("-" * 60)
            
            # 生成问题数据
            problem_data = self._generate_problem_data(problem_config)
            
            # 运行所有算法
            problem_results = {}
            for alg_name, alg_config in self.algorithms.items():
                print(f"  运行算法: {alg_name}")
                
                # 获取对应复杂度的参数
                scale_key = self._get_scale_key(complexity)
                params = alg_config['params'][scale_key]
                
                # 运行算法
                result = self._run_algorithm_experiment(
                    problem_data, 
                    alg_config['class'], 
                    params,
                    runs=5  # 每个算法运行5次
                )
                
                problem_results[alg_name] = result
                
                print(f"    最优加权目标: {result['best_weighted']:.2f}")
                print(f"    平均运行时间: {result['avg_runtime']:.2f}s")
            
            all_results[problem_name] = problem_results
            
            # 绘制该问题的帕累托前沿对比
            self._plot_pareto_comparison(problem_results, problem_name, timestamp)
        
        # 生成综合报告
        self._generate_comprehensive_report(all_results, timestamp)
        
        # 绘制综合性能图表
        self._plot_comprehensive_performance(all_results, timestamp)
        
        print(f"\n🎉 完整对比实验完成！结果保存在: {self.results_dir}/")
        
        return all_results

def main():
    """主函数"""
    print("🔬 启动完整的多目标算法对比实验")
    
    # 创建实验实例
    experiment = ComprehensiveComparisonExperiment()
    
    # 运行完整对比实验
    results = experiment.run_comprehensive_comparison()
    
    print("\n✅ 所有实验完成！")

if __name__ == "__main__":
    main() 
 
# -*- coding: utf-8 -*-
"""
完整的多目标算法对比实验方案
RL-Chaotic-HHO vs 其他主流多目标算法在完全异构MO-DHFSP问题上的性能对比
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Any
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from problem.mo_dhfsp import MO_DHFSP_Problem
from algorithm.rl_chaotic_hho import RL_ChaoticHHO_Optimizer
from algorithm.nsga2 import NSGA2_Optimizer
from algorithm.moead import MOEAD_Optimizer
from algorithm.mopso import MOPSO_Optimizer
from algorithm.mode import MODE_Optimizer
from algorithm.mosa import MOSA_Optimizer
from utils.data_generator import DataGenerator
from utils.performance_metrics import PerformanceEvaluator

class ComprehensiveComparisonExperiment:
    """完整算法对比实验类"""
    
    def __init__(self):
        self.results_dir = "results/comprehensive_comparison"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 完全异构测试问题集
        self.test_problems = self._generate_comprehensive_test_suite()
        
        # 算法配置
        self.algorithms = self._setup_algorithm_configurations()
        
    def _generate_comprehensive_test_suite(self) -> List[Dict]:
        """生成全面的完全异构测试问题集"""
        problems = []
        
        # 小规模问题集 (20作业)
        small_problems = [
            {
                'name': '小规模20×3×3',
                'n_jobs': 20, 'n_factories': 3, 'n_stages': 3,
                'heterogeneous_machines': {0: [2, 2, 2], 1: [2, 3, 3], 2: [2, 3, 4]},
                'complexity': 'low'
            },
            {
                'name': '小规模20×3×4',
                'n_jobs': 20, 'n_factories': 3, 'n_stages': 4,
                'heterogeneous_machines': {0: [1, 2, 2, 2], 1: [2, 3, 3, 2], 2: [3, 4, 4, 2]},
                'complexity': 'low'
            }
        ]
        
        # 中规模问题集 (50作业)
        medium_problems = [
            {
                'name': '中规模50×4×3',
                'n_jobs': 50, 'n_factories': 4, 'n_stages': 3,
                'heterogeneous_machines': {0: [2, 3, 2], 1: [3, 4, 3], 2: [3, 5, 3], 3: [4, 4, 4]},
                'complexity': 'medium'
            },
            {
                'name': '中规模50×4×4',
                'n_jobs': 50, 'n_factories': 4, 'n_stages': 4,
                'heterogeneous_machines': {0: [2, 2, 3, 2], 1: [3, 3, 4, 3], 2: [3, 4, 4, 3], 3: [3, 3, 4, 3]},
                'complexity': 'medium'
            }
        ]
        
        # 大规模问题集 (100作业)
        large_problems = [
            {
                'name': '大规模100×5×3',
                'n_jobs': 100, 'n_factories': 5, 'n_stages': 3,
                'heterogeneous_machines': {0: [2, 2, 3], 1: [3, 3, 4], 2: [3, 4, 4], 3: [4, 3, 5], 4: [3, 3, 4]},
                'complexity': 'high'
            },
            {
                'name': '大规模100×5×4',
                'n_jobs': 100, 'n_factories': 5, 'n_stages': 4,
                'heterogeneous_machines': {0: [1, 2, 2, 1], 1: [2, 3, 3, 2], 2: [2, 3, 4, 2], 3: [3, 4, 3, 2], 4: [2, 3, 4, 2]},
                'complexity': 'high'
            }
        ]
        
        # 超大规模问题集 (200作业)
        extra_large_problems = [
            {
                'name': '超大规模200×6×3',
                'n_jobs': 200, 'n_factories': 6, 'n_stages': 3,
                'heterogeneous_machines': {0: [3, 3, 4], 1: [4, 4, 5], 2: [4, 5, 5], 3: [5, 4, 6], 4: [4, 4, 5], 5: [3, 4, 5]},
                'complexity': 'very_high'
            }
        ]
        
        problems.extend(small_problems)
        problems.extend(medium_problems) 
        problems.extend(large_problems)
        problems.extend(extra_large_problems)
        
        return problems
    
    def _setup_algorithm_configurations(self) -> Dict:
        """设置算法配置"""
        return {
            'RL-Chaotic-HHO': {
                'class': RL_ChaoticHHO_Optimizer,
                'name': 'RL-Chaotic-HHO',
                'description': '基于强化学习协调的混沌哈里斯鹰优化算法',
                'params': {
                    'small': {'max_iterations': 80},
                    'medium': {'max_iterations': 100}, 
                    'large': {'max_iterations': 120},
                    'very_large': {'max_iterations': 150}
                }
            },
            'NSGA-II': {
                'class': NSGA2_Optimizer,
                'name': 'NSGA-II',
                'description': '非支配排序遗传算法II',
                'params': {
                    'small': {'population_size': 60, 'max_generations': 80, 'crossover_prob': 0.9, 'mutation_prob': 0.1},
                    'medium': {'population_size': 80, 'max_generations': 100, 'crossover_prob': 0.9, 'mutation_prob': 0.1},
                    'large': {'population_size': 100, 'max_generations': 120, 'crossover_prob': 0.9, 'mutation_prob': 0.1},
                    'very_large': {'population_size': 120, 'max_generations': 150, 'crossover_prob': 0.9, 'mutation_prob': 0.1}
                }
            },
            'MOEA/D': {
                'class': MOEAD_Optimizer,
                'name': 'MOEA/D',
                'description': '基于分解的多目标进化算法',
                'params': {
                    'small': {'population_size': 60, 'max_generations': 80, 'neighbor_size': 10},
                    'medium': {'population_size': 80, 'max_generations': 100, 'neighbor_size': 15},
                    'large': {'population_size': 100, 'max_generations': 120, 'neighbor_size': 20},
                    'very_large': {'population_size': 120, 'max_generations': 150, 'neighbor_size': 25}
                }
            },
            'MOPSO': {
                'class': MOPSO_Optimizer,
                'name': 'MOPSO',
                'description': '多目标粒子群优化算法',
                'params': {
                    'small': {'swarm_size': 60, 'max_iterations': 80, 'archive_size': 90},
                    'medium': {'swarm_size': 80, 'max_iterations': 100, 'archive_size': 120},
                    'large': {'swarm_size': 100, 'max_iterations': 120, 'archive_size': 150},
                    'very_large': {'swarm_size': 120, 'max_iterations': 150, 'archive_size': 180}
                }
            },
            'MODE': {
                'class': MODE_Optimizer,
                'name': 'MODE',
                'description': '多目标差分进化算法',
                'params': {
                    'small': {'population_size': 60, 'max_generations': 80, 'F': 0.5, 'CR': 0.9},
                    'medium': {'population_size': 80, 'max_generations': 100, 'F': 0.5, 'CR': 0.9},
                    'large': {'population_size': 100, 'max_generations': 120, 'F': 0.5, 'CR': 0.9},
                    'very_large': {'population_size': 120, 'max_generations': 150, 'F': 0.5, 'CR': 0.9}
                }
            },
            'MOSA': {
                'class': MOSA_Optimizer,
                'name': 'MOSA',
                'description': '多目标模拟退火算法',
                'params': {
                    'small': {'max_iterations': 800, 'initial_temperature': 500, 'cooling_rate': 0.98, 'neighborhood_size': 10},
                    'medium': {'max_iterations': 1000, 'initial_temperature': 800, 'cooling_rate': 0.98, 'neighborhood_size': 12},
                    'large': {'max_iterations': 1200, 'initial_temperature': 1000, 'cooling_rate': 0.98, 'neighborhood_size': 15},
                    'very_large': {'max_iterations': 1500, 'initial_temperature': 1200, 'cooling_rate': 0.98, 'neighborhood_size': 18}
                }
            }
        }
    
    def run_comprehensive_comparison(self):
        """运行完整的算法对比实验"""
        print("🚀 开始完整的多目标算法对比实验")
        print("=" * 80)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 总体实验结果
        all_results = {}
        
        # 对每个测试问题运行对比实验
        for problem_config in self.test_problems:
            problem_name = problem_config['name']
            complexity = problem_config['complexity']
            
            print(f"\n🧪 测试问题: {problem_name} (复杂度: {complexity})")
            print("-" * 60)
            
            # 生成问题数据
            problem_data = self._generate_problem_data(problem_config)
            
            # 运行所有算法
            problem_results = {}
            for alg_name, alg_config in self.algorithms.items():
                print(f"  运行算法: {alg_name}")
                
                # 获取对应复杂度的参数
                scale_key = self._get_scale_key(complexity)
                params = alg_config['params'][scale_key]
                
                # 运行算法
                result = self._run_algorithm_experiment(
                    problem_data, 
                    alg_config['class'], 
                    params,
                    runs=5  # 每个算法运行5次
                )
                
                problem_results[alg_name] = result
                
                print(f"    最优加权目标: {result['best_weighted']:.2f}")
                print(f"    平均运行时间: {result['avg_runtime']:.2f}s")
            
            all_results[problem_name] = problem_results
            
            # 绘制该问题的帕累托前沿对比
            self._plot_pareto_comparison(problem_results, problem_name, timestamp)
        
        # 生成综合报告
        self._generate_comprehensive_report(all_results, timestamp)
        
        # 绘制综合性能图表
        self._plot_comprehensive_performance(all_results, timestamp)
        
        print(f"\n🎉 完整对比实验完成！结果保存在: {self.results_dir}/")
        
        return all_results

def main():
    """主函数"""
    print("🔬 启动完整的多目标算法对比实验")
    
    # 创建实验实例
    experiment = ComprehensiveComparisonExperiment()
    
    # 运行完整对比实验
    results = experiment.run_comprehensive_comparison()
    
    print("\n✅ 所有实验完成！")

if __name__ == "__main__":
    main() 
# -*- coding: utf-8 -*-
"""
完整的多目标算法对比实验方案
RL-Chaotic-HHO vs 其他主流多目标算法在完全异构MO-DHFSP问题上的性能对比
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Any
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from problem.mo_dhfsp import MO_DHFSP_Problem
from algorithm.rl_chaotic_hho import RL_ChaoticHHO_Optimizer
from algorithm.nsga2 import NSGA2_Optimizer
from algorithm.moead import MOEAD_Optimizer
from algorithm.mopso import MOPSO_Optimizer
from algorithm.mode import MODE_Optimizer
from algorithm.mosa import MOSA_Optimizer
from utils.data_generator import DataGenerator
from utils.performance_metrics import PerformanceEvaluator

class ComprehensiveComparisonExperiment:
    """完整算法对比实验类"""
    
    def __init__(self):
        self.results_dir = "results/comprehensive_comparison"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 完全异构测试问题集
        self.test_problems = self._generate_comprehensive_test_suite()
        
        # 算法配置
        self.algorithms = self._setup_algorithm_configurations()
        
    def _generate_comprehensive_test_suite(self) -> List[Dict]:
        """生成全面的完全异构测试问题集"""
        problems = []
        
        # 小规模问题集 (20作业)
        small_problems = [
            {
                'name': '小规模20×3×3',
                'n_jobs': 20, 'n_factories': 3, 'n_stages': 3,
                'heterogeneous_machines': {0: [2, 2, 2], 1: [2, 3, 3], 2: [2, 3, 4]},
                'complexity': 'low'
            },
            {
                'name': '小规模20×3×4',
                'n_jobs': 20, 'n_factories': 3, 'n_stages': 4,
                'heterogeneous_machines': {0: [1, 2, 2, 2], 1: [2, 3, 3, 2], 2: [3, 4, 4, 2]},
                'complexity': 'low'
            }
        ]
        
        # 中规模问题集 (50作业)
        medium_problems = [
            {
                'name': '中规模50×4×3',
                'n_jobs': 50, 'n_factories': 4, 'n_stages': 3,
                'heterogeneous_machines': {0: [2, 3, 2], 1: [3, 4, 3], 2: [3, 5, 3], 3: [4, 4, 4]},
                'complexity': 'medium'
            },
            {
                'name': '中规模50×4×4',
                'n_jobs': 50, 'n_factories': 4, 'n_stages': 4,
                'heterogeneous_machines': {0: [2, 2, 3, 2], 1: [3, 3, 4, 3], 2: [3, 4, 4, 3], 3: [3, 3, 4, 3]},
                'complexity': 'medium'
            }
        ]
        
        # 大规模问题集 (100作业)
        large_problems = [
            {
                'name': '大规模100×5×3',
                'n_jobs': 100, 'n_factories': 5, 'n_stages': 3,
                'heterogeneous_machines': {0: [2, 2, 3], 1: [3, 3, 4], 2: [3, 4, 4], 3: [4, 3, 5], 4: [3, 3, 4]},
                'complexity': 'high'
            },
            {
                'name': '大规模100×5×4',
                'n_jobs': 100, 'n_factories': 5, 'n_stages': 4,
                'heterogeneous_machines': {0: [1, 2, 2, 1], 1: [2, 3, 3, 2], 2: [2, 3, 4, 2], 3: [3, 4, 3, 2], 4: [2, 3, 4, 2]},
                'complexity': 'high'
            }
        ]
        
        # 超大规模问题集 (200作业)
        extra_large_problems = [
            {
                'name': '超大规模200×6×3',
                'n_jobs': 200, 'n_factories': 6, 'n_stages': 3,
                'heterogeneous_machines': {0: [3, 3, 4], 1: [4, 4, 5], 2: [4, 5, 5], 3: [5, 4, 6], 4: [4, 4, 5], 5: [3, 4, 5]},
                'complexity': 'very_high'
            }
        ]
        
        problems.extend(small_problems)
        problems.extend(medium_problems) 
        problems.extend(large_problems)
        problems.extend(extra_large_problems)
        
        return problems
    
    def _setup_algorithm_configurations(self) -> Dict:
        """设置算法配置"""
        return {
            'RL-Chaotic-HHO': {
                'class': RL_ChaoticHHO_Optimizer,
                'name': 'RL-Chaotic-HHO',
                'description': '基于强化学习协调的混沌哈里斯鹰优化算法',
                'params': {
                    'small': {'max_iterations': 80},
                    'medium': {'max_iterations': 100}, 
                    'large': {'max_iterations': 120},
                    'very_large': {'max_iterations': 150}
                }
            },
            'NSGA-II': {
                'class': NSGA2_Optimizer,
                'name': 'NSGA-II',
                'description': '非支配排序遗传算法II',
                'params': {
                    'small': {'population_size': 60, 'max_generations': 80, 'crossover_prob': 0.9, 'mutation_prob': 0.1},
                    'medium': {'population_size': 80, 'max_generations': 100, 'crossover_prob': 0.9, 'mutation_prob': 0.1},
                    'large': {'population_size': 100, 'max_generations': 120, 'crossover_prob': 0.9, 'mutation_prob': 0.1},
                    'very_large': {'population_size': 120, 'max_generations': 150, 'crossover_prob': 0.9, 'mutation_prob': 0.1}
                }
            },
            'MOEA/D': {
                'class': MOEAD_Optimizer,
                'name': 'MOEA/D',
                'description': '基于分解的多目标进化算法',
                'params': {
                    'small': {'population_size': 60, 'max_generations': 80, 'neighbor_size': 10},
                    'medium': {'population_size': 80, 'max_generations': 100, 'neighbor_size': 15},
                    'large': {'population_size': 100, 'max_generations': 120, 'neighbor_size': 20},
                    'very_large': {'population_size': 120, 'max_generations': 150, 'neighbor_size': 25}
                }
            },
            'MOPSO': {
                'class': MOPSO_Optimizer,
                'name': 'MOPSO',
                'description': '多目标粒子群优化算法',
                'params': {
                    'small': {'swarm_size': 60, 'max_iterations': 80, 'archive_size': 90},
                    'medium': {'swarm_size': 80, 'max_iterations': 100, 'archive_size': 120},
                    'large': {'swarm_size': 100, 'max_iterations': 120, 'archive_size': 150},
                    'very_large': {'swarm_size': 120, 'max_iterations': 150, 'archive_size': 180}
                }
            },
            'MODE': {
                'class': MODE_Optimizer,
                'name': 'MODE',
                'description': '多目标差分进化算法',
                'params': {
                    'small': {'population_size': 60, 'max_generations': 80, 'F': 0.5, 'CR': 0.9},
                    'medium': {'population_size': 80, 'max_generations': 100, 'F': 0.5, 'CR': 0.9},
                    'large': {'population_size': 100, 'max_generations': 120, 'F': 0.5, 'CR': 0.9},
                    'very_large': {'population_size': 120, 'max_generations': 150, 'F': 0.5, 'CR': 0.9}
                }
            },
            'MOSA': {
                'class': MOSA_Optimizer,
                'name': 'MOSA',
                'description': '多目标模拟退火算法',
                'params': {
                    'small': {'max_iterations': 800, 'initial_temperature': 500, 'cooling_rate': 0.98, 'neighborhood_size': 10},
                    'medium': {'max_iterations': 1000, 'initial_temperature': 800, 'cooling_rate': 0.98, 'neighborhood_size': 12},
                    'large': {'max_iterations': 1200, 'initial_temperature': 1000, 'cooling_rate': 0.98, 'neighborhood_size': 15},
                    'very_large': {'max_iterations': 1500, 'initial_temperature': 1200, 'cooling_rate': 0.98, 'neighborhood_size': 18}
                }
            }
        }
    
    def run_comprehensive_comparison(self):
        """运行完整的算法对比实验"""
        print("🚀 开始完整的多目标算法对比实验")
        print("=" * 80)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 总体实验结果
        all_results = {}
        
        # 对每个测试问题运行对比实验
        for problem_config in self.test_problems:
            problem_name = problem_config['name']
            complexity = problem_config['complexity']
            
            print(f"\n🧪 测试问题: {problem_name} (复杂度: {complexity})")
            print("-" * 60)
            
            # 生成问题数据
            problem_data = self._generate_problem_data(problem_config)
            
            # 运行所有算法
            problem_results = {}
            for alg_name, alg_config in self.algorithms.items():
                print(f"  运行算法: {alg_name}")
                
                # 获取对应复杂度的参数
                scale_key = self._get_scale_key(complexity)
                params = alg_config['params'][scale_key]
                
                # 运行算法
                result = self._run_algorithm_experiment(
                    problem_data, 
                    alg_config['class'], 
                    params,
                    runs=5  # 每个算法运行5次
                )
                
                problem_results[alg_name] = result
                
                print(f"    最优加权目标: {result['best_weighted']:.2f}")
                print(f"    平均运行时间: {result['avg_runtime']:.2f}s")
            
            all_results[problem_name] = problem_results
            
            # 绘制该问题的帕累托前沿对比
            self._plot_pareto_comparison(problem_results, problem_name, timestamp)
        
        # 生成综合报告
        self._generate_comprehensive_report(all_results, timestamp)
        
        # 绘制综合性能图表
        self._plot_comprehensive_performance(all_results, timestamp)
        
        print(f"\n🎉 完整对比实验完成！结果保存在: {self.results_dir}/")
        
        return all_results

def main():
    """主函数"""
    print("🔬 启动完整的多目标算法对比实验")
    
    # 创建实验实例
    experiment = ComprehensiveComparisonExperiment()
    
    # 运行完整对比实验
    results = experiment.run_comprehensive_comparison()
    
    print("\n✅ 所有实验完成！")

if __name__ == "__main__":
    main() 
 
 
 
 
# -*- coding: utf-8 -*-
"""
完整的多目标算法对比实验方案
RL-Chaotic-HHO vs 其他主流多目标算法在完全异构MO-DHFSP问题上的性能对比
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Any
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from problem.mo_dhfsp import MO_DHFSP_Problem
from algorithm.rl_chaotic_hho import RL_ChaoticHHO_Optimizer
from algorithm.nsga2 import NSGA2_Optimizer
from algorithm.moead import MOEAD_Optimizer
from algorithm.mopso import MOPSO_Optimizer
from algorithm.mode import MODE_Optimizer
from algorithm.mosa import MOSA_Optimizer
from utils.data_generator import DataGenerator
from utils.performance_metrics import PerformanceEvaluator

class ComprehensiveComparisonExperiment:
    """完整算法对比实验类"""
    
    def __init__(self):
        self.results_dir = "results/comprehensive_comparison"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 完全异构测试问题集
        self.test_problems = self._generate_comprehensive_test_suite()
        
        # 算法配置
        self.algorithms = self._setup_algorithm_configurations()
        
    def _generate_comprehensive_test_suite(self) -> List[Dict]:
        """生成全面的完全异构测试问题集"""
        problems = []
        
        # 小规模问题集 (20作业)
        small_problems = [
            {
                'name': '小规模20×3×3',
                'n_jobs': 20, 'n_factories': 3, 'n_stages': 3,
                'heterogeneous_machines': {0: [2, 2, 2], 1: [2, 3, 3], 2: [2, 3, 4]},
                'complexity': 'low'
            },
            {
                'name': '小规模20×3×4',
                'n_jobs': 20, 'n_factories': 3, 'n_stages': 4,
                'heterogeneous_machines': {0: [1, 2, 2, 2], 1: [2, 3, 3, 2], 2: [3, 4, 4, 2]},
                'complexity': 'low'
            }
        ]
        
        # 中规模问题集 (50作业)
        medium_problems = [
            {
                'name': '中规模50×4×3',
                'n_jobs': 50, 'n_factories': 4, 'n_stages': 3,
                'heterogeneous_machines': {0: [2, 3, 2], 1: [3, 4, 3], 2: [3, 5, 3], 3: [4, 4, 4]},
                'complexity': 'medium'
            },
            {
                'name': '中规模50×4×4',
                'n_jobs': 50, 'n_factories': 4, 'n_stages': 4,
                'heterogeneous_machines': {0: [2, 2, 3, 2], 1: [3, 3, 4, 3], 2: [3, 4, 4, 3], 3: [3, 3, 4, 3]},
                'complexity': 'medium'
            }
        ]
        
        # 大规模问题集 (100作业)
        large_problems = [
            {
                'name': '大规模100×5×3',
                'n_jobs': 100, 'n_factories': 5, 'n_stages': 3,
                'heterogeneous_machines': {0: [2, 2, 3], 1: [3, 3, 4], 2: [3, 4, 4], 3: [4, 3, 5], 4: [3, 3, 4]},
                'complexity': 'high'
            },
            {
                'name': '大规模100×5×4',
                'n_jobs': 100, 'n_factories': 5, 'n_stages': 4,
                'heterogeneous_machines': {0: [1, 2, 2, 1], 1: [2, 3, 3, 2], 2: [2, 3, 4, 2], 3: [3, 4, 3, 2], 4: [2, 3, 4, 2]},
                'complexity': 'high'
            }
        ]
        
        # 超大规模问题集 (200作业)
        extra_large_problems = [
            {
                'name': '超大规模200×6×3',
                'n_jobs': 200, 'n_factories': 6, 'n_stages': 3,
                'heterogeneous_machines': {0: [3, 3, 4], 1: [4, 4, 5], 2: [4, 5, 5], 3: [5, 4, 6], 4: [4, 4, 5], 5: [3, 4, 5]},
                'complexity': 'very_high'
            }
        ]
        
        problems.extend(small_problems)
        problems.extend(medium_problems) 
        problems.extend(large_problems)
        problems.extend(extra_large_problems)
        
        return problems
    
    def _setup_algorithm_configurations(self) -> Dict:
        """设置算法配置"""
        return {
            'RL-Chaotic-HHO': {
                'class': RL_ChaoticHHO_Optimizer,
                'name': 'RL-Chaotic-HHO',
                'description': '基于强化学习协调的混沌哈里斯鹰优化算法',
                'params': {
                    'small': {'max_iterations': 80},
                    'medium': {'max_iterations': 100}, 
                    'large': {'max_iterations': 120},
                    'very_large': {'max_iterations': 150}
                }
            },
            'NSGA-II': {
                'class': NSGA2_Optimizer,
                'name': 'NSGA-II',
                'description': '非支配排序遗传算法II',
                'params': {
                    'small': {'population_size': 60, 'max_generations': 80, 'crossover_prob': 0.9, 'mutation_prob': 0.1},
                    'medium': {'population_size': 80, 'max_generations': 100, 'crossover_prob': 0.9, 'mutation_prob': 0.1},
                    'large': {'population_size': 100, 'max_generations': 120, 'crossover_prob': 0.9, 'mutation_prob': 0.1},
                    'very_large': {'population_size': 120, 'max_generations': 150, 'crossover_prob': 0.9, 'mutation_prob': 0.1}
                }
            },
            'MOEA/D': {
                'class': MOEAD_Optimizer,
                'name': 'MOEA/D',
                'description': '基于分解的多目标进化算法',
                'params': {
                    'small': {'population_size': 60, 'max_generations': 80, 'neighbor_size': 10},
                    'medium': {'population_size': 80, 'max_generations': 100, 'neighbor_size': 15},
                    'large': {'population_size': 100, 'max_generations': 120, 'neighbor_size': 20},
                    'very_large': {'population_size': 120, 'max_generations': 150, 'neighbor_size': 25}
                }
            },
            'MOPSO': {
                'class': MOPSO_Optimizer,
                'name': 'MOPSO',
                'description': '多目标粒子群优化算法',
                'params': {
                    'small': {'swarm_size': 60, 'max_iterations': 80, 'archive_size': 90},
                    'medium': {'swarm_size': 80, 'max_iterations': 100, 'archive_size': 120},
                    'large': {'swarm_size': 100, 'max_iterations': 120, 'archive_size': 150},
                    'very_large': {'swarm_size': 120, 'max_iterations': 150, 'archive_size': 180}
                }
            },
            'MODE': {
                'class': MODE_Optimizer,
                'name': 'MODE',
                'description': '多目标差分进化算法',
                'params': {
                    'small': {'population_size': 60, 'max_generations': 80, 'F': 0.5, 'CR': 0.9},
                    'medium': {'population_size': 80, 'max_generations': 100, 'F': 0.5, 'CR': 0.9},
                    'large': {'population_size': 100, 'max_generations': 120, 'F': 0.5, 'CR': 0.9},
                    'very_large': {'population_size': 120, 'max_generations': 150, 'F': 0.5, 'CR': 0.9}
                }
            },
            'MOSA': {
                'class': MOSA_Optimizer,
                'name': 'MOSA',
                'description': '多目标模拟退火算法',
                'params': {
                    'small': {'max_iterations': 800, 'initial_temperature': 500, 'cooling_rate': 0.98, 'neighborhood_size': 10},
                    'medium': {'max_iterations': 1000, 'initial_temperature': 800, 'cooling_rate': 0.98, 'neighborhood_size': 12},
                    'large': {'max_iterations': 1200, 'initial_temperature': 1000, 'cooling_rate': 0.98, 'neighborhood_size': 15},
                    'very_large': {'max_iterations': 1500, 'initial_temperature': 1200, 'cooling_rate': 0.98, 'neighborhood_size': 18}
                }
            }
        }
    
    def run_comprehensive_comparison(self):
        """运行完整的算法对比实验"""
        print("🚀 开始完整的多目标算法对比实验")
        print("=" * 80)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 总体实验结果
        all_results = {}
        
        # 对每个测试问题运行对比实验
        for problem_config in self.test_problems:
            problem_name = problem_config['name']
            complexity = problem_config['complexity']
            
            print(f"\n🧪 测试问题: {problem_name} (复杂度: {complexity})")
            print("-" * 60)
            
            # 生成问题数据
            problem_data = self._generate_problem_data(problem_config)
            
            # 运行所有算法
            problem_results = {}
            for alg_name, alg_config in self.algorithms.items():
                print(f"  运行算法: {alg_name}")
                
                # 获取对应复杂度的参数
                scale_key = self._get_scale_key(complexity)
                params = alg_config['params'][scale_key]
                
                # 运行算法
                result = self._run_algorithm_experiment(
                    problem_data, 
                    alg_config['class'], 
                    params,
                    runs=5  # 每个算法运行5次
                )
                
                problem_results[alg_name] = result
                
                print(f"    最优加权目标: {result['best_weighted']:.2f}")
                print(f"    平均运行时间: {result['avg_runtime']:.2f}s")
            
            all_results[problem_name] = problem_results
            
            # 绘制该问题的帕累托前沿对比
            self._plot_pareto_comparison(problem_results, problem_name, timestamp)
        
        # 生成综合报告
        self._generate_comprehensive_report(all_results, timestamp)
        
        # 绘制综合性能图表
        self._plot_comprehensive_performance(all_results, timestamp)
        
        print(f"\n🎉 完整对比实验完成！结果保存在: {self.results_dir}/")
        
        return all_results

def main():
    """主函数"""
    print("🔬 启动完整的多目标算法对比实验")
    
    # 创建实验实例
    experiment = ComprehensiveComparisonExperiment()
    
    # 运行完整对比实验
    results = experiment.run_comprehensive_comparison()
    
    print("\n✅ 所有实验完成！")

if __name__ == "__main__":
    main() 
# -*- coding: utf-8 -*-
"""
完整的多目标算法对比实验方案
RL-Chaotic-HHO vs 其他主流多目标算法在完全异构MO-DHFSP问题上的性能对比
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Any
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from problem.mo_dhfsp import MO_DHFSP_Problem
from algorithm.rl_chaotic_hho import RL_ChaoticHHO_Optimizer
from algorithm.nsga2 import NSGA2_Optimizer
from algorithm.moead import MOEAD_Optimizer
from algorithm.mopso import MOPSO_Optimizer
from algorithm.mode import MODE_Optimizer
from algorithm.mosa import MOSA_Optimizer
from utils.data_generator import DataGenerator
from utils.performance_metrics import PerformanceEvaluator

class ComprehensiveComparisonExperiment:
    """完整算法对比实验类"""
    
    def __init__(self):
        self.results_dir = "results/comprehensive_comparison"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 完全异构测试问题集
        self.test_problems = self._generate_comprehensive_test_suite()
        
        # 算法配置
        self.algorithms = self._setup_algorithm_configurations()
        
    def _generate_comprehensive_test_suite(self) -> List[Dict]:
        """生成全面的完全异构测试问题集"""
        problems = []
        
        # 小规模问题集 (20作业)
        small_problems = [
            {
                'name': '小规模20×3×3',
                'n_jobs': 20, 'n_factories': 3, 'n_stages': 3,
                'heterogeneous_machines': {0: [2, 2, 2], 1: [2, 3, 3], 2: [2, 3, 4]},
                'complexity': 'low'
            },
            {
                'name': '小规模20×3×4',
                'n_jobs': 20, 'n_factories': 3, 'n_stages': 4,
                'heterogeneous_machines': {0: [1, 2, 2, 2], 1: [2, 3, 3, 2], 2: [3, 4, 4, 2]},
                'complexity': 'low'
            }
        ]
        
        # 中规模问题集 (50作业)
        medium_problems = [
            {
                'name': '中规模50×4×3',
                'n_jobs': 50, 'n_factories': 4, 'n_stages': 3,
                'heterogeneous_machines': {0: [2, 3, 2], 1: [3, 4, 3], 2: [3, 5, 3], 3: [4, 4, 4]},
                'complexity': 'medium'
            },
            {
                'name': '中规模50×4×4',
                'n_jobs': 50, 'n_factories': 4, 'n_stages': 4,
                'heterogeneous_machines': {0: [2, 2, 3, 2], 1: [3, 3, 4, 3], 2: [3, 4, 4, 3], 3: [3, 3, 4, 3]},
                'complexity': 'medium'
            }
        ]
        
        # 大规模问题集 (100作业)
        large_problems = [
            {
                'name': '大规模100×5×3',
                'n_jobs': 100, 'n_factories': 5, 'n_stages': 3,
                'heterogeneous_machines': {0: [2, 2, 3], 1: [3, 3, 4], 2: [3, 4, 4], 3: [4, 3, 5], 4: [3, 3, 4]},
                'complexity': 'high'
            },
            {
                'name': '大规模100×5×4',
                'n_jobs': 100, 'n_factories': 5, 'n_stages': 4,
                'heterogeneous_machines': {0: [1, 2, 2, 1], 1: [2, 3, 3, 2], 2: [2, 3, 4, 2], 3: [3, 4, 3, 2], 4: [2, 3, 4, 2]},
                'complexity': 'high'
            }
        ]
        
        # 超大规模问题集 (200作业)
        extra_large_problems = [
            {
                'name': '超大规模200×6×3',
                'n_jobs': 200, 'n_factories': 6, 'n_stages': 3,
                'heterogeneous_machines': {0: [3, 3, 4], 1: [4, 4, 5], 2: [4, 5, 5], 3: [5, 4, 6], 4: [4, 4, 5], 5: [3, 4, 5]},
                'complexity': 'very_high'
            }
        ]
        
        problems.extend(small_problems)
        problems.extend(medium_problems) 
        problems.extend(large_problems)
        problems.extend(extra_large_problems)
        
        return problems
    
    def _setup_algorithm_configurations(self) -> Dict:
        """设置算法配置"""
        return {
            'RL-Chaotic-HHO': {
                'class': RL_ChaoticHHO_Optimizer,
                'name': 'RL-Chaotic-HHO',
                'description': '基于强化学习协调的混沌哈里斯鹰优化算法',
                'params': {
                    'small': {'max_iterations': 80},
                    'medium': {'max_iterations': 100}, 
                    'large': {'max_iterations': 120},
                    'very_large': {'max_iterations': 150}
                }
            },
            'NSGA-II': {
                'class': NSGA2_Optimizer,
                'name': 'NSGA-II',
                'description': '非支配排序遗传算法II',
                'params': {
                    'small': {'population_size': 60, 'max_generations': 80, 'crossover_prob': 0.9, 'mutation_prob': 0.1},
                    'medium': {'population_size': 80, 'max_generations': 100, 'crossover_prob': 0.9, 'mutation_prob': 0.1},
                    'large': {'population_size': 100, 'max_generations': 120, 'crossover_prob': 0.9, 'mutation_prob': 0.1},
                    'very_large': {'population_size': 120, 'max_generations': 150, 'crossover_prob': 0.9, 'mutation_prob': 0.1}
                }
            },
            'MOEA/D': {
                'class': MOEAD_Optimizer,
                'name': 'MOEA/D',
                'description': '基于分解的多目标进化算法',
                'params': {
                    'small': {'population_size': 60, 'max_generations': 80, 'neighbor_size': 10},
                    'medium': {'population_size': 80, 'max_generations': 100, 'neighbor_size': 15},
                    'large': {'population_size': 100, 'max_generations': 120, 'neighbor_size': 20},
                    'very_large': {'population_size': 120, 'max_generations': 150, 'neighbor_size': 25}
                }
            },
            'MOPSO': {
                'class': MOPSO_Optimizer,
                'name': 'MOPSO',
                'description': '多目标粒子群优化算法',
                'params': {
                    'small': {'swarm_size': 60, 'max_iterations': 80, 'archive_size': 90},
                    'medium': {'swarm_size': 80, 'max_iterations': 100, 'archive_size': 120},
                    'large': {'swarm_size': 100, 'max_iterations': 120, 'archive_size': 150},
                    'very_large': {'swarm_size': 120, 'max_iterations': 150, 'archive_size': 180}
                }
            },
            'MODE': {
                'class': MODE_Optimizer,
                'name': 'MODE',
                'description': '多目标差分进化算法',
                'params': {
                    'small': {'population_size': 60, 'max_generations': 80, 'F': 0.5, 'CR': 0.9},
                    'medium': {'population_size': 80, 'max_generations': 100, 'F': 0.5, 'CR': 0.9},
                    'large': {'population_size': 100, 'max_generations': 120, 'F': 0.5, 'CR': 0.9},
                    'very_large': {'population_size': 120, 'max_generations': 150, 'F': 0.5, 'CR': 0.9}
                }
            },
            'MOSA': {
                'class': MOSA_Optimizer,
                'name': 'MOSA',
                'description': '多目标模拟退火算法',
                'params': {
                    'small': {'max_iterations': 800, 'initial_temperature': 500, 'cooling_rate': 0.98, 'neighborhood_size': 10},
                    'medium': {'max_iterations': 1000, 'initial_temperature': 800, 'cooling_rate': 0.98, 'neighborhood_size': 12},
                    'large': {'max_iterations': 1200, 'initial_temperature': 1000, 'cooling_rate': 0.98, 'neighborhood_size': 15},
                    'very_large': {'max_iterations': 1500, 'initial_temperature': 1200, 'cooling_rate': 0.98, 'neighborhood_size': 18}
                }
            }
        }
    
    def run_comprehensive_comparison(self):
        """运行完整的算法对比实验"""
        print("🚀 开始完整的多目标算法对比实验")
        print("=" * 80)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 总体实验结果
        all_results = {}
        
        # 对每个测试问题运行对比实验
        for problem_config in self.test_problems:
            problem_name = problem_config['name']
            complexity = problem_config['complexity']
            
            print(f"\n🧪 测试问题: {problem_name} (复杂度: {complexity})")
            print("-" * 60)
            
            # 生成问题数据
            problem_data = self._generate_problem_data(problem_config)
            
            # 运行所有算法
            problem_results = {}
            for alg_name, alg_config in self.algorithms.items():
                print(f"  运行算法: {alg_name}")
                
                # 获取对应复杂度的参数
                scale_key = self._get_scale_key(complexity)
                params = alg_config['params'][scale_key]
                
                # 运行算法
                result = self._run_algorithm_experiment(
                    problem_data, 
                    alg_config['class'], 
                    params,
                    runs=5  # 每个算法运行5次
                )
                
                problem_results[alg_name] = result
                
                print(f"    最优加权目标: {result['best_weighted']:.2f}")
                print(f"    平均运行时间: {result['avg_runtime']:.2f}s")
            
            all_results[problem_name] = problem_results
            
            # 绘制该问题的帕累托前沿对比
            self._plot_pareto_comparison(problem_results, problem_name, timestamp)
        
        # 生成综合报告
        self._generate_comprehensive_report(all_results, timestamp)
        
        # 绘制综合性能图表
        self._plot_comprehensive_performance(all_results, timestamp)
        
        print(f"\n🎉 完整对比实验完成！结果保存在: {self.results_dir}/")
        
        return all_results

def main():
    """主函数"""
    print("🔬 启动完整的多目标算法对比实验")
    
    # 创建实验实例
    experiment = ComprehensiveComparisonExperiment()
    
    # 运行完整对比实验
    results = experiment.run_comprehensive_comparison()
    
    print("\n✅ 所有实验完成！")

if __name__ == "__main__":
    main() 
 
# -*- coding: utf-8 -*-
"""
完整的多目标算法对比实验方案
RL-Chaotic-HHO vs 其他主流多目标算法在完全异构MO-DHFSP问题上的性能对比
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Any
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from problem.mo_dhfsp import MO_DHFSP_Problem
from algorithm.rl_chaotic_hho import RL_ChaoticHHO_Optimizer
from algorithm.nsga2 import NSGA2_Optimizer
from algorithm.moead import MOEAD_Optimizer
from algorithm.mopso import MOPSO_Optimizer
from algorithm.mode import MODE_Optimizer
from algorithm.mosa import MOSA_Optimizer
from utils.data_generator import DataGenerator
from utils.performance_metrics import PerformanceEvaluator

class ComprehensiveComparisonExperiment:
    """完整算法对比实验类"""
    
    def __init__(self):
        self.results_dir = "results/comprehensive_comparison"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 完全异构测试问题集
        self.test_problems = self._generate_comprehensive_test_suite()
        
        # 算法配置
        self.algorithms = self._setup_algorithm_configurations()
        
    def _generate_comprehensive_test_suite(self) -> List[Dict]:
        """生成全面的完全异构测试问题集"""
        problems = []
        
        # 小规模问题集 (20作业)
        small_problems = [
            {
                'name': '小规模20×3×3',
                'n_jobs': 20, 'n_factories': 3, 'n_stages': 3,
                'heterogeneous_machines': {0: [2, 2, 2], 1: [2, 3, 3], 2: [2, 3, 4]},
                'complexity': 'low'
            },
            {
                'name': '小规模20×3×4',
                'n_jobs': 20, 'n_factories': 3, 'n_stages': 4,
                'heterogeneous_machines': {0: [1, 2, 2, 2], 1: [2, 3, 3, 2], 2: [3, 4, 4, 2]},
                'complexity': 'low'
            }
        ]
        
        # 中规模问题集 (50作业)
        medium_problems = [
            {
                'name': '中规模50×4×3',
                'n_jobs': 50, 'n_factories': 4, 'n_stages': 3,
                'heterogeneous_machines': {0: [2, 3, 2], 1: [3, 4, 3], 2: [3, 5, 3], 3: [4, 4, 4]},
                'complexity': 'medium'
            },
            {
                'name': '中规模50×4×4',
                'n_jobs': 50, 'n_factories': 4, 'n_stages': 4,
                'heterogeneous_machines': {0: [2, 2, 3, 2], 1: [3, 3, 4, 3], 2: [3, 4, 4, 3], 3: [3, 3, 4, 3]},
                'complexity': 'medium'
            }
        ]
        
        # 大规模问题集 (100作业)
        large_problems = [
            {
                'name': '大规模100×5×3',
                'n_jobs': 100, 'n_factories': 5, 'n_stages': 3,
                'heterogeneous_machines': {0: [2, 2, 3], 1: [3, 3, 4], 2: [3, 4, 4], 3: [4, 3, 5], 4: [3, 3, 4]},
                'complexity': 'high'
            },
            {
                'name': '大规模100×5×4',
                'n_jobs': 100, 'n_factories': 5, 'n_stages': 4,
                'heterogeneous_machines': {0: [1, 2, 2, 1], 1: [2, 3, 3, 2], 2: [2, 3, 4, 2], 3: [3, 4, 3, 2], 4: [2, 3, 4, 2]},
                'complexity': 'high'
            }
        ]
        
        # 超大规模问题集 (200作业)
        extra_large_problems = [
            {
                'name': '超大规模200×6×3',
                'n_jobs': 200, 'n_factories': 6, 'n_stages': 3,
                'heterogeneous_machines': {0: [3, 3, 4], 1: [4, 4, 5], 2: [4, 5, 5], 3: [5, 4, 6], 4: [4, 4, 5], 5: [3, 4, 5]},
                'complexity': 'very_high'
            }
        ]
        
        problems.extend(small_problems)
        problems.extend(medium_problems) 
        problems.extend(large_problems)
        problems.extend(extra_large_problems)
        
        return problems
    
    def _setup_algorithm_configurations(self) -> Dict:
        """设置算法配置"""
        return {
            'RL-Chaotic-HHO': {
                'class': RL_ChaoticHHO_Optimizer,
                'name': 'RL-Chaotic-HHO',
                'description': '基于强化学习协调的混沌哈里斯鹰优化算法',
                'params': {
                    'small': {'max_iterations': 80},
                    'medium': {'max_iterations': 100}, 
                    'large': {'max_iterations': 120},
                    'very_large': {'max_iterations': 150}
                }
            },
            'NSGA-II': {
                'class': NSGA2_Optimizer,
                'name': 'NSGA-II',
                'description': '非支配排序遗传算法II',
                'params': {
                    'small': {'population_size': 60, 'max_generations': 80, 'crossover_prob': 0.9, 'mutation_prob': 0.1},
                    'medium': {'population_size': 80, 'max_generations': 100, 'crossover_prob': 0.9, 'mutation_prob': 0.1},
                    'large': {'population_size': 100, 'max_generations': 120, 'crossover_prob': 0.9, 'mutation_prob': 0.1},
                    'very_large': {'population_size': 120, 'max_generations': 150, 'crossover_prob': 0.9, 'mutation_prob': 0.1}
                }
            },
            'MOEA/D': {
                'class': MOEAD_Optimizer,
                'name': 'MOEA/D',
                'description': '基于分解的多目标进化算法',
                'params': {
                    'small': {'population_size': 60, 'max_generations': 80, 'neighbor_size': 10},
                    'medium': {'population_size': 80, 'max_generations': 100, 'neighbor_size': 15},
                    'large': {'population_size': 100, 'max_generations': 120, 'neighbor_size': 20},
                    'very_large': {'population_size': 120, 'max_generations': 150, 'neighbor_size': 25}
                }
            },
            'MOPSO': {
                'class': MOPSO_Optimizer,
                'name': 'MOPSO',
                'description': '多目标粒子群优化算法',
                'params': {
                    'small': {'swarm_size': 60, 'max_iterations': 80, 'archive_size': 90},
                    'medium': {'swarm_size': 80, 'max_iterations': 100, 'archive_size': 120},
                    'large': {'swarm_size': 100, 'max_iterations': 120, 'archive_size': 150},
                    'very_large': {'swarm_size': 120, 'max_iterations': 150, 'archive_size': 180}
                }
            },
            'MODE': {
                'class': MODE_Optimizer,
                'name': 'MODE',
                'description': '多目标差分进化算法',
                'params': {
                    'small': {'population_size': 60, 'max_generations': 80, 'F': 0.5, 'CR': 0.9},
                    'medium': {'population_size': 80, 'max_generations': 100, 'F': 0.5, 'CR': 0.9},
                    'large': {'population_size': 100, 'max_generations': 120, 'F': 0.5, 'CR': 0.9},
                    'very_large': {'population_size': 120, 'max_generations': 150, 'F': 0.5, 'CR': 0.9}
                }
            },
            'MOSA': {
                'class': MOSA_Optimizer,
                'name': 'MOSA',
                'description': '多目标模拟退火算法',
                'params': {
                    'small': {'max_iterations': 800, 'initial_temperature': 500, 'cooling_rate': 0.98, 'neighborhood_size': 10},
                    'medium': {'max_iterations': 1000, 'initial_temperature': 800, 'cooling_rate': 0.98, 'neighborhood_size': 12},
                    'large': {'max_iterations': 1200, 'initial_temperature': 1000, 'cooling_rate': 0.98, 'neighborhood_size': 15},
                    'very_large': {'max_iterations': 1500, 'initial_temperature': 1200, 'cooling_rate': 0.98, 'neighborhood_size': 18}
                }
            }
        }
    
    def run_comprehensive_comparison(self):
        """运行完整的算法对比实验"""
        print("🚀 开始完整的多目标算法对比实验")
        print("=" * 80)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 总体实验结果
        all_results = {}
        
        # 对每个测试问题运行对比实验
        for problem_config in self.test_problems:
            problem_name = problem_config['name']
            complexity = problem_config['complexity']
            
            print(f"\n🧪 测试问题: {problem_name} (复杂度: {complexity})")
            print("-" * 60)
            
            # 生成问题数据
            problem_data = self._generate_problem_data(problem_config)
            
            # 运行所有算法
            problem_results = {}
            for alg_name, alg_config in self.algorithms.items():
                print(f"  运行算法: {alg_name}")
                
                # 获取对应复杂度的参数
                scale_key = self._get_scale_key(complexity)
                params = alg_config['params'][scale_key]
                
                # 运行算法
                result = self._run_algorithm_experiment(
                    problem_data, 
                    alg_config['class'], 
                    params,
                    runs=5  # 每个算法运行5次
                )
                
                problem_results[alg_name] = result
                
                print(f"    最优加权目标: {result['best_weighted']:.2f}")
                print(f"    平均运行时间: {result['avg_runtime']:.2f}s")
            
            all_results[problem_name] = problem_results
            
            # 绘制该问题的帕累托前沿对比
            self._plot_pareto_comparison(problem_results, problem_name, timestamp)
        
        # 生成综合报告
        self._generate_comprehensive_report(all_results, timestamp)
        
        # 绘制综合性能图表
        self._plot_comprehensive_performance(all_results, timestamp)
        
        print(f"\n🎉 完整对比实验完成！结果保存在: {self.results_dir}/")
        
        return all_results

def main():
    """主函数"""
    print("🔬 启动完整的多目标算法对比实验")
    
    # 创建实验实例
    experiment = ComprehensiveComparisonExperiment()
    
    # 运行完整对比实验
    results = experiment.run_comprehensive_comparison()
    
    print("\n✅ 所有实验完成！")

if __name__ == "__main__":
    main() 
# -*- coding: utf-8 -*-
"""
完整的多目标算法对比实验方案
RL-Chaotic-HHO vs 其他主流多目标算法在完全异构MO-DHFSP问题上的性能对比
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Any
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from problem.mo_dhfsp import MO_DHFSP_Problem
from algorithm.rl_chaotic_hho import RL_ChaoticHHO_Optimizer
from algorithm.nsga2 import NSGA2_Optimizer
from algorithm.moead import MOEAD_Optimizer
from algorithm.mopso import MOPSO_Optimizer
from algorithm.mode import MODE_Optimizer
from algorithm.mosa import MOSA_Optimizer
from utils.data_generator import DataGenerator
from utils.performance_metrics import PerformanceEvaluator

class ComprehensiveComparisonExperiment:
    """完整算法对比实验类"""
    
    def __init__(self):
        self.results_dir = "results/comprehensive_comparison"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 完全异构测试问题集
        self.test_problems = self._generate_comprehensive_test_suite()
        
        # 算法配置
        self.algorithms = self._setup_algorithm_configurations()
        
    def _generate_comprehensive_test_suite(self) -> List[Dict]:
        """生成全面的完全异构测试问题集"""
        problems = []
        
        # 小规模问题集 (20作业)
        small_problems = [
            {
                'name': '小规模20×3×3',
                'n_jobs': 20, 'n_factories': 3, 'n_stages': 3,
                'heterogeneous_machines': {0: [2, 2, 2], 1: [2, 3, 3], 2: [2, 3, 4]},
                'complexity': 'low'
            },
            {
                'name': '小规模20×3×4',
                'n_jobs': 20, 'n_factories': 3, 'n_stages': 4,
                'heterogeneous_machines': {0: [1, 2, 2, 2], 1: [2, 3, 3, 2], 2: [3, 4, 4, 2]},
                'complexity': 'low'
            }
        ]
        
        # 中规模问题集 (50作业)
        medium_problems = [
            {
                'name': '中规模50×4×3',
                'n_jobs': 50, 'n_factories': 4, 'n_stages': 3,
                'heterogeneous_machines': {0: [2, 3, 2], 1: [3, 4, 3], 2: [3, 5, 3], 3: [4, 4, 4]},
                'complexity': 'medium'
            },
            {
                'name': '中规模50×4×4',
                'n_jobs': 50, 'n_factories': 4, 'n_stages': 4,
                'heterogeneous_machines': {0: [2, 2, 3, 2], 1: [3, 3, 4, 3], 2: [3, 4, 4, 3], 3: [3, 3, 4, 3]},
                'complexity': 'medium'
            }
        ]
        
        # 大规模问题集 (100作业)
        large_problems = [
            {
                'name': '大规模100×5×3',
                'n_jobs': 100, 'n_factories': 5, 'n_stages': 3,
                'heterogeneous_machines': {0: [2, 2, 3], 1: [3, 3, 4], 2: [3, 4, 4], 3: [4, 3, 5], 4: [3, 3, 4]},
                'complexity': 'high'
            },
            {
                'name': '大规模100×5×4',
                'n_jobs': 100, 'n_factories': 5, 'n_stages': 4,
                'heterogeneous_machines': {0: [1, 2, 2, 1], 1: [2, 3, 3, 2], 2: [2, 3, 4, 2], 3: [3, 4, 3, 2], 4: [2, 3, 4, 2]},
                'complexity': 'high'
            }
        ]
        
        # 超大规模问题集 (200作业)
        extra_large_problems = [
            {
                'name': '超大规模200×6×3',
                'n_jobs': 200, 'n_factories': 6, 'n_stages': 3,
                'heterogeneous_machines': {0: [3, 3, 4], 1: [4, 4, 5], 2: [4, 5, 5], 3: [5, 4, 6], 4: [4, 4, 5], 5: [3, 4, 5]},
                'complexity': 'very_high'
            }
        ]
        
        problems.extend(small_problems)
        problems.extend(medium_problems) 
        problems.extend(large_problems)
        problems.extend(extra_large_problems)
        
        return problems
    
    def _setup_algorithm_configurations(self) -> Dict:
        """设置算法配置"""
        return {
            'RL-Chaotic-HHO': {
                'class': RL_ChaoticHHO_Optimizer,
                'name': 'RL-Chaotic-HHO',
                'description': '基于强化学习协调的混沌哈里斯鹰优化算法',
                'params': {
                    'small': {'max_iterations': 80},
                    'medium': {'max_iterations': 100}, 
                    'large': {'max_iterations': 120},
                    'very_large': {'max_iterations': 150}
                }
            },
            'NSGA-II': {
                'class': NSGA2_Optimizer,
                'name': 'NSGA-II',
                'description': '非支配排序遗传算法II',
                'params': {
                    'small': {'population_size': 60, 'max_generations': 80, 'crossover_prob': 0.9, 'mutation_prob': 0.1},
                    'medium': {'population_size': 80, 'max_generations': 100, 'crossover_prob': 0.9, 'mutation_prob': 0.1},
                    'large': {'population_size': 100, 'max_generations': 120, 'crossover_prob': 0.9, 'mutation_prob': 0.1},
                    'very_large': {'population_size': 120, 'max_generations': 150, 'crossover_prob': 0.9, 'mutation_prob': 0.1}
                }
            },
            'MOEA/D': {
                'class': MOEAD_Optimizer,
                'name': 'MOEA/D',
                'description': '基于分解的多目标进化算法',
                'params': {
                    'small': {'population_size': 60, 'max_generations': 80, 'neighbor_size': 10},
                    'medium': {'population_size': 80, 'max_generations': 100, 'neighbor_size': 15},
                    'large': {'population_size': 100, 'max_generations': 120, 'neighbor_size': 20},
                    'very_large': {'population_size': 120, 'max_generations': 150, 'neighbor_size': 25}
                }
            },
            'MOPSO': {
                'class': MOPSO_Optimizer,
                'name': 'MOPSO',
                'description': '多目标粒子群优化算法',
                'params': {
                    'small': {'swarm_size': 60, 'max_iterations': 80, 'archive_size': 90},
                    'medium': {'swarm_size': 80, 'max_iterations': 100, 'archive_size': 120},
                    'large': {'swarm_size': 100, 'max_iterations': 120, 'archive_size': 150},
                    'very_large': {'swarm_size': 120, 'max_iterations': 150, 'archive_size': 180}
                }
            },
            'MODE': {
                'class': MODE_Optimizer,
                'name': 'MODE',
                'description': '多目标差分进化算法',
                'params': {
                    'small': {'population_size': 60, 'max_generations': 80, 'F': 0.5, 'CR': 0.9},
                    'medium': {'population_size': 80, 'max_generations': 100, 'F': 0.5, 'CR': 0.9},
                    'large': {'population_size': 100, 'max_generations': 120, 'F': 0.5, 'CR': 0.9},
                    'very_large': {'population_size': 120, 'max_generations': 150, 'F': 0.5, 'CR': 0.9}
                }
            },
            'MOSA': {
                'class': MOSA_Optimizer,
                'name': 'MOSA',
                'description': '多目标模拟退火算法',
                'params': {
                    'small': {'max_iterations': 800, 'initial_temperature': 500, 'cooling_rate': 0.98, 'neighborhood_size': 10},
                    'medium': {'max_iterations': 1000, 'initial_temperature': 800, 'cooling_rate': 0.98, 'neighborhood_size': 12},
                    'large': {'max_iterations': 1200, 'initial_temperature': 1000, 'cooling_rate': 0.98, 'neighborhood_size': 15},
                    'very_large': {'max_iterations': 1500, 'initial_temperature': 1200, 'cooling_rate': 0.98, 'neighborhood_size': 18}
                }
            }
        }
    
    def run_comprehensive_comparison(self):
        """运行完整的算法对比实验"""
        print("🚀 开始完整的多目标算法对比实验")
        print("=" * 80)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 总体实验结果
        all_results = {}
        
        # 对每个测试问题运行对比实验
        for problem_config in self.test_problems:
            problem_name = problem_config['name']
            complexity = problem_config['complexity']
            
            print(f"\n🧪 测试问题: {problem_name} (复杂度: {complexity})")
            print("-" * 60)
            
            # 生成问题数据
            problem_data = self._generate_problem_data(problem_config)
            
            # 运行所有算法
            problem_results = {}
            for alg_name, alg_config in self.algorithms.items():
                print(f"  运行算法: {alg_name}")
                
                # 获取对应复杂度的参数
                scale_key = self._get_scale_key(complexity)
                params = alg_config['params'][scale_key]
                
                # 运行算法
                result = self._run_algorithm_experiment(
                    problem_data, 
                    alg_config['class'], 
                    params,
                    runs=5  # 每个算法运行5次
                )
                
                problem_results[alg_name] = result
                
                print(f"    最优加权目标: {result['best_weighted']:.2f}")
                print(f"    平均运行时间: {result['avg_runtime']:.2f}s")
            
            all_results[problem_name] = problem_results
            
            # 绘制该问题的帕累托前沿对比
            self._plot_pareto_comparison(problem_results, problem_name, timestamp)
        
        # 生成综合报告
        self._generate_comprehensive_report(all_results, timestamp)
        
        # 绘制综合性能图表
        self._plot_comprehensive_performance(all_results, timestamp)
        
        print(f"\n🎉 完整对比实验完成！结果保存在: {self.results_dir}/")
        
        return all_results

def main():
    """主函数"""
    print("🔬 启动完整的多目标算法对比实验")
    
    # 创建实验实例
    experiment = ComprehensiveComparisonExperiment()
    
    # 运行完整对比实验
    results = experiment.run_comprehensive_comparison()
    
    print("\n✅ 所有实验完成！")

if __name__ == "__main__":
    main() 
 
 
 
 
 
 
 
 