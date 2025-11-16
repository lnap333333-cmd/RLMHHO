#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
结果可视化工具
用于绘制帕累托前沿和收敛曲线
"""

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple
from problem.mo_dhfsp import Solution

class ResultVisualizer:
    """结果可视化器"""
    
    def __init__(self):
        """初始化可视化器"""
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文显示
        plt.rcParams['axes.unicode_minus'] = False
    
    def calculate_inverted_generational_distance(self, pareto_front: List[Solution], 
                                                reference_front: List[Solution]) -> float:
        """
        计算反世代距离 (Inverted Generational Distance, IGD)
        
        Args:
            pareto_front: 算法得到的帕累托前沿
            reference_front: 参考帕累托前沿 (通常是所有算法的合并前沿)
        
        Returns:
            IGD值，越小越好
        """
        if not pareto_front or not reference_front:
            return float('inf')
        
        total_distance = 0.0
        
        for ref_sol in reference_front:
            min_distance = float('inf')
            
            for pf_sol in pareto_front:
                # 计算欧几里得距离
                distance = np.sqrt(
                    (ref_sol.makespan - pf_sol.makespan)**2 + 
                    (ref_sol.total_tardiness - pf_sol.total_tardiness)**2
                )
                min_distance = min(min_distance, distance)
            
            total_distance += min_distance
        
        return total_distance / len(reference_front)
    
    def calculate_hypervolume(self, pareto_front: List[Solution], 
                             reference_point: Tuple[float, float] = None) -> float:
        """
        计算超体积 (Hypervolume, HV)
        
        Args:
            pareto_front: 帕累托前沿
            reference_point: 参考点 (makespan, tardiness)
        
        Returns:
            超体积值，越大越好
        """
        if not pareto_front:
            return 0.0
        
        # 如果没有提供参考点，使用所有解的最大值
        if reference_point is None:
            max_makespan = max(sol.makespan for sol in pareto_front) * 1.1
            max_tardiness = max(sol.total_tardiness for sol in pareto_front) * 1.1
            reference_point = (max_makespan, max_tardiness)
        
        # 简化的超体积计算 (2D情况)
        points = [(sol.makespan, sol.total_tardiness) for sol in pareto_front]
        points.sort()  # 按第一个目标排序
        
        hypervolume = 0.0
        prev_x = 0
        
        for x, y in points:
            if x < reference_point[0] and y < reference_point[1]:
                width = x - prev_x
                height = reference_point[1] - y
                hypervolume += width * height
                prev_x = x
        
        return hypervolume
    
    def plot_detailed_algorithm_comparison(self, results: Dict, problem_name: str):
        """绘制详细的算法对比图"""
        algorithms = list(results['algorithms'].keys())
        
        if len(algorithms) < 2:
            print("至少需要两个算法进行对比")
            return
        
        # 创建更大的图形，包含6个子图
        fig = plt.figure(figsize=(20, 15))
        fig.suptitle(f'{problem_name} - 详细算法对比分析', fontsize=18, fontweight='bold')
        
        # 颜色和标记样式
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']
        markers = ['o', 's', '^', 'D', 'v', '<']
        
        # 1. 帕累托前沿对比 (左上)
        ax1 = plt.subplot(2, 3, 1)
        for i, alg_name in enumerate(algorithms):
            solutions = results['algorithms'][alg_name]['pareto_solutions']
            if solutions:
                makespans = [sol.makespan for sol in solutions]
                tardiness = [sol.total_tardiness for sol in solutions]
                ax1.scatter(makespans, tardiness, 
                           c=colors[i % len(colors)], 
                           marker=markers[i % len(markers)],
                           label=alg_name, 
                           alpha=0.8, 
                           s=60,
                           edgecolors='black',
                           linewidth=0.5)
        
        ax1.set_xlabel('完工时间 (Makespan)', fontsize=12)
        ax1.set_ylabel('总拖期 (Total Tardiness)', fontsize=12)
        ax1.set_title('帕累托前沿对比', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 2. 完工时间收敛对比 (右上)
        ax2 = plt.subplot(2, 3, 2)
        for i, alg_name in enumerate(algorithms):
            conv_data = results['algorithms'][alg_name]['convergence_data']
            makespan_history = conv_data.get('makespan_history', [])
            if makespan_history:
                ax2.plot(makespan_history, 
                        color=colors[i % len(colors)], 
                        label=alg_name, 
                        linewidth=2.5,
                        alpha=0.8)
        
        ax2.set_xlabel('迭代次数', fontsize=12)
        ax2.set_ylabel('最优完工时间', fontsize=12)
        ax2.set_title('完工时间收敛对比', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 3. 总拖期收敛对比 (中上)
        ax3 = plt.subplot(2, 3, 3)
        for i, alg_name in enumerate(algorithms):
            conv_data = results['algorithms'][alg_name]['convergence_data']
            tardiness_history = conv_data.get('tardiness_history', [])
            if tardiness_history:
                ax3.plot(tardiness_history, 
                        color=colors[i % len(colors)], 
                        label=alg_name, 
                        linewidth=2.5,
                        alpha=0.8)
        
        ax3.set_xlabel('迭代次数', fontsize=12)
        ax3.set_ylabel('最优总拖期', fontsize=12)
        ax3.set_title('总拖期收敛对比', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # 4. 反世代距离对比 (左下)
        ax4 = plt.subplot(2, 3, 4)
        
        # 计算合并的参考前沿
        all_solutions = []
        for alg_name in algorithms:
            all_solutions.extend(results['algorithms'][alg_name]['pareto_solutions'])
        
        # 获取真正的帕累托前沿作为参考
        reference_front = self._get_true_pareto_front(all_solutions)
        
        igd_values = []
        alg_names = []
        
        for alg_name in algorithms:
            solutions = results['algorithms'][alg_name]['pareto_solutions']
            if solutions:
                igd = self.calculate_inverted_generational_distance(solutions, reference_front)
                igd_values.append(igd)
                alg_names.append(alg_name)
        
        bars = ax4.bar(alg_names, igd_values, 
                      color=[colors[i % len(colors)] for i in range(len(alg_names))], 
                      alpha=0.8,
                      edgecolor='black',
                      linewidth=1)
        
        # 添加数值标签
        for bar, value in zip(bars, igd_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        ax4.set_ylabel('反世代距离 (IGD)', fontsize=12)
        ax4.set_title('反世代距离对比 (越小越好)', fontsize=14, fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. 超体积对比 (右下)
        ax5 = plt.subplot(2, 3, 5)
        
        hv_values = []
        alg_names_hv = []
        
        for alg_name in algorithms:
            solutions = results['algorithms'][alg_name]['pareto_solutions']
            if solutions:
                hv = self.calculate_hypervolume(solutions)
                hv_values.append(hv)
                alg_names_hv.append(alg_name)
        
        bars = ax5.bar(alg_names_hv, hv_values, 
                      color=[colors[i % len(colors)] for i in range(len(alg_names_hv))], 
                      alpha=0.8,
                      edgecolor='black',
                      linewidth=1)
        
        # 添加数值标签
        for bar, value in zip(bars, hv_values):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                    f'{value:.1f}', ha='center', va='bottom', fontsize=10)
        
        ax5.set_ylabel('超体积 (HV)', fontsize=12)
        ax5.set_title('超体积对比 (越大越好)', fontsize=14, fontweight='bold')
        ax5.tick_params(axis='x', rotation=45)
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. 综合性能雷达图 (中下)
        ax6 = plt.subplot(2, 3, 6, projection='polar')
        
        # 性能指标
        metrics = ['运行时间\n(归一化)', '帕累托解数\n(归一化)', 'IGD\n(反向归一化)', 'HV\n(归一化)']
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        for i, alg_name in enumerate(algorithms):
            alg_data = results['algorithms'][alg_name]
            
            # 提取和归一化性能数据
            runtime = alg_data['execution_time']
            pareto_size = len(alg_data['pareto_solutions'])
            
            # 获取对应的IGD和HV值
            try:
                igd_idx = alg_names.index(alg_name)
                igd = igd_values[igd_idx]
            except (ValueError, IndexError):
                igd = 0
            
            try:
                hv_idx = alg_names_hv.index(alg_name)
                hv = hv_values[hv_idx]
            except (ValueError, IndexError):
                hv = 0
            
            # 归一化 (0-1范围)
            max_runtime = max(results['algorithms'][a]['execution_time'] for a in algorithms)
            max_pareto = max(len(results['algorithms'][a]['pareto_solutions']) for a in algorithms)
            max_igd = max(igd_values) if igd_values else 1
            max_hv = max(hv_values) if hv_values else 1
            
            normalized_values = [
                1 - (runtime / max_runtime) if max_runtime > 0 else 0,  # 运行时间越小越好
                pareto_size / max_pareto if max_pareto > 0 else 0,       # 解数量越多越好
                1 - (igd / max_igd) if max_igd > 0 else 0,               # IGD越小越好
                hv / max_hv if max_hv > 0 else 0                         # HV越大越好
            ]
            
            normalized_values += normalized_values[:1]  # 闭合图形
            
            ax6.plot(angles, normalized_values, 
                    color=colors[i % len(colors)], 
                    linewidth=2, 
                    label=alg_name)
            ax6.fill(angles, normalized_values, 
                    color=colors[i % len(colors)], 
                    alpha=0.25)
        
        ax6.set_xticks(angles[:-1])
        ax6.set_xticklabels(metrics, fontsize=10)
        ax6.set_ylim(0, 1)
        ax6.set_title('综合性能雷达图', fontsize=14, fontweight='bold', pad=20)
        ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存详细对比图 - 添加时间戳确保唯一性
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        detailed_filename = f"results/{problem_name}_详细算法对比_{timestamp}.png"
        plt.savefig(detailed_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   详细算法对比图已保存: {detailed_filename}")
        
        return detailed_filename
    
    def _get_true_pareto_front(self, solutions: List[Solution]) -> List[Solution]:
        """获取真正的帕累托前沿"""
        if not solutions:
            return []
        
        pareto_front = []
        
        for sol in solutions:
            is_dominated = False
            
            for other in solutions:
                if (other.makespan <= sol.makespan and 
                    other.total_tardiness <= sol.total_tardiness and
                    (other.makespan < sol.makespan or other.total_tardiness < sol.total_tardiness)):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_front.append(sol)
        
        return pareto_front
    
    def plot_pareto_front(self, solutions: List[Solution], filename: str = None):
        """
        绘制帕累托前沿
        
        Args:
            solutions: 解列表
            filename: 保存文件名
        """
        if not solutions:
            print("没有解可以绘制")
            return
        
        # 提取目标函数值
        makespans = [sol.makespan for sol in solutions]
        tardiness = [sol.total_tardiness for sol in solutions]
        
        # 创建图形
        plt.figure(figsize=(10, 8))
        plt.scatter(makespans, tardiness, c='red', marker='o', s=50, alpha=0.7)
        
        plt.xlabel('最大完工时间 (Makespan)')
        plt.ylabel('总拖期 (Total Tardiness)')
        plt.title(f'帕累托前沿 (共{len(solutions)}个解)')
        plt.grid(True, alpha=0.3)
        
        # 添加注释
        for i, (ms, td) in enumerate(zip(makespans, tardiness)):
            if i % max(1, len(solutions)//10) == 0:  # 只标注部分点
                plt.annotate(f'({ms:.1f},{td:.1f})', 
                           (ms, td), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"帕累托前沿图已保存到: {filename}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_convergence(self, convergence_data: Dict, filename: str = None):
        """
        绘制收敛曲线
        
        Args:
            convergence_data: 收敛数据
            filename: 保存文件名
        """
        makespan_history = convergence_data.get('makespan_history', [])
        tardiness_history = convergence_data.get('tardiness_history', [])
        
        if not makespan_history:
            print("没有收敛数据可以绘制")
            return
        
        iterations = list(range(len(makespan_history)))
        
        # 创建子图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 绘制最大完工时间收敛曲线
        ax1.plot(iterations, makespan_history, 'b-', linewidth=2, label='最大完工时间')
        ax1.set_xlabel('迭代次数')
        ax1.set_ylabel('最大完工时间')
        ax1.set_title('最大完工时间收敛曲线')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 绘制总拖期收敛曲线
        ax2.plot(iterations, tardiness_history, 'r-', linewidth=2, label='总拖期')
        ax2.set_xlabel('迭代次数')
        ax2.set_ylabel('总拖期')
        ax2.set_title('总拖期收敛曲线')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"收敛曲线图已保存到: {filename}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_combined_results(self, results_dict: Dict, filename: str = None):
        """
        绘制多个规模的综合结果比较
        
        Args:
            results_dict: 多个规模的结果字典
            filename: 保存文件名
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        scales = list(results_dict.keys())
        
        # 提取数据
        best_makespans = []
        best_tardiness = []
        pareto_sizes = []
        runtimes = []
        
        for scale in scales:
            result = results_dict[scale]
            if result['best_solutions']:
                best_makespans.append(min(sol.makespan for sol in result['best_solutions']))
                best_tardiness.append(min(sol.total_tardiness for sol in result['best_solutions']))
            else:
                best_makespans.append(float('inf'))
                best_tardiness.append(float('inf'))
            
            pareto_sizes.append(len(result['best_solutions']))
            runtimes.append(result['runtime'])
        
        # 绘制最优完工时间比较
        ax1.bar(scales, best_makespans, color='skyblue', alpha=0.7)
        ax1.set_title('不同规模的最优完工时间')
        ax1.set_ylabel('最大完工时间')
        ax1.tick_params(axis='x', rotation=45)
        
        # 绘制最优拖期比较
        ax2.bar(scales, best_tardiness, color='lightcoral', alpha=0.7)
        ax2.set_title('不同规模的最优总拖期')
        ax2.set_ylabel('总拖期')
        ax2.tick_params(axis='x', rotation=45)
        
        # 绘制帕累托解数量比较
        ax3.bar(scales, pareto_sizes, color='lightgreen', alpha=0.7)
        ax3.set_title('不同规模的帕累托解数量')
        ax3.set_ylabel('解数量')
        ax3.tick_params(axis='x', rotation=45)
        
        # 绘制运行时间比较
        ax4.bar(scales, runtimes, color='gold', alpha=0.7)
        ax4.set_title('不同规模的运行时间')
        ax4.set_ylabel('时间 (秒)')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"综合结果图已保存到: {filename}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_algorithm_comparison(self, results: Dict, problem_name: str):
        """绘制算法对比图（保持向后兼容）"""
        return self.plot_detailed_algorithm_comparison(results, problem_name)
    
    def plot_separate_pareto_comparison(self, results: Dict, problem_name: str):
        """
        单独生成帕累托前沿对比图，使用不同形状标记区分算法
        
        Args:
            results: 实验结果字典
            problem_name: 问题名称
        """
        algorithms = list(results['algorithms'].keys())
        
        if len(algorithms) < 2:
            print("至少需要两个算法进行对比")
            return
        
        # 创建图形
        plt.figure(figsize=(10, 8))
        
        # 定义不同的标记形状和颜色（更亮的配色方案）
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']  # 圆形、方形、三角形、菱形等
        colors = ['#FF1744', '#00E676', '#2196F3', '#FF6F00', '#E91E63', '#9C27B0']
        
        # 为每个算法绘制帕累托前沿
        for i, alg_name in enumerate(algorithms):
            solutions = results['algorithms'][alg_name]['pareto_solutions']
            if solutions:
                makespans = [sol.makespan for sol in solutions]
                tardiness = [sol.total_tardiness for sol in solutions]
                
                plt.scatter(makespans, tardiness, 
                           c=colors[i % len(colors)], 
                           marker=markers[i % len(markers)],
                           label=alg_name, 
                           alpha=0.8, 
                           s=80,  # 增大点的大小
                           edgecolors='black',
                           linewidth=1.0)
        
        # 设置图表属性
        plt.xlabel('完工时间 (Makespan)', fontsize=14, fontweight='bold')
        plt.ylabel('总拖期 (Total Tardiness)', fontsize=14, fontweight='bold')
        plt.title(f'{problem_name} - 帕累托前沿对比', fontsize=16, fontweight='bold')
        
        # 设置图例
        plt.legend(fontsize=12, loc='upper right', frameon=True, 
                  fancybox=True, shadow=True)
        
        # 设置网格
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # 设置坐标轴
        plt.tick_params(axis='both', which='major', labelsize=12)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存单独的帕累托前沿对比图
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        pareto_filename = f"results/{problem_name}_帕累托前沿对比_{timestamp}.png"
        plt.savefig(pareto_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   帕累托前沿对比图已保存: {pareto_filename}")
        
        return pareto_filename
    
    def plot_three_algorithm_pareto_comparison(self, results: Dict, problem_name: str, filename: str):
        """
        生成三算法帕累托前沿对比图
        
        Args:
            results: 实验结果字典
            problem_name: 问题名称
            filename: 保存文件名
        """
        algorithms = list(results.keys())
        
        if len(algorithms) < 2:
            print("至少需要两个算法进行对比")
            return
        
        # 创建图形
        plt.figure(figsize=(12, 9))
        
        # 定义不同的标记形状和颜色
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']  # 圆形、方形、三角形等
        colors = ['#FF1744', '#00E676', '#2196F3', '#FF6F00', '#E91E63', '#9C27B0']
        
        # 为每个算法绘制帕累托前沿
        for i, alg_name in enumerate(algorithms):
            if results[alg_name] and results[alg_name]['pareto_solutions']:
                solutions = results[alg_name]['pareto_solutions']
                makespans = [sol.makespan for sol in solutions]
                tardiness = [sol.total_tardiness for sol in solutions]
                
                plt.scatter(makespans, tardiness, 
                           c=colors[i % len(colors)], 
                           marker=markers[i % len(markers)],
                           label=alg_name, 
                           alpha=0.8, 
                           s=100,  # 点的大小
                           edgecolors='black',
                           linewidth=1.2)
        
        # 设置图表属性
        plt.xlabel('完工时间 (Makespan)', fontsize=14, fontweight='bold')
        plt.ylabel('总拖期 (Total Tardiness)', fontsize=14, fontweight='bold')
        plt.title(f'{problem_name}', fontsize=16, fontweight='bold')
        
        # 设置图例
        plt.legend(fontsize=13, loc='upper right', frameon=True, 
                  fancybox=True, shadow=True, borderpad=1)
        
        # 设置网格
        plt.grid(True, alpha=0.4, linestyle='--')
        
        # 设置坐标轴
        plt.tick_params(axis='both', which='major', labelsize=12)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"三算法帕累托前沿对比图已保存: {filename}")
        return filename
    
    def plot_simple_three_algorithm_comparison(self, results: Dict, problem_name: str, filename: str):
        """
        生成简化的三算法对比图
        
        Args:
            results: 实验结果字典 {alg_name: {pareto_solutions, convergence_data, runtime}}
            problem_name: 问题名称
            filename: 保存文件名
        """
        algorithms = list(results.keys())
        
        if len(algorithms) < 2:
            print("至少需要两个算法进行对比")
            return
        
        # 创建图形
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{problem_name} - 三算法对比分析', fontsize=16, fontweight='bold')
        
        # 颜色和标记样式
        colors = ['#FF1744', '#00E676', '#2196F3', '#FF6F00', '#E91E63', '#9C27B0']
        markers = ['o', 's', '^', 'D', 'v', '<']
        
        # 1. 帕累托前沿对比
        for i, alg_name in enumerate(algorithms):
            if results[alg_name] and results[alg_name]['pareto_solutions']:
                solutions = results[alg_name]['pareto_solutions']
                makespans = [sol.makespan for sol in solutions]
                tardiness = [sol.total_tardiness for sol in solutions]
                
                ax1.scatter(makespans, tardiness, 
                           c=colors[i % len(colors)], 
                           marker=markers[i % len(markers)],
                           label=alg_name, 
                           alpha=0.8, 
                           s=80,
                           edgecolors='black',
                           linewidth=1.0)
        
        ax1.set_xlabel('完工时间 (Makespan)', fontsize=12)
        ax1.set_ylabel('总拖期 (Total Tardiness)', fontsize=12)
        ax1.set_title('帕累托前沿对比', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # 2. 完工时间收敛对比
        for i, alg_name in enumerate(algorithms):
            if results[alg_name] and results[alg_name]['convergence_data']:
                makespan_history = results[alg_name]['convergence_data'].get('best_makespan', [])
                if makespan_history:
                    ax2.plot(makespan_history, 
                            color=colors[i % len(colors)], 
                            label=alg_name, 
                            linewidth=2.5,
                            alpha=0.8)
        
        ax2.set_xlabel('迭代次数', fontsize=12)
        ax2.set_ylabel('最优完工时间', fontsize=12)
        ax2.set_title('完工时间收敛对比', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # 3. 总拖期收敛对比
        for i, alg_name in enumerate(algorithms):
            if results[alg_name] and results[alg_name]['convergence_data']:
                tardiness_history = results[alg_name]['convergence_data'].get('best_tardiness', [])
                if tardiness_history:
                    ax3.plot(tardiness_history, 
                            color=colors[i % len(colors)], 
                            label=alg_name, 
                            linewidth=2.5,
                            alpha=0.8)
        
        ax3.set_xlabel('迭代次数', fontsize=12)
        ax3.set_ylabel('最优总拖期', fontsize=12)
        ax3.set_title('总拖期收敛对比', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3)
        
        # 4. 综合性能对比
        alg_names = []
        pareto_sizes = []
        runtimes = []
        
        for alg_name in algorithms:
            if results[alg_name] and results[alg_name]['pareto_solutions']:
                alg_names.append(alg_name)
                pareto_sizes.append(len(results[alg_name]['pareto_solutions']))
                runtimes.append(results[alg_name]['runtime'])
        
        # 绘制性能条形图
        x = np.arange(len(alg_names))
        width = 0.35
        
        # 归一化数据用于显示
        max_pareto = max(pareto_sizes) if pareto_sizes else 1
        max_runtime = max(runtimes) if runtimes else 1
        
        normalized_pareto = [p / max_pareto for p in pareto_sizes]
        normalized_runtime = [1 - (r / max_runtime) for r in runtimes]  # 运行时间越小越好
        
        bars1 = ax4.bar(x - width/2, normalized_pareto, width, 
                       label='帕累托解数量(归一化)', 
                       color=colors[0], alpha=0.8)
        bars2 = ax4.bar(x + width/2, normalized_runtime, width, 
                       label='运行效率(归一化)', 
                       color=colors[1], alpha=0.8)
        
        ax4.set_xlabel('算法', fontsize=12)
        ax4.set_ylabel('归一化性能值', fontsize=12)
        ax4.set_title('综合性能对比', fontsize=14, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(alg_names)
        ax4.legend(fontsize=11)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, value in zip(bars1, pareto_sizes):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{value}', ha='center', va='bottom', fontsize=10)
        
        for bar, value in zip(bars2, runtimes):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{value:.1f}s', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"三算法对比图已保存: {filename}")
        return filename 