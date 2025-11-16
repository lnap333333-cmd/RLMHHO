#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版帕累托图可视化模块
支持高清晰度PNG和矢量图格式（PDF、SVG）
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import matplotlib.font_manager as fm

# 设置高质量绘图参数
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.fancybox'] = True
plt.rcParams['legend.shadow'] = True
plt.rcParams['grid.linewidth'] = 0.8
plt.rcParams['grid.alpha'] = 0.3

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class EnhancedParetoVisualizer:
    """增强版帕累托图可视化器"""
    
    def __init__(self):
        """初始化可视化器"""
        self.algorithm_styles = {
            'RL-Chaotic-HHO': {
                'color': '#FF1744',  # 鲜艳红色
                'marker': 'o',
                'size': 100,
                'edgecolor': '#D50000',
                'linewidth': 1.5,
                'alpha': 0.85
            },
            'I-NSGA-II': {
                'color': '#00E676',  # 鲜艳绿色
                'marker': 's',
                'size': 90,
                'edgecolor': '#00C853',
                'linewidth': 1.5,
                'alpha': 0.85
            },
            'MOPSO': {
                'color': '#2196F3',  # 蓝色
                'marker': '^',
                'size': 100,
                'edgecolor': '#1976D2',
                'linewidth': 1.5,
                'alpha': 0.85
            },
            'MODE': {
                'color': '#FF9800',  # 橙色
                'marker': 'v',
                'size': 100,
                'edgecolor': '#F57C00',
                'linewidth': 1.5,
                'alpha': 0.85
            },
            'DQN': {
                'color': '#9C27B0',  # 紫色
                'marker': '<',
                'size': 100,
                'edgecolor': '#7B1FA2',
                'linewidth': 1.5,
                'alpha': 0.85
            },
            'QL-ABC': {
                'color': '#8D6E63',  # 棕色
                'marker': '>',
                'size': 100,
                'edgecolor': '#6D4C41',
                'linewidth': 1.5,
                'alpha': 0.85
            }
        }
        
        # 创建结果目录
        os.makedirs('results', exist_ok=True)
        os.makedirs('results/vector', exist_ok=True)
        os.makedirs('results/high_res', exist_ok=True)
    
    def plot_enhanced_pareto_comparison(self, all_results: Dict, scale: str, 
                                      save_formats: List[str] = ['png', 'pdf', 'svg'],
                                      figsize: Tuple[int, int] = (14, 10)) -> List[str]:
        """
        绘制增强版帕累托前沿对比图
        
        Args:
            all_results: 所有算法结果
            scale: 数据集规模名称
            save_formats: 保存格式列表 ['png', 'pdf', 'svg']
            figsize: 图形尺寸
            
        Returns:
            保存的文件路径列表
        """
        # 创建高质量图形
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=300)
        
        print(f"\n🎨 绘制{scale}的增强版帕累托前沿对比图...")
        
        plot_count = 0
        legend_elements = []
        
        for algorithm_name, result in all_results.items():
            print(f"  处理算法: {algorithm_name}")
            
            if result and 'pareto_solutions' in result and result['pareto_solutions']:
                pareto_solutions = result['pareto_solutions']
                makespan_values = [sol.makespan for sol in pareto_solutions]
                tardiness_values = [sol.total_tardiness for sol in pareto_solutions]
                
                print(f"    解集数量: {len(pareto_solutions)}")
                print(f"    完工时间范围: {min(makespan_values):.2f} - {max(makespan_values):.2f}")
                print(f"    总拖期范围: {min(tardiness_values):.2f} - {max(tardiness_values):.2f}")
                
                # 获取算法样式
                style = self.algorithm_styles.get(algorithm_name, {
                    'color': '#666666',
                    'marker': 'o',
                    'size': 80,
                    'edgecolor': '#444444',
                    'linewidth': 1.0,
                    'alpha': 0.8
                })
                
                # 转换算法显示名称
                display_name = algorithm_name
                if algorithm_name == 'RL-Chaotic-HHO':
                    display_name = 'RLMHHO'
                
                # 绘制散点图
                scatter = ax.scatter(makespan_values, tardiness_values,
                                   c=style['color'],
                                   marker=style['marker'],
                                   s=style['size'],
                                   alpha=style['alpha'],
                                   edgecolors=style['edgecolor'],
                                   linewidth=style['linewidth'],
                                   label=display_name)
                
                plot_count += 1
            else:
                print(f"    ❌ 没有有效的pareto解集")
        
        if plot_count == 0:
            print("    ⚠️  警告：没有任何算法产生有效的pareto解集")
            plt.close()
            return []
        else:
            print(f"    ✅ 成功绘制了{plot_count}个算法的结果")
        
        # 设置坐标轴
        ax.set_xlabel('最大完工时间 (Makespan)', fontsize=16, fontweight='bold', labelpad=10)
        ax.set_ylabel('最大延迟时间 (Total Tardiness)', fontsize=16, fontweight='bold', labelpad=10)
        ax.set_title(f'{scale} - 帕累托前沿对比', fontsize=18, fontweight='bold', pad=20)
        
        # 设置图例
        legend = ax.legend(loc='upper right', frameon=True, fancybox=True, 
                          shadow=True, fontsize=13, bbox_to_anchor=(1.02, 1.0))
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.95)
        legend.get_frame().set_linewidth(1.5)
        
        # 设置网格
        ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
        ax.set_axisbelow(True)
        
        # 设置坐标轴样式
        ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=6)
        ax.tick_params(axis='both', which='minor', width=1.0, length=3)
        
        # 设置边框
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color('#333333')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存不同格式的图片
        saved_files = []
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for fmt in save_formats:
            if fmt.lower() == 'png':
                # 高分辨率PNG
                filename = f'results/high_res/pareto_comparison_{scale}_{timestamp}.png'
                plt.savefig(filename, dpi=400, bbox_inches='tight', 
                           facecolor='white', edgecolor='none', 
                           pad_inches=0.1)
                saved_files.append(filename)
                print(f"    📊 高分辨率PNG已保存: {filename}")
                
            elif fmt.lower() == 'pdf':
                # 矢量PDF
                filename = f'results/vector/pareto_comparison_{scale}_{timestamp}.pdf'
                plt.savefig(filename, format='pdf', bbox_inches='tight',
                           facecolor='white', edgecolor='none',
                           pad_inches=0.1)
                saved_files.append(filename)
                print(f"    📊 矢量PDF已保存: {filename}")
                
            elif fmt.lower() == 'svg':
                # 矢量SVG
                filename = f'results/vector/pareto_comparison_{scale}_{timestamp}.svg'
                plt.savefig(filename, format='svg', bbox_inches='tight',
                           facecolor='white', edgecolor='none',
                           pad_inches=0.1)
                saved_files.append(filename)
                print(f"    📊 矢量SVG已保存: {filename}")
        
        plt.close()
        return saved_files
    
    def plot_single_algorithm_pareto(self, solutions: List, algorithm_name: str,
                                   scale: str, save_formats: List[str] = ['png', 'pdf'],
                                   figsize: Tuple[int, int] = (10, 8)) -> List[str]:
        """
        绘制单个算法的帕累托前沿
        
        Args:
            solutions: 解列表
            algorithm_name: 算法名称
            scale: 数据集规模
            save_formats: 保存格式
            figsize: 图形尺寸
            
        Returns:
            保存的文件路径列表
        """
        if not solutions:
            print(f"❌ {algorithm_name}没有解可以绘制")
            return []
        
        # 创建图形
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=300)
        
        # 提取目标函数值
        makespans = [sol.makespan for sol in solutions]
        tardiness = [sol.total_tardiness for sol in solutions]
        
        # 获取算法样式
        style = self.algorithm_styles.get(algorithm_name, {
            'color': '#FF1744',
            'marker': 'o',
            'size': 120,
            'edgecolor': '#D50000',
            'linewidth': 1.5,
            'alpha': 0.8
        })
        
        # 绘制散点图
        ax.scatter(makespans, tardiness,
                  c=style['color'],
                  marker=style['marker'],
                  s=style['size'],
                  alpha=style['alpha'],
                  edgecolors=style['edgecolor'],
                  linewidth=style['linewidth'])
        
        # 转换算法显示名称
        display_name = algorithm_name
        if algorithm_name == 'RL-Chaotic-HHO':
            display_name = 'RLMHHO'
        
        # 设置坐标轴
        ax.set_xlabel('最大完工时间 (Makespan)', fontsize=16, fontweight='bold')
        ax.set_ylabel('最大延迟时间 (Total Tardiness)', fontsize=16, fontweight='bold')
        ax.set_title(f'{display_name} - {scale} 帕累托前沿', 
                    fontsize=18, fontweight='bold')
        
        # 设置网格
        ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
        ax.set_axisbelow(True)
        
        # 设置坐标轴样式
        ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=6)
        
        # 设置边框
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color('#333333')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片
        saved_files = []
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for fmt in save_formats:
            if fmt.lower() == 'png':
                filename = f'results/high_res/{algorithm_name}_{scale}_pareto_{timestamp}.png'
                plt.savefig(filename, dpi=400, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
                saved_files.append(filename)
                
            elif fmt.lower() == 'pdf':
                filename = f'results/vector/{algorithm_name}_{scale}_pareto_{timestamp}.pdf'
                plt.savefig(filename, format='pdf', bbox_inches='tight',
                           facecolor='white', edgecolor='none')
                saved_files.append(filename)
        
        plt.close()
        print(f"✅ {algorithm_name}帕累托前沿图已保存")
        return saved_files
    
    def create_publication_quality_plot(self, all_results: Dict, scale: str,
                                      figsize: Tuple[int, int] = (16, 12)) -> List[str]:
        """
        创建发表质量的帕累托前沿图
        
        Args:
            all_results: 所有算法结果
            scale: 数据集规模
            figsize: 图形尺寸
            
        Returns:
            保存的文件路径列表
        """
        # 创建高质量图形
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=400)
        
        print(f"\n🎨 创建{scale}的发表质量帕累托前沿图...")
        
        plot_count = 0
        
        for algorithm_name, result in all_results.items():
            if result and 'pareto_solutions' in result and result['pareto_solutions']:
                pareto_solutions = result['pareto_solutions']
                makespan_values = [sol.makespan for sol in pareto_solutions]
                tardiness_values = [sol.total_tardiness for sol in pareto_solutions]
                
                # 获取算法样式
                style = self.algorithm_styles.get(algorithm_name, {
                    'color': '#666666',
                    'marker': 'o',
                    'size': 150,
                    'edgecolor': '#444444',
                    'linewidth': 2.0,
                    'alpha': 0.9
                })
                
                # 转换算法显示名称
                display_name = algorithm_name
                if algorithm_name == 'RL-Chaotic-HHO':
                    display_name = 'RLMHHO'
                
                # 绘制散点图
                ax.scatter(makespan_values, tardiness_values,
                          c=style['color'],
                          marker=style['marker'],
                          s=style['size'],
                          alpha=style['alpha'],
                          edgecolors=style['edgecolor'],
                          linewidth=style['linewidth'],
                          label=display_name)
                
                plot_count += 1
        
        if plot_count == 0:
            print("    ⚠️  警告：没有任何算法产生有效的pareto解集")
            plt.close()
            return []
        
        # 设置坐标轴
        ax.set_xlabel('最大完工时间 (Makespan)', fontsize=20, fontweight='bold', labelpad=15)
        ax.set_ylabel('最大延迟时间 (Total Tardiness)', fontsize=20, fontweight='bold', labelpad=15)
        ax.set_title(f'{scale} - 帕累托前沿对比', fontsize=24, fontweight='bold', pad=25)
        
        # 设置图例
        legend = ax.legend(loc='upper right', frameon=True, fancybox=True, 
                          shadow=True, fontsize=16, bbox_to_anchor=(1.02, 1.0))
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.95)
        legend.get_frame().set_linewidth(2.0)
        
        # 设置网格
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=1.0)
        ax.set_axisbelow(True)
        
        # 设置坐标轴样式
        ax.tick_params(axis='both', which='major', labelsize=16, width=2.0, length=8)
        ax.tick_params(axis='both', which='minor', width=1.5, length=4)
        
        # 设置边框
        for spine in ax.spines.values():
            spine.set_linewidth(2.0)
            spine.set_color('#000000')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存高质量图片
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        saved_files = []
        
        # 超高分辨率PNG
        filename_png = f'results/high_res/publication_pareto_{scale}_{timestamp}.png'
        plt.savefig(filename_png, dpi=600, bbox_inches='tight',
                   facecolor='white', edgecolor='none', pad_inches=0.15)
        saved_files.append(filename_png)
        
        # 矢量PDF
        filename_pdf = f'results/vector/publication_pareto_{scale}_{timestamp}.pdf'
        plt.savefig(filename_pdf, format='pdf', bbox_inches='tight',
                   facecolor='white', edgecolor='none', pad_inches=0.15)
        saved_files.append(filename_pdf)
        
        # 矢量SVG
        filename_svg = f'results/vector/publication_pareto_{scale}_{timestamp}.svg'
        plt.savefig(filename_svg, format='svg', bbox_inches='tight',
                   facecolor='white', edgecolor='none', pad_inches=0.15)
        saved_files.append(filename_svg)
        
        plt.close()
        
        print(f"    ✅ 发表质量图片已保存:")
        for filename in saved_files:
            print(f"       {filename}")
        
        return saved_files

def test_enhanced_visualization():
    """测试增强版可视化功能"""
    from problem.mo_dhfsp import MO_DHFSP_Problem
    from algorithm.rl_chaotic_hho import RL_ChaoticHHO_Optimizer
    from algorithm.improved_nsga2 import ImprovedNSGA2_Optimizer
    from algorithm.mopso import MOPSO_Optimizer
    from utils.data_generator import generate_heterogeneous_problem_data
    
    print("🧪 测试增强版帕累托图可视化功能...")
    
    # 创建可视化器
    visualizer = EnhancedParetoVisualizer()
    
    # 生成测试数据
    config = {
        'scale': '50J4S3F',
        'n_jobs': 50,
        'n_factories': 3,
        'n_stages': 4,
        'machines_per_stage': [3, 4, 3, 4],
        'urgency_ddt': [0.5, 1.0, 1.5],
        'processing_time_range': (1, 20),
        'heterogeneous_machines': {
            0: [3, 4, 3, 4],
            1: [4, 3, 4, 3],
            2: [3, 3, 4, 4]
        }
    }
    
    problem_data = generate_heterogeneous_problem_data(config)
    
    # 运行算法
    algorithms = {
        'RL-Chaotic-HHO': (RL_ChaoticHHO_Optimizer, {
            'population_size': 50,
            'max_iterations': 30,
            'pareto_size_limit': 100
        }),
        'I-NSGA-II': (ImprovedNSGA2_Optimizer, {
            'population_size': 50,
            'max_generations': 30
        }),
        'MOPSO': (MOPSO_Optimizer, {
            'swarm_size': 50,
            'max_iterations': 30
        })
    }
    
    results = {}
    
    for alg_name, (alg_class, params) in algorithms.items():
        print(f"运行算法: {alg_name}")
        try:
            optimizer = alg_class(problem_data, **params)
            result = optimizer.optimize()
            results[alg_name] = result
        except Exception as e:
            print(f"算法{alg_name}运行失败: {e}")
            results[alg_name] = None
    
    # 测试不同格式的绘图
    print("\n📊 测试不同格式的帕累托图...")
    
    # 1. 标准增强版
    files1 = visualizer.plot_enhanced_pareto_comparison(
        results, config['scale'], save_formats=['png', 'pdf', 'svg']
    )
    
    # 2. 发表质量版
    files2 = visualizer.create_publication_quality_plot(results, config['scale'])
    
    # 3. 单个算法图
    for alg_name, result in results.items():
        if result and result['pareto_solutions']:
            files3 = visualizer.plot_single_algorithm_pareto(
                result['pareto_solutions'], alg_name, config['scale']
            )
    
    print(f"\n✅ 测试完成！共生成{len(files1) + len(files2)}个文件")
    print("文件保存在 results/high_res/ 和 results/vector/ 目录中")

if __name__ == "__main__":
    test_enhanced_visualization() 