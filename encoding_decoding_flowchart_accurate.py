#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于实际实现的RL-Chaotic-HHO算法编码与解码流程图
准确反映算法的真实编码结构和解码过程
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

# 设置中文字体和高质量显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 14
plt.rcParams['figure.dpi'] = 300

def create_accurate_flowchart():
    """创建基于实际实现的编码解码流程图"""
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(18, 12))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 12)
    ax.set_aspect('equal')
    
    # 隐藏坐标轴
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # 绘制各部分
    draw_actual_encoding_section(ax)
    draw_actual_decoding_steps(ax)
    draw_actual_decoding_results(ax)
    draw_actual_connections(ax)
    
    # 添加标题
    ax.text(9, 11.5, 'RL-Chaotic-HHO算法实际编码解码流程图', 
            fontsize=22, fontweight='bold', ha='center')
    
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('编码解码流程图_实际实现版.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig('编码解码流程图_实际实现版.pdf', bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    plt.show()

def draw_actual_encoding_section(ax):
    """绘制实际的编码结构"""
    
    # 编码主框架
    encoding_box = Rectangle((1, 6), 6, 4.5, linewidth=3, 
                            edgecolor='black', facecolor='#F8F8F8')
    ax.add_patch(encoding_box)
    
    # 编码标题
    ax.text(4, 10, '实际编码结构', fontsize=20, fontweight='bold', ha='center')
    
    # 1. 工厂分配向量 (factory_assignment)
    ax.text(1.5, 9.2, '工厂分配向量', fontsize=16, fontweight='bold', color='blue')
    ax.text(1.5, 8.8, 'factory_assignment: List[int]', fontsize=12, style='italic')
    
    factory_values = [0, 2, 1, 0, 1, 2]
    factory_text = '[' + ', '.join([f'F{v}' for v in factory_values]) + ']'
    
    factory_box = FancyBboxPatch((2, 8.2), 4.5, 0.6, boxstyle="round,pad=0.1",
                                facecolor='#E5F2FF', edgecolor='blue', linewidth=2)
    ax.add_patch(factory_box)
    ax.text(4.25, 8.5, factory_text, ha='center', va='center', fontsize=12, fontweight='bold')
    
    # 2. 作业序列 (job_sequences)
    ax.text(1.5, 7.5, '作业序列', fontsize=16, fontweight='bold', color='green')
    ax.text(1.5, 7.1, 'job_sequences: List[List[int]]', fontsize=12, style='italic')
    
    # 各工厂的作业序列
    sequences = [
        "F0: [J0, J3]",
        "F1: [J2, J4]", 
        "F2: [J1, J5]"
    ]
    
    for i, seq in enumerate(sequences):
        seq_box = FancyBboxPatch((2, 6.5 - i*0.4), 4.5, 0.3, boxstyle="round,pad=0.05",
                                facecolor='#E5F9E5', edgecolor='green', linewidth=1)
        ax.add_patch(seq_box)
        ax.text(4.25, 6.65 - i*0.4, seq, ha='center', va='center', fontsize=11, fontweight='bold')

def draw_actual_decoding_steps(ax):
    """绘制实际的解码处理步骤"""
    
    # 解码处理标题
    ax.text(10, 10, '解码处理步骤', fontsize=20, fontweight='bold', ha='center')
    
    # 定义实际的解码步骤
    steps = [
        {
            'title': '步骤1：初始化调度环境',
            'desc': '为每个工厂创建机器完工时间矩阵',
            'color': '#FFE5E5',
            'edge_color': 'red'
        },
        {
            'title': '步骤2：按序列处理工件',
            'desc': '根据job_sequences顺序处理各工厂工件',
            'color': '#E5F2FF',
            'edge_color': 'blue'
        },
        {
            'title': '步骤3：贪心机器选择',
            'desc': '选择最早可用的机器进行加工',
            'color': '#E5F9E5',
            'edge_color': 'green'
        },
        {
            'title': '步骤4：计算开始时间',
            'desc': '考虑工序约束和机器可用性',
            'color': '#FFF0E5',
            'edge_color': 'orange'
        },
        {
            'title': '步骤5：更新完工时间',
            'desc': '更新工件和机器的完工时间',
            'color': '#F0E5FF',
            'edge_color': 'purple'
        }
    ]
    
    y_positions = [9.2, 8.2, 7.2, 6.2, 5.2]
    
    for i, (step, y_pos) in enumerate(zip(steps, y_positions)):
        # 步骤标题框
        title_box = FancyBboxPatch((8.5, y_pos - 0.15), 3, 0.5, boxstyle="round,pad=0.05",
                                  facecolor=step['color'], edgecolor=step['edge_color'], linewidth=2)
        ax.add_patch(title_box)
        ax.text(10, y_pos + 0.1, step['title'], ha='center', va='center', fontsize=11, fontweight='bold')
        
        # 详细描述
        ax.text(10, y_pos - 0.4, step['desc'], ha='center', va='center', fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        
        # 步骤间连接箭头
        if i < len(steps) - 1:
            arrow = patches.FancyArrowPatch((10, y_pos - 0.6), (10, y_positions[i+1] + 0.2),
                                          arrowstyle='->', mutation_scale=15,
                                          color='gray', linewidth=2)
            ax.add_patch(arrow)

def draw_actual_decoding_results(ax):
    """绘制实际的解码结果"""
    
    # 解码结果标题
    ax.text(15, 10, '解码结果', fontsize=20, fontweight='bold', ha='center')
    
    # 调度结果示例
    results = [
        {
            'title': '工厂F0调度',
            'jobs': [
                'J0: 0.0→5.0→8.0→15.0',
                'J3: 5.0→9.0→15.0→20.0'
            ],
            'color': '#FFE5E5'
        },
        {
            'title': '工厂F1调度', 
            'jobs': [
                'J2: 0.0→6.0→10.0→17.0',
                'J4: 6.0→11.0→17.0→24.0'
            ],
            'color': '#E5F2FF'
        },
        {
            'title': '工厂F2调度',
            'jobs': [
                'J1: 0.0→4.0→9.0→14.0',
                'J5: 4.0→12.0→14.0→22.0'
            ],
            'color': '#E5F9E5'
        }
    ]
    
    y_positions = [8.5, 6.5, 4.5]
    
    for (result, y_pos) in zip(results, y_positions):
        # 工厂框
        factory_height = len(result['jobs']) * 0.4 + 0.8
        factory_box = FancyBboxPatch((13, y_pos - factory_height/2), 4.5, factory_height,
                                   boxstyle="round,pad=0.1", facecolor=result['color'], 
                                   edgecolor='black', linewidth=2)
        ax.add_patch(factory_box)
        
        # 工厂名称
        ax.text(15.25, y_pos + factory_height/2 - 0.3, result['title'], ha='center', va='center', 
                fontsize=14, fontweight='bold')
        
        # 工件调度详情
        for i, job in enumerate(result['jobs']):
            job_y = y_pos + factory_height/2 - 0.8 - i * 0.4
            ax.text(15.25, job_y, job, ha='center', va='center', fontsize=10)
    
    # 目标函数值
    obj_box = FancyBboxPatch((13, 2), 4.5, 1.2, boxstyle="round,pad=0.1",
                           facecolor='#FFF0E5', edgecolor='orange', linewidth=2)
    ax.add_patch(obj_box)
    ax.text(15.25, 2.8, '目标函数值', ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(15.25, 2.4, 'Makespan = 24.0', ha='center', va='center', fontsize=12)
    ax.text(15.25, 2.0, 'Total Tardiness = 8.0', ha='center', va='center', fontsize=12)

def draw_actual_connections(ax):
    """绘制实际的连接线"""
    
    # 从编码到解码步骤的主要连接
    main_arrow = patches.FancyArrowPatch((7, 8), (8.5, 8),
                                       arrowstyle='->', mutation_scale=25,
                                       color='black', linewidth=3)
    ax.add_patch(main_arrow)
    
    # 从解码步骤到结果的连接
    result_arrow = patches.FancyArrowPatch((11.5, 7), (13, 7),
                                         arrowstyle='->', mutation_scale=25,
                                         color='black', linewidth=3)
    ax.add_patch(result_arrow)
    
    # 编码各部分到对应步骤的连接
    connections = [
        ((6.5, 8.5), (8.5, 8.2), 'blue'),     # 工厂分配到步骤2
        ((6.5, 7), (8.5, 7.2), 'green'),      # 作业序列到步骤3
    ]
    
    for (start, end, color) in connections:
        arrow = patches.FancyArrowPatch(start, end, arrowstyle='->', mutation_scale=15,
                                      color=color, linewidth=2, alpha=0.7)
        ax.add_patch(arrow)
    
    # 添加说明文字
    ax.text(9, 4, '关键特点：', fontsize=14, fontweight='bold')
    ax.text(9, 3.5, '• 机器选择：贪心策略（选择最早可用机器）', fontsize=12)
    ax.text(9, 3.1, '• 加工时间：预定义processing_times矩阵', fontsize=12)
    ax.text(9, 2.7, '• 调度约束：工序顺序 + 机器容量', fontsize=12)
    ax.text(9, 2.3, '• 目标优化：最小化Makespan和总拖期', fontsize=12)

if __name__ == "__main__":
    create_accurate_flowchart()
    print("基于实际实现的编码解码流程图已生成完成！")
    print("输出文件：")
    print("- 编码解码流程图_实际实现版.png")
    print("- 编码解码流程图_实际实现版.pdf")
    print("\n主要修正：")
    print("1. 编码结构：只包含factory_assignment和job_sequences")
    print("2. 解码过程：反映_decode_solution方法的实际逻辑")
    print("3. 机器选择：贪心策略而非显式编码")
    print("4. 加工时间：使用预定义矩阵而非权重调整") 