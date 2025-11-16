#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高分辨率RL-Chaotic-HHO算法编码与解码流程图
优化文字清晰度和可读性
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

# 设置中文字体和高质量显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 14  # 增大基础字体大小
plt.rcParams['figure.dpi'] = 300  # 高DPI

def create_high_res_flowchart():
    """创建高分辨率编码解码流程图"""
    
    # 创建更大的图形
    fig, ax = plt.subplots(1, 1, figsize=(20, 14))  # 增大图形尺寸
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 14)
    ax.set_aspect('equal')
    
    # 隐藏坐标轴
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # 绘制各部分
    draw_encoding_section_hd(ax)
    draw_decoding_steps_hd(ax)
    draw_decoding_results_hd(ax)
    draw_connections_hd(ax)
    
    plt.tight_layout()
    
    # 保存高质量图片
    plt.savefig('编码解码流程图_高清版.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig('编码解码流程图_高清版.pdf', bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    plt.show()

def draw_encoding_section_hd(ax):
    """绘制编码部分（高清版）"""
    
    # 编码主框架
    encoding_box = Rectangle((1, 3), 7, 8, linewidth=3, 
                            edgecolor='black', facecolor='#F8F8F8')
    ax.add_patch(encoding_box)
    
    # 编码标题
    ax.text(4.5, 10.5, '编码结构', fontsize=24, fontweight='bold', ha='center')
    
    # 工件优先级 X1
    ax.text(1.5, 9.5, '工件优先级 X1', fontsize=16, fontweight='bold')
    priority_values = [0.2, 0.8, 0.8, 1.1, 1.4]
    priority_text = '[' + ', '.join([str(v) for v in priority_values]) + ']'
    
    # 绘制优先级框
    priority_box = FancyBboxPatch((2, 8.8), 5.5, 0.8, boxstyle="round,pad=0.1",
                                 facecolor='#FFE5E5', edgecolor='red', linewidth=2)
    ax.add_patch(priority_box)
    ax.text(4.75, 9.2, priority_text, ha='center', va='center', fontsize=14, fontweight='bold')
    
    # 工厂分配 X2
    ax.text(1.5, 8, '工厂分配 X2', fontsize=16, fontweight='bold')
    factory_values = [1, 3, 2, 1, 2]
    factory_text = '[' + ', '.join([str(v) for v in factory_values]) + ']'
    
    factory_box = FancyBboxPatch((2, 7.3), 5.5, 0.8, boxstyle="round,pad=0.1",
                                facecolor='#E5F2FF', edgecolor='blue', linewidth=2)
    ax.add_patch(factory_box)
    ax.text(4.75, 7.7, factory_text, ha='center', va='center', fontsize=14, fontweight='bold')
    
    # 机器分配 AM
    ax.text(1.5, 6.5, '机器分配 AM', fontsize=16, fontweight='bold')
    machine_values = [1, 1.2, 1.2, 1.2, 1.5]
    machine_text = '[' + ', '.join([str(v) for v in machine_values]) + ']'
    
    machine_box = FancyBboxPatch((2, 5.8), 5.5, 0.8, boxstyle="round,pad=0.1",
                                facecolor='#E5F9E5', edgecolor='green', linewidth=2)
    ax.add_patch(machine_box)
    ax.text(4.75, 6.2, machine_text, ha='center', va='center', fontsize=14, fontweight='bold')
    
    # 加工时间量 AS
    ax.text(1.5, 5, '加工时间量 AS', fontsize=16, fontweight='bold')
    time_values = [0.79, 0.23, 0.75, 0.54, 0.89]
    time_text = '[' + ', '.join([f'{v:.2f}' for v in time_values]) + ']'
    
    time_box = FancyBboxPatch((2, 4.3), 5.5, 0.8, boxstyle="round,pad=0.1",
                             facecolor='#FFF0E5', edgecolor='orange', linewidth=2)
    ax.add_patch(time_box)
    ax.text(4.75, 4.7, time_text, ha='center', va='center', fontsize=14, fontweight='bold')

def draw_decoding_steps_hd(ax):
    """绘制解码处理步骤（高清版）"""
    
    # 解码处理标题
    ax.text(10, 10.5, '解码处理步骤', fontsize=20, fontweight='bold', ha='center')
    
    # 定义解码步骤
    steps = [
        ('步骤1：工件优先级排序', 'J1(0.2) < J3(0.8) < J2(0.8) < J4(1.1) < J5(1.4)', '#FFE5E5', 'red'),
        ('步骤2：工厂分配解析', 'F1: {J1,J4}, F2: {J2,J5}, F3: {J3}', '#E5F2FF', 'blue'),
        ('步骤3：机器选择解码', '根据AM值选择具体机器编号', '#E5F9E5', 'green'),
        ('步骤4：调度序列生成', '考虑优先级和工厂约束', '#FFF0E5', 'orange'),
        ('步骤5：时间计算调度', '结合AS计算最终加工时间', '#F0E5FF', 'purple')
    ]
    
    y_positions = [9.2, 8.2, 7.2, 6.2, 5.2]
    
    for i, ((title, desc, color, edge_color), y_pos) in enumerate(zip(steps, y_positions)):
        # 步骤标题框
        title_box = FancyBboxPatch((8.5, y_pos - 0.1), 3, 0.5, boxstyle="round,pad=0.05",
                                  facecolor=color, edgecolor=edge_color, linewidth=2)
        ax.add_patch(title_box)
        ax.text(10, y_pos + 0.15, title, ha='center', va='center', fontsize=12, fontweight='bold')
        
        # 详细描述
        ax.text(10, y_pos - 0.4, desc, ha='center', va='center', fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # 步骤间连接箭头
        if i < len(steps) - 1:
            arrow = patches.FancyArrowPatch((10, y_pos - 0.6), (10, y_positions[i+1] + 0.3),
                                          arrowstyle='->', mutation_scale=15,
                                          color='gray', linewidth=2)
            ax.add_patch(arrow)

def draw_decoding_results_hd(ax):
    """绘制解码结果（高清版）"""
    
    # 解码结果标题
    ax.text(16, 10.5, '解码结果', fontsize=20, fontweight='bold', ha='center')
    
    # 工厂调度结果
    factories = [
        ('工厂1', ['J1: M1,1(0.79)→M1,2(0.23)', 'J4: M1,2(0.54)→M1,1(0.89)', 'J5: M1,2(0.54)→M1,2(0.67)'], '#FFE5E5'),
        ('工厂2', ['J3: M2,1(0.30)→M2,2(0.16)', 'J5: M2,2(0.53)→M2,1(0.79)', 'J2: M2,1(0.75)→M2,1(0.83)'], '#E5F2FF'),
        ('工厂3', ['J2: M3,1(0.75)→M3,2(0.49)', 'J6: M3,2(0.67)→M3,2(0.16)'], '#E5F9E5')
    ]
    
    y_positions = [8.5, 6.5, 4.5]
    
    for (factory_name, jobs, color), y_pos in zip(factories, y_positions):
        # 工厂框
        factory_height = len(jobs) * 0.4 + 0.8
        factory_box = FancyBboxPatch((13.5, y_pos - factory_height/2), 5, factory_height,
                                   boxstyle="round,pad=0.1", facecolor=color, 
                                   edgecolor='black', linewidth=2)
        ax.add_patch(factory_box)
        
        # 工厂名称
        ax.text(16, y_pos + factory_height/2 - 0.3, factory_name, ha='center', va='center', 
                fontsize=16, fontweight='bold')
        
        # 工件调度详情
        for i, job in enumerate(jobs):
            job_y = y_pos + factory_height/2 - 0.8 - i * 0.4
            ax.text(16, job_y, job, ha='center', va='center', fontsize=11, fontweight='bold')

def draw_connections_hd(ax):
    """绘制连接线（高清版）"""
    
    # 从编码到解码步骤的主要连接
    main_arrow = patches.FancyArrowPatch((8, 7), (8.5, 7),
                                       arrowstyle='->', mutation_scale=25,
                                       color='black', linewidth=3)
    ax.add_patch(main_arrow)
    
    # 从解码步骤到结果的连接
    result_arrow = patches.FancyArrowPatch((11.5, 6.5), (13.5, 6.5),
                                         arrowstyle='->', mutation_scale=25,
                                         color='black', linewidth=3)
    ax.add_patch(result_arrow)
    
    # 编码各部分到对应步骤的连接
    connections = [
        ((7.5, 9.2), (8.5, 9.2), 'red'),      # 优先级到步骤1
        ((7.5, 7.7), (8.5, 8.2), 'blue'),     # 工厂分配到步骤2
        ((7.5, 6.2), (8.5, 7.2), 'green'),    # 机器分配到步骤3
        ((7.5, 4.7), (8.5, 5.2), 'orange'),   # 时间到步骤5
    ]
    
    for (start, end, color) in connections:
        arrow = patches.FancyArrowPatch(start, end, arrowstyle='->', mutation_scale=15,
                                      color=color, linewidth=2, alpha=0.7)
        ax.add_patch(arrow)

if __name__ == "__main__":
    create_high_res_flowchart()
    print("高分辨率编码解码流程图已生成完成！")
    print("输出文件：")
    print("- 编码解码流程图_高清版.png")
    print("- 编码解码流程图_高清版.pdf") 