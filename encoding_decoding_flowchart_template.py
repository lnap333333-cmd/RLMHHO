#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于模板的RL-Chaotic-HHO算法编码与解码流程图
左侧：编码结构，右侧：解码结果
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_template_flowchart():
    """创建基于模板的编码解码流程图"""
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    
    # 隐藏坐标轴
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # 绘制编码部分（左侧）
    draw_encoding_section(ax)
    
    # 绘制解码部分（右侧）
    draw_decoding_section(ax)
    
    # 绘制连接箭头和中间处理步骤
    draw_arrows_and_steps(ax)
    
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('编码解码流程图_模板版.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig('编码解码流程图_模板版.pdf', bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    plt.show()

def draw_encoding_section(ax):
    """绘制编码部分（左侧）"""
    
    # 编码主框架
    encoding_box = Rectangle((0.5, 2), 5, 6, linewidth=2, 
                            edgecolor='black', facecolor='white')
    ax.add_patch(encoding_box)
    
    # 编码标题
    ax.text(3, 7.5, '编码', fontsize=16, fontweight='bold', ha='center')
    
    # 工件优先级 X₁
    ax.text(1, 6.8, '工件优先级', fontsize=12, fontweight='bold')
    ax.text(1, 6.5, 'X₁', fontsize=12, fontweight='bold')
    
    # 绘制优先级数值框
    priority_values = [0.2, 0.8, 0.8, 1.1, 1.4]
    for i, val in enumerate(priority_values):
        x_pos = 2.5 + i * 0.5
        rect = Rectangle((x_pos - 0.2, 6.4), 0.4, 0.4, 
                        linewidth=1, edgecolor='black', facecolor='lightblue')
        ax.add_patch(rect)
        ax.text(x_pos, 6.6, f'{val}', ha='center', va='center', fontsize=9)
    
    # 工厂分配 X₂
    ax.text(1, 5.8, '工厂分配', fontsize=12, fontweight='bold')
    ax.text(1, 5.5, 'X₂', fontsize=12, fontweight='bold')
    
    # 绘制工厂分配数值框
    factory_values = [1, 3, 2, 1, 2]
    for i, val in enumerate(factory_values):
        x_pos = 2.5 + i * 0.5
        rect = Rectangle((x_pos - 0.2, 5.4), 0.4, 0.4, 
                        linewidth=1, edgecolor='black', facecolor='lightgreen')
        ax.add_patch(rect)
        ax.text(x_pos, 5.6, f'{val}', ha='center', va='center', fontsize=9)
    
    # 机器分配 AM
    ax.text(1, 4.8, '机器分配', fontsize=12, fontweight='bold')
    ax.text(1, 4.5, 'AM', fontsize=12, fontweight='bold')
    
    # 绘制机器分配数值框
    machine_values = [1, 1.2, 1.2, 1.2, 1.5]
    for i, val in enumerate(machine_values):
        x_pos = 2.5 + i * 0.5
        rect = Rectangle((x_pos - 0.2, 4.4), 0.4, 0.4, 
                        linewidth=1, edgecolor='black', facecolor='lightyellow')
        ax.add_patch(rect)
        ax.text(x_pos, 4.6, f'{val}', ha='center', va='center', fontsize=8)
    
    # 加工时间量 AS
    ax.text(1, 3.8, '加工时间量', fontsize=12, fontweight='bold')
    ax.text(1, 3.5, 'AS', fontsize=12, fontweight='bold')
    
    # 绘制加工时间数值框
    time_values = [0.79, 0.23, 0.75, 0.54, 0.89]
    for i, val in enumerate(time_values):
        x_pos = 2.5 + i * 0.5
        rect = Rectangle((x_pos - 0.2, 3.4), 0.4, 0.4, 
                        linewidth=1, edgecolor='black', facecolor='lightcoral')
        ax.add_patch(rect)
        ax.text(x_pos, 3.6, f'{val:.2f}', ha='center', va='center', fontsize=8)

def draw_decoding_section(ax):
    """绘制解码部分（右侧）"""
    
    # 解码主框架
    decoding_box = Rectangle((8, 1.5), 5.5, 7, linewidth=2, 
                           edgecolor='black', facecolor='white')
    ax.add_patch(decoding_box)
    
    # 解码标题
    ax.text(10.75, 8, '解码', fontsize=16, fontweight='bold', ha='center')
    
    # 工厂1
    factory1_box = Rectangle((8.2, 6.5), 5.1, 1.2, linewidth=1, 
                           edgecolor='black', facecolor='#FFE5E5')
    ax.add_patch(factory1_box)
    ax.text(8.4, 7.5, '工厂 1', fontsize=12, fontweight='bold')
    
    # 工厂1的工件调度
    jobs_f1 = [
        ('J₁', 'M₁,₁(0.79)→M₁,₂(0.23)'),
        ('J₄', 'M₁,₂(0.54)→M₁,₁(0.89)'),
        ('J₅', 'M₁,₂(0.54)→M₁,₂(0.67)')
    ]
    
    y_start = 7.2
    ax.text(8.4, y_start, '工', fontsize=10, ha='center')
    ax.text(8.4, y_start - 0.1, '件', fontsize=10, ha='center')
    ax.text(8.4, y_start - 0.2, '调', fontsize=10, ha='center')
    ax.text(8.4, y_start - 0.3, '度', fontsize=10, ha='center')
    ax.text(8.4, y_start - 0.4, '及', fontsize=10, ha='center')
    ax.text(8.4, y_start - 0.5, '完', fontsize=10, ha='center')
    ax.text(8.4, y_start - 0.6, '工', fontsize=10, ha='center')
    ax.text(8.4, y_start - 0.7, '时', fontsize=10, ha='center')
    ax.text(8.4, y_start - 0.8, '间', fontsize=10, ha='center')
    ax.text(8.4, y_start - 0.9, '计', fontsize=10, ha='center')
    ax.text(8.4, y_start - 1.0, '算', fontsize=10, ha='center')
    
    for i, (job, schedule) in enumerate(jobs_f1):
        y_pos = 7.2 - i * 0.25
        ax.text(8.8, y_pos, job, fontsize=10, fontweight='bold')
        ax.text(9.2, y_pos, schedule, fontsize=9)
    
    # 工厂2
    factory2_box = Rectangle((8.2, 4.8), 5.1, 1.2, linewidth=1, 
                           edgecolor='black', facecolor='#E5F2FF')
    ax.add_patch(factory2_box)
    ax.text(8.4, 5.8, '工厂 2', fontsize=12, fontweight='bold')
    
    # 工厂2的工件调度
    jobs_f2 = [
        ('J₃', 'M₂,₁(0.30)→M₂,₂(0.16)'),
        ('J₅', 'M₂,₂(0.53)→M₂,₁(0.79)'),
        ('J₂', 'M₂,₁(0.75)→M₂,₁(0.83)')
    ]
    
    for i, (job, schedule) in enumerate(jobs_f2):
        y_pos = 5.5 - i * 0.25
        ax.text(8.8, y_pos, job, fontsize=10, fontweight='bold')
        ax.text(9.2, y_pos, schedule, fontsize=9)
    
    # 工厂3
    factory3_box = Rectangle((8.2, 3.1), 5.1, 1.2, linewidth=1, 
                           edgecolor='black', facecolor='#E5F9E5')
    ax.add_patch(factory3_box)
    ax.text(8.4, 4.1, '工厂 3', fontsize=12, fontweight='bold')
    
    # 工厂3的工件调度
    jobs_f3 = [
        ('J₂', 'M₃,₁(0.75)→M₃,₂(0.49)'),
        ('J₆', 'M₃,₂(0.67)→M₃,₂(0.16)')
    ]
    
    for i, (job, schedule) in enumerate(jobs_f3):
        y_pos = 3.8 - i * 0.25
        ax.text(8.8, y_pos, job, fontsize=10, fontweight='bold')
        ax.text(9.2, y_pos, schedule, fontsize=9)

def draw_arrows_and_steps(ax):
    """绘制连接箭头和中间处理步骤"""
    
    # 主连接箭头
    main_arrow = patches.FancyArrowPatch((5.5, 5), (8, 5),
                                       arrowstyle='->', mutation_scale=20,
                                       color='black', linewidth=2)
    ax.add_patch(main_arrow)
    
    # 详细的解码处理步骤
    steps = [
        ('步骤1：工件优先级排序', (6.75, 7.5), '#FFE5E5'),
        ('步骤2：工厂分配解析', (6.75, 6.8), '#E5F2FF'),
        ('步骤3：机器选择解码', (6.75, 6.1), '#E5F9E5'),
        ('步骤4：调度序列生成', (6.75, 5.4), '#FFF0E5'),
        ('步骤5：时间计算调度', (6.75, 4.7), '#F0E5FF')
    ]
    
    # 绘制步骤框
    for i, (step_text, pos, color) in enumerate(steps):
        x, y = pos
        step_box = Rectangle((x - 0.85, y - 0.25), 1.7, 0.5, linewidth=1, 
                           edgecolor='black', facecolor=color)
        ax.add_patch(step_box)
        ax.text(x, y, step_text, fontsize=9, ha='center', va='center', fontweight='bold')
    
    # 从编码到各个步骤的箭头
    # 从优先级向量到步骤1
    arrow1 = patches.FancyArrowPatch((5.5, 6.6), (5.9, 7.5),
                                   arrowstyle='->', mutation_scale=12,
                                   color='red', linewidth=1.5)
    ax.add_patch(arrow1)
    
    # 从工厂分配到步骤2
    arrow2 = patches.FancyArrowPatch((5.5, 5.6), (5.9, 6.8),
                                   arrowstyle='->', mutation_scale=12,
                                   color='blue', linewidth=1.5)
    ax.add_patch(arrow2)
    
    # 从机器分配到步骤3
    arrow3 = patches.FancyArrowPatch((5.5, 4.6), (5.9, 6.1),
                                   arrowstyle='->', mutation_scale=12,
                                   color='green', linewidth=1.5)
    ax.add_patch(arrow3)
    
    # 从加工时间到步骤5
    arrow4 = patches.FancyArrowPatch((5.5, 3.6), (5.9, 4.7),
                                   arrowstyle='->', mutation_scale=12,
                                   color='purple', linewidth=1.5)
    ax.add_patch(arrow4)
    
    # 步骤间的连接箭头（垂直向下）
    for i in range(len(steps) - 1):
        y1 = steps[i][1][1] - 0.25
        y2 = steps[i+1][1][1] + 0.25
        x = 6.75
        
        arrow = patches.FancyArrowPatch((x, y1), (x, y2),
                                      arrowstyle='->', mutation_scale=12,
                                      color='gray', linewidth=1.5)
        ax.add_patch(arrow)
    
    # 从最后步骤到解码结果的箭头
    final_arrow = patches.FancyArrowPatch((7.6, 4.7), (8, 5),
                                        arrowstyle='->', mutation_scale=15,
                                        color='black', linewidth=2)
    ax.add_patch(final_arrow)
    
    # 添加详细说明文本框
    add_step_details(ax)

def add_step_details(ax):
    """添加步骤详细说明"""
    
    # 步骤说明框
    detail_box = Rectangle((0.5, 0.2), 5, 1.5, linewidth=1, 
                         edgecolor='gray', facecolor='#F8F8F8')
    ax.add_patch(detail_box)
    
    detail_text = """解码处理步骤说明：
1. 根据X₁排序：J₁(0.2) < J₃(0.8) < J₂(0.8) < J₄(1.1) < J₅(1.4)
2. 按X₂分配：F1{J₁,J₄}, F2{J₂,J₅}, F3{J₃}
3. 用AM选机器：每个工序选择对应机器编号
4. 生成调度：考虑优先级和工厂约束
5. 结合AS计算：最终加工时间和完工时间"""
    
    ax.text(3, 0.95, detail_text, fontsize=8, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

if __name__ == "__main__":
    create_template_flowchart()
    print("基于模板的编码解码流程图已生成完成！")
    print("输出文件：")
    print("- 编码解码流程图_模板版.png")
    print("- 编码解码流程图_模板版.pdf") 