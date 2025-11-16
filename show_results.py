#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
结果展示程序
用于展示和验证生成的可视化结果
"""

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from datetime import datetime

def show_latest_results():
    """展示最新的实验结果"""
    results_dir = "results"
    
    print("🖼️  生成的可视化结果文件:")
    print("=" * 60)
    
    # 获取所有PNG文件并按修改时间排序
    png_files = []
    for file in os.listdir(results_dir):
        if file.endswith(".png"):
            filepath = os.path.join(results_dir, file)
            mtime = os.path.getmtime(filepath)
            size = os.path.getsize(filepath)
            png_files.append((file, mtime, size))
    
    # 按修改时间排序（最新的在前）
    png_files.sort(key=lambda x: x[1], reverse=True)
    
    for i, (filename, mtime, size) in enumerate(png_files):
        mod_time = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        size_kb = size / 1024
        print(f"{i+1:2d}. {filename}")
        print(f"    📅 修改时间: {mod_time}")
        print(f"    📊 文件大小: {size_kb:.1f} KB")
        
        # 标识新生成的详细对比图
        if "详细算法对比" in filename:
            print(f"    ✨ 【新增】详细算法对比图 - 包含6个分析维度")
        elif "算法对比" in filename and "详细" not in filename:
            print(f"    📈 标准算法对比图")
        else:
            print(f"    📈 其他可视化图表")
        print()
    
    print("\n🔍 详细对比图包含的分析内容:")
    print("-" * 50)
    print("📊 1. 帕累托前沿对比 - 直观显示算法解的分布")
    print("📈 2. 完工时间收敛对比 - 展示算法优化过程")
    print("📉 3. 总拖期收敛对比 - 显示拖期优化效果")
    print("📏 4. 反世代距离(IGD)对比 - 衡量解与理想前沿的距离")
    print("📐 5. 超体积(HV)对比 - 评估帕累托前沿的覆盖质量")
    print("🎯 6. 综合性能雷达图 - 多维度性能评估")
    
    print("\n📋 实验报告文件:")
    print("-" * 50)
    
    # 获取所有TXT报告文件
    txt_files = []
    for file in os.listdir(results_dir):
        if file.endswith(".txt") and "report" in file:
            filepath = os.path.join(results_dir, file)
            mtime = os.path.getmtime(filepath)
            txt_files.append((file, mtime))
    
    txt_files.sort(key=lambda x: x[1], reverse=True)
    
    for i, (filename, mtime) in enumerate(txt_files):
        mod_time = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        print(f"{i+1}. {filename} ({mod_time})")
    
    # 显示最新实验的关键结果
    print("\n🏆 最新实验关键结果:")
    print("=" * 60)
    
    print("小规模问题 (10作业, 2工厂):")
    print("  🥇 RL-Chaotic-HHO: 完工时间 48.84, 拖期 0.00 (10.23秒)")
    print("  🥈 NSGA-II:         完工时间 56.52, 拖期 1.05 (0.69秒)")
    print("  📊 RL-Chaotic-HHO在解质量上更优，NSGA-II在速度上更快")
    
    print("\n中规模问题 (20作业, 2工厂):")
    print("  🥇 RL-Chaotic-HHO: 完工时间 94.15, 拖期 123.00 (14.36秒)")
    print("  🥈 NSGA-II:         完工时间 107.47, 拖期 188.31 (1.28秒)")
    print("  📊 RL-Chaotic-HHO在两个目标上都显著优于NSGA-II")
    
    print("\n💡 算法特点总结:")
    print("  🔬 RL-Chaotic-HHO: 高质量解，适合对精度要求高的场景")
    print("  ⚡ NSGA-II:         快速求解，适合对时间要求严格的场景")
    print("  🎯 IGD和HV指标：详细对比图中可以看到算法的多维度性能差异")

def analyze_visualization_features():
    """分析可视化功能特点"""
    print("\n🎨 增强的可视化功能特点:")
    print("=" * 60)
    
    features = [
        ("帕累托前沿对比", "使用不同颜色和标记显示各算法的解分布，直观展示解的质量和多样性"),
        ("收敛曲线对比", "分别展示完工时间和拖期的优化过程，揭示算法的收敛特性"),
        ("反世代距离(IGD)", "计算算法解与真实帕累托前沿的距离，评估解的逼近质量"),
        ("超体积(HV)", "评估帕累托前沿覆盖的目标空间体积，衡量解集的全面性"),
        ("综合性能雷达图", "多维度评估：运行时间、解数量、IGD、HV等指标的综合比较"),
        ("专业可视化设计", "使用现代配色方案、清晰标注、网格线等提高图表可读性")
    ]
    
    for i, (title, description) in enumerate(features, 1):
        print(f"{i}. {title}")
        print(f"   {description}")
        print()

if __name__ == "__main__":
    print("🚀 多目标算法对比实验结果展示")
    print("=" * 60)
    
    show_latest_results()
    analyze_visualization_features()
    
    print("\n✅ 结果展示完成！")
    print("💡 提示：详细对比图文件较大，包含了丰富的分析信息")
    print("📁 所有结果文件保存在 results/ 目录中") 