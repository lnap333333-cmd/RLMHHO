#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自定义规模配置使用示例
演示如何使用 table_format_comparison_specific_scales3.py 中的自定义规模功能
"""

from table_format_comparison_specific_scales3 import run_specific_scale_experiments

def example_1_default_scales():
    """示例1: 使用默认的三个规模配置"""
    print("=" * 60)
    print("示例1: 使用默认的三个规模配置")
    print("=" * 60)
    
    # 不传递参数，使用默认配置
    run_specific_scale_experiments()

def example_2_custom_small_scales():
    """示例2: 使用自定义的小规模配置"""
    print("=" * 60)
    print("示例2: 使用自定义的小规模配置")
    print("=" * 60)
    
    # 定义三个小规模配置
    custom_scales = [
        {'n_jobs': 15, 'n_stages': 2, 'n_factories': 2, 'name': '微型规模'},
        {'n_jobs': 25, 'n_stages': 3, 'n_factories': 2, 'name': '小型规模'},
        {'n_jobs': 35, 'n_stages': 3, 'n_factories': 3, 'name': '中小规模'}
    ]
    
    run_specific_scale_experiments(custom_scales)

def example_3_custom_large_scales():
    """示例3: 使用自定义的大规模配置"""
    print("=" * 60)
    print("示例3: 使用自定义的大规模配置")
    print("=" * 60)
    
    # 定义三个大规模配置
    custom_scales = [
        {'n_jobs': 80, 'n_stages': 4, 'n_factories': 3, 'name': '较大规模'},
        {'n_jobs': 120, 'n_stages': 5, 'n_factories': 4, 'name': '大规模'},
        {'n_jobs': 150, 'n_stages': 5, 'n_factories': 5, 'name': '超大规模'}
    ]
    
    run_specific_scale_experiments(custom_scales)

def example_4_mixed_scales():
    """示例4: 使用混合规模配置"""
    print("=" * 60)
    print("示例4: 使用混合规模配置")
    print("=" * 60)
    
    # 定义混合规模配置
    custom_scales = [
        {'n_jobs': 20, 'n_stages': 2, 'n_factories': 2, 'name': '简单规模'},
        {'n_jobs': 60, 'n_stages': 4, 'n_factories': 3, 'name': '复杂规模'},
        {'n_jobs': 100, 'n_stages': 6, 'n_factories': 5, 'name': '高复杂度规模'}
    ]
    
    run_specific_scale_experiments(custom_scales)

def example_5_user_defined_scales():
    """示例5: 用户自定义规模配置"""
    print("=" * 60)
    print("示例5: 用户自定义规模配置")
    print("=" * 60)
    
    # 用户可以根据需要修改这些参数
    custom_scales = [
        # 第一个规模：小规模测试
        {'n_jobs': 30, 'n_stages': 3, 'n_factories': 2, 'name': '测试规模1'},
        
        # 第二个规模：中等规模
        {'n_jobs': 70, 'n_stages': 4, 'n_factories': 3, 'name': '测试规模2'},
        
        # 第三个规模：大规模
        {'n_jobs': 110, 'n_stages': 5, 'n_factories': 4, 'name': '测试规模3'}
    ]
    
    # 可以在这里添加自定义逻辑
    print("自定义规模配置：")
    for i, scale in enumerate(custom_scales, 1):
        print(f"  规模{i}: {scale['name']} - {scale['n_jobs']}工件 {scale['n_stages']}阶段 {scale['n_factories']}工厂")
    
    run_specific_scale_experiments(custom_scales)

if __name__ == "__main__":
    print("自定义规模配置使用示例")
    print("=" * 80)
    print("本文件演示了如何使用自定义规模配置进行算法对比实验")
    print("请选择要运行的示例：")
    print("1. 默认规模配置")
    print("2. 小规模配置")
    print("3. 大规模配置")
    print("4. 混合规模配置")
    print("5. 用户自定义规模配置")
    print()
    
    # 运行所有示例（注意：这会运行很长时间）
    print("注意：运行所有示例将需要很长时间，建议根据需要选择特定示例")
    print()
    
    # 取消注释下面的行来运行特定示例
    # example_1_default_scales()
    # example_2_custom_small_scales()
    # example_3_custom_large_scales()
    # example_4_mixed_scales()
    # example_5_user_defined_scales()
    
    print("请取消注释主函数中的相应示例函数来运行实验")
    print("例如：取消注释 example_1_default_scales() 来运行默认规模配置") 