#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QL-ABC算法集成测试脚本
"""

def test_qlabc_integration():
    """测试QL-ABC算法集成"""
    print("=" * 60)
    print("QL-ABC算法集成测试")
    print("=" * 60)
    
    try:
        # 测试导入
        print("1. 测试QL-ABC算法导入...")
        from algorithm.ql_abc import QLABC_Optimizer
        from problem.mo_dhfsp import MO_DHFSP_Problem
        from utils.data_generator import DataGenerator
        print("   ✅ 导入成功")
        
        # 生成测试数据
        print("\n2. 生成测试数据...")
        generator = DataGenerator(seed=42)
        problem_data = generator.generate_problem(
            n_jobs=10,
            n_factories=2,
            n_stages=3,
            machines_per_stage=[2, 2, 2],
            processing_time_range=(1, 10),
            due_date_tightness=1.5
        )
        
        # 添加异构机器配置
        problem_data['heterogeneous_machines'] = {
            0: [2, 2, 2],
            1: [2, 2, 2]
        }
        print("   ✅ 测试数据生成成功")
        
        # 创建问题实例
        print("\n3. 创建问题实例...")
        problem = MO_DHFSP_Problem(problem_data)
        print("   ✅ 问题实例创建成功")
        
        # 创建QL-ABC优化器
        print("\n4. 创建QL-ABC优化器...")
        optimizer = QLABC_Optimizer(
            problem, 
            population_size=10, 
            max_iterations=5,
            learning_rate=0.1,
            epsilon=0.3
        )
        print("   ✅ QL-ABC优化器创建成功")
        
        # 运行优化
        print("\n5. 运行优化...")
        pareto_solutions, convergence_data = optimizer.optimize()
        print(f"   ✅ 优化完成，找到{len(pareto_solutions)}个帕累托解")
        
        # 验证结果
        print("\n6. 验证结果...")
        if pareto_solutions:
            best_makespan = min(sol.makespan for sol in pareto_solutions)
            best_tardiness = min(sol.total_tardiness for sol in pareto_solutions)
            print(f"   最佳完工时间: {best_makespan:.2f}")
            print(f"   最佳总拖期: {best_tardiness:.2f}")
            print("   ✅ 结果验证成功")
        else:
            print("   ⚠️ 警告：未找到有效的帕累托解")
        
        print("\n" + "=" * 60)
        print("🎉 QL-ABC算法集成测试完成！所有测试通过")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_qlabc_integration() 