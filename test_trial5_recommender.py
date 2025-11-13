#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test Trial 5 recommender system performance
直接测试Trial 5推荐效果
"""

import os
import sys
import json

def main():
    print("=" * 70)
    print("🧪 Testing Trial 5 Recommender System")
    print("=" * 70)
    
    try:
        # 导入推荐系统
        from recommender_system import AcademicRecommender
        from evaluate_recommender import RecommenderEvaluator, analyze_data_issues
        
        print("\n📥 Initializing recommender system with Trial 5 model...")
        recommender = AcademicRecommender(
            model_path="models/trial5/han_embeddings.pth",
            neo4j_uri="neo4j://127.0.0.1:7687",
            neo4j_username="neo4j",
            neo4j_password="87654321"
        )
        print("✅ Recommender system initialized successfully!")
        
        # 创建评估器 - 使用filtered策略
        print("\n📊 Creating evaluator with 'filtered' strategy...")
        evaluator = RecommenderEvaluator(
            recommender,
            k_values=[5, 10, 20],
            evaluation_strategy="filtered"
        )
        
        # 加载测试数据
        test_data_path = "data/processed/test/papers.csv"
        if not os.path.exists(test_data_path):
            print(f"❌ Test data not found: {test_data_path}")
            print("💡 Available paths:")
            for root, dirs, files in os.walk("data/processed/test"):
                for file in files:
                    print(f"   - {os.path.join(root, file)}")
            return
        
        print(f"\n📋 Loading test data from {test_data_path}...")
        evaluator.load_test_data(test_data_path)
        evaluator.prepare_ground_truth()
        
        # 分析数据覆盖情况
        print("\n📊 Analyzing data coverage...")
        data_analysis = analyze_data_issues(recommender, test_data_path)
        print(f"   Coverage: {data_analysis['coverage']:.1%}")
        print(f"   Common papers: {data_analysis['common_papers']}")
        print(f"   Missing papers: {data_analysis['missing_papers']}")
        
        # 获取可用论文
        available_papers = evaluator._get_available_test_papers()
        print(f"\n📝 Available test papers: {len(available_papers)}")
        
        if not available_papers:
            print("❌ No test papers available for evaluation")
            return
        
        # 评估协同过滤方法
        print("\n" + "=" * 70)
        print("📊 Evaluating Collaborative Filtering")
        print("=" * 70)
        
        sample_size = min(20, len(available_papers))
        print(f"\n🔄 Evaluating {sample_size} papers...")
        
        collab_results = evaluator.evaluate_batch(
            available_papers[:sample_size], 
            "collaborative"
        )
        
        if "average_metrics" in collab_results:
            print("\n📈 Results Summary:")
            for k in [5, 10, 20]:
                if k in collab_results["average_metrics"]:
                    metrics = collab_results["average_metrics"][k]
                    print(f"\n  K={k}:")
                    print(f"    Precision:  {metrics['precision']['mean']:.4f} ± {metrics['precision']['std']:.4f}")
                    print(f"    Recall:     {metrics['recall']['mean']:.4f} ± {metrics['recall']['std']:.4f}")
                    print(f"    F1:         {metrics['f1']['mean']:.4f} ± {metrics['f1']['std']:.4f}")
                    print(f"    NDCG:       {metrics['ndcg']['mean']:.4f} ± {metrics['ndcg']['std']:.4f}")
                    print(f"    Hit Rate:   {metrics['hit_rate']['mean']:.4f}")
                    print(f"    Coverage:   {metrics['coverage']['mean']:.1%}")
        
        # 尝试评估内容基础方法
        print("\n" + "=" * 70)
        print("📊 Evaluating Content-Based Method")
        print("=" * 70)
        
        print(f"\n🔄 Evaluating {sample_size} papers...")
        
        try:
            content_results = evaluator.evaluate_batch(
                available_papers[:sample_size],
                "content"
            )
            
            if "average_metrics" in content_results:
                print("\n📈 Results Summary:")
                for k in [5, 10, 20]:
                    if k in content_results["average_metrics"]:
                        metrics = content_results["average_metrics"][k]
                        print(f"\n  K={k}:")
                        print(f"    Precision:  {metrics['precision']['mean']:.4f} ± {metrics['precision']['std']:.4f}")
                        print(f"    Recall:     {metrics['recall']['mean']:.4f} ± {metrics['recall']['std']:.4f}")
                        print(f"    F1:         {metrics['f1']['mean']:.4f} ± {metrics['f1']['std']:.4f}")
                        print(f"    NDCG:       {metrics['ndcg']['mean']:.4f} ± {metrics['ndcg']['std']:.4f}")
        except Exception as e:
            print(f"⚠️  Content-based evaluation failed: {e}")
        
        # 保存完整结果
        print("\n" + "=" * 70)
        print("💾 Saving Results")
        print("=" * 70)
        
        final_results = {
            "model": "Trial 5",
            "model_path": "models/trial5/han_embeddings.pth",
            "evaluation_strategy": "filtered",
            "data_coverage": data_analysis,
            "collaborative_filtering": collab_results,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        output_dir = "evaluation_results"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "trial5_evaluation.json")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # 转换numpy类型为可序列化的格式
            import numpy as np
            def convert_to_serializable(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_to_serializable(item) for item in obj]
                return obj
            
            json.dump(convert_to_serializable(final_results), f, indent=2, ensure_ascii=False)
        
        print(f"✅ Results saved to {output_path}")
        
        # 最终总结
        print("\n" + "=" * 70)
        print("📋 FINAL SUMMARY")
        print("=" * 70)
        print(f"Model: Trial 5")
        print(f"Evaluation Strategy: Filtered (only papers in database)")
        print(f"Test Papers: {sample_size}/{len(available_papers)}")
        print(f"Data Coverage: {data_analysis['coverage']:.1%}")
        
        if "average_metrics" in collab_results:
            k = 10
            if k in collab_results["average_metrics"]:
                metrics = collab_results["average_metrics"][k]
                print(f"\nBest Results (K=10):")
                print(f"  Precision@10:  {metrics['precision']['mean']:.4f}")
                print(f"  Recall@10:     {metrics['recall']['mean']:.4f}")
                print(f"  NDCG@10:       {metrics['ndcg']['mean']:.4f}")
                print(f"  Hit Rate@10:   {metrics['hit_rate']['mean']:.2%}")
        
        print("\n✅ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    import pandas as pd
    sys.exit(main())
