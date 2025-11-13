# [file name]: evaluate_recommender_fixed.py
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, ndcg_score
import json
import os
from typing import List, Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class RecommenderEvaluator:
    def __init__(self, 
                 recommender_system,
                 test_data_path: str = None,
                 k_values: List[int] = [5, 10, 20],
                 evaluation_strategy: str = "filtered"):
        """
        初始化推荐系统评估器
        
        Args:
            evaluation_strategy: 评估策略
                - "filtered": 只用数据库中存在的ground truth计算指标（推荐）
                - "full": 用全部ground truth计算指标（会显示偏低但可看出数据缺失程度）
        """
        self.recommender = recommender_system
        self.k_values = k_values
        self.test_data = None
        self.ground_truth = {}
        self.evaluation_strategy = evaluation_strategy
        
        if test_data_path and os.path.exists(test_data_path):
            self.load_test_data(test_data_path)
    
    def load_test_data(self, test_data_path: str):
        """加载测试数据"""
        print(f"📊 Loading test data from {test_data_path}...")
        try:
            # 支持多种格式：CSV, JSON等
            if test_data_path.endswith('.csv'):
                self.test_data = pd.read_csv(test_data_path)
            elif test_data_path.endswith('.jsonl') or test_data_path.endswith('.ndjson'):
                # line-delimited JSON
                self.test_data = pd.read_json(test_data_path, lines=True)
            elif test_data_path.endswith('.json'):
                with open(test_data_path, 'r', encoding='utf-8') as f:
                    # support lists of objects or dict-of-lists
                    raw = json.load(f)
                    if isinstance(raw, list):
                        self.test_data = pd.DataFrame(raw)
                    else:
                        self.test_data = pd.DataFrame(raw)
            print(f"✅ Loaded {len(self.test_data)} test samples")
        except Exception as e:
            print(f"❌ Failed to load test data: {e}")
            self.test_data = None
    
    def prepare_ground_truth(self, citation_data: Dict[str, List[str]] = None):
        """
        准备ground truth数据 - 修复ID格式问题
        """
        if citation_data:
            self.ground_truth = citation_data
        elif self.test_data is not None:
            # 从测试数据中提取ground truth，确保ID格式一致
            cols = set(self.test_data.columns)
            for _, row in self.test_data.iterrows():
                # paper id may be under 'paper_id' or 'id' depending on dataset
                if 'paper_id' in cols and pd.notna(row.get('paper_id')):
                    paper_id = str(row['paper_id']).strip()
                elif 'id' in cols and pd.notna(row.get('id')):
                    paper_id = str(row['id']).strip()
                else:
                    # skip rows without identifiable id
                    continue

                # 处理不同的ground truth格式: try several common field names
                citations = []
                gt_citations = None
                for key in ('ground_truth_citations', 'references', 'citations'):
                    if key in cols:
                        raw = row.get(key)
                        # skip null/NaN
                        if raw is None:
                            continue
                        try:
                            if isinstance(raw, float) and np.isnan(raw):
                                continue
                        except Exception:
                            pass
                        gt_citations = raw
                        break

                if gt_citations is None:
                    citations = []
                elif isinstance(gt_citations, str):
                    # sometimes string-encoded lists or comma-separated ids
                    try:
                        parsed = json.loads(gt_citations)
                        if isinstance(parsed, list):
                            citations = parsed
                        else:
                            citations = [str(parsed)]
                    except Exception:
                        # fallback to eval or comma-split
                        try:
                            citations = eval(gt_citations)
                        except Exception:
                            citations = [x.strip() for x in str(gt_citations).split(',') if x.strip()]
                elif isinstance(gt_citations, list):
                    citations = gt_citations
                else:
                    citations = [str(gt_citations)]

                # 确保所有citation ID都是字符串格式
                self.ground_truth[paper_id] = [str(cite).strip() for cite in citations if cite]
        
        print(f"✅ Prepared ground truth for {len(self.ground_truth)} papers")
        print(f"   Sample ground truth: {list(self.ground_truth.items())[:2]}")
    
    def _check_paper_in_embeddings(self, paper_id: str) -> bool:
        """检查论文是否在嵌入中"""
        return paper_id in self.recommender.id_maps['paper']
    
    def _get_available_test_papers(self) -> List[str]:
        """获取在嵌入中可用的测试论文"""
        if not self.ground_truth:
            return []
        
        available_papers = []
        for paper_id in self.ground_truth.keys():
            if self._check_paper_in_embeddings(paper_id):
                available_papers.append(paper_id)
        
        print(f"📋 Available test papers in embeddings: {len(available_papers)}/{len(self.ground_truth)}")
        return available_papers
    
    def evaluate_single_paper(self, paper_id: str, method: str = "collaborative") -> Dict[str, Any]:
        """
        评估单篇论文的推荐效果 - 修复版本
        """
        # 检查论文是否在嵌入中
        if not self._check_paper_in_embeddings(paper_id):
            return {"error": f"Paper {paper_id} not in embeddings", "paper_id": paper_id}
        
        if paper_id not in self.ground_truth:
            return {"error": f"No ground truth for paper {paper_id}", "paper_id": paper_id}
        
        true_citations = set(self.ground_truth[paper_id])
        
        # 获取推荐结果
        try:
            if method == "collaborative":
                recommendations = self.recommender.enhanced_collaborative_recommendation(
                    paper_id, top_k=max(self.k_values)
                )
            elif method == "content":
                # 使用论文标题进行基于内容的推荐
                paper_metadata = self.recommender.get_paper_metadata([paper_id])
                title = paper_metadata.get(paper_id, {}).get('title', '')
                if title:
                    recommendations = self.recommender.content_based_paper_recommendation(
                        title, top_k=max(self.k_values)
                    )
                else:
                    return {"error": f"No title available for paper {paper_id}", "paper_id": paper_id}
            elif method == "hybrid":
                hybrid_result = self.recommender.optimized_hybrid_recommendation(
                    target_paper_id=paper_id, top_k=max(self.k_values)
                )
                recommendations = hybrid_result['recommendations']
            else:
                return {"error": f"Unknown method: {method}", "paper_id": paper_id}
        except Exception as e:
            return {"error": f"Recommendation failed: {e}", "paper_id": paper_id}
        
        if not recommendations:
            return {"error": f"No recommendations generated for paper {paper_id}", "paper_id": paper_id}
        
        # 提取推荐的论文ID - 确保格式一致
        recommended_ids = [str(rec['paper_id']).strip() for rec in recommendations]
        
        # 计算各种指标
        results = {
            "paper_id": paper_id,
            "method": method,
            "true_citations_count": len(true_citations),
            "recommendations_count": len(recommended_ids),
            "recommended_ids": recommended_ids[:10]  # 保存前10个推荐用于调试
        }
        
        # 为每个K值计算指标
        for k in self.k_values:
            k_recs = recommended_ids[:k]
            k_results = self._compute_metrics_at_k(k_recs, true_citations, k)
            results.update({f"{metric}_at_{k}": value for metric, value in k_results.items()})
        
        # 添加coverage信息到结果中
        if true_citations:
            available_gt = {gt for gt in true_citations 
                           if gt in self.recommender.id_maps['paper']}
            results['ground_truth_total'] = len(true_citations)
            results['ground_truth_available'] = len(available_gt)
            results['ground_truth_coverage'] = len(available_gt) / len(true_citations)
        
        # 调试信息
        if len(true_citations) > 0:
            hits = sum(1 for rec in recommended_ids[:10] if rec in true_citations)
            print(f"   Paper {paper_id}: {hits} hits in top-10, {len(true_citations)} ground truth")
        
        return results
    
    def _compute_metrics_at_k(self, recommendations: List[str], ground_truth: set, k: int) -> Dict[str, float]:
        """在特定K值下计算评估指标
        
        支持两种评估策略：
        - filtered: 只用数据库中存在的ground truth计算
        - full: 用全部ground truth计算
        """
        # 处理两种评估策略
        if self.evaluation_strategy == "filtered":
            # 只保留在embeddings中存在的ground truth
            available_gt = {gt for gt in ground_truth 
                           if gt in self.recommender.id_maps['paper']}
            coverage = len(available_gt) / len(ground_truth) if ground_truth else 0
        else:  # full strategy
            available_gt = ground_truth
            coverage = 1.0
        
        if not available_gt:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "hit_rate": 0.0,
                "ndcg": 0.0,
                "coverage": coverage
            }
        
        # 计算命中情况
        hits = [1 if rec in available_gt else 0 for rec in recommendations]
        num_hits = sum(hits)
        
        # Precision@K
        precision = num_hits / k if k > 0 else 0.0
        
        # Recall@K
        recall = num_hits / len(available_gt) if available_gt else 0.0
        
        # F1@K
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Hit Rate@K
        hit_rate = 1.0 if num_hits > 0 else 0.0
        
        # NDCG@K
        ndcg = self._compute_ndcg(hits, available_gt, k)
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "hit_rate": hit_rate,
            "ndcg": ndcg,
            "coverage": coverage  # 数据库覆盖率
        }
    
    def _compute_ndcg(self, hits: List[int], ground_truth: set, k: int) -> float:
        """计算NDCG"""
        def dcg(scores):
            return sum(score / np.log2(i + 2) for i, score in enumerate(scores))
        
        # 实际DCG
        actual_dcg = dcg(hits)
        
        # 理想DCG - 所有相关项在前
        ideal_ranking = [1] * min(len(ground_truth), k) + [0] * max(0, k - len(ground_truth))
        ideal_dcg = dcg(ideal_ranking)
        
        return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0
    
    def evaluate_batch(self, paper_ids: List[str] = None, method: str = "collaborative") -> Dict[str, Any]:
        """
        批量评估多篇论文 - 修复版本
        """
        if paper_ids is None:
            paper_ids = self._get_available_test_papers()
        
        if not paper_ids:
            return {"error": "No available test papers in embeddings"}
        
        print(f"🔍 Evaluating {len(paper_ids)} papers using {method} method...")
        
        results = []
        successful_evals = 0
        error_details = []
        
        for i, paper_id in enumerate(paper_ids):
            if i % 10 == 0:
                print(f"   Progress: {i}/{len(paper_ids)}")
            
            result = self.evaluate_single_paper(paper_id, method)
            if "error" not in result:
                results.append(result)
                successful_evals += 1
            else:
                error_details.append(result)
        
        print(f"✅ Completed evaluation of {successful_evals} papers")
        print(f"   Errors: {len(error_details)}")
        
        if not results:
            return {
                "error": "No successful evaluations", 
                "error_details": error_details
            }
        
        # 计算平均指标
        avg_metrics = self._compute_average_metrics(results)
        
        return {
            "method": method,
            "total_papers": len(paper_ids),
            "successful_evaluations": successful_evals,
            "error_details": error_details,
            "average_metrics": avg_metrics,
            "detailed_results": results
        }
    
    def _compute_average_metrics(self, results: List[Dict]) -> Dict[str, Any]:
        """计算平均指标"""
        avg_metrics = {}
        
        for k in self.k_values:
            metrics_at_k = {
                "precision": [],
                "recall": [], 
                "f1": [],
                "hit_rate": [],
                "ndcg": [],
                "coverage": []
            }
            
            for result in results:
                for metric in metrics_at_k.keys():
                    value = result.get(f"{metric}_at_{k}", 0)
                    metrics_at_k[metric].append(value)
            
            # 计算平均值和标准差
            avg_metrics[k] = {
                metric: {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values)
                }
                for metric, values in metrics_at_k.items()
            }
        
        return avg_metrics
    
    def compare_methods(self, paper_ids: List[str] = None, methods: List[str] = None) -> Dict[str, Any]:
        """
        比较不同推荐方法的性能
        """
        if methods is None:
            methods = ["collaborative", "content", "hybrid"]
        
        comparison_results = {}
        
        for method in methods:
            print(f"\n📊 Evaluating {method} method...")
            results = self.evaluate_batch(paper_ids, method)
            comparison_results[method] = results
        
        # 生成比较报告
        comparison_report = self._generate_comparison_report(comparison_results)
        
        return {
            "comparison_results": comparison_results,
            "comparison_report": comparison_report
        }
    
    def _generate_comparison_report(self, comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """生成方法比较报告"""
        report = {}
        
        for k in self.k_values:
            report[f"K={k}"] = {}
            for method, results in comparison_results.items():
                if "average_metrics" in results and k in results["average_metrics"]:
                    report[f"K={k}"][method] = results["average_metrics"][k]
        
        return report
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """保存评估结果 - 修复路径问题"""
        try:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            # 转换为可JSON序列化的格式
            serializable_results = self._make_serializable(results)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Results saved to {output_path}")
        except Exception as e:
            print(f"❌ Failed to save results: {e}")
            # 回退到当前目录
            fallback_path = "evaluation_results_fallback.json"
            with open(fallback_path, 'w', encoding='utf-8') as f:
                json.dump(self._make_serializable(results), f, indent=2, ensure_ascii=False)
            print(f"💾 Results saved to {fallback_path} as fallback")
    
    def _make_serializable(self, obj):
        """确保对象可JSON序列化"""
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return str(obj)

def analyze_data_issues(recommender, test_data_path: str):
    """分析数据问题"""
    print("\n🔍 Analyzing data issues...")
    
    # 检查测试数据
    # reuse loading logic: support csv, json, jsonl
    try:
        if test_data_path.endswith('.csv'):
            test_data = pd.read_csv(test_data_path)
        elif test_data_path.endswith('.jsonl') or test_data_path.endswith('.ndjson'):
            test_data = pd.read_json(test_data_path, lines=True)
        elif test_data_path.endswith('.json'):
            with open(test_data_path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
                test_data = pd.DataFrame(raw)
        else:
            test_data = pd.read_csv(test_data_path)
        print(f"📊 Test data: {len(test_data)} samples")
    except Exception as e:
        print(f"❌ Failed to load test data for analysis: {e}")
        return {"common_papers": 0, "missing_papers": 0, "coverage": 0}
    
    # 检查ID映射
    paper_ids_in_embeddings = set(recommender.id_maps['paper'].keys())
    # test file may use 'paper_id' or 'id'
    if 'paper_id' in test_data.columns:
        paper_ids_in_test = set(str(pid).strip() for pid in test_data['paper_id'] if pd.notna(pid))
    elif 'id' in test_data.columns:
        paper_ids_in_test = set(str(pid).strip() for pid in test_data['id'] if pd.notna(pid))
    else:
        # fallback: try to use first column as id
        first_col = test_data.columns[0]
        paper_ids_in_test = set(str(pid).strip() for pid in test_data[first_col] if pd.notna(pid))
    
    common_papers = paper_ids_in_embeddings & paper_ids_in_test
    missing_papers = paper_ids_in_test - paper_ids_in_embeddings
    
    print(f"📋 Papers in both embeddings and test data: {len(common_papers)}")
    print(f"❌ Papers in test data but not in embeddings: {len(missing_papers)}")
    
    if missing_papers:
        print(f"   Sample missing papers: {list(missing_papers)[:5]}")
    
    return {
        "common_papers": len(common_papers),
        "missing_papers": len(missing_papers),
        "coverage": len(common_papers) / len(paper_ids_in_test) if paper_ids_in_test else 0
    }

def main():
    """主评估函数 - 修复版本"""
    print("🎯 Starting Recommender System Evaluation (Fixed Version)")
    print("=" * 70)
    
    try:
        # 初始化推荐系统
        from recommender_system import AcademicRecommender
        
        print("📥 Initializing recommender system...")
        recommender = AcademicRecommender()
        
        # 支持两种评估策略
        print("\n🔄 Evaluation Strategy Options:")
        print("   1. 'filtered': 只用数据库中存在的ground truth计算（推荐）")
        print("   2. 'full': 用全部ground truth计算（显示真实数据覆盖情况）")
        
        # 创建评估器 - 默认使用filtered策略
        evaluator = RecommenderEvaluator(
            recommender, 
            k_values=[5, 10, 20],
            evaluation_strategy="filtered"  # 可改为"full"查看数据缺失影响
        )
        
        print(f"\n📊 Using evaluation strategy: {evaluator.evaluation_strategy}")
        
        # 检查测试数据
        test_data_path = "test_data.csv"  # 替换为您的真实ground truth文件
        if not os.path.exists(test_data_path):
            print(f"❌ Test data file not found: {test_data_path}")
            print("💡 Please provide your ground truth data in the correct format")
            return
        
        # 分析数据问题
        data_analysis = analyze_data_issues(recommender, test_data_path)
        
        if data_analysis["coverage"] < 0.5:
            print(f"⚠️ Low coverage: only {data_analysis['coverage']:.1%} of test papers are in embeddings")
            print("💡 Consider retraining the model with more comprehensive data")
        
        # 加载测试数据
        evaluator.load_test_data(test_data_path)
        evaluator.prepare_ground_truth()
        
        # 只评估在嵌入中可用的论文
        available_papers = evaluator._get_available_test_papers()
        
        if not available_papers:
            print("❌ No test papers available in embeddings. Evaluation cannot proceed.")
            return
        
        print(f"📝 Using {len(available_papers)} available papers for evaluation")
        
        # 评估单个方法
        print("\n" + "=" * 70)
        print("📊 Evaluating Collaborative Filtering Method")
        print("=" * 70)
        
        collaborative_results = evaluator.evaluate_batch(available_papers[:20], "collaborative")
        
        if "average_metrics" in collaborative_results:
            print("\n📈 Collaborative Filtering Results:")
            print(f"   Evaluation Strategy: {evaluator.evaluation_strategy}")
            for k, metrics in collaborative_results["average_metrics"].items():
                print(f"  K={k}:")
                for metric, stats in metrics.items():
                    print(f"    {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
            
            # 显示数据覆盖率
            if 'coverage' in collaborative_results["average_metrics"].get(list(collaborative_results["average_metrics"].keys())[0], {}):
                avg_coverage = np.mean([
                    collaborative_results["average_metrics"][k]['coverage']['mean']
                    for k in collaborative_results["average_metrics"].keys()
                ])
                print(f"\n  📊 Average Ground Truth Coverage: {avg_coverage:.1%}")
        
        # 比较不同方法
        print("\n" + "=" * 70)
        print("🔄 Comparing Different Recommendation Methods")
        print("=" * 70)
        
        # 使用少量样本进行比较（为了速度）
        comparison_sample = available_papers[:10]
        comparison_results = evaluator.compare_methods(
            comparison_sample,
            methods=["collaborative", "content"]  # 暂时只比较这两种方法
        )
        
        # 显示比较结果
        if "comparison_report" in comparison_results:
            print("\n📊 Method Comparison Report:")
            for k, methods_data in comparison_results["comparison_report"].items():
                print(f"\n  {k}:")
                for method, metrics in methods_data.items():
                    print(f"    {method}:")
                    for metric, stats in metrics.items():
                        print(f"      {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        # 保存结果
        print("\n" + "=" * 70)
        print("💾 Saving Evaluation Results")
        print("=" * 70)
        
        all_results = {
            "evaluation_strategy": evaluator.evaluation_strategy,
            "data_analysis": data_analysis,
            "collaborative_results": collaborative_results,
            "comparison_results": comparison_results,
            "evaluation_config": {
                "k_values": evaluator.k_values,
                "test_samples": len(available_papers),
                "available_samples": len(available_papers),
                "evaluation_strategy": evaluator.evaluation_strategy,
                "timestamp": pd.Timestamp.now().isoformat()
            }
        }
        
        evaluator.save_results(all_results, "evaluation_results/evaluation_report.json")
        
        print("\n✅ Evaluation completed successfully!")
        
        # 显示简要总结
        print("\n📋 Summary:")
        print(f"  Test coverage: {data_analysis['coverage']:.1%}")
        print(f"  Evaluation strategy: {evaluator.evaluation_strategy}")
        if "average_metrics" in collaborative_results:
            k = 10
            metrics = collaborative_results["average_metrics"].get(k, {})
            precision = metrics.get('precision', {}).get('mean', 0)
            recall = metrics.get('recall', {}).get('mean', 0)
            coverage = metrics.get('coverage', {}).get('mean', 1.0)
            print(f"  Precision@{k}: {precision:.4f}")
            print(f"  Recall@{k}: {recall:.4f}")
            print(f"  Ground truth coverage: {coverage:.1%}")
            print(f"  Successful evaluations: {collaborative_results.get('successful_evaluations', 0)}")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()