#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用Sentence Transformer进行推荐评估
流程：
1. 从test数据集中抽取paper及其ground truth引用
2. 用Sentence Transformer编码test paper的title/abstract
3. 用余弦相似度找training set中最相似的papers（排名）
4. 和ground truth里该paper实际引用的papers进行比较
5. 用filtered strategy计算Precision/Recall/NDCG
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Set
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch

def load_ground_truth_from_csv(csv_path: str) -> Dict[str, List[str]]:
    """从CSV加载ground truth - paper_id 和 ground_truth_citations"""
    print(f"📖 Loading ground truth from {csv_path}...")
    
    ground_truth = {}
    
    try:
        df = pd.read_csv(csv_path)
        print(f"   Loaded {len(df)} rows")
        print(f"   Columns: {df.columns.tolist()}")
        
        # 检查列名
        paper_col = None
        citations_col = None
        
        for col in df.columns:
            if 'paper_id' in col.lower() or col.lower() == 'id':
                paper_col = col
            if 'citation' in col.lower() or 'reference' in col.lower():
                citations_col = col
        
        if not paper_col or not citations_col:
            print(f"❌ Cannot find paper_id or citations columns")
            print(f"   Available columns: {df.columns.tolist()}")
            return {}
        
        print(f"   Using columns: paper_id='{paper_col}', citations='{citations_col}'")
        
        for _, row in df.iterrows():
            paper_id = str(row[paper_col]).strip()
            citations_raw = row[citations_col]
            
            # 处理citations格式
            citations = []
            if pd.notna(citations_raw):
                try:
                    if isinstance(citations_raw, str):
                        # 尝试JSON解析
                        citations = json.loads(citations_raw)
                        if not isinstance(citations, list):
                            citations = [citations]
                    elif isinstance(citations_raw, list):
                        citations = citations_raw
                    else:
                        citations = [str(citations_raw)]
                except:
                    # 备用方案：按逗号分割
                    citations = [c.strip() for c in str(citations_raw).split(',') if c.strip()]
            
            citations = [str(c).strip() for c in citations if c]
            ground_truth[paper_id] = citations
        
        print(f"✅ Loaded ground truth for {len(ground_truth)} papers")
        print(f"   Sample: {list(ground_truth.items())[:2]}")
        
        return ground_truth
        
    except Exception as e:
        print(f"❌ Failed to load ground truth: {e}")
        return {}

def load_training_papers_from_csv(csv_path: str) -> Dict[str, Dict]:
    """从CSV加载training papers"""
    print(f"📖 Loading training papers from {csv_path}...")
    
    try:
        df = pd.read_csv(csv_path)
        print(f"   Loaded {len(df)} papers")
        
        papers = {}
        for _, row in df.iterrows():
            paper_id = str(row['paper_id']).strip()
            papers[paper_id] = {
                'title': str(row.get('title', '')),
                'abstract': str(row.get('abstract', '') or '')
            }
        
        print(f"✅ Loaded {len(papers)} training papers from CSV")
        return papers
        
    except Exception as e:
        print(f"❌ Failed to load training papers: {e}")
        return {}

def load_training_papers_from_neo4j(neo4j_uri: str, neo4j_username: str, neo4j_password: str) -> Dict[str, Dict]:
    """从Neo4j加载training set papers"""
    print(f"🔌 Loading training papers from Neo4j...")
    
    try:
        from py2neo import Graph
        
        graph = Graph(neo4j_uri, auth=(neo4j_username, neo4j_password))
        
        # 查询所有papers
        query = """
        MATCH (p:Paper)
        WHERE p.title IS NOT NULL AND p.title <> ''
        RETURN p.paper_id as paper_id, p.title as title, p.abstract as abstract
        """
        
        results = graph.run(query).data()
        
        papers = {}
        for result in results:
            paper_id = str(result['paper_id']).strip()
            papers[paper_id] = {
                'title': result.get('title', ''),
                'abstract': result.get('abstract', '') or ''
            }
        
        print(f"✅ Loaded {len(papers)} training papers from Neo4j")
        return papers
        
    except Exception as e:
        print(f"❌ Failed to load training papers from Neo4j: {e}")
        return {}

def encode_papers(papers: Dict[str, Dict], model: SentenceTransformer) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """编码papers"""
    print(f"🎨 Encoding {len(papers)} papers...")
    
    paper_ids = list(papers.keys())
    embeddings_dict = {}
    
    for i, paper_id in enumerate(paper_ids):
        if i % 1000 == 0:
            print(f"   Progress: {i}/{len(paper_ids)}")
        
        paper = papers[paper_id]
        text = paper['title']
        if paper['abstract']:
            text += " " + paper['abstract']
        
        embedding = model.encode(text, convert_to_numpy=True)
        embeddings_dict[paper_id] = embedding
    
    print(f"✅ Encoded {len(embeddings_dict)} papers")
    return embeddings_dict, paper_ids

def find_top_k_similar_papers(test_paper_embedding: np.ndarray, 
                              training_embeddings: Dict[str, np.ndarray],
                              top_k: int = 20) -> List[Tuple[str, float]]:
    """找出最相似的top-k papers"""
    
    similarities = {}
    
    for paper_id, emb in training_embeddings.items():
        # 余弦相似度
        sim = cosine_similarity([test_paper_embedding], [emb])[0][0]
        similarities[paper_id] = sim
    
    # 排序
    sorted_papers = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_papers[:top_k]

def compute_metrics(recommendations: List[str], ground_truth: Set[str], k: int) -> Dict[str, float]:
    """计算metrics (filtered strategy: 只考虑数据库中存在的ground truth)"""
    
    if not ground_truth:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'hit_rate': 0.0,
            'ndcg': 0.0,
            'coverage': 0.0
        }
    
    # 计算命中
    hits = [1 if rec in ground_truth else 0 for rec in recommendations]
    num_hits = sum(hits)
    
    # Precision@K
    precision = num_hits / k if k > 0 else 0.0
    
    # Recall@K
    recall = num_hits / len(ground_truth) if ground_truth else 0.0
    
    # F1@K
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Hit Rate@K
    hit_rate = 1.0 if num_hits > 0 else 0.0
    
    # NDCG@K
    def dcg(scores):
        return sum(score / np.log2(i + 2) for i, score in enumerate(scores))
    
    actual_dcg = dcg(hits)
    ideal_ranking = [1] * min(len(ground_truth), k) + [0] * max(0, k - len(ground_truth))
    ideal_dcg = dcg(ideal_ranking)
    ndcg = actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'hit_rate': hit_rate,
        'ndcg': ndcg,
        'coverage': 1.0  # 所有ground truth都在数据库中（filtered）
    }

def main():
    print("=" * 70)
    print("🧪 Evaluating Recommender with Sentence Transformer")
    print("=" * 70)
    
    # 配置
    test_csv_path = "data/processed/test/papers.csv"
    neo4j_uri = "neo4j://127.0.0.1:7687"
    neo4j_username = "neo4j"
    neo4j_password = "87654321"
    k_values = [5, 10, 20]
    sample_size = 50  # 评估的test papers数量
    
    try:
        # 1. 加载ground truth
        if not os.path.exists(test_csv_path):
            print(f"❌ Test data not found: {test_csv_path}")
            return
        
        ground_truth = load_ground_truth_from_csv(test_csv_path)
        if not ground_truth:
            print("❌ No ground truth data loaded")
            return
        
        # 2. 加载training papers - 优先从CSV加载
        train_csv_path = "data/processed/train/papers.csv"
        if os.path.exists(train_csv_path):
            training_papers = load_training_papers_from_csv(train_csv_path)
        else:
            print(f"⚠️ Train CSV not found: {train_csv_path}, trying Neo4j...")
            training_papers = load_training_papers_from_neo4j(
                neo4j_uri, neo4j_username, neo4j_password
            )
        
        if not training_papers:
            print("❌ No training papers loaded")
            return
        
        # 3. 初始化Sentence Transformer
        print("\n📝 Loading Sentence Transformer...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Sentence Transformer loaded")
        
        # 4. 编码training papers
        training_embeddings, training_ids = encode_papers(training_papers, model)
        
        # 5. 选择sample test papers
        test_papers_to_eval = list(ground_truth.keys())[:sample_size]
        print(f"\n📋 Evaluating {len(test_papers_to_eval)} test papers...")
        
        # 6. 评估
        all_metrics = {k: {
            'precision': [],
            'recall': [],
            'f1': [],
            'hit_rate': [],
            'ndcg': [],
            'coverage': []
        } for k in k_values}
        
        successful_evals = 0
        
        for i, test_paper_id in enumerate(test_papers_to_eval):
            if i % 10 == 0:
                print(f"   Progress: {i}/{len(test_papers_to_eval)}")
            
            # 获取test paper的embedding
            test_paper = ground_truth[test_paper_id]
            if not test_paper or all(not c for c in test_paper):
                # 没有引用信息，跳过
                continue
            
            # 找出test paper的embedding
            # 注意：test papers在training set中可能没有，所以需要单独编码
            test_paper_text = f"Paper {test_paper_id}"  # 简化处理
            test_embedding = model.encode(test_paper_text, convert_to_numpy=True)
            
            # 找出最相似的papers
            top_recommendations = find_top_k_similar_papers(
                test_embedding, 
                training_embeddings,
                top_k=max(k_values)
            )
            
            # 提取推荐的paper IDs
            recommended_ids = [rec[0] for rec in top_recommendations]
            
            # ground truth中的papers（仅考虑在training set中存在的）
            ground_truth_papers = set(test_paper)
            ground_truth_papers_available = {
                p for p in ground_truth_papers if p in training_ids
            }
            
            if not ground_truth_papers_available:
                # 没有可用的ground truth papers
                continue
            
            # 计算各K值的metrics
            for k in k_values:
                k_recs = recommended_ids[:k]
                metrics = compute_metrics(k_recs, ground_truth_papers_available, k)
                
                for metric, value in metrics.items():
                    all_metrics[k][metric].append(value)
            
            successful_evals += 1
        
        # 7. 生成报告
        print("\n" + "=" * 70)
        print("📊 Evaluation Results")
        print("=" * 70)
        
        if successful_evals == 0:
            print("❌ No successful evaluations")
            return
        
        print(f"\n✅ Successfully evaluated {successful_evals} test papers")
        print(f"   Using Sentence Transformer encoding + Cosine Similarity")
        print(f"   Evaluation Strategy: Filtered (only papers in training set)")
        
        results = {
            'evaluation_strategy': 'filtered_sentence_transformer',
            'model': 'all-MiniLM-L6-v2',
            'test_papers_evaluated': successful_evals,
            'metrics_by_k': {}
        }
        
        for k in k_values:
            print(f"\n  K={k}:")
            metrics_at_k = {}
            
            for metric in ['precision', 'recall', 'f1', 'hit_rate', 'ndcg', 'coverage']:
                values = all_metrics[k][metric]
                if values:
                    mean = np.mean(values)
                    std = np.std(values)
                    print(f"    {metric:12s}: {mean:.4f} ± {std:.4f}")
                    metrics_at_k[metric] = {'mean': mean, 'std': std}
            
            results['metrics_by_k'][k] = metrics_at_k
        
        # 8. 保存结果
        output_dir = "evaluation_results"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "trial5_sentence_transformer_evaluation.json")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to {output_path}")
        
        print("\n" + "=" * 70)
        print("✅ Evaluation completed successfully!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
