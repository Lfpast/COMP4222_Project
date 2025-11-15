"""
混合推荐系统 (Hybrid Recommender)

结合两种 embeddings：
1. Original Sentence-BERT embeddings (语义相似性)
2. HAN-trained embeddings (图结构相似性)

用法：
    python hybrid_recommender.py
"""

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from py2neo import Graph
import pandas as pd
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class HybridRecommender:
    """混合推荐器 - 结合语义和图结构信息"""
    
    def __init__(self, model_path: str, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        """
        初始化混合推荐器
        
        Args:
            model_path: HAN 模型路径（包含两种 embeddings）
            neo4j_uri: Neo4j 连接 URI
            neo4j_user: Neo4j 用户名
            neo4j_password: Neo4j 密码
        """
        print("=" * 70)
        print("Initializing Hybrid Recommender System")
        print("=" * 70)
        
        # 1. 加载模型和 embeddings
        print("\n[1/4] Loading model and embeddings...")
        checkpoint = torch.load(model_path, map_location='cpu')
        
        self.original_embeddings = checkpoint.get('original_embeddings', {})
        self.han_embeddings = checkpoint.get('embeddings', {})
        self.id_maps = checkpoint['id_maps']
        
        # 创建反向映射（index -> id）
        self.reverse_maps = {
            'paper': {idx: pid for pid, idx in self.id_maps['paper'].items()},
            'author': {idx: aid for aid, idx in self.id_maps['author'].items()},
            'keyword': {idx: kid for kid, idx in self.id_maps['keyword'].items()}
        }
        
        print(f"   ✓ Original embeddings: {list(self.original_embeddings.keys())}")
        print(f"   ✓ HAN embeddings: {list(self.han_embeddings.keys())}")
        print(f"   ✓ Papers: {len(self.id_maps['paper'])}")
        
        # 2. 转换为 numpy 数组
        print("\n[2/4] Converting embeddings to numpy arrays...")
        self.original_paper_emb = self._to_numpy(self.original_embeddings['paper'])
        self.han_paper_emb = self._to_numpy(self.han_embeddings['paper'])
        print(f"   ✓ Original shape: {self.original_paper_emb.shape}")
        print(f"   ✓ HAN shape: {self.han_paper_emb.shape}")
        
        # 3. 加载 Sentence Transformer（用于编码查询）
        print("\n[3/4] Loading Sentence Transformer...")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("   ✓ Model loaded")
        
        # 4. 连接 Neo4j
        print("\n[4/4] Connecting to Neo4j...")
        self.graph = Graph(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.graph.run("RETURN 1")
        print("   ✓ Connected")
        
        print("\n" + "=" * 70)
        print("Hybrid Recommender Ready!")
        print("=" * 70 + "\n")
    
    def _to_numpy(self, tensor):
        """将 tensor 转换为 numpy 数组"""
        if isinstance(tensor, torch.Tensor):
            return tensor.cpu().numpy()
        return np.array(tensor)
    
    def get_paper_metadata(self, paper_ids: List[str]) -> Dict[str, Dict]:
        """从 Neo4j 获取论文元数据"""
        if not paper_ids:
            return {}
        
        formatted_ids = [f"'{str(pid)}'" for pid in paper_ids]
        ids_str = ', '.join(formatted_ids)
        
        query = f"""
        MATCH (p:Paper)
        WHERE p.paper_id IN [{ids_str}]
        RETURN p.paper_id as paper_id, p.title as title,
               p.abstract as abstract, p.year as year,
               p.venue as venue, p.n_citation as citations
        """
        
        results = self.graph.run(query).data()
        
        metadata_map = {}
        for result in results:
            paper_id = str(result['paper_id'])
            metadata_map[paper_id] = {
                'paper_id': paper_id,
                'title': result.get('title', f'Paper {paper_id}'),
                'abstract': result.get('abstract', 'N/A')[:200] + '...',
                'year': result.get('year', 'N/A'),
                'venue': result.get('venue', 'N/A'),
                'citations': result.get('citations', 0)
            }
        
        return metadata_map
    
    def semantic_search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        使用 Original Sentence-BERT embeddings 进行语义搜索
        
        Returns:
            List of (paper_id, score) tuples
        """
        # 编码查询
        query_emb = self.sentence_model.encode([query])
        
        # 计算余弦相似度
        similarities = cosine_similarity(query_emb, self.original_paper_emb)[0]
        
        # 获取 top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            paper_id = self.reverse_maps['paper'].get(idx)
            if paper_id and similarities[idx] > 0:
                results.append((paper_id, float(similarities[idx])))
        
        return results
    
    def structural_search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        使用 HAN embeddings 进行图结构搜索
        
        这会找到在引用网络中结构相似的论文
        
        Returns:
            List of (paper_id, score) tuples
        """
        # 编码查询
        query_emb = self.sentence_model.encode([query])
        
        # HAN embeddings 维度可能不同，需要投影
        if query_emb.shape[1] != self.han_paper_emb.shape[1]:
            # 简单截断或填充到相同维度
            min_dim = min(query_emb.shape[1], self.han_paper_emb.shape[1])
            query_emb = query_emb[:, :min_dim]
            han_emb_proj = self.han_paper_emb[:, :min_dim]
        else:
            han_emb_proj = self.han_paper_emb
        
        # 计算余弦相似度
        similarities = cosine_similarity(query_emb, han_emb_proj)[0]
        
        # 获取 top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            paper_id = self.reverse_maps['paper'].get(idx)
            if paper_id and similarities[idx] > 0:
                results.append((paper_id, float(similarities[idx])))
        
        return results
    
    def hybrid_search(self, query: str, top_k: int = 10, alpha: float = 0.7) -> List[Tuple[str, float, Dict]]:
        """
        混合搜索：结合语义和结构相似性
        
        Args:
            query: 查询文本
            top_k: 返回的论文数量
            alpha: 语义权重 (0-1)，结构权重 = 1-alpha
                  alpha=1.0 -> 纯语义搜索
                  alpha=0.0 -> 纯结构搜索
                  alpha=0.7 -> 70% 语义 + 30% 结构（推荐）
        
        Returns:
            List of (paper_id, combined_score, score_breakdown) tuples
        """
        print(f"\n{'='*70}")
        print(f"Hybrid Search: '{query}'")
        print(f"{'='*70}")
        print(f"Configuration: alpha={alpha:.2f} ({alpha*100:.0f}% semantic + {(1-alpha)*100:.0f}% structural)")
        print()
        
        # 1. 语义搜索
        print("[1/3] Semantic search (Original Sentence-BERT)...")
        semantic_results = self.semantic_search(query, top_k=top_k*2)  # 多取一些候选
        semantic_scores = {pid: score for pid, score in semantic_results}
        print(f"   ✓ Found {len(semantic_results)} semantic matches")
        
        # 2. 结构搜索
        print("[2/3] Structural search (HAN embeddings)...")
        structural_results = self.structural_search(query, top_k=top_k*2)
        structural_scores = {pid: score for pid, score in structural_results}
        print(f"   ✓ Found {len(structural_results)} structural matches")
        
        # DEBUG: 打印structural结果的paper_id类型和示例
        if structural_results:
            sample_struct_id = structural_results[0][0]
            print(f"   DEBUG: Structural paper_id type: {type(sample_struct_id)}, example: {sample_struct_id}")
        if semantic_results:
            sample_sem_id = semantic_results[0][0]
            print(f"   DEBUG: Semantic paper_id type: {type(sample_sem_id)}, example: {sample_sem_id}")
        
        # 3. 合并结果
        print("[3/3] Combining results...")
        all_paper_ids = set(semantic_scores.keys()) | set(structural_scores.keys())
        
        combined = []
        for paper_id in all_paper_ids:
            sem_score = semantic_scores.get(paper_id, 0.0)
            struct_score = structural_scores.get(paper_id, 0.0)
            
            # 加权组合
            combined_score = alpha * sem_score + (1 - alpha) * struct_score
            
            score_breakdown = {
                'semantic': sem_score,
                'structural': struct_score,
                'combined': combined_score
            }
            
            combined.append((paper_id, combined_score, score_breakdown))
        
        # 按组合分数排序
        combined.sort(key=lambda x: x[1], reverse=True)
        
        print(f"   ✓ Combined {len(combined)} unique papers\n")
        
        return combined[:top_k]
    
    def compare_methods(self, query: str, top_k: int = 5):
        """
        对比三种方法的推荐结果
        
        Args:
            query: 查询文本
            top_k: 每种方法返回的论文数量
        """
        print("\n" + "=" * 70)
        print(f"COMPARISON: Different Recommendation Methods")
        print("=" * 70)
        print(f"Query: '{query}'")
        print(f"Top-{top_k} results for each method\n")
        
        # 1. 纯语义
        print("\n" + "-" * 70)
        print("METHOD 1: Pure Semantic (Original Sentence-BERT)")
        print("-" * 70)
        semantic_results = self.semantic_search(query, top_k=top_k)
        paper_ids_sem = [pid for pid, _ in semantic_results]
        metadata_sem = self.get_paper_metadata(paper_ids_sem)
        
        for i, (paper_id, score) in enumerate(semantic_results, 1):
            meta = metadata_sem.get(paper_id, {})
            print(f"\n{i}. [{score:.4f}] {meta.get('title', paper_id)}")
            print(f"   Year: {meta.get('year', 'N/A')} | Citations: {meta.get('citations', 'N/A')}")
        
        # 2. 纯结构
        print("\n\n" + "-" * 70)
        print("METHOD 2: Pure Structural (HAN Embeddings)")
        print("-" * 70)
        structural_results = self.structural_search(query, top_k=top_k)
        paper_ids_struct = [pid for pid, _ in structural_results]
        metadata_struct = self.get_paper_metadata(paper_ids_struct)
        
        for i, (paper_id, score) in enumerate(structural_results, 1):
            meta = metadata_struct.get(paper_id, {})
            print(f"\n{i}. [{score:.4f}] {meta.get('title', paper_id)}")
            print(f"   Year: {meta.get('year', 'N/A')} | Citations: {meta.get('citations', 'N/A')}")
        
        # 3. 混合（70% 语义 + 30% 结构）
        print("\n\n" + "-" * 70)
        print("METHOD 3: Hybrid (70% Semantic + 30% Structural)")
        print("-" * 70)
        hybrid_results = self.hybrid_search(query, top_k=top_k, alpha=0.7)
        paper_ids_hybrid = [pid for pid, _, _ in hybrid_results]
        metadata_hybrid = self.get_paper_metadata(paper_ids_hybrid)
        
        for i, (paper_id, combined_score, breakdown) in enumerate(hybrid_results, 1):
            meta = metadata_hybrid.get(paper_id, {})
            print(f"\n{i}. [Combined: {combined_score:.4f}] {meta.get('title', paper_id)}")
            print(f"   Semantic: {breakdown['semantic']:.4f} | Structural: {breakdown['structural']:.4f}")
            print(f"   Year: {meta.get('year', 'N/A')} | Citations: {meta.get('citations', 'N/A')}")
        
        # 4. 分析差异
        print("\n\n" + "=" * 70)
        print("ANALYSIS: Method Differences")
        print("=" * 70)
        
        set_sem = set(paper_ids_sem)
        set_struct = set(paper_ids_struct)
        set_hybrid = set(paper_ids_hybrid)
        
        print(f"\nOverlap between methods:")
        print(f"  • Semantic ∩ Structural: {len(set_sem & set_struct)} papers")
        print(f"  • Semantic ∩ Hybrid: {len(set_sem & set_hybrid)} papers")
        print(f"  • Structural ∩ Hybrid: {len(set_struct & set_hybrid)} papers")
        print(f"  • All three methods: {len(set_sem & set_struct & set_hybrid)} papers")
        
        print(f"\nUnique to each method:")
        print(f"  • Only in Semantic: {len(set_sem - set_struct - set_hybrid)} papers")
        print(f"  • Only in Structural: {len(set_struct - set_sem - set_hybrid)} papers")
        print(f"  • Only in Hybrid: {len(set_hybrid - set_sem - set_struct)} papers")
        
        print("\n" + "=" * 70 + "\n")


def main():
    """主函数：演示混合推荐系统"""
    
    # 配置
    MODEL_PATH = "../training/models/link_prediction_v3/han_embeddings.pth"
    NEO4J_URI = "neo4j://127.0.0.1:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "12345678"
    
    # 初始化推荐器
    recommender = HybridRecommender(
        model_path=MODEL_PATH,
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PASSWORD
    )
    
    # 测试查询
    test_queries = [
        "Graph Neural Networks for recommender systems",
        "Attention mechanisms in deep learning",
        "Link prediction in social networks"
    ]
    
    for query in test_queries:
        recommender.compare_methods(query, top_k=5)
        input("\nPress Enter to continue to next query...\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
