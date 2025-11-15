# [file name]: recommender_system.py
import torch
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import networkx as nx
from py2neo import Graph
import dgl
import json
import os
from typing import List, Dict, Tuple, Any
from sentence_transformers import SentenceTransformer
import heapq
from collections import defaultdict
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

class AcademicRecommender:
    def __init__(self, 
                 model_path=r"training\models\trial5\han_embeddings.pth", 
                 neo4j_uri="neo4j://127.0.0.1:7687",
                 neo4j_username="neo4j",
                 neo4j_password="87654321"):
        """初始化学术推荐系统"""
        self.device = torch.device('cuda')
        
        # 加载训练好的嵌入（封装到 helper，支持容错与缓存）
        self._emb_np_cache = {}
        print("📥 Loading trained embeddings...")
        try:
            self._load_checkpoint(model_path)
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
        
        # 连接Neo4j获取元数据
        print("🔌 Connecting to Neo4j...")
        try:
            self.graph_db = Graph(neo4j_uri, auth=(neo4j_username, neo4j_password))
            # 测试连接
            self.graph_db.run("RETURN 1")
            print("✅ Neo4j connection successful")
        except Exception as e:
            print(f"❌ Failed to connect to Neo4j: {e}")
            # 创建空的图数据库连接，但标记为不可用
            self.graph_db = None
        
        # 初始化文本编码器
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Sentence transformer loaded")
        except Exception as e:
            print(f"❌ Failed to load sentence transformer: {e}")
            self.sentence_model = None
        
        # 构建反向映射
        self.reverse_maps = {}
        for node_type, id_map in self.id_maps.items():
            self.reverse_maps[node_type] = {v: k for k, v in id_map.items()}
        
        print("✅ Recommender system initialized!")

    def _load_checkpoint(self, model_path: str):
        """加载 checkpoint，并准备 embeddings 的 numpy 缓存"""
        # 支持多种相对/绝对路径尝试
        tried = []
        paths_to_try = [model_path]
        # 常见可能的相对位置
        paths_to_try.extend([
            os.path.join(os.path.dirname(__file__), model_path),
            os.path.join(os.path.dirname(__file__), '..', model_path),
        ])

        checkpoint = None
        for p in paths_to_try:
            if not p:
                continue
            tried.append(p)
            try:
                if os.path.exists(p):
                    checkpoint = torch.load(p, map_location=self.device)
                    model_path = p
                    break
            except Exception:
                continue

        if checkpoint is None:
            # 最后尝试直接 load（让 torch 抛出更详细异常）
            checkpoint = torch.load(model_path, map_location=self.device)

        self.embeddings = checkpoint['embeddings']
        self.id_maps = checkpoint['id_maps']
        self.config = checkpoint.get('config', {})

        # 预填充 numpy 缓存（延迟转换：只缓存存在的类型）
        for ntype in self.embeddings.keys():
            try:
                arr = self.embeddings[ntype]
                # 支持 torch.Tensor 或 numpy.ndarray
                if hasattr(arr, 'numpy'):
                    self._emb_np_cache[ntype] = arr.cpu().numpy()
                elif isinstance(arr, np.ndarray):
                    self._emb_np_cache[ntype] = arr
                else:
                    # 尝试转换为 ndarray
                    self._emb_np_cache[ntype] = np.array(arr)
            except Exception:
                # 忽略不支持的类型，按需生成
                pass

        print(f"✅ Loaded embeddings from: {model_path}; types: {list(self.embeddings.keys())}")

    def _get_numpy_emb(self, node_type: str):
        """返回指定 node_type 的 numpy 嵌入数组（缓存）"""
        if node_type in self._emb_np_cache:
            return self._emb_np_cache[node_type]

        if node_type in self.embeddings:
            arr = self.embeddings[node_type]
            if hasattr(arr, 'cpu') and hasattr(arr, 'numpy'):
                try:
                    np_arr = arr.cpu().numpy()
                except Exception:
                    np_arr = np.array(arr)
            else:
                np_arr = np.array(arr)

            self._emb_np_cache[node_type] = np_arr
            return np_arr

        raise KeyError(f"Embedding for node type '{node_type}' not found")

    def _safe_neo4j_query(self, query):
        """安全的Neo4j查询执行"""
        if self.graph_db is None:
            return []
        try:
            return self.graph_db.run(query).data()
        except Exception as e:
            print(f"⚠️ Neo4j query failed: {e}")
            return []
    
    def get_paper_metadata(self, paper_ids: List[str]) -> Dict[str, Any]:
        """改进的元数据获取 - 处理各种ID格式"""
        if not paper_ids or self.graph_db is None:
            return self._get_enhanced_fallback_metadata(paper_ids)

        try:
            # 使用字符串 ID 查询（更鲁棒，支持字符串或数字形式的 ID）
            batch_size = 50
            all_results = []

            for i in range(0, len(paper_ids), batch_size):
                batch = paper_ids[i:i + batch_size]
                # 格式化为带引号的字符串列表，确保 cypher 可识别字符串 id
                formatted_ids = [f"'{str(pid)}'" for pid in batch]
                ids_str = ', '.join(formatted_ids)

                query = f"""
                MATCH (p:Paper)
                WHERE p.paper_id IN [{ids_str}]
                RETURN p.paper_id as paper_id, p.title as title,
                    p.abstract as abstract, p.year as year,
                    p.venue as venue, p.n_citation as citations
                """

                batch_results = self._safe_neo4j_query(query)
                all_results.extend(batch_results)

            # 创建元数据映射
            metadata_map = {}
            for result in all_results:
                paper_id = result['paper_id']
                metadata_map[str(paper_id)] = {
                    'paper_id': str(paper_id),
                    'title': result.get('title', f'Paper {paper_id}'),
                    'abstract': result.get('abstract', 'Abstract not available'),
                    'year': result.get('year', 'Unknown'),
                    'venue': result.get('venue', 'Unknown'),
                    'citation_count': result.get('citations', 0)
                }

            # 为缺失的论文添加备用元数据
            for pid in paper_ids:
                if str(pid) not in metadata_map:
                    metadata_map[str(pid)] = self._create_fallback_metadata(pid)

            return metadata_map

        except Exception as e:
            print(f"❌ Failed to get paper metadata: {e}")
            return self._get_enhanced_fallback_metadata(paper_ids)

    def _create_fallback_metadata(self, pid):
        """创建备用元数据"""
        pid_str = str(pid)
        return {
            'paper_id': pid_str,
            'title': f"Research Paper {pid_str}",
            'abstract': "Abstract not available in current database.",
            'year': "2023", 
            'venue': "Academic Conference",
            'citation_count': 0,
            'is_fallback': True
        }

    def _get_enhanced_fallback_metadata(self, paper_ids: List[str]) -> Dict[str, Any]:
        """增强的备用元数据"""
        metadata = {}
        for pid in paper_ids:
            if pid is not None:
                pid_str = str(pid)
                metadata[pid_str] = self._create_fallback_metadata(pid)
        return metadata
    
    def get_author_metadata(self, author_ids: List[str]) -> Dict[str, Any]:
        """从Neo4j获取作者元数据"""
        if not author_ids or self.graph_db is None:
            return {}
        
        try:
            # 处理ID格式
            formatted_ids = [f"'{aid}'" for aid in author_ids if aid]
            if not formatted_ids:
                return {}
                
            ids_str = ', '.join(formatted_ids)
            
            query = f"""
            MATCH (a:Author)
            WHERE a.author_id IN [{ids_str}]
            RETURN a.author_id as author_id, a.name as name
            """
            results = self._safe_neo4j_query(query)
            return {result['author_id']: dict(result) for result in results}
        except Exception as e:
            print(f"⚠️ Failed to get author metadata: {e}")
            return {}
    
    def _diversify_recommendations(self, recommendations: List[Dict], top_k: int) -> List[Dict]:
        """多样性重排序策略"""
        if len(recommendations) <= top_k:
            return recommendations
        
        # 按venue分组确保主题多样性
        venue_groups = defaultdict(list)
        for rec in recommendations:
            venue = rec.get('venue', 'Unknown')
            venue_groups[venue].append(rec)
        
        diversified = []
        max_per_venue = max(1, top_k // len(venue_groups))
        
        # 从每个主题组中选择代表性论文
        for venue, group in venue_groups.items():
            # 按分数排序并选择前几个
            group_sorted = sorted(group, key=lambda x: x.get('similarity_score', 0), reverse=True)
            diversified.extend(group_sorted[:max_per_venue])
        
        # 如果多样性策略导致结果不足，用最高分补足
        if len(diversified) < top_k:
            remaining = [r for r in recommendations if r not in diversified]
            remaining_sorted = sorted(remaining, key=lambda x: x.get('similarity_score', 0), reverse=True)
            diversified.extend(remaining_sorted[:top_k - len(diversified)])
        
        # 重新分配排名
        for i, rec in enumerate(diversified[:top_k]):
            rec['rank'] = i + 1
            rec['diversity_boost'] = True  # 标记经过多样性优化
        
        return diversified[:top_k]
    
    def content_based_paper_recommendation(self, query_text: str, top_k: int = 10) -> List[Dict]:
        """基于内容的论文推荐 - 改进投影方法"""
        print(f"📚 Content-based paper recommendation for: {query_text}")
        
        if self.sentence_model is None:
            print("❌ Sentence transformer not available")
            return []
        
        # 编码查询文本
        try:
            query_embedding = self.sentence_model.encode([query_text])
            print(f"   Query embedding shape: {query_embedding.shape}")
        except Exception as e:
            print(f"❌ Failed to encode query: {e}")
            return []
        
        # 获取论文嵌入
        if 'paper' not in self.embeddings:
            print("❌ Paper embeddings not found")
            return []
        paper_embeddings = self._get_numpy_emb('paper')
        print(f"   Paper embeddings shape: {paper_embeddings.shape}")

        # 改进的投影方法
        if query_embedding.shape[1] != paper_embeddings.shape[1]:
            print(f"⚠️ Dimension mismatch: query {query_embedding.shape[1]}D vs paper {paper_embeddings.shape[1]}D")
            print("   Using improved projection...")

            # 方法1: 使用PCA进行更好的投影
            from sklearn.decomposition import PCA

            try:
                # 如果查询维度更高，使用PCA降维
                if query_embedding.shape[1] > paper_embeddings.shape[1]:
                    pca = PCA(n_components=paper_embeddings.shape[1])
                    # 使用论文嵌入来拟合PCA（模拟在相同空间）
                    pca.fit(paper_embeddings)
                    query_projected = pca.transform(query_embedding)
                    paper_projected = paper_embeddings
                else:
                    # 如果论文维度更高，提升查询维度
                    query_projected = np.pad(query_embedding,
                                        ((0,0), (0, paper_embeddings.shape[1] - query_embedding.shape[1])),
                                        mode='constant')
                    paper_projected = paper_embeddings

                # 计算相似度
                query_norm = query_projected / np.linalg.norm(query_projected, axis=1, keepdims=True)
                paper_norm = paper_projected / np.linalg.norm(paper_projected, axis=1, keepdims=True)
                similarities = np.dot(query_norm, paper_norm.T)[0]

            except Exception as e:
                print(f"⚠️ PCA projection failed, using simple truncation: {e}")
                # 回退到简单的截断方法
                min_dim = min(query_embedding.shape[1], paper_embeddings.shape[1])
                query_projected = query_embedding[:, :min_dim]
                paper_projected = paper_embeddings[:, :min_dim]

                query_norm = query_projected / np.linalg.norm(query_projected, axis=1, keepdims=True)
                paper_norm = paper_projected / np.linalg.norm(paper_projected, axis=1, keepdims=True)
                similarities = np.dot(query_norm, paper_norm.T)[0]
        else:
            # 维度匹配，正常计算；优先使用矢量化点积
            try:
                q = np.array(query_embedding)
                p = paper_embeddings
                qn = q / np.linalg.norm(q, axis=1, keepdims=True)
                pn = p / np.linalg.norm(p, axis=1, keepdims=True)
                similarities = (qn @ pn.T)[0]
            except Exception:
                similarities = cosine_similarity(query_embedding, paper_embeddings)[0]
        
        # 获取top-K推荐
        paper_ids = [self.reverse_maps['paper'].get(i) for i in range(len(paper_embeddings))]
        valid_indices = [i for i, pid in enumerate(paper_ids) if pid is not None and similarities[i] > 0]
        
        if not valid_indices:
            print("❌ No valid recommendations found")
            return []
        
        # 按相似度排序
        top_indices = sorted(valid_indices, key=lambda i: similarities[i], reverse=True)[:top_k]
        
        recommendations = []
        for rank, idx in enumerate(top_indices, 1):
            paper_id = paper_ids[idx]
            recommendations.append({
                'paper_id': paper_id,
                'similarity_score': float(similarities[idx]),
                'rank': rank
            })
        
        # 改进元数据获取
        print("   Fetching paper metadata...")
        paper_metadata = self.get_paper_metadata([rec['paper_id'] for rec in recommendations])
        
        for rec in recommendations:
            metadata = paper_metadata.get(rec['paper_id'], {})
            rec.update(metadata)
            if 'title' not in rec or not rec['title']:
                rec['title'] = f"Paper {rec['paper_id']}"
        
        print(f"   ✅ Generated {len(recommendations)} recommendations with metadata")
        for rec in recommendations[:3]:
            print(f"      {rec['rank']}. {rec['title']} (score: {rec['similarity_score']:.3f})")
        
        return recommendations
    
    def collaborative_paper_recommendation(self, target_paper_id: str, top_k: int = 10) -> List[Dict]:
        """基于协同过滤的论文推荐（使用图结构）- 修复版本"""
        print(f"🔗 Collaborative paper recommendation for: {target_paper_id}")
        
        # 添加详细的调试信息
        print(f"   Checking if target paper exists in embeddings...")
        print(f"   Available paper IDs in id_maps: {len(self.id_maps['paper'])}")
        
        if 'paper' not in self.id_maps:
            print("❌ Paper id_maps not found")
            return []
        
        if target_paper_id not in self.id_maps['paper']:
            print(f"⚠️ Target paper {target_paper_id} not found in embeddings")
            print(f"   Sample paper IDs: {list(self.id_maps['paper'].keys())[:5]}")
            return []
        
        target_idx = self.id_maps['paper'][target_paper_id]
        paper_embeddings = self._get_numpy_emb('paper')

        print(f"   Target paper index: {target_idx}")
        print(f"   Paper embeddings shape: {paper_embeddings.shape}")

        # 计算与目标论文的相似度（使用向量化点积以提高速度）
        try:
            print(f"   Calculating cosine similarities...")
            t = paper_embeddings[target_idx:target_idx+1]
            if np.isnan(t).any() or np.isinf(t).any():
                print("❌ Target embedding contains NaN or Inf values")
                return []

            t_norm = t / np.linalg.norm(t, axis=1, keepdims=True)
            emb_norm = paper_embeddings / np.linalg.norm(paper_embeddings, axis=1, keepdims=True)
            similarities = (t_norm @ emb_norm.T)[0]

            print(f"   Similarities range: {similarities.min():.3f} to {similarities.max():.3f}")
            print(f"   Number of positive similarities: {np.sum(similarities > 0)}")

        except Exception as e:
            print(f"❌ Failed to calculate similarities: {e}")
            return []
        
        # 排除目标论文本身
        similarities[target_idx] = -1
        
        # 获取top-K推荐 - 改进选择逻辑
        paper_ids = [self.reverse_maps['paper'].get(i) for i in range(len(paper_embeddings))]
        
        # 只选择有效的论文ID和正相似度
        valid_indices = []
        for i in range(len(paper_ids)):
            if (paper_ids[i] is not None and 
                similarities[i] > 0.1 and  # 设置相似度阈值，避免推荐不相关的论文
                i != target_idx):
                valid_indices.append(i)
        
        print(f"   Valid candidate papers: {len(valid_indices)}")
        
        if not valid_indices:
            print("⚠️ No valid recommendations found (all similarities <= 0.1)")
            return []
        
        # 按相似度排序并选择top-K
        valid_indices_sorted = sorted(valid_indices, key=lambda i: similarities[i], reverse=True)[:top_k]
        
        recommendations = []
        for rank, idx in enumerate(valid_indices_sorted, 1):
            paper_id = paper_ids[idx]
            recommendations.append({
                'paper_id': paper_id,
                'similarity_score': float(similarities[idx]),
                'rank': rank
            })
        
        print(f"   Generated {len(recommendations)} collaborative recommendations")
        
        # 获取元数据
        if recommendations:
            print(f"   Fetching metadata for {len(recommendations)} recommendations...")
            recommendation_ids = [rec['paper_id'] for rec in recommendations]
            paper_metadata = self.get_paper_metadata(recommendation_ids)
            print(f"   Retrieved metadata for {len(paper_metadata)} papers")
        
        for rec in recommendations:
            # 使用字符串ID查找元数据
            paper_id_str = str(rec['paper_id'])
            metadata = paper_metadata.get(paper_id_str, {})
            
            # 如果从Neo4j获取到元数据，使用它；否则使用备用元数据
            if metadata and metadata.get('title'):
                rec.update(metadata)
            else:
                # 使用备用标题
                rec['title'] = f"Research Paper {paper_id_str}"
                rec['year'] = "Unknown"
                rec['venue'] = "Unknown"
        
        # 显示最终推荐结果
        print(f"   Final recommendations with metadata:")
        for rec in recommendations[:3]:
            title = rec.get('title', 'Unknown')
            score = rec['similarity_score']
            print(f"      {rec['rank']}. {title} (score: {score:.3f})")
        
        return recommendations
    
    def author_based_recommendation(self, author_id: str, top_k: int = 10) -> List[Dict]:
        """基于作者相似性的论文推荐"""
        print(f"👤 Author-based recommendation for author: {author_id}")
        
        if 'author' not in self.id_maps or author_id not in self.id_maps['author']:
            print(f"⚠️ Author {author_id} not found in embeddings")
            return []
        
        # 获取作者嵌入
        author_embeddings = self._get_numpy_emb('author')
        target_idx = self.id_maps['author'][author_id]

        # 找到相似作者（向量化）
        try:
            a = author_embeddings[target_idx:target_idx+1]
            a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
            emb_norm = author_embeddings / np.linalg.norm(author_embeddings, axis=1, keepdims=True)
            author_similarities = (a_norm @ emb_norm.T)[0]
        except Exception as e:
            print(f"❌ Failed to calculate author similarities: {e}")
            return []
        
        # 排除目标作者
        author_similarities[target_idx] = -1
        
        # 获取相似作者的论文
        similar_author_indices = np.argsort(author_similarities)[-5:][::-1]  # Top 5相似作者
        similar_author_ids = [self.reverse_maps['author'].get(i) for i in similar_author_indices]
        similar_author_ids = [aid for aid in similar_author_ids if aid is not None]
        
        if not similar_author_ids or self.graph_db is None:
            return []
        
        # 从Neo4j获取这些作者的论文
        try:
            formatted_ids = [f"'{aid}'" for aid in similar_author_ids]
            ids_str = ', '.join(formatted_ids)
            
            query = f"""
            MATCH (a:Author)-[:WRITTEN_BY]-(p:Paper)
            WHERE a.author_id IN [{ids_str}]
            RETURN a.author_id as author_id, p.paper_id as paper_id,
                   p.title as title, p.year as year, p.venue as venue
            ORDER BY p.year DESC
            LIMIT {top_k * 2}
            """
            results = self._safe_neo4j_query(query)
        except Exception as e:
            print(f"❌ Failed to query author papers: {e}")
            return []
        
        recommendations = []
        seen_papers = set()
        
        for result in results:
            paper_id = result['paper_id']
            if paper_id not in seen_papers:
                author_id_used = result['author_id']
                author_idx = self.id_maps['author'].get(author_id_used)
                author_sim = author_similarities[author_idx] if author_idx is not None else 0
                
                recommendations.append({
                    'paper_id': paper_id,
                    'title': result.get('title', 'Unknown'),
                    'year': result.get('year', 'Unknown'),
                    'venue': result.get('venue', 'Unknown'),
                    'recommended_by_author': author_id_used,
                    'similar_author_score': float(author_sim),
                    'rank': len(recommendations) + 1
                })
                seen_papers.add(paper_id)
            
            if len(recommendations) >= top_k:
                break
        
        return recommendations
    
    def hybrid_paper_recommendation(self, query: str = None, target_paper_id: str = None, 
                                  author_id: str = None, top_k: int = 10) -> Dict[str, Any]:
        """混合推荐：结合内容、协同过滤和作者信息"""
        print("🔄 Generating hybrid recommendations...")
        
        all_recommendations = []
        
        # 1. 基于内容的推荐
        if query:
            content_recs = self.content_based_paper_recommendation(query, top_k*2)
            for rec in content_recs:
                rec['method'] = 'content_based'
                rec['final_score'] = rec['similarity_score'] * 0.4
            all_recommendations.extend(content_recs)
        
        # 2. 基于协同过滤的推荐
        if target_paper_id:
            collab_recs = self.collaborative_paper_recommendation(target_paper_id, top_k*2)
            for rec in collab_recs:
                rec['method'] = 'collaborative'
                rec['final_score'] = rec['similarity_score'] * 0.4
            all_recommendations.extend(collab_recs)
        
        # 3. 基于作者的推荐
        if author_id:
            author_recs = self.author_based_recommendation(author_id, top_k*2)
            for rec in author_recs:
                rec['method'] = 'author_based'
                rec['final_score'] = rec['similar_author_score'] * 0.2
            all_recommendations.extend(author_recs)
        
        # 如果没有推荐方法，使用基于内容的默认推荐
        if not all_recommendations and query:
            content_recs = self.content_based_paper_recommendation(query, top_k)
            for rec in content_recs:
                rec['method'] = 'content_based'
                rec['final_score'] = rec['similarity_score']
            all_recommendations.extend(content_recs)
        
        # 合并和重排序
        paper_scores = defaultdict(float)
        paper_details = {}
        
        for rec in all_recommendations:
            paper_id = rec['paper_id']
            paper_scores[paper_id] += rec['final_score']
            if paper_id not in paper_details:
                paper_details[paper_id] = rec
        
        # 获取top-K推荐
        top_papers = heapq.nlargest(top_k, paper_scores.items(), key=lambda x: x[1])
        
        final_recommendations = []
        for paper_id, score in top_papers:
            rec = paper_details[paper_id].copy()
            rec['final_score'] = score
            rec['rank'] = len(final_recommendations) + 1
            final_recommendations.append(rec)
        
        return {
            'recommendations': final_recommendations,
            'query': query,
            'target_paper': target_paper_id,
            'target_author': author_id,
            'total_recommendations': len(final_recommendations)
        }

    def enhanced_collaborative_recommendation(self, target_paper_id: str, top_k: int = 10) -> List[Dict]:
        """增强的协同过滤推荐 - 解决数据稀疏性问题"""
        print(f"🔗 Enhanced collaborative recommendation for: {target_paper_id}")
        
        if 'paper' not in self.id_maps or target_paper_id not in self.id_maps['paper']:
            print(f"⚠️ Target paper {target_paper_id} not found")
            return []
        
        target_idx = self.id_maps['paper'][target_paper_id]
        paper_embeddings = self._get_numpy_emb('paper')

        # 方法1: 基于嵌入的相似度（向量化点积）
        try:
            t = paper_embeddings[target_idx:target_idx+1]
            t_norm = t / np.linalg.norm(t, axis=1, keepdims=True)
            emb_norm = paper_embeddings / np.linalg.norm(paper_embeddings, axis=1, keepdims=True)
            similarities = (t_norm @ emb_norm.T)[0]
        except Exception as e:
            print(f"❌ Embedding similarity failed: {e}")
            return []
        
        # 方法2: 基于图结构的相似度（如果可用）
        graph_similarities = self._calculate_graph_similarity(target_paper_id)
        
        # 融合两种相似度
        final_similarities = similarities.copy()
        if graph_similarities:
            for paper_id, graph_sim in graph_similarities.items():
                if paper_id in self.id_maps['paper']:
                    idx = self.id_maps['paper'][paper_id]
                    # 加权融合：嵌入相似度70%，图相似度30%
                    final_similarities[idx] = 0.7 * similarities[idx] + 0.3 * graph_sim
        
        # 排除目标论文
        final_similarities[target_idx] = -1
        
        # 获取推荐
        paper_ids = [self.reverse_maps['paper'].get(i) for i in range(len(paper_embeddings))]
        
        valid_indices = []
        for i, pid in enumerate(paper_ids):
            if (pid is not None and 
                pid != target_paper_id and 
                final_similarities[i] > 0.05):  # 提高相似度阈值
                valid_indices.append(i)
        
        if not valid_indices:
            print("⚠️ No valid recommendations found")
            return []
        
        # 排序并选择top-K
        top_indices = sorted(valid_indices, key=lambda i: final_similarities[i], reverse=True)[:top_k]
        
        recommendations = []
        for rank, idx in enumerate(top_indices, 1):
            paper_id = paper_ids[idx]
            recommendations.append({
                'paper_id': paper_id,
                'similarity_score': float(final_similarities[idx]),
                'embedding_similarity': float(similarities[idx]),
                'graph_similarity': graph_similarities.get(paper_id, 0.0),
                'rank': rank
            })
        
        # 获取元数据
        paper_metadata = self.get_paper_metadata([rec['paper_id'] for rec in recommendations])
        for rec in recommendations:
            metadata = paper_metadata.get(str(rec['paper_id']), {})
            rec.update(metadata)
        
        print(f"✅ Enhanced collaborative: {len(recommendations)} recommendations")
        return recommendations

    def _calculate_graph_similarity(self, target_paper_id: str) -> Dict[str, float]:
        """基于图结构计算论文相似度"""
        if self.graph_db is None:
            return {}
        
        try:
            # 查询共同引用、共同作者等图关系
            query = """
            MATCH (p1:Paper {paper_id: $target_id})
            OPTIONAL MATCH (p1)-[:CITES]-(common:Paper)-[:CITES]-(p2:Paper)
            WHERE p2.paper_id <> $target_id
            WITH p2, COUNT(common) as common_citations
            OPTIONAL MATCH (p1)-[:WRITTEN_BY]-(a:Author)-[:WRITTEN_BY]-(p2)
            WITH p2, common_citations, COUNT(a) as common_authors
            RETURN p2.paper_id as paper_id, 
                (common_citations * 0.6 + common_authors * 0.4) as graph_similarity
            ORDER BY graph_similarity DESC
            LIMIT 20
            """
            
            results = self.graph_db.run(query, target_id=str(target_paper_id)).data()
            return {str(result['paper_id']): float(result['graph_similarity']) for result in results}
            
        except Exception as e:
            print(f"⚠️ Graph similarity calculation failed: {e}")
            return {}
        
    def optimized_hybrid_recommendation(self, query: str = None, target_paper_id: str = None, 
                                    author_id: str = None, top_k: int = 10) -> Dict[str, Any]:
        """优化的混合推荐 - 动态权重调整"""
        print("🔄 Running optimized hybrid recommendation...")
        
        all_recommendations = []
        method_weights = self._calculate_adaptive_weights(query, target_paper_id, author_id)
        
        print(f"   Dynamic weights: {method_weights}")
        
        # 1. 基于内容的推荐
        if query and method_weights['content'] > 0:
            content_recs = self.content_based_paper_recommendation(query, top_k*3)
            for rec in content_recs:
                rec['method'] = 'content_based'
                rec['final_score'] = rec['similarity_score'] * method_weights['content']
                rec['method_weight'] = method_weights['content']
            all_recommendations.extend(content_recs)
        
        # 2. 增强的协同过滤推荐
        if target_paper_id and method_weights['collaborative'] > 0:
            collab_recs = self.enhanced_collaborative_recommendation(target_paper_id, top_k*3)
            for rec in collab_recs:
                rec['method'] = 'collaborative_enhanced'
                rec['final_score'] = rec['similarity_score'] * method_weights['collaborative']
                rec['method_weight'] = method_weights['collaborative']
            all_recommendations.extend(collab_recs)
        
        # 3. 基于作者的推荐
        if author_id and method_weights['author'] > 0:
            author_recs = self.author_based_recommendation(author_id, top_k*2)
            for rec in author_recs:
                rec['method'] = 'author_based'
                rec['final_score'] = rec['similar_author_score'] * method_weights['author']
                rec['method_weight'] = method_weights['author']
            all_recommendations.extend(author_recs)
        
        # 多样性重排序
        final_recommendations = self._diversified_reranking(all_recommendations, top_k)
        
        return {
            'recommendations': final_recommendations,
            'query': query,
            'target_paper': target_paper_id,
            'target_author': author_id,
            'method_weights': method_weights,
            'total_recommendations': len(final_recommendations)
        }

    def _calculate_adaptive_weights(self, query: str = None, target_paper_id: str = None, 
                              author_id: str = None) -> Dict[str, float]:
        """自适应权重计算"""
        weights = {'content': 0.0, 'collaborative': 0.0, 'author': 0.0}
        
        # 基于输入质量调整权重
        if query and len(query.strip()) > 10:  # 查询较长，内容权重更高
            weights['content'] += 0.5
        elif query:
            weights['content'] += 0.3
        
        if target_paper_id:
            # 检查目标论文是否有足够的连接
            connection_strength = self._evaluate_paper_connections(target_paper_id)
            weights['collaborative'] += 0.3 + (0.2 * connection_strength)
        
        if author_id:
            # 检查作者的活跃度
            author_activity = self._evaluate_author_activity(author_id)
            weights['author'] += 0.2 + (0.1 * author_activity)
        
        # 归一化
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        else:
            # 默认权重
            weights = {'content': 0.5, 'collaborative': 0.3, 'author': 0.2}
        
        return weights

    def _evaluate_paper_connections(self, paper_id: str) -> float:
        """评估论文的连接强度"""
        if self.graph_db is None:
            return 0.5  # 默认中等强度
        
        try:
            query = """
            MATCH (p:Paper {paper_id: $paper_id})
            OPTIONAL MATCH (p)-[:CITES]-(cited)
            OPTIONAL MATCH (p)-[:WRITTEN_BY]-(authors)
            WITH COUNT(DISTINCT cited) as citation_count, 
                COUNT(DISTINCT authors) as author_count
            RETURN (citation_count * 0.7 + author_count * 0.3) as connection_strength
            """
            result = self.graph_db.run(query, paper_id=str(paper_id)).data()
            if result and result[0]['connection_strength']:
                return min(1.0, result[0]['connection_strength'] / 10.0)  # 归一化
        except:
            pass
        
        return 0.5

    def _evaluate_author_activity(self, author_id: str) -> float:
        """评估作者活跃度"""
        if self.graph_db is None:
            return 0.5
        
        try:
            query = """
            MATCH (a:Author {author_id: $author_id})-[:WRITTEN_BY]-(p:Paper)
            WITH COUNT(p) as paper_count
            RETURN CASE 
                WHEN paper_count > 10 THEN 1.0
                WHEN paper_count > 5 THEN 0.7
                WHEN paper_count > 2 THEN 0.4
                ELSE 0.2
            END as activity_level
            """
            result = self.graph_db.run(query, author_id=author_id).data()
            if result:
                return result[0]['activity_level']
        except:
            pass
        
        return 0.5

    def _diversified_reranking(self, recommendations, top_k):
        """多样性重排序"""
        if not recommendations:
            return []
        
        # 按分数排序
        sorted_by_score = sorted(recommendations, key=lambda x: x.get('final_score', 0), reverse=True)
        
        # 确保方法多样性
        final_recs = []
        method_count = {'content_based': 0, 'collaborative_enhanced': 0, 'author_based': 0}
        max_per_method = max(1, top_k // 3)  # 每种方法最多占1/3
        
        for rec in sorted_by_score:
            method = rec.get('method', 'unknown')
            if method_count.get(method, 0) < max_per_method:
                final_recs.append(rec)
                method_count[method] = method_count.get(method, 0) + 1
            
            if len(final_recs) >= top_k:
                break
        
        # 如果多样性限制导致结果不足，用最高分补足
        if len(final_recs) < top_k:
            for rec in sorted_by_score:
                if rec not in final_recs:
                    final_recs.append(rec)
                if len(final_recs) >= top_k:
                    break
        
        # 重新分配排名
        for i, rec in enumerate(final_recs):
            rec['rank'] = i + 1
        
        return final_recs

    def _generate_recommendations_from_scores(self, scores, target_idx, top_k, method):
        """从分数生成推荐结果"""
        paper_ids = [self.reverse_maps['paper'].get(i) for i in range(len(scores))]
        
        valid_indices = []
        for i, pid in enumerate(paper_ids):
            if (pid is not None and 
                i != target_idx and 
                scores[i] > 0.05):
                valid_indices.append(i)
        
        if not valid_indices:
            return []
        
        top_indices = sorted(valid_indices, key=lambda i: scores[i], reverse=True)[:top_k]
        
        recommendations = []
        for rank, idx in enumerate(top_indices, 1):
            paper_id = paper_ids[idx]
            recommendations.append({
                'paper_id': paper_id,
                'similarity_score': float(scores[idx]),
                'rank': rank,
                'method': method
            })
        
        # 获取元数据
        paper_metadata = self.get_paper_metadata([rec['paper_id'] for rec in recommendations])
        for rec in recommendations:
            metadata = paper_metadata.get(str(rec['paper_id']), {})
            rec.update(metadata)
        
        return recommendations

class CollaboratorRecommender:
    def __init__(self, 
                 model_path=r"training\models\trial5\han_embeddings.pth",
                 neo4j_uri="neo4j://127.0.0.1:7687",
                 neo4j_username="neo4j", 
                 neo4j_password="12345678"):
        """初始化合作者推荐系统"""
        self.device = torch.device('cpu')
        
        # 加载嵌入
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.embeddings = checkpoint['embeddings']
            self.id_maps = checkpoint['id_maps']
            print(f"✅ Loaded collaborator embeddings: {[f'{k}: {v.shape}' for k, v in self.embeddings.items() if 'author' in k]}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
        
        # 连接Neo4j
        try:
            self.graph_db = Graph(neo4j_uri, auth=(neo4j_username, neo4j_password))
            self.graph_db.run("RETURN 1")
            print("✅ Neo4j connection successful")
        except Exception as e:
            print(f"❌ Failed to connect to Neo4j: {e}")
            self.graph_db = None
        
        # 构建反向映射
        self.reverse_maps = {}
        for node_type, id_map in self.id_maps.items():
            self.reverse_maps[node_type] = {v: k for k, v in id_map.items()}
        
        # 构建合作网络
        self._build_collaboration_network()
    
    def _safe_neo4j_query(self, query):
        """安全的Neo4j查询执行"""
        if self.graph_db is None:
            return []
        try:
            return self.graph_db.run(query).data()
        except Exception as e:
            print(f"⚠️ Neo4j query failed: {e}")
            return []
    
    def _build_collaboration_network(self):
        """构建作者合作网络"""
        print("🔨 Building collaboration network...")
        
        import time
        start_time = time.time()
        
        try:
            # 方法1: 使用更简单的查询，限制数据量
            query = """
            MATCH (a1:Author)-[:WRITTEN_BY]-(p:Paper)-[:WRITTEN_BY]-(a2:Author)
            WHERE a1.author_id <> a2.author_id
            WITH a1, a2, COUNT(p) as collaboration_count
            WHERE collaboration_count >= 1  // 至少合作过一次
            RETURN a1.author_id as author1, a2.author_id as author2, collaboration_count
            LIMIT 5000  // 限制数量避免超时
            """
            
            print("   Executing optimized Neo4j query...")
            results = self._safe_neo4j_query(query)
            
            if not results:
                print("⚠️ No collaboration data found in Neo4j, creating empty network")
                self.collab_network = nx.Graph()
                return
            
            print(f"   Retrieved {len(results)} collaboration records")
            
            # 创建合作网络
            self.collab_network = nx.Graph()
            
            # 快速构建网络，不显示进度条
            for result in results:
                author1 = result['author1']
                author2 = result['author2']
                count = result['collaboration_count']
                
                self.collab_network.add_edge(author1, author2, weight=count)
            
            elapsed_time = time.time() - start_time
            print(f"✅ Collaboration network built in {elapsed_time:.2f}s: {self.collab_network.number_of_nodes()} authors, "
                f"{self.collab_network.number_of_edges()} collaborations")
            
        except Exception as e:
            print(f"❌ Failed to build collaboration network: {e}")
            print("⚠️ Creating empty collaboration network as fallback")
            # 创建空网络作为回退
            self.collab_network = nx.Graph()
    
    def embedding_based_collaborator_recommendation(self, author_id: str, top_k: int = 10) -> List[Dict]:
        """基于嵌入相似性的合作者推荐"""
        if 'author' not in self.id_maps or author_id not in self.id_maps['author']:
            print(f"⚠️ Author {author_id} not found in embeddings")
            return []
        
        # 兼容 torch tensor 或 numpy array
        arr = self.embeddings['author']
        if hasattr(arr, 'cpu') and hasattr(arr, 'numpy'):
            author_embeddings = arr.cpu().numpy()
        elif isinstance(arr, np.ndarray):
            author_embeddings = arr
        else:
            author_embeddings = np.array(arr)
        target_idx = self.id_maps['author'][author_id]
        
        # 计算作者相似度
        try:
            a = author_embeddings[target_idx:target_idx+1]
            a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
            emb_norm = author_embeddings / np.linalg.norm(author_embeddings, axis=1, keepdims=True)
            similarities = (a_norm @ emb_norm.T)[0]
        except Exception as e:
            print(f"❌ Failed to calculate author similarities: {e}")
            return []
        
        # 排除目标作者和已有合作者
        similarities[target_idx] = -1
        
        # 获取现有合作者
        if author_id in self.collab_network:
            existing_collaborators = list(self.collab_network[author_id].keys())
            for collab_id in existing_collaborators:
                if collab_id in self.id_maps['author']:
                    collab_idx = self.id_maps['author'][collab_id]
                    similarities[collab_idx] = -1
        
        # 获取top-K推荐
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        author_ids = [self.reverse_maps['author'].get(i) for i in range(len(author_embeddings))]
        author_ids = [aid for aid in author_ids if aid is not None]
        
        recommendations = []
        for idx in top_indices:
            if (idx < len(author_ids) and author_ids[idx] is not None and 
                similarities[idx] > 0):  # 只保留正相似度
                rec_author_id = author_ids[idx]
                recommendations.append({
                    'author_id': rec_author_id,
                    'similarity_score': float(similarities[idx]),
                    'rank': len(recommendations) + 1
                })
        
        # 获取作者元数据
        author_metadata = self.get_author_metadata([rec['author_id'] for rec in recommendations])
        for rec in recommendations:
            rec.update(author_metadata.get(rec['author_id'], {}))
        
        return recommendations
    
    def network_based_collaborator_recommendation(self, author_id: str, top_k: int = 10) -> List[Dict]:
        """基于网络结构的合作者推荐（共同合作者）"""
        if author_id not in self.collab_network:
            print(f"⚠️ Author {author_id} not found in collaboration network")
            return []
        
        # 使用共同邻居作为推荐依据
        recommendations = []
        seen_authors = set([author_id])
        
        # 直接合作者（已存在）
        direct_collaborators = list(self.collab_network[author_id].keys())
        seen_authors.update(direct_collaborators)
        
        # 推荐共同合作者（朋友的朋友）
        for collab in direct_collaborators:
            if collab in self.collab_network:
                for potential_collab in self.collab_network[collab]:
                    if (potential_collab not in seen_authors and 
                        potential_collab != author_id):
                        
                        # 计算共同合作者数量
                        common_collabs = []
                        if author_id in self.collab_network and potential_collab in self.collab_network:
                            common_collabs = set(self.collab_network[author_id].keys()) & \
                                           set(self.collab_network[potential_collab].keys())
                        
                        jaccard_similarity = len(common_collabs) / len(
                            set(self.collab_network[author_id].keys()) | 
                            set(self.collab_network[potential_collab].keys())
                        ) if (author_id in self.collab_network and potential_collab in self.collab_network and 
                              len(self.collab_network[author_id]) > 0 and len(self.collab_network[potential_collab]) > 0) else 0
                        
                        recommendations.append({
                            'author_id': potential_collab,
                            'common_collaborators': len(common_collabs),
                            'jaccard_similarity': jaccard_similarity,
                            'recommended_via': collab
                        })
                        seen_authors.add(potential_collab)
        
        # 按共同合作者数量排序
        recommendations.sort(key=lambda x: (x['common_collaborators'], x['jaccard_similarity']), reverse=True)
        
        # 限制数量并添加排名
        final_recommendations = []
        for i, rec in enumerate(recommendations[:top_k]):
            rec['rank'] = i + 1
            final_recommendations.append(rec)
        
        # 获取作者元数据
        author_metadata = self.get_author_metadata([rec['author_id'] for rec in final_recommendations])
        for rec in final_recommendations:
            rec.update(author_metadata.get(rec['author_id'], {}))
        
        return final_recommendations
    
    def community_based_recommendation(self, author_id: str, top_k: int = 10) -> List[Dict]:
        """基于社区检测的合作者推荐"""
        if author_id not in self.collab_network:
            return []
        
        # 使用Louvain方法检测社区
        try:
            communities = nx.community.louvain_communities(self.collab_network)
        except:
            # 如果Louvain失败，使用连通组件
            communities = list(nx.connected_components(self.collab_network))
        
        # 找到目标作者的社区
        target_community = None
        for community in communities:
            if author_id in community:
                target_community = community
                break
        
        if not target_community:
            return []
        
        recommendations = []
        for candidate in target_community:
            if (candidate != author_id and 
                (author_id not in self.collab_network or candidate not in self.collab_network[author_id])):
                
                # 在社区内但尚未合作
                recommendations.append({
                    'author_id': candidate,
                    'same_community': True,
                    'community_size': len(target_community)
                })
        
        # 限制数量
        recommendations = recommendations[:top_k]
        
        # 获取作者元数据
        author_metadata = self.get_author_metadata([rec['author_id'] for rec in recommendations])
        for i, rec in enumerate(recommendations):
            rec.update(author_metadata.get(rec['author_id'], {}))
            rec['rank'] = i + 1
        
        return recommendations
    
    def hybrid_collaborator_recommendation(self, author_id: str, top_k: int = 10) -> Dict[str, Any]:
        """混合合作者推荐"""
        print(f"👥 Hybrid collaborator recommendation for: {author_id}")
        
        all_recommendations = []
        
        # 1. 基于嵌入的推荐
        embedding_recs = self.embedding_based_collaborator_recommendation(author_id, top_k*2)
        for rec in embedding_recs:
            rec['method'] = 'embedding_based'
            rec['final_score'] = rec['similarity_score'] * 0.5
            all_recommendations.append(rec)
        
        # 2. 基于网络的推荐
        network_recs = self.network_based_collaborator_recommendation(author_id, top_k*2)
        for rec in network_recs:
            rec['method'] = 'network_based'
            rec['final_score'] = (rec['common_collaborators'] * 0.05 + 
                                rec['jaccard_similarity'] * 0.3)
            all_recommendations.append(rec)
        
        # 3. 基于社区的推荐
        community_recs = self.community_based_recommendation(author_id, top_k)
        for rec in community_recs:
            rec['method'] = 'community_based'
            rec['final_score'] = 0.2  # 基础分数
            all_recommendations.append(rec)
        
        # 合并和重排序
        author_scores = defaultdict(float)
        author_details = {}
        
        for rec in all_recommendations:
            author_id_rec = rec['author_id']
            author_scores[author_id_rec] += rec['final_score']
            if author_id_rec not in author_details:
                author_details[author_id_rec] = rec
        
        # 获取top-K推荐
        top_authors = heapq.nlargest(top_k, author_scores.items(), key=lambda x: x[1])
        
        final_recommendations = []
        for author_id_rec, score in top_authors:
            rec = author_details[author_id_rec].copy()
            rec['final_score'] = score
            rec['rank'] = len(final_recommendations) + 1
            final_recommendations.append(rec)
        
        return {
            'recommendations': final_recommendations,
            'target_author': author_id,
            'total_recommendations': len(final_recommendations),
            'collaboration_network_stats': {
                'total_collaborators': len(self.collab_network[author_id]) if author_id in self.collab_network else 0,
                'network_size': self.collab_network.number_of_nodes(),
                'total_collaborations': self.collab_network.number_of_edges()
            }
        }
    
    def get_author_metadata(self, author_ids: List[str]) -> Dict[str, Any]:
        """获取作者元数据"""
        if not author_ids or self.graph_db is None:
            return {}
        
        try:
            # 处理ID格式
            formatted_ids = [f"'{aid}'" for aid in author_ids if aid]
            if not formatted_ids:
                return {}
                
            ids_str = ', '.join(formatted_ids)
            
            query = f"""
            MATCH (a:Author)
            WHERE a.author_id IN [{ids_str}]
            RETURN a.author_id as author_id, a.name as name
            """
            results = self._safe_neo4j_query(query)
            return {result['author_id']: dict(result) for result in results}
        except Exception as e:
            print(f"⚠️ Failed to get author metadata: {e}")
            return {}

class RecommendationMonitor:
    """推荐质量监控器"""
    
    def __init__(self):
        self.performance_history = []
    
    def log_recommendation_quality(self, recommendations: List[Dict], method: str):
        """记录推荐质量指标"""
        if not recommendations:
            return
        
        quality_metrics = {
            'method': method,
            'timestamp': datetime.now().isoformat(),
            'count': len(recommendations),
            'avg_score': np.mean([r.get('similarity_score', 0) for r in recommendations]),
            'score_std': np.std([r.get('similarity_score', 0) for r in recommendations]),
            'diversity': self._calculate_diversity(recommendations),
            'venue_coverage': len(set(r.get('venue', 'Unknown') for r in recommendations))
        }
        
        self.performance_history.append(quality_metrics)
        
        # 保持历史记录大小
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    def _calculate_diversity(self, recommendations: List[Dict]) -> float:
        """计算推荐多样性"""
        if not recommendations:
            return 0.0
        
        venues = [r.get('venue', 'Unknown') for r in recommendations]
        unique_venues = len(set(venues))
        return unique_venues / len(venues)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """生成性能报告"""
        if not self.performance_history:
            return {}
        
        df = pd.DataFrame(self.performance_history)
        report = {
            'total_recommendations': len(self.performance_history),
            'methods_used': df['method'].value_counts().to_dict(),
            'average_scores_by_method': df.groupby('method')['avg_score'].mean().to_dict(),
            'average_diversity_by_method': df.groupby('method')['diversity'].mean().to_dict()
        }
        
        return report

def main():
    """测试推荐系统"""
    print("🎯 Testing Academic Recommendation System")
    
    try:
        # 初始化推荐器
        paper_recommender = AcademicRecommender()
        collaborator_recommender = CollaboratorRecommender()
        
        # 测试论文推荐
        print("\n" + "="*70)
        print("📚 Testing Paper Recommendation")
        print("="*70)
        
        # 基于内容的推荐
        content_recs = paper_recommender.content_based_paper_recommendation(
            "graph neural networks for recommender systems", top_k=3
        )
        print(f"Content-based recommendations: {len(content_recs)} papers")
        for rec in content_recs[:2]:
            print(f"  - {rec.get('title', 'Unknown')} (score: {rec['similarity_score']:.3f})")
        
        # 测试合作者推荐
        print("\n" + "="*70)
        print("👥 Testing Collaborator Recommendation")
        print("="*70)
        
        if collaborator_recommender.id_maps['author']:
            sample_author = list(collaborator_recommender.id_maps['author'].keys())[0]
            collab_recs = collaborator_recommender.hybrid_collaborator_recommendation(
                sample_author, top_k=3
            )
            print(f"Collaborator recommendations for author: {len(collab_recs['recommendations'])} authors")
            for rec in collab_recs['recommendations'][:2]:
                print(f"  - {rec.get('name', 'Unknown')} (score: {rec['final_score']:.3f})")
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()