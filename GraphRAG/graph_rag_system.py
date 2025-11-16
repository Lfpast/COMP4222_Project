# [file name]: graph_rag_system.py
#
# Combined Academic Recommender + GraphRAG Engine
# This file contains the core implementation for academic paper recommendation
# and GraphRAG-based question answering system.

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
from openai import OpenAI
warnings.filterwarnings('ignore')


class AcademicRecommender:
    """
    Academic Paper Recommender System
    
    Uses HAN-trained graph embeddings to recommend papers based on:
    - Content-based filtering (semantic similarity)
    - Collaborative filtering (graph structure)
    - Author-based recommendations
    """
    
    def __init__(self, 
                 model_path=r"training\models\focused_v1\han_embeddings.pth", 
                 neo4j_uri="neo4j://127.0.0.1:7687",
                 neo4j_username="neo4j",
                 neo4j_password="87654321"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self._emb_np_cache = {}
        print("📥 Loading trained embeddings...")
        try:
            self._load_checkpoint(model_path)
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
        
        print("🔌 Connecting to Neo4j...")
        try:
            self.graph_db = Graph(neo4j_uri, auth=(neo4j_username, neo4j_password))
            self.graph_db.run("RETURN 1")
            print("✅ Neo4j connection successful")
        except Exception as e:
            print(f"❌ Failed to connect to Neo4j: {e}")
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
        paths_to_try = [
            model_path,
            os.path.join(os.path.dirname(__file__), model_path),
            os.path.join(os.path.dirname(__file__), '..', model_path),
        ]

        checkpoint = None
        for p in paths_to_try:
            if not p:
                continue
            try:
                if os.path.exists(p):
                    checkpoint = torch.load(p, map_location=self.device)
                    model_path = p
                    break
            except Exception:
                continue

        if checkpoint is None:
            checkpoint = torch.load(model_path, map_location=self.device)

        self.embeddings = checkpoint['embeddings']
        self.original_embeddings = checkpoint.get('original_embeddings', checkpoint['embeddings'])
        self.id_maps = checkpoint['id_maps']
        self.config = checkpoint.get('config', {})

        # Cache numpy arrays for both original and HAN embeddings
        self._han_emb_cache = {}
        for ntype in self.original_embeddings.keys():
            try:
                arr = self.original_embeddings[ntype]
                if hasattr(arr, 'numpy'):
                    self._emb_np_cache[ntype] = arr.cpu().numpy()
                elif isinstance(arr, np.ndarray):
                    self._emb_np_cache[ntype] = arr
                else:
                    self._emb_np_cache[ntype] = np.array(arr)
            except Exception:
                pass

        for ntype in self.embeddings.keys():
            try:
                arr = self.embeddings[ntype]
                if hasattr(arr, 'numpy'):
                    self._han_emb_cache[ntype] = arr.cpu().numpy()
                elif isinstance(arr, np.ndarray):
                    self._han_emb_cache[ntype] = arr
                else:
                    self._han_emb_cache[ntype] = np.array(arr)
            except Exception:
                pass

        print(f"✅ Loaded embeddings from: {model_path}")
        print(f"   Original (Sentence-BERT) types: {list(self.original_embeddings.keys())}")
        print(f"   HAN-trained types: {list(self.embeddings.keys())}")

    def _get_numpy_emb(self, node_type: str, use_original=True):
        cache = self._emb_np_cache if use_original else self._han_emb_cache
        embeddings_dict = self.original_embeddings if use_original else self.embeddings
        
        if node_type in cache:
            return cache[node_type]

        if node_type in embeddings_dict:
            arr = embeddings_dict[node_type]
            if hasattr(arr, 'cpu') and hasattr(arr, 'numpy'):
                try:
                    np_arr = arr.cpu().numpy()
                except Exception:
                    np_arr = np.array(arr)
            else:
                np_arr = np.array(arr)

            cache[node_type] = np_arr
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
        """获取论文元数据"""
        if not paper_ids or self.graph_db is None:
            return self._get_enhanced_fallback_metadata(paper_ids)

        try:
            batch_size = 50
            all_results = []

            for i in range(0, len(paper_ids), batch_size):
                batch = paper_ids[i:i + batch_size]
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
    
    def content_based_paper_recommendation(self, query_text: str, top_k: int = 10) -> List[Dict]:
        """基于内容的论文推荐 - 使用 HAN-trained embeddings"""
        print(f"📚 Content-based paper recommendation for: {query_text}")
        
        if self.sentence_model is None:
            print("❌ Sentence transformer not available")
            return []
        
        try:
            query_embedding = self.sentence_model.encode([query_text])
            print(f"   Query embedding shape: {query_embedding.shape}")
        except Exception as e:
            print(f"❌ Failed to encode query: {e}")
            return []
        
        if 'paper' not in self.embeddings:
            print("❌ Paper embeddings not found")
            return []
            
        paper_embeddings = self._get_numpy_emb('paper', use_original=False)
        print(f"   Paper embeddings shape: {paper_embeddings.shape}")
        print(f"   Using HAN-trained embeddings for graph-aware semantic search")

        # Calculate similarity
        if query_embedding.shape[1] != paper_embeddings.shape[1]:
            print(f"⚠️ Dimension mismatch: query {query_embedding.shape[1]}D vs paper {paper_embeddings.shape[1]}D")
            min_dim = min(query_embedding.shape[1], paper_embeddings.shape[1])
            query_projected = query_embedding[:, :min_dim]
            paper_projected = paper_embeddings[:, :min_dim]
            
            query_norm = query_projected / np.linalg.norm(query_projected, axis=1, keepdims=True)
            paper_norm = paper_projected / np.linalg.norm(paper_projected, axis=1, keepdims=True)
            similarities = np.dot(query_norm, paper_norm.T)[0]
        else:
            try:
                q = np.array(query_embedding)
                p = paper_embeddings
                qn = q / np.linalg.norm(q, axis=1, keepdims=True)
                pn = p / np.linalg.norm(p, axis=1, keepdims=True)
                similarities = (qn @ pn.T)[0]
            except Exception:
                similarities = cosine_similarity(query_embedding, paper_embeddings)[0]
        
        paper_ids = [self.reverse_maps['paper'].get(i) for i in range(len(paper_embeddings))]
        valid_indices = [i for i, pid in enumerate(paper_ids) if pid is not None and similarities[i] > 0]
        
        if not valid_indices:
            print("❌ No valid recommendations found")
            return []
        
        top_indices = sorted(valid_indices, key=lambda i: similarities[i], reverse=True)[:top_k]
        
        recommendations = []
        for rank, idx in enumerate(top_indices, 1):
            paper_id = paper_ids[idx]
            recommendations.append({
                'paper_id': paper_id,
                'similarity_score': float(similarities[idx]),
                'rank': rank
            })
        
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
    
    def collaborative_paper_recommendation(self, target_paper_id: str, top_k: int = 10, boost_citations: bool = True) -> List[Dict]:
        """
        Collaborative paper recommendation using citation graph structure
        
        Args:
            target_paper_id: Target paper ID
            top_k: Number of recommendations
            boost_citations: If True, boost papers with citation relationships
        
        Returns:
            List of recommended papers with metadata
        """
        print(f"🔗 Collaborative paper recommendation for: {target_paper_id}")
        
        if 'paper' not in self.id_maps:
            print("❌ Paper id_maps not found")
            return []
        
        if target_paper_id not in self.id_maps['paper']:
            print(f"⚠️ Target paper {target_paper_id} not found in embeddings")
            return []
        
        # Get citation neighborhood from graph
        cited_papers = set()
        citing_papers = set()
        cocited_papers = set()
        
        if self.graph_db and boost_citations:
            try:
                # Papers cited by target
                query_cited = f"""
                MATCH (target:Paper {{paper_id: '{target_paper_id}'}})-[:CITES]->(cited:Paper)
                RETURN cited.paper_id as paper_id
                LIMIT 50
                """
                results = self.graph_db.run(query_cited).data()
                cited_papers = {r['paper_id'] for r in results if r['paper_id'] in self.id_maps['paper']}
                
                # Papers citing target
                query_citing = f"""
                MATCH (citing:Paper)-[:CITES]->(target:Paper {{paper_id: '{target_paper_id}'}})
                RETURN citing.paper_id as paper_id
                LIMIT 50
                """
                results = self.graph_db.run(query_citing).data()
                citing_papers = {r['paper_id'] for r in results if r['paper_id'] in self.id_maps['paper']}
                
                # Papers with co-citations (cite same papers)
                query_cocite = f"""
                MATCH (target:Paper {{paper_id: '{target_paper_id}'}})-[:CITES]->(common:Paper)<-[:CITES]-(cocited:Paper)
                WHERE cocited.paper_id <> '{target_paper_id}'
                RETURN cocited.paper_id as paper_id, COUNT(common) as shared_citations
                ORDER BY shared_citations DESC
                LIMIT 30
                """
                results = self.graph_db.run(query_cocite).data()
                cocited_papers = {r['paper_id'] for r in results if r['paper_id'] in self.id_maps['paper']}
                
                print(f"   Citation context: {len(cited_papers)} cited, {len(citing_papers)} citing, {len(cocited_papers)} co-cited")
                
            except Exception as e:
                print(f"   ⚠️ Citation query failed: {e}")
        
        # Calculate semantic similarity using HAN embeddings
        target_idx = self.id_maps['paper'][target_paper_id]
        paper_embeddings = self._get_numpy_emb('paper', use_original=False)

        try:
            t = paper_embeddings[target_idx:target_idx+1]
            if np.isnan(t).any() or np.isinf(t).any():
                print("❌ Target embedding contains NaN or Inf values")
                return []

            t_norm = t / np.linalg.norm(t, axis=1, keepdims=True)
            emb_norm = paper_embeddings / np.linalg.norm(paper_embeddings, axis=1, keepdims=True)
            similarities = (t_norm @ emb_norm.T)[0]

        except Exception as e:
            print(f"❌ Failed to calculate similarities: {e}")
            return []
        
        # Boost scores for papers in citation neighborhood
        if boost_citations and (cited_papers or citing_papers or cocited_papers):
            paper_ids = [self.reverse_maps['paper'].get(i) for i in range(len(paper_embeddings))]
            for i, pid in enumerate(paper_ids):
                if pid in cited_papers:
                    similarities[i] *= 1.5  # Strong boost for cited papers
                elif pid in citing_papers:
                    similarities[i] *= 1.3  # Medium boost for citing papers
                elif pid in cocited_papers:
                    similarities[i] *= 1.2  # Light boost for co-cited papers
        
        similarities[target_idx] = -1
        
        paper_ids = [self.reverse_maps['paper'].get(i) for i in range(len(paper_embeddings))]
        valid_indices = [i for i in range(len(paper_ids))
                        if paper_ids[i] is not None and similarities[i] > 0.1 and i != target_idx]
        
        if not valid_indices:
            print("⚠️ No valid recommendations found")
            return []
        
        valid_indices_sorted = sorted(valid_indices, key=lambda i: similarities[i], reverse=True)[:top_k]
        
        recommendations = []
        for rank, idx in enumerate(valid_indices_sorted, 1):
            paper_id = paper_ids[idx]
            
            # Mark citation relationship
            citation_type = None
            if paper_id in cited_papers:
                citation_type = "cited_by_target"
            elif paper_id in citing_papers:
                citation_type = "cites_target"
            elif paper_id in cocited_papers:
                citation_type = "co-cited"
            
            recommendations.append({
                'paper_id': paper_id,
                'similarity_score': float(similarities[idx]),
                'rank': rank,
                'citation_relationship': citation_type
            })
        
        if recommendations:
            recommendation_ids = [rec['paper_id'] for rec in recommendations]
            paper_metadata = self.get_paper_metadata(recommendation_ids)
        
            for rec in recommendations:
                paper_id_str = str(rec['paper_id'])
                metadata = paper_metadata.get(paper_id_str, {})
                if metadata and metadata.get('title'):
                    rec.update(metadata)
                else:
                    rec['title'] = f"Research Paper {paper_id_str}"
                    rec['year'] = "Unknown"
                    rec['venue'] = "Unknown"
        
        return recommendations


class GraphRAGEngine:
    """
    GraphRAG Engine combining semantic and structural retrieval
    with LLM-based answer synthesis.
    """
    
    def __init__(self, recommender: AcademicRecommender, llm_api_key: str, llm_base_url: str = "https://api.deepseek.com"):
        """
        Initialize GraphRAG Engine
        
        Args:
            recommender: AcademicRecommender instance
            llm_api_key: API key for LLM (e.g., Deepseek, OpenAI)
            llm_base_url: Base URL for LLM API
        """
        if not hasattr(recommender, 'content_based_paper_recommendation'):
            raise ValueError("Invalid recommender object")
        if not hasattr(recommender, '_safe_neo4j_query'):
            raise ValueError("Invalid recommender object")
            
        self.recommender = recommender
        
        try:
            self.llm_client = OpenAI(
                api_key=llm_api_key,
                base_url=llm_base_url
            )
            self.llm_client.models.list()
            print("✅ LLM client initialized successfully")
        except Exception as e:
            print(f"❌ LLM client initialization failed: {e}")
            raise
        
        print("✅ GraphRAG Engine is ready")

    def _get_structural_context(self, seed_paper_ids: List[str]) -> Dict[str, List[str]]:
        """
        Graph-Structural Retrieval: Extract subgraph information from Neo4j
        """
        print(f"   Phase 4.2b: Graph-Structural Retrieval for {len(seed_paper_ids)} seeds...")
        
        if not seed_paper_ids or self.recommender.graph_db is None:
            return {}

        formatted_ids = [f"'{str(pid)}'" for pid in seed_paper_ids]
        ids_str = ', '.join(formatted_ids)
        
        context_dict = {
            "related_authors": [],
            "citation_links": [],
            "common_keywords": [],
            "key_papers_cited_by_seeds": [],
            "influential_co_authors": []
        }
        
        # Query Authors
        try:
            query_authors = f"""
            MATCH (p:Paper)-[:WRITTEN_BY]->(a:Author)
            WHERE p.paper_id IN [{ids_str}]
            RETURN p.title as paper_title, a.name as author_name
            LIMIT 10
            """
            author_results = self.recommender._safe_neo4j_query(query_authors) 
            for res in author_results:
                context_dict["related_authors"].append(f"Author '{res['author_name']}' wrote paper '{res['paper_title']}'.")
        except Exception as e:
            print(f"⚠️ Author graph query failed: {e}")

        # Query Citations
        try:
            query_cites = f"""
            MATCH (p1:Paper)-[:CITES]->(p2:Paper)
            WHERE p1.paper_id IN [{ids_str}]
            RETURN p1.title as citing_paper, p2.title as cited_paper
            LIMIT 10
            """
            cite_results = self.recommender._safe_neo4j_query(query_cites)
            for res in cite_results:
                context_dict["citation_links"].append(f"Paper '{res['citing_paper']}' cites paper '{res['cited_paper']}'.")
        except Exception as e:
            print(f"⚠️ Citation graph query failed: {e}")
            
        # Query Keywords
        try:
            query_keywords = f"""
            MATCH (p:Paper)-[:HAS_KEYWORD]->(k:Keyword)
            WHERE p.paper_id IN [{ids_str}]
            RETURN p.title as paper_title, k.keyword as keyword
            LIMIT 15
            """
            keyword_results = self.recommender._safe_neo4j_query(query_keywords)
            kw_set = set()
            for res in keyword_results:
                kw_set.add(f"Paper '{res['paper_title']}' is associated with keyword '{res['keyword']}'.")
            context_dict["common_keywords"] = list(kw_set)
        except Exception as e:
            print(f"⚠️ Keyword graph query failed: {e}")
            
        # Query Co-Cited Papers
        try:
            query_cocited = f"""
            MATCH (p:Paper)-[:CITES]->(cited_paper:Paper)
            WHERE p.paper_id IN [{ids_str}]
            WITH cited_paper, COUNT(p) as citation_count
            WHERE citation_count > 1
            RETURN cited_paper.title as paper_title, citation_count
            ORDER BY citation_count DESC
            LIMIT 5
            """
            cocited_results = self.recommender._safe_neo4j_query(query_cocited)
            for res in cocited_results:
                context_dict["key_papers_cited_by_seeds"].append(f"Key Paper '{res['paper_title']}' was co-cited by {res['citation_count']} seed papers.")
        except Exception as e:
            print(f"⚠️ Co-citation graph query failed: {e}")
            
        # Query Influential Co-Authors
        try:
            query_coauthors = f"""
            MATCH (p:Paper)-[:WRITTEN_BY]->(a1:Author)-[:WRITTEN_BY]-(:Paper)-[:WRITTEN_BY]-(a2:Author)
            WHERE p.paper_id IN [{ids_str}] AND a1 <> a2
            WITH a2, COUNT(DISTINCT p) as shared_work_count
            RETURN a2.name as co_author_name, shared_work_count
            ORDER BY shared_work_count DESC
            LIMIT 5
            """
            coauthor_results = self.recommender._safe_neo4j_query(query_coauthors)
            for res in coauthor_results:
                context_dict["influential_co_authors"].append(f"Influential co-author '{res['co_author_name']}' collaborated on {res['shared_work_count']} related paper(s).")
        except Exception as e:
            print(f"⚠️ Co-author graph query failed: {e}")
            
        total_info = sum(len(v) for v in context_dict.values())
        print(f"   ✅ Successfully fetched {total_info} pieces of structured graph info")
        return context_dict

    def query(self, user_question: str, top_k_seeds: int = 5, enable_graph_retrieval: bool = True) -> str:
        """
        Execute full GraphRAG query
        
        Args:
            user_question: Natural language question
            top_k_seeds: Number of seed papers to retrieve
            enable_graph_retrieval: Enable/disable graph context
        """
        print("\n" + "="*70)
        print(f"🚀 Received new query: {user_question}")
        print(f"   Graph Retrieval Enabled: {enable_graph_retrieval}")
        print("="*70)
        
        # Phase 1: Semantic Retrieval
        print(f"Phase 4.1: Semantic Retrieval...")
        
        seed_papers = self.recommender.content_based_paper_recommendation(
            query_text=user_question, 
            top_k=top_k_seeds
        )
        
        if not seed_papers:
            print("❌ Failed to find relevant papers")
            return "Sorry, I could not find any papers in the knowledge base related to your query."
        
        seed_paper_ids = [str(p['paper_id']) for p in seed_papers]
        print(f"   ✅ Found {len(seed_paper_ids)} seed papers")

        # Phase 2: Context Aggregation
        print("Phase 4.2: Context Aggregation...")
        
        # 2a. Semantic Context
        print(f"   Phase 4.2a: Semantic Context (Abstracts)...")
        paper_metadata = self.recommender.get_paper_metadata(seed_paper_ids)
        
        semantic_context_parts = []
        for paper in seed_papers:
            pid = str(paper['paper_id'])
            meta = paper_metadata.get(pid, {})
            context = (
                f"Paper (ID: {pid}, Retrieval Score: {paper.get('similarity_score', 0):.3f}):\n"
                f"Title: {meta.get('title', 'N/A')}\n"
                f"Abstract: {meta.get('abstract', 'N/A')}\n"
                f"---"
            )
            semantic_context_parts.append(context)
        
        # 2b. Structural Context
        structural_context_parts = []
        if enable_graph_retrieval:
            structural_context_dict = self._get_structural_context(seed_paper_ids)
            
            if structural_context_dict["related_authors"]:
                structural_context_parts.append("--- Related Authors ---\n" + "\n".join(structural_context_dict["related_authors"]))
            if structural_context_dict["citation_links"]:
                structural_context_parts.append("--- Citation Links ---\n" + "\n".join(structural_context_dict["citation_links"]))
            if structural_context_dict["common_keywords"]:
                structural_context_parts.append("--- Common Keywords ---\n" + "\n".join(structural_context_dict["common_keywords"]))
            if structural_context_dict["key_papers_cited_by_seeds"]:
                structural_context_parts.append("--- Key Papers Cited by Seeds (Co-citation) ---\n" + "\n".join(structural_context_dict["key_papers_cited_by_seeds"]))
            if structural_context_dict["influential_co_authors"]:
                structural_context_parts.append("--- Influential Co-Authors ---\n" + "\n".join(structural_context_dict["influential_co_authors"]))
        else:
            print("   Phase 4.2b: Graph-Structural Retrieval DISABLED (Baseline Vector RAG)")
        
        full_context = (
            "--- SEMANTIC CONTEXT (Retrieved Relevant Papers) ---\n"
            + "\n".join(semantic_context_parts)
        )
        
        if structural_context_parts:
            full_context += ("\n\n--- GRAPH-STRUCTURAL CONTEXT (Relationships) ---\n" 
                             + "\n\n".join(structural_context_parts))

        # Phase 3: Answer Synthesis
        print("Phase 4.3: Answer Synthesis (Calling LLM API)...")
        
        SYSTEM_PROMPT = """
        You are a world-class academic research assistant.
        Your task is to answer the user's 'Question' based *only* on the 'Context' provided below.
        You must strictly follow these rules:
        1.  **Use Only Provided Context**: Do not use any external knowledge.
        2.  **Synthesize the Answer**: Do not just list information. Combine the semantic context (abstracts) and structural context (relationships) to form a coherent answer.
        3.  **Cite Sources**: If possible, mention paper titles or authors.
        4.  **Be Factual**: If the context does not contain enough information to answer the question, state clearly, 'Based on the provided context, I cannot answer this question.'
        """

        USER_PROMPT = f"""
        --- CONTEXT START ---

        {full_context}

        --- CONTEXT END ---

        Question: {user_question}

        As an academic research assistant, please answer this question based *only* on the context provided above.
        """

        try:
            response = self.llm_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT}
                ]
            )
            
            final_answer = response.choices[0].message.content
            print("   ✅ LLM generated answer successfully")
            return final_answer
        
        except Exception as e:
            print(f"❌ LLM API call failed: {e}")
            return f"Error: An error occurred while calling the LLM API: {e}"
