# [file name]: evaluate_retrieval.py
#
# Evaluation suite comparing HAN+GraphRAG vs Pure Semantic retrieval
# Measures: Diversity, Novelty, Coverage, and Ranking Quality

import os
import numpy as np
from typing import List, Dict, Set, Tuple
from collections import defaultdict
from dotenv import load_dotenv
from graph_rag_system import AcademicRecommender
from scipy.stats import ttest_rel, wilcoxon
from sklearn.feature_extraction.text import TfidfVectorizer


class RetrievalEvaluator:
    """
    Evaluate and compare retrieval quality between:
    1. HAN-based retrieval (graph-aware)
    2. Pure semantic retrieval (original SBERT)
    """
    
    def __init__(self, recommender: AcademicRecommender):
        self.recommender = recommender
        self.results = {
            'han': {},
            'semantic': {}
        }
    
    def get_han_recommendations(self, query: str, top_k: int = 10) -> List[Dict]:
        """Get recommendations using HAN embeddings (current implementation)"""
        return self.recommender.content_based_paper_recommendation(query, top_k)
    
    def get_semantic_recommendations(self, query: str, top_k: int = 10) -> List[Dict]:
        """Get recommendations using pure semantic (SBERT) embeddings"""
        print(f"📚 Pure semantic recommendation for: {query}")
        
        if self.recommender.sentence_model is None:
            print("❌ Sentence transformer not available")
            return []
        
        # Encode query
        try:
            query_embedding = self.recommender.sentence_model.encode([query])
        except Exception as e:
            print(f"❌ Failed to encode query: {e}")
            return []
        
        # Use ORIGINAL SBERT embeddings instead of HAN
        if 'paper' not in self.recommender.original_embeddings:
            print("❌ Original embeddings not found")
            return []
            
        paper_embeddings = self.recommender._get_numpy_emb('paper', use_original=True)
        
        # Calculate cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(query_embedding, paper_embeddings)[0]
        
        # Get top-k
        paper_ids = [self.recommender.reverse_maps['paper'].get(i) for i in range(len(paper_embeddings))]
        valid_indices = [i for i, pid in enumerate(paper_ids) if pid is not None and similarities[i] > 0]
        
        if not valid_indices:
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
        
        # Get metadata
        paper_metadata = self.recommender.get_paper_metadata([rec['paper_id'] for rec in recommendations])
        for rec in recommendations:
            metadata = paper_metadata.get(rec['paper_id'], {})
            rec.update(metadata)
        
        print(f"   ✅ Generated {len(recommendations)} pure semantic recommendations")
        return recommendations
    
    def calculate_diversity(self, recommendations: List[Dict]) -> Dict[str, float]:
        """
        Measure diversity of recommendations
        - Venue diversity: How many unique venues
        - Year diversity: Spread across years
        - Author diversity: Unique authors
        """
        if not recommendations:
            return {'venue_diversity': 0, 'year_diversity': 0, 'unique_venues': 0, 'unique_years': 0}
        
        venues = [rec.get('venue', 'Unknown') for rec in recommendations]
        years = [rec.get('year', 'Unknown') for rec in recommendations if rec.get('year') != 'Unknown']
        
        unique_venues = len(set(venues))
        unique_years = len(set(years))
        
        # Normalized diversity (0 to 1)
        venue_diversity = unique_venues / len(recommendations)
        year_diversity = unique_years / len(recommendations) if years else 0
        
        return {
            'venue_diversity': venue_diversity,
            'year_diversity': year_diversity,
            'unique_venues': unique_venues,
            'unique_years': unique_years
        }
    
    def calculate_novelty(self, recommendations: List[Dict], query_terms: Set[str]) -> float:
        """
        Measure novelty: how many recommendations introduce new concepts
        beyond the query terms
        
        Higher novelty = retrieves papers with broader/related concepts
        """
        if not recommendations:
            return 0.0
        
        novel_papers = 0
        for rec in recommendations:
            title = rec.get('title', '').lower()
            abstract = rec.get('abstract', '').lower()
            
            # Extract words from title/abstract
            title_words = set(title.split())
            abstract_words = set(abstract.split())
            paper_words = title_words | abstract_words
            
            # Check if paper introduces words not in query
            new_concepts = paper_words - query_terms
            if len(new_concepts) > 5:  # Threshold for "novel"
                novel_papers += 1
        
        return novel_papers / len(recommendations)
    
    def calculate_citation_coverage(self, recommendations: List[Dict]) -> Dict[str, float]:
        """
        Measure how well recommendations cover the citation graph
        - High citation papers: influential works
        - Citation diversity: mix of highly/lowly cited
        """
        if not recommendations:
            return {'avg_citations': 0, 'max_citations': 0, 'citation_std': 0}
        
        citations = [rec.get('citation_count', 0) for rec in recommendations]
        
        return {
            'avg_citations': np.mean(citations),
            'max_citations': max(citations),
            'min_citations': min(citations),
            'citation_std': np.std(citations)
        }
    
    def calculate_ranking_correlation(self, han_recs: List[Dict], semantic_recs: List[Dict]) -> Dict[str, float]:
        """
        Calculate how different the rankings are
        - Overlap: papers appearing in both
        - Rank correlation: Spearman correlation of ranks
        - Unique to each: papers only in one system
        """
        han_ids = {rec['paper_id']: rec['rank'] for rec in han_recs}
        semantic_ids = {rec['paper_id']: rec['rank'] for rec in semantic_recs}
        
        overlap_ids = set(han_ids.keys()) & set(semantic_ids.keys())
        han_unique = set(han_ids.keys()) - set(semantic_ids.keys())
        semantic_unique = set(semantic_ids.keys()) - set(han_ids.keys())
        
        overlap_ratio = len(overlap_ids) / max(len(han_ids), 1)
        
        # Spearman correlation for overlapping papers
        if len(overlap_ids) > 1:
            from scipy.stats import spearmanr
            han_ranks = [han_ids[pid] for pid in overlap_ids]
            sem_ranks = [semantic_ids[pid] for pid in overlap_ids]
            correlation, p_value = spearmanr(han_ranks, sem_ranks)
        else:
            correlation = 0.0
            p_value = 1.0
        
        return {
            'overlap_ratio': overlap_ratio,
            'overlap_count': len(overlap_ids),
            'han_unique_count': len(han_unique),
            'semantic_unique_count': len(semantic_unique),
            'rank_correlation': correlation,
            'correlation_pvalue': p_value
        }
    
    def calculate_graph_structure_score(self, recommendations: List[Dict]) -> float:
        """
        Measure how well recommendations leverage graph structure
        by checking co-citation patterns in Neo4j
        """
        if not self.recommender.graph_db or not recommendations:
            return 0.0
        
        paper_ids = [str(rec['paper_id']) for rec in recommendations]
        formatted_ids = [f"'{pid}'" for pid in paper_ids]
        ids_str = ', '.join(formatted_ids)
        
        try:
            # Query co-citation strength
            query = f"""
            MATCH (p1:Paper)-[:CITES]->(common:Paper)<-[:CITES]-(p2:Paper)
            WHERE p1.paper_id IN [{ids_str}] AND p2.paper_id IN [{ids_str}]
            AND p1.paper_id < p2.paper_id
            RETURN COUNT(common) as cocitations
            """
            result = self.recommender.graph_db.run(query).data()
            
            if result:
                cocitations = result[0]['cocitations']
                # Normalize by maximum possible co-citations
                max_possible = len(recommendations) * (len(recommendations) - 1) / 2
                return cocitations / max_possible if max_possible > 0 else 0
        except Exception as e:
            print(f"⚠️ Graph structure query failed: {e}")
        
        return 0.0
    
    def calculate_author_network_metrics(self, recommendations: List[Dict]) -> Dict[str, float]:
        """
        分析推荐论文的作者合作网络质量
        - 作者多样性：独立作者数
        - 跨团队合作：不同作者组之间的合作
        """
        if not self.recommender.graph_db or not recommendations:
            return {'author_diversity': 0, 'unique_authors': 0, 'cross_team_score': 0}
        
        paper_ids = [str(rec['paper_id']) for rec in recommendations]
        formatted_ids = [f"'{pid}'" for pid in paper_ids]
        ids_str = ', '.join(formatted_ids)
        
        try:
            # 查询作者信息
            query = f"""
            MATCH (p:Paper)-[:WRITTEN_BY]->(a:Author)
            WHERE p.paper_id IN [{ids_str}]
            RETURN p.paper_id as paper_id, collect(a.author_id) as authors
            """
            result = self.recommender.graph_db.run(query).data()
            
            all_authors = set()
            paper_author_sets = []
            
            for row in result:
                authors = set(row['authors'])
                all_authors.update(authors)
                paper_author_sets.append(authors)
            
            # 计算跨团队合作分数（不同论文作者集合的差异度）
            cross_team_pairs = 0
            total_pairs = 0
            for i in range(len(paper_author_sets)):
                for j in range(i + 1, len(paper_author_sets)):
                    total_pairs += 1
                    # 如果两篇论文的作者没有交集，说明是跨团队
                    if len(paper_author_sets[i] & paper_author_sets[j]) == 0:
                        cross_team_pairs += 1
            
            cross_team_score = cross_team_pairs / total_pairs if total_pairs > 0 else 0
            author_diversity = len(all_authors) / max(len(recommendations), 1)
            
            return {
                'author_diversity': author_diversity,
                'unique_authors': len(all_authors),
                'cross_team_score': cross_team_score
            }
        except Exception as e:
            print(f"⚠️ Author network query failed: {e}")
            return {'author_diversity': 0, 'unique_authors': 0, 'cross_team_score': 0}
    
    def calculate_citation_network_density(self, recommendations: List[Dict]) -> Dict[str, float]:
        """
        计算推荐论文之间的引用网络密度
        - 直接引用：推荐论文之间的直接引用关系
        - 网络密度：实际引用边数 / 最大可能引用边数
        """
        if not self.recommender.graph_db or not recommendations:
            return {'direct_citations': 0, 'network_density': 0}
        
        paper_ids = [str(rec['paper_id']) for rec in recommendations]
        formatted_ids = [f"'{pid}'" for pid in paper_ids]
        ids_str = ', '.join(formatted_ids)
        
        try:
            query = f"""
            MATCH (p1:Paper)-[:CITES]->(p2:Paper)
            WHERE p1.paper_id IN [{ids_str}] AND p2.paper_id IN [{ids_str}]
            RETURN COUNT(*) as direct_citations
            """
            result = self.recommender.graph_db.run(query).data()
            
            if result:
                direct_citations = result[0]['direct_citations']
                max_possible = len(recommendations) * (len(recommendations) - 1)
                network_density = direct_citations / max_possible if max_possible > 0 else 0
                
                return {
                    'direct_citations': direct_citations,
                    'network_density': network_density
                }
        except Exception as e:
            print(f"⚠️ Citation network query failed: {e}")
        
        return {'direct_citations': 0, 'network_density': 0}
    
    def calculate_semantic_relevance_score(self, recommendations: List[Dict], query: str) -> Dict[str, float]:
        """
        使用TF-IDF计算推荐论文与查询的语义相关性
        更精确地衡量语义匹配度
        """
        if not recommendations:
            return {'avg_tfidf_score': 0, 'min_tfidf_score': 0, 'max_tfidf_score': 0}
        
        # 收集所有文本
        texts = [query]
        for rec in recommendations:
            title = rec.get('title', '')
            abstract = rec.get('abstract', '')
            texts.append(f"{title} {abstract}")
        
        try:
            # 计算TF-IDF
            vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # 计算查询与每篇论文的余弦相似度
            from sklearn.metrics.pairwise import cosine_similarity
            query_vec = tfidf_matrix[0:1]
            doc_vecs = tfidf_matrix[1:]
            similarities = cosine_similarity(query_vec, doc_vecs)[0]
            
            return {
                'avg_tfidf_score': float(np.mean(similarities)),
                'min_tfidf_score': float(np.min(similarities)),
                'max_tfidf_score': float(np.max(similarities)),
                'std_tfidf_score': float(np.std(similarities))
            }
        except Exception as e:
            print(f"⚠️ TF-IDF calculation failed: {e}")
            return {'avg_tfidf_score': 0, 'min_tfidf_score': 0, 'max_tfidf_score': 0, 'std_tfidf_score': 0}
    
    def calculate_coverage_score(self, recommendations: List[Dict]) -> float:
        """
        计算推荐的覆盖率：推荐论文在整个知识图谱中的分布广度
        通过计算推荐论文在图中的平均最短路径来衡量
        """
        if not self.recommender.graph_db or not recommendations or len(recommendations) < 2:
            return 0.0
        
        paper_ids = [str(rec['paper_id']) for rec in recommendations]
        formatted_ids = [f"'{pid}'" for pid in paper_ids]
        ids_str = ', '.join(formatted_ids)
        
        try:
            # 计算推荐论文之间的平均最短路径长度
            query = f"""
            MATCH (p1:Paper), (p2:Paper)
            WHERE p1.paper_id IN [{ids_str}] AND p2.paper_id IN [{ids_str}]
            AND p1.paper_id < p2.paper_id
            MATCH path = shortestPath((p1)-[*..5]-(p2))
            RETURN AVG(length(path)) as avg_distance, COUNT(*) as pairs
            """
            result = self.recommender.graph_db.run(query).data()
            
            if result and result[0]['pairs'] > 0:
                avg_distance = result[0]['avg_distance']
                # 距离越大，覆盖范围越广
                # 归一化到0-1，假设最大距离为5
                return min(avg_distance / 5.0, 1.0)
        except Exception as e:
            print(f"⚠️ Coverage calculation failed: {e}")
        
        return 0.0
    
    def calculate_ndcg(self, recommendations: List[Dict], k: int = 10) -> float:
        """
        计算NDCG（归一化折损累计增益）
        使用引用数作为相关性分数
        """
        if not recommendations:
            return 0.0
        
        def dcg_at_k(scores, k):
            scores = scores[:k]
            return sum([score / np.log2(idx + 2) for idx, score in enumerate(scores)])
        
        # 使用引用数作为相关性
        relevance_scores = [rec.get('citation_count', 0) for rec in recommendations[:k]]
        
        # 理想排序（按引用数降序）
        ideal_scores = sorted(relevance_scores, reverse=True)
        
        dcg = dcg_at_k(relevance_scores, k)
        idcg = dcg_at_k(ideal_scores, k)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def evaluate_query(self, query: str, top_k: int = 10) -> Dict[str, Dict]:
        """
        Evaluate a single query with both methods
        """
        print("\n" + "="*70)
        print(f"📊 Evaluating Query: {query}")
        print("="*70)
        
        # Get recommendations from both methods
        print("\n1️⃣ HAN-based retrieval...")
        han_recs = self.get_han_recommendations(query, top_k)
        
        print("\n2️⃣ Pure semantic retrieval...")
        semantic_recs = self.get_semantic_recommendations(query, top_k)
        
        # Extract query terms for novelty calculation
        query_terms = set(query.lower().split())
        
        # Calculate metrics
        print("\n3️⃣ Computing metrics...")
        
        results = {
            'han': {
                'recommendations': han_recs,
                'diversity': self.calculate_diversity(han_recs),
                'novelty': self.calculate_novelty(han_recs, query_terms),
                'citation_coverage': self.calculate_citation_coverage(han_recs),
                'graph_structure_score': self.calculate_graph_structure_score(han_recs),
                'author_network': self.calculate_author_network_metrics(han_recs),
                'citation_network': self.calculate_citation_network_density(han_recs),
                'semantic_relevance': self.calculate_semantic_relevance_score(han_recs, query),
                'coverage_score': self.calculate_coverage_score(han_recs),
                'ndcg': self.calculate_ndcg(han_recs, top_k)
            },
            'semantic': {
                'recommendations': semantic_recs,
                'diversity': self.calculate_diversity(semantic_recs),
                'novelty': self.calculate_novelty(semantic_recs, query_terms),
                'citation_coverage': self.calculate_citation_coverage(semantic_recs),
                'graph_structure_score': self.calculate_graph_structure_score(semantic_recs),
                'author_network': self.calculate_author_network_metrics(semantic_recs),
                'citation_network': self.calculate_citation_network_density(semantic_recs),
                'semantic_relevance': self.calculate_semantic_relevance_score(semantic_recs, query),
                'coverage_score': self.calculate_coverage_score(semantic_recs),
                'ndcg': self.calculate_ndcg(semantic_recs, top_k)
            },
            'comparison': self.calculate_ranking_correlation(han_recs, semantic_recs)
        }
        
        return results
    
    def print_evaluation_report(self, results: Dict[str, Dict]):
        """Print a formatted evaluation report"""
        print("\n" + "="*70)
        print("📊 EVALUATION REPORT")
        print("="*70)
        
        han = results['han']
        sem = results['semantic']
        comp = results['comparison']
        
        print("\n--- DIVERSITY METRICS ---")
        print(f"HAN:      Venue: {han['diversity']['venue_diversity']:.2%} ({han['diversity']['unique_venues']} unique)")
        print(f"Semantic: Venue: {sem['diversity']['venue_diversity']:.2%} ({sem['diversity']['unique_venues']} unique)")
        print(f"Winner: {'🏆 HAN' if han['diversity']['venue_diversity'] > sem['diversity']['venue_diversity'] else '🏆 Semantic' if sem['diversity']['venue_diversity'] > han['diversity']['venue_diversity'] else '🤝 Tie'}")
        
        print(f"\nHAN:      Year: {han['diversity']['year_diversity']:.2%} ({han['diversity']['unique_years']} unique)")
        print(f"Semantic: Year: {sem['diversity']['year_diversity']:.2%} ({sem['diversity']['unique_years']} unique)")
        print(f"Winner: {'🏆 HAN' if han['diversity']['year_diversity'] > sem['diversity']['year_diversity'] else '🏆 Semantic' if sem['diversity']['year_diversity'] > han['diversity']['year_diversity'] else '🤝 Tie'}")
        
        print("\n--- NOVELTY ---")
        print(f"HAN:      {han['novelty']:.2%} papers introduce novel concepts")
        print(f"Semantic: {sem['novelty']:.2%} papers introduce novel concepts")
        print(f"Winner: {'🏆 HAN' if han['novelty'] > sem['novelty'] else '🏆 Semantic' if sem['novelty'] > han['novelty'] else '🤝 Tie'}")
        
        print("\n--- CITATION COVERAGE ---")
        print(f"HAN:      Avg: {han['citation_coverage']['avg_citations']:.1f}, Max: {han['citation_coverage']['max_citations']}")
        print(f"Semantic: Avg: {sem['citation_coverage']['avg_citations']:.1f}, Max: {sem['citation_coverage']['max_citations']}")
        print(f"Winner: {'🏆 HAN' if han['citation_coverage']['avg_citations'] > sem['citation_coverage']['avg_citations'] else '🏆 Semantic' if sem['citation_coverage']['avg_citations'] > han['citation_coverage']['avg_citations'] else '🤝 Tie'}")
        
        print("\n--- GRAPH STRUCTURE UTILIZATION ---")
        print(f"HAN:      Co-citation score: {han['graph_structure_score']:.2%}")
        print(f"Semantic: Co-citation score: {sem['graph_structure_score']:.2%}")
        print(f"Winner: {'🏆 HAN' if han['graph_structure_score'] > sem['graph_structure_score'] else '🏆 Semantic' if sem['graph_structure_score'] > han['graph_structure_score'] else '🤝 Tie'}")
        
        print("\n--- RANKING COMPARISON ---")
        print(f"Overlap: {comp['overlap_count']}/{len(han['recommendations'])} papers ({comp['overlap_ratio']:.1%})")
        print(f"Unique to HAN: {comp['han_unique_count']}")
        print(f"Unique to Semantic: {comp['semantic_unique_count']}")
        print(f"Rank Correlation: {comp['rank_correlation']:.3f} (p={comp['correlation_pvalue']:.3f})")
        
        print("\n--- AUTHOR NETWORK METRICS ---")
        print(f"HAN:      Author diversity: {han['author_network']['author_diversity']:.2f} ({han['author_network']['unique_authors']} unique)")
        print(f"Semantic: Author diversity: {sem['author_network']['author_diversity']:.2f} ({sem['author_network']['unique_authors']} unique)")
        print(f"HAN:      Cross-team score: {han['author_network']['cross_team_score']:.2%}")
        print(f"Semantic: Cross-team score: {sem['author_network']['cross_team_score']:.2%}")
        print(f"Winner: {'🏆 HAN' if han['author_network']['author_diversity'] > sem['author_network']['author_diversity'] else '🏆 Semantic' if sem['author_network']['author_diversity'] > han['author_network']['author_diversity'] else '🤝 Tie'}")
        
        print("\n--- CITATION NETWORK DENSITY ---")
        print(f"HAN:      Direct citations: {han['citation_network']['direct_citations']}, Density: {han['citation_network']['network_density']:.2%}")
        print(f"Semantic: Direct citations: {sem['citation_network']['direct_citations']}, Density: {sem['citation_network']['network_density']:.2%}")
        print(f"Winner: {'🏆 HAN' if han['citation_network']['network_density'] > sem['citation_network']['network_density'] else '🏆 Semantic' if sem['citation_network']['network_density'] > han['citation_network']['network_density'] else '🤝 Tie'}")
        
        print("\n--- SEMANTIC RELEVANCE (TF-IDF) ---")
        print(f"HAN:      Avg: {han['semantic_relevance']['avg_tfidf_score']:.3f}, Range: [{han['semantic_relevance']['min_tfidf_score']:.3f}, {han['semantic_relevance']['max_tfidf_score']:.3f}]")
        print(f"Semantic: Avg: {sem['semantic_relevance']['avg_tfidf_score']:.3f}, Range: [{sem['semantic_relevance']['min_tfidf_score']:.3f}, {sem['semantic_relevance']['max_tfidf_score']:.3f}]")
        print(f"Winner: {'🏆 HAN' if han['semantic_relevance']['avg_tfidf_score'] > sem['semantic_relevance']['avg_tfidf_score'] else '🏆 Semantic' if sem['semantic_relevance']['avg_tfidf_score'] > han['semantic_relevance']['avg_tfidf_score'] else '🤝 Tie'}")
        
        print("\n--- COVERAGE SCORE ---")
        print(f"HAN:      {han['coverage_score']:.2%}")
        print(f"Semantic: {sem['coverage_score']:.2%}")
        print(f"Winner: {'🏆 HAN' if han['coverage_score'] > sem['coverage_score'] else '🏆 Semantic' if sem['coverage_score'] > han['coverage_score'] else '🤝 Tie'}")
        
        print("\n--- NDCG (Ranking Quality) ---")
        print(f"HAN:      {han['ndcg']:.3f}")
        print(f"Semantic: {sem['ndcg']:.3f}")
        print(f"Winner: {'🏆 HAN' if han['ndcg'] > sem['ndcg'] else '🏆 Semantic' if sem['ndcg'] > han['ndcg'] else '🤝 Tie'}")
        
        # Determine overall winner with expanded metrics
        print("\n" + "="*70)
        scores = {
            'han': 0,
            'semantic': 0
        }
        
        # Original metrics
        if han['diversity']['venue_diversity'] > sem['diversity']['venue_diversity']:
            scores['han'] += 1
        elif sem['diversity']['venue_diversity'] > han['diversity']['venue_diversity']:
            scores['semantic'] += 1
        
        if han['novelty'] > sem['novelty']:
            scores['han'] += 1
        elif sem['novelty'] > han['novelty']:
            scores['semantic'] += 1
        
        if han['citation_coverage']['avg_citations'] > sem['citation_coverage']['avg_citations']:
            scores['han'] += 1
        elif sem['citation_coverage']['avg_citations'] > han['citation_coverage']['avg_citations']:
            scores['semantic'] += 1
        
        if han['graph_structure_score'] > sem['graph_structure_score']:
            scores['han'] += 1
        elif sem['graph_structure_score'] > han['graph_structure_score']:
            scores['semantic'] += 1
        
        # New metrics
        if han['author_network']['author_diversity'] > sem['author_network']['author_diversity']:
            scores['han'] += 1
        elif sem['author_network']['author_diversity'] > han['author_network']['author_diversity']:
            scores['semantic'] += 1
        
        if han['citation_network']['network_density'] > sem['citation_network']['network_density']:
            scores['han'] += 1
        elif sem['citation_network']['network_density'] > han['citation_network']['network_density']:
            scores['semantic'] += 1
        
        if han['coverage_score'] > sem['coverage_score']:
            scores['han'] += 1
        elif sem['coverage_score'] > han['coverage_score']:
            scores['semantic'] += 1
        
        if han['ndcg'] > sem['ndcg']:
            scores['han'] += 1
        elif sem['ndcg'] > han['ndcg']:
            scores['semantic'] += 1
        
        total_metrics = 8
        print(f"OVERALL WINNER: ", end="")
        if scores['han'] > scores['semantic']:
            print(f"🏆 HAN ({scores['han']}/{total_metrics} metrics)")
        elif scores['semantic'] > scores['han']:
            print(f"🏆 Semantic ({scores['semantic']}/{total_metrics} metrics)")
        else:
            print(f"🤝 TIE ({scores['han']}/{total_metrics} metrics each)")
        print("="*70)
        
        return scores
    
    def print_top_papers_comparison(self, results: Dict[str, Dict], top_n: int = 5):
        """Print top-N papers from both methods side by side"""
        han_recs = results['han']['recommendations'][:top_n]
        sem_recs = results['semantic']['recommendations'][:top_n]
        
        print("\n" + "="*70)
        print(f"📄 TOP-{top_n} PAPERS COMPARISON")
        print("="*70)
        
        print("\n🔷 HAN-based Recommendations:")
        for rec in han_recs:
            print(f"  {rec['rank']}. {rec.get('title', 'Unknown')}")
            print(f"     Score: {rec['similarity_score']:.3f} | Venue: {rec.get('venue', 'N/A')}")
        
        print("\n🔶 Pure Semantic Recommendations:")
        for rec in sem_recs:
            print(f"  {rec['rank']}. {rec.get('title', 'Unknown')}")
            print(f"     Score: {rec['similarity_score']:.3f} | Venue: {rec.get('venue', 'N/A')}")


def main():
    """Run evaluation on test queries"""
    print("="*70)
    print("🔬 Retrieval Evaluation: HAN vs Pure Semantic")
    print("="*70)
    
    load_dotenv()
    
    # Initialize recommender
    MODEL_PATH = os.environ.get("MODEL_PATH", r"training\models\focused_v1\han_embeddings.pth")
    NEO4J_URI = os.environ.get("NEO4J_URI", "neo4j://127.0.0.1:7687")
    NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "87654321")
    
    try:
        recommender = AcademicRecommender(
            model_path=MODEL_PATH,
            neo4j_uri=NEO4J_URI,
            neo4j_username=NEO4J_USERNAME,
            neo4j_password=NEO4J_PASSWORD
        )
    except Exception as e:
        print(f"❌ Failed to initialize recommender: {e}")
        return
    
    evaluator = RetrievalEvaluator(recommender)
    
    # Test queries
    test_queries = [
        "graph neural networks for recommender systems",
        "attention mechanisms in deep learning",
        "knowledge graph embedding methods",
        "social network analysis using graph theory"
    ]
    
    all_results = []
    
    for query in test_queries:
        results = evaluator.evaluate_query(query, top_k=10)
        evaluator.print_evaluation_report(results)
        evaluator.print_top_papers_comparison(results, top_n=5)
        all_results.append(results)
    
    # Aggregate statistics
    print("\n" + "="*70)
    print("📈 AGGREGATE STATISTICS (Across All Queries)")
    print("="*70)
    
    han_wins = sum(1 for r in all_results if 
                   r['han']['diversity']['venue_diversity'] > r['semantic']['diversity']['venue_diversity'])
    sem_wins = sum(1 for r in all_results if 
                   r['semantic']['diversity']['venue_diversity'] > r['han']['diversity']['venue_diversity'])
    
    print(f"\nVenue Diversity: HAN wins {han_wins}/{len(test_queries)}, Semantic wins {sem_wins}/{len(test_queries)}")
    
    han_novel = np.mean([r['han']['novelty'] for r in all_results])
    sem_novel = np.mean([r['semantic']['novelty'] for r in all_results])
    print(f"Avg Novelty: HAN {han_novel:.2%}, Semantic {sem_novel:.2%}")
    
    han_citations = np.mean([r['han']['citation_coverage']['avg_citations'] for r in all_results])
    sem_citations = np.mean([r['semantic']['citation_coverage']['avg_citations'] for r in all_results])
    print(f"Avg Citations: HAN {han_citations:.1f}, Semantic {sem_citations:.1f}")
    
    han_graph = np.mean([r['han']['graph_structure_score'] for r in all_results])
    sem_graph = np.mean([r['semantic']['graph_structure_score'] for r in all_results])
    print(f"Avg Graph Score: HAN {han_graph:.2%}, Semantic {sem_graph:.2%}")
    
    avg_overlap = np.mean([r['comparison']['overlap_ratio'] for r in all_results])
    print(f"\nAvg Ranking Overlap: {avg_overlap:.1%}")
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
