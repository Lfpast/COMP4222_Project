# [file name]: evaluate_retrieval.py
#
# Sanity check: Verify HAN embeddings use graph structure better than pure SBERT

import os
import numpy as np
from typing import List, Dict
from dotenv import load_dotenv
from graph_rag_system import AcademicRecommender


class RetrievalEvaluator:
    """
    Evaluate HAN vs Pure Semantic retrieval on graph structure utilization
    """
    
    def __init__(self, recommender: AcademicRecommender):
        self.recommender = recommender
    
    def get_han_recommendations(self, query: str, top_k: int = 10) -> List[Dict]:
        """Get recommendations using HAN embeddings (graph-aware)"""
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
    
    def calculate_graph_structure_score(self, recommendations: List[Dict]) -> Dict[str, float]:
        """
        Measure how well recommendations leverage graph structure
        
        Returns 4 graph-aware metrics:
        1. Co-citation score: papers citing same references
        2. Citation connectivity: direct citations between papers
        3. Author overlap: shared authors
        4. Keyword coherence: shared keywords
        """
        if not self.recommender.graph_db or not recommendations:
            return {
                'cocitation_score': 0.0,
                'citation_connectivity': 0.0,
                'author_overlap_score': 0.0,
                'keyword_coherence': 0.0
            }
        
        paper_ids = [str(rec['paper_id']) for rec in recommendations]
        formatted_ids = [f"'{pid}'" for pid in paper_ids]
        ids_str = ', '.join(formatted_ids)
        
        metrics = {}
        
        try:
            # 1. Co-citation strength (papers citing same references)
            query_cocite = f"""
            MATCH (p1:Paper)-[:CITES]->(common:Paper)<-[:CITES]-(p2:Paper)
            WHERE p1.paper_id IN [{ids_str}] AND p2.paper_id IN [{ids_str}]
            AND p1.paper_id < p2.paper_id
            RETURN COUNT(common) as cocitations
            """
            result = self.recommender.graph_db.run(query_cocite).data()
            
            if result:
                cocitations = result[0]['cocitations']
                max_possible = len(recommendations) * (len(recommendations) - 1) / 2
                metrics['cocitation_score'] = cocitations / max_possible if max_possible > 0 else 0
            else:
                metrics['cocitation_score'] = 0.0
            
            # 2. Direct citation connectivity (papers citing each other)
            query_direct = f"""
            MATCH (p1:Paper)-[:CITES]->(p2:Paper)
            WHERE p1.paper_id IN [{ids_str}] AND p2.paper_id IN [{ids_str}]
            RETURN COUNT(*) as direct_citations
            """
            result = self.recommender.graph_db.run(query_direct).data()
            
            if result:
                direct = result[0]['direct_citations']
                metrics['citation_connectivity'] = direct / len(recommendations) if recommendations else 0
            else:
                metrics['citation_connectivity'] = 0.0
            
            # 3. Author overlap (shared authors indicate related research)
            query_authors = f"""
            MATCH (p1:Paper)-[:WRITTEN_BY]->(a:Author)<-[:WRITTEN_BY]-(p2:Paper)
            WHERE p1.paper_id IN [{ids_str}] AND p2.paper_id IN [{ids_str}]
            AND p1.paper_id < p2.paper_id
            RETURN COUNT(DISTINCT a) as shared_authors
            """
            result = self.recommender.graph_db.run(query_authors).data()
            
            if result:
                shared = result[0]['shared_authors']
                metrics['author_overlap_score'] = shared / len(recommendations) if recommendations else 0
            else:
                metrics['author_overlap_score'] = 0.0
            
            # 4. Keyword coherence (shared keywords indicate topical coherence)
            query_keywords = f"""
            MATCH (p:Paper)-[:HAS_KEYWORD]->(k:Keyword)
            WHERE p.paper_id IN [{ids_str}]
            WITH k, COUNT(p) as paper_count
            WHERE paper_count > 1
            RETURN COUNT(k) as shared_keywords, SUM(paper_count) as total_shares
            """
            result = self.recommender.graph_db.run(query_keywords).data()
            
            if result and result[0]['shared_keywords']:
                shared_kw = result[0]['shared_keywords']
                total = result[0]['total_shares']
                metrics['keyword_coherence'] = total / (len(recommendations) * 2) if recommendations else 0
            else:
                metrics['keyword_coherence'] = 0.0
                
        except Exception as e:
            print(f"⚠️ Graph structure query failed: {e}")
            metrics = {
                'cocitation_score': 0.0,
                'citation_connectivity': 0.0,
                'author_overlap_score': 0.0,
                'keyword_coherence': 0.0
            }
        
        return metrics


def main():
    """Simple test: HAN vs SBERT graph utilization"""
    print("="*70)
    print("🔬 HAN Graph Utilization Sanity Check")
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
    
    # Test query
    test_query = "graph neural networks for recommender systems"
    print(f"\n📊 Test Query: {test_query}\n")
    
    # Get recommendations
    han_recs = evaluator.get_han_recommendations(test_query, top_k=10)
    sem_recs = evaluator.get_semantic_recommendations(test_query, top_k=10)
    
    if not han_recs or not sem_recs:
        print("❌ Failed to get recommendations")
        return
    
    # Calculate graph structure scores
    han_graph = evaluator.calculate_graph_structure_score(han_recs)
    sem_graph = evaluator.calculate_graph_structure_score(sem_recs)
    
    # Calculate totals
    han_total = sum(han_graph.values())
    sem_total = sum(sem_graph.values())
    
    # Print results
    print("="*70)
    print("GRAPH STRUCTURE UTILIZATION COMPARISON")
    print("="*70)
    
    print(f"\n🔷 HAN Results:")
    print(f"  Co-citation score:      {han_graph['cocitation_score']:.3f}")
    print(f"  Citation connectivity:  {han_graph['citation_connectivity']:.3f}")
    print(f"  Author overlap:         {han_graph['author_overlap_score']:.3f}")
    print(f"  Keyword coherence:      {han_graph['keyword_coherence']:.3f}")
    print(f"  → Total:                {han_total:.3f}")
    
    print(f"\n🔶 Pure Semantic Results:")
    print(f"  Co-citation score:      {sem_graph['cocitation_score']:.3f}")
    print(f"  Citation connectivity:  {sem_graph['citation_connectivity']:.3f}")
    print(f"  Author overlap:         {sem_graph['author_overlap_score']:.3f}")
    print(f"  Keyword coherence:      {sem_graph['keyword_coherence']:.3f}")
    print(f"  → Total:                {sem_total:.3f}")
    
    # Verdict
    print("\n" + "="*70)
    if han_total > sem_total:
        improvement = ((han_total / sem_total) - 1) * 100 if sem_total > 0 else float('inf')
        print(f"✅ PASS: HAN utilizes graph structure better (+{improvement:.1f}%)")
    elif han_total == sem_total:
        print("⚠️ WARN: HAN and Semantic show equal graph utilization")
    else:
        degradation = ((sem_total / han_total) - 1) * 100 if han_total > 0 else float('inf')
        print(f"❌ FAIL: HAN underperforms pure semantic (-{degradation:.1f}%)")
    print("="*70)


if __name__ == "__main__":
    main()
