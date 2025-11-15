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
                'graph_structure_score': self.calculate_graph_structure_score(han_recs)
            },
            'semantic': {
                'recommendations': semantic_recs,
                'diversity': self.calculate_diversity(semantic_recs),
                'novelty': self.calculate_novelty(semantic_recs, query_terms),
                'citation_coverage': self.calculate_citation_coverage(semantic_recs),
                'graph_structure_score': self.calculate_graph_structure_score(semantic_recs)
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
        
        # Determine overall winner
        print("\n" + "="*70)
        scores = {
            'han': 0,
            'semantic': 0
        }
        
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
        
        print(f"OVERALL WINNER: ", end="")
        if scores['han'] > scores['semantic']:
            print(f"🏆 HAN ({scores['han']}/4 metrics)")
        elif scores['semantic'] > scores['han']:
            print(f"🏆 Semantic ({scores['semantic']}/4 metrics)")
        else:
            print(f"🤝 TIE ({scores['han']}/4 metrics each)")
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
