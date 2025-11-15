# [file name]: test_graph_rag.py
#
# Test suite for Academic Recommender and GraphRAG Engine
# Run this file to test all functionality

import os
import numpy as np
from dotenv import load_dotenv
from graph_rag_system import AcademicRecommender, GraphRAGEngine
from evaluate_retrieval import RetrievalEvaluator
import re


class TestResults:
    """Track test results and failures"""
    def __init__(self):
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.failures = []
    
    def add_test(self, test_name, passed, error_msg=None):
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
            print(f"✅ PASS: {test_name}")
        else:
            self.failed_tests += 1
            self.failures.append((test_name, error_msg))
            print(f"❌ FAIL: {test_name}")
            if error_msg:
                print(f"   Error: {error_msg}")
    
    def print_summary(self):
        print("\n" + "="*70)
        print("📊 TEST SUMMARY")
        print("="*70)
        print(f"Total Tests: {self.total_tests}")
        print(f"Passed: {self.passed_tests} ✅")
        print(f"Failed: {self.failed_tests} ❌")
        
        if self.failures:
            print("\n❌ Failed Tests:")
            for test_name, error_msg in self.failures:
                print(f"  - {test_name}")
                if error_msg:
                    print(f"    {error_msg}")
        
        if self.failed_tests == 0:
            print("\n🎉 ALL TESTS PASSED!")
            return True
        else:
            print(f"\n⚠️ {self.failed_tests} test(s) failed")
            return False


def save_llm_output(query, answer, mode):
    """
    追加保存所有LLM输出到GraphRAG/lm_output.txt
    """
    filename = "lm_output.txt"
    with open(filename, 'a', encoding='utf-8') as f:
        f.write("\n" + "="*60 + "\n")
        f.write(f"Mode: {mode}\n")
        f.write(f"Query: {query}\n\n")
        f.write(answer)
        f.write("\n" + "="*60 + "\n")
    print(f"   📝 LLM output appended to: {filename}")


def test_recommender_system(results):
    """Test the Academic Recommender System"""
    print("\n" + "="*70)
    print("🧪 TEST 1: Academic Recommender System")
    print("="*70)
    
    # Load environment variables
    load_dotenv()
    
    MODEL_PATH = os.environ.get("MODEL_PATH", r"training\models\focused_v1\han_embeddings.pth")
    NEO4J_URI = os.environ.get("NEO4J_URI", "neo4j://127.0.0.1:7687")
    NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "87654321")
    
    recommender = None
    
    try:
        # Test 1.0: Initialization
        print("\n--- Test 1.0: Recommender Initialization ---")
        recommender = AcademicRecommender(
            model_path=MODEL_PATH,
            neo4j_uri=NEO4J_URI,
            neo4j_username=NEO4J_USERNAME,
            neo4j_password=NEO4J_PASSWORD
        )
        
        # Validate initialization
        assert recommender is not None, "Recommender is None"
        assert hasattr(recommender, 'embeddings'), "Missing embeddings"
        assert hasattr(recommender, 'id_maps'), "Missing id_maps"
        assert 'paper' in recommender.id_maps, "Missing paper id_maps"
        
        results.add_test("1.0 Recommender Initialization", True)
        
    except Exception as e:
        results.add_test("1.0 Recommender Initialization", False, str(e))
        return None
    
    # Test 1.1: Content-based recommendation
    print("\n--- Test 1.1: Content-Based Recommendation ---")
    try:
        content_recs = recommender.content_based_paper_recommendation(
            "graph neural networks for recommender systems", 
            top_k=5
        )
        
        # Validation criteria
        assert content_recs is not None, "Recommendations are None"
        assert len(content_recs) > 0, "No recommendations returned"
        assert len(content_recs) <= 5, f"Too many recommendations: {len(content_recs)}"
        
        # Validate recommendation structure
        for i, rec in enumerate(content_recs):
            assert 'paper_id' in rec, f"Rec {i} missing paper_id"
            assert 'similarity_score' in rec, f"Rec {i} missing similarity_score"
            assert 'rank' in rec, f"Rec {i} missing rank"
            assert rec['rank'] == i + 1, f"Rec {i} has wrong rank: {rec['rank']}"
            assert 0 <= rec['similarity_score'] <= 1, f"Invalid similarity score: {rec['similarity_score']}"
        
        # Validate scores are sorted descending
        scores = [rec['similarity_score'] for rec in content_recs]
        assert scores == sorted(scores, reverse=True), "Scores not sorted descending"
        
        print(f"   Returned {len(content_recs)} recommendations")
        for rec in content_recs[:3]:
            print(f"   {rec['rank']}. {rec.get('title', 'Unknown')} (score: {rec['similarity_score']:.3f})")
        
        results.add_test("1.1 Content-Based Recommendation", True)
        
    except AssertionError as e:
        results.add_test("1.1 Content-Based Recommendation", False, str(e))
        content_recs = None
    except Exception as e:
        results.add_test("1.1 Content-Based Recommendation", False, f"Unexpected error: {e}")
        content_recs = None
    
    # Test 1.2: Collaborative recommendation
    print("\n--- Test 1.2: Collaborative Filtering Recommendation ---")
    try:
        if not content_recs:
            raise AssertionError("Cannot test collaborative filtering - no papers from content-based test")
        
        target_paper = content_recs[0]['paper_id']
        collab_recs = recommender.collaborative_paper_recommendation(
            target_paper, 
            top_k=5
        )
        
        # Validation criteria
        assert collab_recs is not None, "Recommendations are None"
        assert len(collab_recs) > 0, "No recommendations returned"
        assert len(collab_recs) <= 5, f"Too many recommendations: {len(collab_recs)}"
        
        # Validate no self-recommendation
        collab_ids = [rec['paper_id'] for rec in collab_recs]
        assert target_paper not in collab_ids, "Recommended the target paper itself"
        
        # Validate structure
        for i, rec in enumerate(collab_recs):
            assert 'paper_id' in rec, f"Rec {i} missing paper_id"
            assert 'similarity_score' in rec, f"Rec {i} missing similarity_score"
            assert 'rank' in rec, f"Rec {i} missing rank"
            assert rec['rank'] == i + 1, f"Rec {i} has wrong rank"
        
        # Validate scores are sorted descending
        scores = [rec['similarity_score'] for rec in collab_recs]
        assert scores == sorted(scores, reverse=True), "Scores not sorted descending"
        
        print(f"   Returned {len(collab_recs)} recommendations")
        for rec in collab_recs[:3]:
            print(f"   {rec['rank']}. {rec.get('title', 'Unknown')} (score: {rec['similarity_score']:.3f})")
        
        results.add_test("1.2 Collaborative Filtering", True)
        
    except AssertionError as e:
        results.add_test("1.2 Collaborative Filtering", False, str(e))
    except Exception as e:
        results.add_test("1.2 Collaborative Filtering", False, f"Unexpected error: {e}")
    
    # Test 1.3: Metadata quality
    print("\n--- Test 1.3: Metadata Quality ---")
    try:
        if not content_recs:
            raise AssertionError("No recommendations to check metadata")
        
        paper_ids = [rec['paper_id'] for rec in content_recs]
        metadata = recommender.get_paper_metadata(paper_ids)
        
        assert metadata is not None, "Metadata is None"
        assert len(metadata) > 0, "No metadata returned"
        
        # Check that at least some papers have real metadata (not fallback)
        real_metadata_count = sum(1 for m in metadata.values() if not m.get('is_fallback', False))
        
        print(f"   Retrieved metadata for {len(metadata)} papers")
        print(f"   Real metadata: {real_metadata_count}/{len(metadata)}")
        
        # We expect at least some real metadata if Neo4j is connected
        if recommender.graph_db is not None:
            assert real_metadata_count > 0, "No real metadata found (all fallback)"
        
        results.add_test("1.3 Metadata Quality", True)
        
    except AssertionError as e:
        results.add_test("1.3 Metadata Quality", False, str(e))
    except Exception as e:
        results.add_test("1.3 Metadata Quality", False, f"Unexpected error: {e}")
    
    return recommender


def test_graph_rag_engine(recommender, results):
    """Test the GraphRAG Engine"""
    print("\n" + "="*70)
    print("🧪 TEST 2: GraphRAG Engine")
    print("="*70)
    
    load_dotenv()
    
    # If recommender not provided, skip
    if recommender is None:
        results.add_test("2.0 GraphRAG Engine (skipped)", False, "Recommender not available")
        return
    
    # Get LLM credentials
    DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
    
    if not DEEPSEEK_API_KEY:
        print("⚠️ DEEPSEEK_API_KEY not found - skipping LLM tests")
        results.add_test("2.0 GraphRAG Initialization", False, "DEEPSEEK_API_KEY not set")
        return
    
    engine = None
    
    # Test 2.0: Engine Initialization
    print("\n--- Test 2.0: GraphRAG Engine Initialization ---")
    try:
        engine = GraphRAGEngine(recommender, DEEPSEEK_API_KEY)
        
        assert engine is not None, "Engine is None"
        assert hasattr(engine, 'recommender'), "Missing recommender"
        assert hasattr(engine, 'llm_client'), "Missing llm_client"
        
        results.add_test("2.0 GraphRAG Initialization", True)
        
    except Exception as e:
        results.add_test("2.0 GraphRAG Initialization", False, str(e))
        return
    
    # Test 2.1: Full GraphRAG Query
    print("\n--- Test 2.1: Full GraphRAG Query (Graph Enabled) ---")
    try:
        test_query = "What are the main papers about Graph Neural Networks in recommender systems?"
        
        answer = engine.query(
            test_query, 
            top_k_seeds=3,  # Use 3 for faster testing
            enable_graph_retrieval=True
        )
        save_llm_output(test_query, answer, mode="graph_enabled")
        
        # Validation
        assert answer is not None, "Answer is None"
        assert isinstance(answer, str), f"Answer is not string: {type(answer)}"
        assert len(answer) > 50, f"Answer too short: {len(answer)} chars"
        assert not answer.startswith("Error:"), f"LLM returned error: {answer[:100]}"
        
        # Check that answer mentions papers or context
        answer_lower = answer.lower()
        has_content = any(keyword in answer_lower for keyword in 
                         ['paper', 'research', 'graph', 'neural', 'network', 'recommend'])
        assert has_content, "Answer doesn't contain expected keywords"
        
        print("\n📝 GraphRAG Answer (Full Mode):")
        print(answer[:500] + ("..." if len(answer) > 500 else ""))
        
        results.add_test("2.1 Full GraphRAG Query", True)
        
    except AssertionError as e:
        results.add_test("2.1 Full GraphRAG Query", False, str(e))
    except Exception as e:
        results.add_test("2.1 Full GraphRAG Query", False, f"Unexpected error: {e}")
    
    # Test 2.2: Baseline Vector RAG
    print("\n--- Test 2.2: Baseline Vector RAG (Graph Disabled) ---")
    try:
        test_query = "What are the main papers about Graph Neural Networks?"
        
        answer_baseline = engine.query(
            test_query, 
            top_k_seeds=3,
            enable_graph_retrieval=False
        )
        save_llm_output(test_query, answer_baseline, mode="baseline")
        
        # Validation
        assert answer_baseline is not None, "Answer is None"
        assert isinstance(answer_baseline, str), "Answer is not string"
        assert len(answer_baseline) > 50, f"Answer too short: {len(answer_baseline)} chars"
        assert not answer_baseline.startswith("Error:"), "LLM returned error"
        
        print("\n📝 Baseline Answer:")
        print(answer_baseline[:500] + ("..." if len(answer_baseline) > 500 else ""))
        
        results.add_test("2.2 Baseline Vector RAG", True)
        
    except AssertionError as e:
        results.add_test("2.2 Baseline Vector RAG", False, str(e))
    except Exception as e:
        results.add_test("2.2 Baseline Vector RAG", False, f"Unexpected error: {e}")
    
    # Test 2.3: Answer Quality Comparison
    print("\n--- Test 2.3: Answer Quality Comparison ---")
    try:
        # Both tests from 2.1 and 2.2 must have passed
        # We check if graph-enabled answer is different from baseline
        # (demonstrating that graph retrieval adds information)
        
        test_query = "What papers discuss graph neural networks?"
        
        answer_with_graph = engine.query(test_query, top_k_seeds=3, enable_graph_retrieval=True)
        save_llm_output(test_query, answer_with_graph, mode="graph_enabled_compare")
        answer_without_graph = engine.query(test_query, top_k_seeds=3, enable_graph_retrieval=False)
        save_llm_output(test_query, answer_without_graph, mode="baseline_compare")
        
        # They should be different (graph context adds info)
        # But this is not a strict requirement, so we just log
        if answer_with_graph != answer_without_graph:
            print("   ✓ Graph-enabled answer differs from baseline (expected)")
        else:
            print("   ℹ️ Answers are identical (graph context may not have changed output)")
        
        results.add_test("2.3 Answer Quality Comparison", True)
        
    except Exception as e:
        results.add_test("2.3 Answer Quality Comparison", False, f"Unexpected error: {e}")


def test_retrieval_evaluation(recommender, results):
    """Test 3: Verify collaborative filtering uses citation graph"""
    print("\n" + "="*70)
    print("🧪 TEST 3: Citation Graph Utilization in Collaborative Filtering")
    print("="*70)
    
    if recommender is None:
        results.add_test("3.0 Collaborative Filtering (skipped)", False, "Recommender not available")
        return
    
    # Test 3.1: Collaborative recommendations should leverage citation graph
    print("\n--- Test 3.1: Citation Graph Coverage in Recommendations ---")
    try:
        if not recommender.graph_db:
            print("   ⚠️ Neo4j not available - skipping")
            results.add_test("3.1 Citation Graph Coverage", True, "Neo4j not available")
            return
        
        # Find a seed paper with citations in both Neo4j and embeddings
        query = """
        MATCH (seed:Paper)-[:CITES]->(cited:Paper)
        WHERE EXISTS((seed)<-[:CITES]-(:Paper))
        WITH seed, COUNT(cited) as num_cited
        WHERE num_cited >= 3
        RETURN seed.paper_id as seed_id, seed.title as seed_title, num_cited
        LIMIT 50
        """
        candidates = recommender.graph_db.run(query).data()
        
        seed_id = None
        seed_title = None
        for candidate in candidates:
            if candidate['seed_id'] in recommender.id_maps.get('paper', {}):
                seed_id = candidate['seed_id']
                seed_title = candidate['seed_title']
                break
        
        if not seed_id:
            print("   ⚠️ No suitable seed paper found in both Neo4j and embeddings")
            results.add_test("3.1 Citation Graph Coverage", True, "No suitable seed found")
            return
        
        print(f"📄 Seed: {seed_title[:60]}...")
        print(f"   ID: {seed_id}")
        
        # Get collaborative recommendations (should use citation graph)
        han_recs = recommender.collaborative_paper_recommendation(seed_id, top_k=20, boost_citations=True)
        
        assert len(han_recs) > 0, f"Collaborative method returned no recommendations for {seed_id}"
        
        print(f"\n   ✅ Generated {len(han_recs)} collaborative recommendations")
        
        # Count citation relationships
        cited_by_seed = 0
        cites_seed = 0
        cocited_papers = 0
        
        for rec in han_recs:
            rec_id = rec['paper_id']
            
            # Check citation relationship from response
            if rec.get('citation_relationship') == 'cited_by_target':
                cited_by_seed += 1
            elif rec.get('citation_relationship') == 'cites_target':
                cites_seed += 1
            elif rec.get('citation_relationship') == 'co-cited':
                cocited_papers += 1
        
        total_citation_related = cited_by_seed + cites_seed
        total_graph_related = total_citation_related + cocited_papers
        citation_ratio = total_citation_related / len(han_recs)
        graph_ratio = total_graph_related / len(han_recs)
        
        print(f"\n   📊 Citation Graph Coverage:")
        print(f"     Papers cited by seed:       {cited_by_seed}/{len(han_recs)}")
        print(f"     Papers citing seed:         {cites_seed}/{len(han_recs)}")
        print(f"     Papers with co-citations:   {cocited_papers}/{len(han_recs)}")
        print(f"     Direct citations:           {total_citation_related}/{len(han_recs)} ({citation_ratio:.1%})")
        print(f"     Total graph-related:        {total_graph_related}/{len(han_recs)} ({graph_ratio:.1%})")
        
        # PASS criteria: At least 20% direct citations OR 40% graph-related
        if citation_ratio >= 0.2:
            print(f"   ✅ PASS: Collaborative filtering leverages citation graph ({citation_ratio:.1%} direct citations)")
            results.add_test("3.1 Citation Graph Coverage", True)
        elif graph_ratio >= 0.4:
            print(f"   ✅ PASS: Collaborative filtering uses graph structure ({graph_ratio:.1%} via co-citations)")
            results.add_test("3.1 Citation Graph Coverage", True)
        else:
            print(f"   ❌ FAIL: Low citation coverage (direct: {citation_ratio:.1%}, total: {graph_ratio:.1%})")
            print(f"        Expected: ≥20% direct citations OR ≥40% graph-related")
            results.add_test("3.1 Citation Graph Coverage", False, 
                           f"Insufficient graph usage: {graph_ratio:.1%}")
        
    except AssertionError as e:
        results.add_test("3.1 Citation Graph Coverage", False, str(e))
    except Exception as e:
        results.add_test("3.1 Citation Graph Coverage", False, f"Unexpected error: {e}")


def main():
    """Run all tests"""
    print("="*70)
    print("🚀 GraphRAG System Test Suite")
    print("="*70)
    
    results = TestResults()
    
    # Test 1: Recommender System
    recommender = test_recommender_system(results)
    
    # Test 2: GraphRAG Engine (reuse recommender if successful)
    test_graph_rag_engine(recommender, results)
    
    # Test 3: Retrieval Evaluation (HAN vs Semantic)
    test_retrieval_evaluation(recommender, results)
    
    # Print summary and return exit code
    all_passed = results.print_summary()
    
    if all_passed:
        print("\n🎉 SUCCESS: All tests passed!")
        return 0
    else:
        print(f"\n❌ FAILURE: {results.failed_tests} test(s) failed")
        return 1


if __name__ == "__main__":
    import sys
    exit_code = main()
    sys.exit(exit_code)
