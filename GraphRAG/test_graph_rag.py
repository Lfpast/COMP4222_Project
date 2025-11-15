# [file name]: test_graph_rag.py
#
# Test suite for Academic Recommender and GraphRAG Engine
# Run this file to test all functionality

import os
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
    """Test 3: Compare HAN vs Pure Semantic Retrieval"""
    print("\n" + "="*70)
    print("🧪 TEST 3: Retrieval Quality Evaluation (HAN vs Semantic)")
    print("="*70)
    
    if recommender is None:
        results.add_test("3.0 Retrieval Evaluation (skipped)", False, "Recommender not available")
        return
    
    evaluator = RetrievalEvaluator(recommender)
    
    # Test 3.1: Single query evaluation
    print("\n--- Test 3.1: Query Evaluation ---")
    try:
        test_query = "graph neural networks for recommender systems"
        eval_results = evaluator.evaluate_query(test_query, top_k=10)
        
        # Validation
        assert eval_results is not None, "Evaluation results are None"
        assert 'han' in eval_results, "Missing HAN results"
        assert 'semantic' in eval_results, "Missing semantic results"
        assert 'comparison' in eval_results, "Missing comparison"
        
        # Check that both methods returned recommendations
        han_recs = eval_results['han']['recommendations']
        sem_recs = eval_results['semantic']['recommendations']
        
        assert len(han_recs) > 0, "HAN returned no recommendations"
        assert len(sem_recs) > 0, "Semantic returned no recommendations"
        
        results.add_test("3.1 Query Evaluation", True)
        
    except AssertionError as e:
        results.add_test("3.1 Query Evaluation", False, str(e))
        return
    except Exception as e:
        results.add_test("3.1 Query Evaluation", False, f"Unexpected error: {e}")
        return
    
    # Test 3.2: Metrics calculation
    print("\n--- Test 3.2: Metrics Calculation ---")
    try:
        han = eval_results['han']
        sem = eval_results['semantic']
        
        # Validate all metrics exist
        assert 'diversity' in han, "Missing diversity metrics"
        assert 'novelty' in han, "Missing novelty metric"
        assert 'citation_coverage' in han, "Missing citation metrics"
        assert 'graph_structure_score' in han, "Missing graph score"
        
        # Validate metric ranges
        assert 0 <= han['diversity']['venue_diversity'] <= 1, "Invalid diversity"
        assert 0 <= han['novelty'] <= 1, "Invalid novelty"
        assert han['citation_coverage']['avg_citations'] >= 0, "Invalid citations"
        
        # Validate graph structure score (now a dict with multiple metrics)
        graph_score = han['graph_structure_score']
        assert isinstance(graph_score, dict), "Graph score should be a dict"
        assert 'cocitation_score' in graph_score, "Missing cocitation_score"
        assert 0 <= graph_score['cocitation_score'] <= 1, "Invalid cocitation score"
        
        results.add_test("3.2 Metrics Calculation", True)
        
    except AssertionError as e:
        results.add_test("3.2 Metrics Calculation", False, str(e))
        return
    except Exception as e:
        results.add_test("3.2 Metrics Calculation", False, f"Unexpected error: {e}")
        return
    
    # Test 3.3: Comparison metrics
    print("\n--- Test 3.3: Ranking Comparison ---")
    try:
        comp = eval_results['comparison']
        
        assert 'overlap_ratio' in comp, "Missing overlap ratio"
        assert 'rank_correlation' in comp, "Missing rank correlation"
        
        # Validate ranges
        assert 0 <= comp['overlap_ratio'] <= 1, "Invalid overlap ratio"
        assert -1 <= comp['rank_correlation'] <= 1, "Invalid correlation"
        
        # Print summary
        evaluator.print_evaluation_report(eval_results)
        
        results.add_test("3.3 Ranking Comparison", True)
        
    except AssertionError as e:
        results.add_test("3.3 Ranking Comparison", False, str(e))
    except Exception as e:
        results.add_test("3.3 Ranking Comparison", False, f"Unexpected error: {e}")
    
    # Test 3.4: HAN Advantage Score
    print("\n--- Test 3.4: HAN Advantage Analysis ---")
    try:
        han_recs = eval_results['han']['recommendations']
        sem_recs = eval_results['semantic']['recommendations']
        
        advantage = evaluator.calculate_han_advantage_score(han_recs, sem_recs)
        
        # Validation
        assert advantage is not None, "Advantage score is None"
        assert 'han_graph_advantage' in advantage, "Missing graph advantage"
        assert 'han_finds_influential' in advantage, "Missing influential flag"
        assert 'han_builds_coherent_set' in advantage, "Missing coherence flag"
        
        # Print results
        print(f"   Graph Advantage: {advantage['han_graph_advantage']:.2%}")
        print(f"   Finds Influential: {'✓' if advantage['han_finds_influential'] else '✗'}")
        print(f"   Builds Coherent Set: {'✓' if advantage['han_builds_coherent_set'] else '✗'}")
        print(f"   Explanation: {advantage['explanation']}")
        
        # HAN should ideally outperform on at least one dimension
        has_advantage = (advantage['han_finds_influential'] or 
                        advantage['han_builds_coherent_set'] or 
                        advantage['han_graph_advantage'] > 0)
        
        if has_advantage:
            print("   ✅ HAN demonstrates graph-aware advantage!")
        else:
            print("   ⚠️ HAN did not outperform semantic on this query")
        
        results.add_test("3.4 HAN Advantage Analysis", True)
        
    except AssertionError as e:
        results.add_test("3.4 HAN Advantage Analysis", False, str(e))
    except Exception as e:
        results.add_test("3.4 HAN Advantage Analysis", False, f"Unexpected error: {e}")


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
