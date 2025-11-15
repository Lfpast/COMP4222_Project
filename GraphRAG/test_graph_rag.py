# [file name]: test_graph_rag.py
#
# Test suite for Academic Recommender and GraphRAG Engine
# Run this file to test all functionality

import os
from dotenv import load_dotenv
from graph_rag_system import AcademicRecommender, GraphRAGEngine


def test_recommender_system():
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
    
    try:
        # Initialize recommender
        print("\n📚 Initializing Academic Recommender...")
        recommender = AcademicRecommender(
            model_path=MODEL_PATH,
            neo4j_uri=NEO4J_URI,
            neo4j_username=NEO4J_USERNAME,
            neo4j_password=NEO4J_PASSWORD
        )
        
        # Test 1: Content-based recommendation
        print("\n--- Test 1.1: Content-Based Recommendation ---")
        content_recs = recommender.content_based_paper_recommendation(
            "graph neural networks for recommender systems", 
            top_k=5
        )
        print(f"\n✅ Content-based recommendations: {len(content_recs)} papers")
        for rec in content_recs[:3]:
            print(f"  {rec['rank']}. {rec.get('title', 'Unknown')}")
            print(f"     Score: {rec['similarity_score']:.3f}")
        
        # Test 2: Collaborative recommendation
        if content_recs:
            print("\n--- Test 1.2: Collaborative Filtering Recommendation ---")
            target_paper = content_recs[0]['paper_id']
            collab_recs = recommender.collaborative_paper_recommendation(
                target_paper, 
                top_k=5
            )
            print(f"\n✅ Collaborative recommendations: {len(collab_recs)} papers")
            for rec in collab_recs[:3]:
                print(f"  {rec['rank']}. {rec.get('title', 'Unknown')}")
                print(f"     Score: {rec['similarity_score']:.3f}")
        
        print("\n✅ Recommender system tests passed!")
        return recommender
        
    except Exception as e:
        print(f"\n❌ Recommender test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_graph_rag_engine(recommender=None):
    """Test the GraphRAG Engine"""
    print("\n" + "="*70)
    print("🧪 TEST 2: GraphRAG Engine")
    print("="*70)
    
    load_dotenv()
    
    # If recommender not provided, create one
    if recommender is None:
        MODEL_PATH = os.environ.get("MODEL_PATH", r"training\models\focused_v1\han_embeddings.pth")
        NEO4J_URI = os.environ.get("NEO4J_URI", "neo4j://127.0.0.1:7687")
        NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
        NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "87654321")
        
        recommender = AcademicRecommender(
            model_path=MODEL_PATH,
            neo4j_uri=NEO4J_URI,
            neo4j_username=NEO4J_USERNAME,
            neo4j_password=NEO4J_PASSWORD
        )
    
    # Get LLM credentials
    DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
    
    if not DEEPSEEK_API_KEY:
        print("❌ Error: DEEPSEEK_API_KEY not found in environment variables")
        print("   Please add it to your .env file")
        return
    
    try:
        # Initialize GraphRAG Engine
        print("\n🚀 Initializing GraphRAG Engine...")
        engine = GraphRAGEngine(recommender, DEEPSEEK_API_KEY)
        
        # Test 1: Full GraphRAG Query
        print("\n--- Test 2.1: Full GraphRAG Query (Graph Enabled) ---")
        test_query = "What are the main papers about Graph Neural Networks in recommender systems?"
        
        answer = engine.query(
            test_query, 
            top_k_seeds=5, 
            enable_graph_retrieval=True
        )
        
        print("\n" + "="*70)
        print("📝 GraphRAG Answer (Full Mode):")
        print("="*70)
        print(answer)
        
        # Test 2: Baseline Vector RAG (Ablation Study)
        print("\n--- Test 2.2: Baseline Vector RAG (Graph Disabled) ---")
        
        answer_baseline = engine.query(
            test_query, 
            top_k_seeds=5, 
            enable_graph_retrieval=False
        )
        
        print("\n" + "="*70)
        print("📝 GraphRAG Answer (Baseline Mode):")
        print("="*70)
        print(answer_baseline)
        
        # Test 3: Complex Query
        print("\n--- Test 2.3: Complex Query ---")
        complex_query = "Recommend papers using GNNs in recommender systems for data sparsity, listing common graph mining algorithms and two collaborators."
        
        answer_complex = engine.query(
            complex_query, 
            top_k_seeds=5, 
            enable_graph_retrieval=True
        )
        
        print("\n" + "="*70)
        print("📝 GraphRAG Answer (Complex Query):")
        print("="*70)
        print(answer_complex)
        
        print("\n✅ GraphRAG engine tests passed!")
        
    except Exception as e:
        print(f"\n❌ GraphRAG test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests"""
    print("="*70)
    print("🚀 GraphRAG System Test Suite")
    print("="*70)
    
    # Test 1: Recommender System
    recommender = test_recommender_system()
    
    # Test 2: GraphRAG Engine (reuse recommender if successful)
    if recommender is not None:
        test_graph_rag_engine(recommender)
    else:
        print("\n⚠️ Skipping GraphRAG tests due to recommender initialization failure")
    
    print("\n" + "="*70)
    print("✅ All tests completed!")
    print("="*70)


if __name__ == "__main__":
    main()
