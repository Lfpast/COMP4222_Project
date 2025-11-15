# [file name]: graph_rag_engine.py
# 
# This is the core implementation file for Project Phase 4 (GraphRAG).
# It depends on the AcademicRecommender class provided by `recommender_system.py`.
#
# How to run:
# 1. Ensure `recommender_system.py` is in the same directory or Python path.
# 2. Ensure `openai` and `python-dotenv` libraries are installed: 
#    pip install openai python-dotenv
# 3. Create a .env file (by renaming config.txt) and fill in your keys.
# 4. Ensure your Neo4j database is running.
# 5. Ensure the path to `han_embeddings.pth` is correct in your .env file.
# 6. Run this file: python graph_rag_engine.py

import os
from typing import List, Dict, Any
import warnings
from openai import OpenAI  # We can still use the OpenAI client
from dotenv import load_dotenv  # Import dotenv

# Import your existing core recommender
from recommender_system import AcademicRecommender
warnings.filterwarnings('ignore')

class GraphRAGEngine:
    """
    GraphRAG Engine (Phase 4 Implementation)
    
    This engine orchestrates the entire GraphRAG pipeline:
    1. Semantic Retrieval: Uses AcademicRecommender to find "seed" papers.
    2. Graph-Structural Retrieval: Starts from seeds to extract subgraph info from Neo4j.
    3. Answer Synthesis: Feeds all context into an LLM API to generate an answer.
    """
    
    def __init__(self, recommender: AcademicRecommender, deepseek_api_key: str):
        """
        Initializes the GraphRAG Engine for Deepseek.

        Args:
            recommender: An instantiated `AcademicRecommender` object.
            deepseek_api_key: The API key for calling the Deepseek API.
        """
        if not hasattr(recommender, 'content_based_paper_recommendation'):
            raise ValueError("Passed recommender object is incomplete, missing `content_based_paper_recommendation` method.")
        if not hasattr(recommender, '_safe_neo4j_query'):
            raise ValueError("Passed recommender object is incomplete, missing `_safe_neo4j_query` method.")
            
        self.recommender = recommender
        
        try:
            # --- MODIFIED FOR DEEPSEEK ---
            self.llm_client = OpenAI(
                api_key=deepseek_api_key,
                base_url="https://api.deepseek.com"  # Point to Deepseek's endpoint
            )
            # --- END MODIFICATION ---

            self.llm_client.models.list() # Test API key
            print("✅ Deepseek LLM client initialized successfully.")
        except Exception as e:
            print(f"❌ Deepseek client initialization failed: {e}")
            print("   Please check if your DEEPSEEK_API_KEY environment variable is set correctly.")
            raise
        
        print("✅ GraphRAG Engine is ready.")

    def _get_structural_context(self, seed_paper_ids: List[str]) -> Dict[str, List[str]]:
        """
        Private method: Executing 'Graph-Structural Retrieval' (Phase 4.2).
        
        Starting from 'seed' papers, querying Neo4j for related authors, citations, 
        keywords, co-citations, and influential co-authors (multi-hop).
        """
        print(f"   Phase 4.2b: Graph-Structural Retrieval for {len(seed_paper_ids)} seeds...")
        
        if not seed_paper_ids or self.recommender.graph_db is None:
            return {}

        # Formatting paper_id list into a Cypher query string
        formatted_ids = [f"'{str(pid)}'" for pid in seed_paper_ids]
        ids_str = ', '.join(formatted_ids)
        
        context_dict = {
            "related_authors": [],
            "citation_links": [],
            "common_keywords": [],
            "key_papers_cited_by_seeds": [],
            "influential_co_authors": []
        }
        
        # 1. Query Authors (Schema: Paper-WRITTEN_BY->Author)
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

        # 2. Query Citations (Schema: Paper-CITES->Paper)
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
            
        # 3. Query Keywords (Schema: Paper-HAS_KEYWORD->Keyword)
        try:
            query_keywords = f"""
            MATCH (p:Paper)-[:HAS_KEYWORD]->(k:Keyword)
            WHERE p.paper_id IN [{ids_str}]
            RETURN p.title as paper_title, k.keyword as keyword
            LIMIT 15
            """
            keyword_results = self.recommender._safe_neo4j_query(query_keywords)
            # Use a set to avoid duplicate keywords for the same paper
            kw_set = set()
            for res in keyword_results:
                kw_set.add(f"Paper '{res['paper_title']}' is associated with keyword '{res['keyword']}'.")
            context_dict["common_keywords"] = list(kw_set)
        except Exception as e:
            print(f"⚠️ Keyword graph query failed: {e}")
            
        # 4. [EXPANDED] Query Co-Cited Papers (Multi-hop)
        try:
            query_cocited = f"""
            MATCH (p:Paper)-[:CITES]->(cited_paper:Paper)
            WHERE p.paper_id IN [{ids_str}]
            WITH cited_paper, COUNT(p) as citation_count
            WHERE citation_count > 1 // Must be cited by at least 2 seed papers
            RETURN cited_paper.title as paper_title, citation_count
            ORDER BY citation_count DESC
            LIMIT 5
            """
            cocited_results = self.recommender._safe_neo4j_query(query_cocited)
            for res in cocited_results:
                context_dict["key_papers_cited_by_seeds"].append(f"Key Paper '{res['paper_title']}' was co-cited by {res['citation_count']} seed papers.")
        except Exception as e:
            print(f"⚠️ Co-citation graph query failed: {e}")
            
        # 5. [EXPANDED] Query Influential Co-Authors (Multi-hop)
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
        print(f"   ✅ Successfully fetched {total_info} pieces of structured graph info.")
        return context_dict

    def query(self, user_question: str, top_k_seeds: int = 5, enable_graph_retrieval: bool = True) -> str:
        """
        Executing a full GraphRAG query.
        
        Args:
            user_question: The natural language question from the user.
            top_k_seeds: The number of seed papers to retrieve.
            enable_graph_retrieval: Flag to enable/disable graph context (for ablation study).
        """
        print("\n" + "="*70)
        print(f"🚀 Received new query: {user_question}")
        print(f"   Graph Retrieval Enabled: {enable_graph_retrieval}")
        print("="*70)
        
        # === 1. 语义检索 (Semantic Retrieval) ===
        # Reusing your content_based_paper_recommendation
        # This perfectly implements Plan A and Phase 4.1 from your proposal
        print(f"Phase 4.1: Semantic Retrieval...")
        
        seed_papers = self.recommender.content_based_paper_recommendation(
            query_text=user_question, 
            top_k=top_k_seeds
        )
        
        if not seed_papers:
            print("❌ Failed to find relevant papers during Semantic Retrieval phase.")
            return "Sorry, I could not find any papers in the knowledge base related to your query."
        
        seed_paper_ids = [str(p['paper_id']) for p in seed_papers]
        print(f"   ✅ Found {len(seed_paper_ids)} seed papers: {seed_paper_ids}")

        # === 2. 上下文聚合 (Context Aggregation) ===
        print("Phase 4.2: Context Aggregation...")
        
        # 2a. Semantic Context (from Paper Abstracts)
        print(f"   Phase 4.2a: Semantic Context (Abstracts)...")
        # Reusing your get_paper_metadata
        paper_metadata = self.recommender.get_paper_metadata(seed_paper_ids)
        
        semantic_context_parts = []
        for paper in seed_papers:
            pid = str(paper['paper_id'])
            meta = paper_metadata.get(pid, {})
            # Attaching context info to help LLM understand why this paper was selected
            context = (
                f"Paper (ID: {pid}, Retrieval Score: {paper.get('similarity_score', 0):.3f}):\n"
                f"Title: {meta.get('title', 'N/A')}\n"
                f"Abstract: {meta.get('abstract', 'N/A')}\n"
                f"---"
            )
            semantic_context_parts.append(context)
        
        # 2b. Structural Context (from Neo4j Subgraph) - NOW CONDITIONAL
        structural_context_parts = []
        if enable_graph_retrieval:
            structural_context_dict = self._get_structural_context(seed_paper_ids)
            
            # Format the dict into a structured string for the prompt
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
            print("   Phase 4.2b: Graph-Structural Retrieval DISABLED (Baseline Vector RAG).")
        
        # Combine all context
        full_context = (
            "--- SEMANTIC CONTEXT (Retrieved Relevant Papers) ---\n"
            + "\n".join(semantic_context_parts)
        )
        
        if structural_context_parts:
            full_context += ("\n\n--- GRAPH-STRUCTURAL CONTEXT (Relationships) ---\n" 
                             + "\n\n".join(structural_context_parts))

        # === 3. 答案综合 (Answer Synthesis) ===
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
            # Calling Deepseek API
            response = self.llm_client.chat.completions.create(
                # --- MODIFIED FOR DEEPSEEK ---
                model="deepseek-chat",  # Use a Deepseek model
                # --- END MODIFICATION ---
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT}
                ]
            )
            
            final_answer = response.choices[0].message.content
            print("   ✅ LLM generated answer successfully.")
            return final_answer
        
        except Exception as e:
            print(f"❌ LLM API call failed: {e}")
            return f"Error: An error occurred while calling the LLM API: {e}"

def main():
    """
    Main function to test the GraphRAGEngine
    """
    # --- NEW: Load environment variables from .env file ---
    load_dotenv()
    print("Attempting to load environment variables from .env file...")
    # --- END NEW ---

    print("=" * 70)
    print("🚀 GraphRAG Engine Test")
    print("=" * 70)
    
    # --- MODIFIED TO USE ENVIRONMENT VARIABLES ---
    # Load all configuration from environment variables
    # Provide defaults to match your previous hard-coded values
    
    MODEL_PATH = os.environ.get("MODEL_PATH", r"training\models\trial5\han_embeddings.pth")
    
    NEO4J_URI = os.environ.get("NEO4J_URI", "neo4j://127.0.0.1:7687")
    NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD") # No default for password
    
    # Check for Deepseek key
    DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
    
    if not DEEPSEEK_API_KEY:
        print("❌ Error: DEEPSEEK_API_KEY environment variable not found.")
        print("   Please create a .env file and add: DEEPSEEK_API_KEY='your_api_key_here'")
        return
        
    if not NEO4J_PASSWORD:
        print("❌ Error: NEO4J_PASSWORD environment variable not found.")
        print("   Please create a .env file and add: NEO4J_PASSWORD='your_neo4j_password'")
        return

    print("\n--- Configuration Loaded ---")
    print(f"   Model Path: {MODEL_PATH}")
    print(f"   Neo4j URI: {NEO4J_URI}")
    print(f"   Neo4j User: {NEO4J_USERNAME}")
    print("   Neo4j Pass: [HIDDEN]")
    print("   Deepseek Key: [HIDDEN]")
    print("------------------------------\n")
    # --- END MODIFICATION ---
    
    try:
        # 1. Initializing underlying recommender
        print("Initializing AcademicRecommender (Phase 1-3)...")
        recommender = AcademicRecommender(
            model_path=MODEL_PATH,
            neo4j_uri=NEO4J_URI,
            neo4j_username=NEO4J_USERNAME,
            neo4j_password=NEO4J_PASSWORD
        )
        print("\n")
        
        # 2. Initializing GraphRAGEngine (Phase 4)
        print("Initializing GraphRAGEngine (Phase 4)...")
        engine = GraphRAGEngine(recommender, DEEPSEEK_API_KEY) # Pass the Deepseek key
        
        # 3. Running a complex test query (FULL GraphRAG)
        # This matches your proposal's query
        test_query = "Recommend papers using GNNs in recommender systems for data sparsity, listing common graph mining algorithms and two collaborators."
        
        print("\n" + "="*70)
        print("🚀 Test 1: Running FULL GraphRAG Query...")
        print("="*70)
        final_answer_graphrag = engine.query(test_query, top_k_seeds=5, enable_graph_retrieval=True)
        
        # 4. Printing final answer
        print("\n" + "="*40)
        print("🏁 GraphRAG Final Answer (Graph Enabled):")
        print("="*40)
        print(final_answer_graphrag)
        
        # 5. Running Baseline Vector RAG (Ablation Study)
        # Uses the SAME query but disables graph context retrieval
        print("\n" + "="*70)
        print("🚀 Test 2: Running Baseline Vector RAG (Ablation Study)...")
        print("="*70)
        final_answer_vector = engine.query(test_query, top_k_seeds=5, enable_graph_retrieval=False)
        
        print("\n" + "="*40)
        print("🏁 GraphRAG Final Answer (Graph DISABLED):")
        print("="*40)
        print(final_answer_vector)
        
        
        # 6. Running a simple test query
        test_query_simple = "What are the main papers about Graph Attention Networks?"
        print("\n" + "="*70)
        print("🚀 Test 3: Running Simple Query (Full GraphRAG)...")
        print("="*7)
        final_answer_simple = engine.query(test_query_simple, top_k_seeds=3, enable_graph_retrieval=True)
        print("\n" + "="*40)
        print("🏁 GraphRAG Final Answer (Simple Query):")
        print("="*40)
        print(final_answer_simple)
        
    except Exception as e:
        print(f"\n❌ Engine test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()