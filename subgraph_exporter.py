import pandas as pd
from py2neo import Graph
import os
from tqdm import tqdm
import math

class GraphExporter:
    """
    Connects to an existing large Neo4j database, runs queries to find a
    focused subgraph (seeds + 1-hop), and exports that subgraph to
    a new set of CSV files.
    """
    
    def __init__(self, uri="bolt://localhost:7687", username="neo4j", password="your_password", output_dir="data/focused_v1"):
        """
        Initialize Neo4j connection and set output directory.
        """
        try:
            self.graph = Graph(uri, auth=(username, password))
            self.graph.run("RETURN 1")
            print(f"✅ Successfully connected to Neo4j at {uri}")
        except Exception as e:
            print(f"❌ Failed to connect to Neo4j: {e}")
            raise

        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"📁 Created output directory: {self.output_dir}")
        
    def create_indices_for_export(self):
        """
        Creates indices on Paper.year and Paper.venue to speed up
        the initial seed paper query. This is critical for large graphs.
        """
        print("\n🔧 Creating indices for faster exporting (this may take a few minutes)...")
        try:
            self.graph.run("CREATE INDEX paper_year IF NOT EXISTS FOR (p:Paper) ON (p.year)")
            print("   ✅ Index on Paper(year) created.")
            self.graph.run("CREATE INDEX paper_venue IF NOT EXISTS FOR (p:Paper) ON (p.venue)")
            print("   ✅ Index on Paper(venue) created.")
        except Exception as e:
            print(f"   ⚠️  Could not create indices (might be a permissions issue): {e}")

    def _find_seed_papers(self, venue_list, years):
        """
        Finds the initial set of seed paper IDs based on fuzzy venue
        matching and a list of years.
        """
        print("\nStep 1: Finding Seed Papers (CVPR/ICLR/ICML 2020-2022)...")
        
        # Build the fuzzy matching query for venues
        venue_query_parts = []
        for venue in venue_list:
            venue_query_parts.append(f"lower(p.venue) CONTAINS '{venue}'")
        venue_query = " OR ".join(venue_query_parts)
        
        query = f"""
        MATCH (p:Paper)
        WHERE (p.year IN {years}) AND ({venue_query})
        RETURN p.paper_id as paper_id
        """
        
        print("   Running Cypher query to find seeds...")
        try:
            results = self.graph.run(query).data()
            seed_ids = set([r['paper_id'] for r in results])
            print(f"   ✅ Found {len(seed_ids):,} seed papers.")
            return seed_ids
        except Exception as e:
            print(f"   ❌ Query failed: {e}")
            return set()

    def _find_1_hop_papers(self, seed_ids):
        """
        Finds all papers 1-hop away (CITES and CITED_BY) from the seeds.
        """
        print("\nStep 2: Finding 1-Hop Neighborhood Papers...")
        if not seed_ids:
            return set()
        
        seed_ids_list = list(seed_ids)
        all_paper_ids = set(seed_ids)
        
        # We must batch this to avoid query errors
        batch_size = 20000
        
        # Query 1: Find papers CITED BY seeds
        print("   Querying papers CITED BY seeds (forward links)...")
        with tqdm(total=len(seed_ids_list), desc="Cited By") as pbar:
            for i in range(0, len(seed_ids_list), batch_size):
                batch = seed_ids_list[i:i+batch_size]
                query = """
                MATCH (p_seed:Paper)-[:CITES]->(p_cited:Paper)
                WHERE p_seed.paper_id IN $batch
                RETURN DISTINCT p_cited.paper_id as paper_id
                """
                results = self.graph.run(query, batch=batch).data()
                all_paper_ids.update([r['paper_id'] for r in results])
                pbar.update(len(batch))
        
        # Query 2: Find papers that CITE seeds
        print("   Querying papers that CITE seeds (backward links)...")
        with tqdm(total=len(seed_ids_list), desc="Cites") as pbar:
            for i in range(0, len(seed_ids_list), batch_size):
                batch = seed_ids_list[i:i+batch_size]
                query = """
                MATCH (p_cites:Paper)-[:CITES]->(p_seed:Paper)
                WHERE p_seed.paper_id IN $batch
                RETURN DISTINCT p_cites.paper_id as paper_id
                """
                results = self.graph.run(query, batch=batch).data()
                all_paper_ids.update([r['paper_id'] for r in results])
                pbar.update(len(batch))

        print(f"   ✅ Total focused paper set size (seeds + 1-hop): {len(all_paper_ids):,}")
        return all_paper_ids

    def _export_nodes(self, paper_ids_set):
        """
        Exports all nodes (Papers, Authors, Keywords) for the
        focused graph to new CSV files.
        """
        print("\nStep 3: Exporting Node CSVs...")
        paper_ids_list = list(paper_ids_set)
        
        # --- Export papers.csv ---
        print("   Exporting papers.csv...")
        paper_df_list = []
        batch_size = 20000
        
        with tqdm(total=len(paper_ids_list), desc="Papers") as pbar:
            for i in range(0, len(paper_ids_list), batch_size):
                batch = paper_ids_list[i:i+batch_size]
                query = """
                MATCH (p:Paper)
                WHERE p.paper_id IN $batch
                RETURN p.paper_id AS paper_id, p.title AS title, 
                       p.abstract AS abstract, p.year AS year, 
                       p.venue AS venue, p.n_citation AS n_citation
                """
                results = self.graph.run(query, batch=batch).data()
                paper_df_list.append(pd.DataFrame(results))
                pbar.update(len(batch))
        
        papers_df = pd.concat(paper_df_list, ignore_index=True)
        papers_df.to_csv(f"{self.output_dir}/papers.csv", index=False)
        print(f"   ✅ Saved {self.output_dir}/papers.csv ({len(papers_df):,} rows)")
        
        # --- Find and Export authors.csv ---
        print("   Finding & exporting authors.csv...")
        all_author_ids = set()
        author_df_list = []
        
        with tqdm(total=len(paper_ids_list), desc="Authors") as pbar:
            for i in range(0, len(paper_ids_list), batch_size):
                batch = paper_ids_list[i:i+batch_size]
                query = """
                MATCH (p:Paper)-[:WRITTEN_BY]->(a:Author)
                WHERE p.paper_id IN $batch
                RETURN DISTINCT a.author_id AS author_id, a.name AS name, a.org AS org
                """
                results = self.graph.run(query, batch=batch).data()
                batch_df = pd.DataFrame(results)
                new_authors = set(batch_df['author_id']) - all_author_ids
                author_df_list.append(batch_df[batch_df['author_id'].isin(new_authors)])
                all_author_ids.update(new_authors)
                pbar.update(len(batch))
        
        authors_df = pd.concat(author_df_list, ignore_index=True).drop_duplicates(subset=['author_id'])
        authors_df.to_csv(f"{self.output_dir}/authors.csv", index=False)
        print(f"   ✅ Saved {self.output_dir}/authors.csv ({len(authors_df):,} rows)")
        
        # --- Find and Export keywords.csv ---
        print("   Finding & exporting keywords.csv...")
        all_keyword_ids = set()
        keyword_df_list = []

        with tqdm(total=len(paper_ids_list), desc="Keywords") as pbar:
            for i in range(0, len(paper_ids_list), batch_size):
                batch = paper_ids_list[i:i+batch_size]
                query = """
                MATCH (p:Paper)-[:HAS_KEYWORD]->(k:Keyword)
                WHERE p.paper_id IN $batch
                RETURN DISTINCT k.keyword_id AS keyword_id, k.keyword AS keyword
                """
                results = self.graph.run(query, batch=batch).data()
                batch_df = pd.DataFrame(results)
                new_keywords = set(batch_df['keyword_id']) - all_keyword_ids
                keyword_df_list.append(batch_df[batch_df['keyword_id'].isin(new_keywords)])
                all_keyword_ids.update(new_keywords)
                pbar.update(len(batch))

        keywords_df = pd.concat(keyword_df_list, ignore_index=True).drop_duplicates(subset=['keyword_id'])
        keywords_df.to_csv(f"{self.output_dir}/keywords.csv", index=False)
        print(f"   ✅ Saved {self.output_dir}/keywords.csv ({len(keywords_df):,} rows)")

        return all_author_ids, all_keyword_ids

    def _export_relationships(self, paper_ids_set, author_ids_set, keyword_ids_set):
        """
        Exports all relationships (WRITTEN_BY, CITES, HAS_KEYWORD)
        that exist *within* the focused subgraph.
        """
        print("\nStep 4: Exporting Relationship CSVs...")
        paper_ids_list = list(paper_ids_set)
        
        # We need the node ID sets for filtering
        author_ids_list = list(author_ids_set)
        keyword_ids_list = list(keyword_ids_set)

        batch_size = 20000

        # --- Export written_by.csv ---
        print("   Exporting written_by.csv...")
        rels_list = []
        with tqdm(total=len(paper_ids_list), desc="WRITTEN_BY") as pbar:
            for i in range(0, len(paper_ids_list), batch_size):
                batch = paper_ids_list[i:i+batch_size]
                query = """
                MATCH (p:Paper)-[r:WRITTEN_BY]->(a:Author)
                WHERE p.paper_id IN $batch
                RETURN p.paper_id AS paper_id, a.author_id AS author_id, r.author_order AS author_order
                """
                results = self.graph.run(query, batch=batch).data()
                rels_list.extend(results)
                pbar.update(len(batch))
        
        written_by_df = pd.DataFrame(rels_list)
        # Filter: only keep rels where the author is also in our set
        written_by_df = written_by_df[written_by_df['author_id'].isin(author_ids_set)]
        written_by_df.to_csv(f"{self.output_dir}/written_by.csv", index=False)
        print(f"   ✅ Saved {self.output_dir}/written_by.csv ({len(written_by_df):,} rows)")
        
        # --- Export has_keyword.csv ---
        print("   Exporting has_keyword.csv...")
        rels_list = []
        with tqdm(total=len(paper_ids_list), desc="HAS_KEYWORD") as pbar:
            for i in range(0, len(paper_ids_list), batch_size):
                batch = paper_ids_list[i:i+batch_size]
                query = """
                MATCH (p:Paper)-[r:HAS_KEYWORD]->(k:Keyword)
                WHERE p.paper_id IN $batch
                RETURN p.paper_id AS paper_id, k.keyword_id AS keyword_id
                """
                results = self.graph.run(query, batch=batch).data()
                rels_list.extend(results)
                pbar.update(len(batch))
                
        has_keyword_df = pd.DataFrame(rels_list)
        # Filter: only keep rels where the keyword is also in our set
        has_keyword_df = has_keyword_df[has_keyword_df['keyword_id'].isin(keyword_ids_set)]
        has_keyword_df.to_csv(f"{self.output_dir}/has_keyword.csv", index=False)
        print(f"   ✅ Saved {self.output_dir}/has_keyword.csv ({len(has_keyword_df):,} rows)")

        # --- Export cites.csv ---
        print("   Exporting cites.csv...")
        rels_list = []
        with tqdm(total=len(paper_ids_list), desc="CITES") as pbar:
            for i in range(0, len(paper_ids_list), batch_size):
                batch = paper_ids_list[i:i+batch_size]
                query = """
                MATCH (p1:Paper)-[r:CITES]->(p2:Paper)
                WHERE p1.paper_id IN $batch
                RETURN p1.paper_id as citing_paper, p2.paper_id as cited_paper
                """
                results = self.graph.run(query, batch=batch).data()
                rels_list.extend(results)
                pbar.update(len(batch))
        
        cites_df = pd.DataFrame(rels_list)
        # Filter: only keep rels where BOTH papers are in our set
        cites_df = cites_df[cites_df['cited_paper'].isin(paper_ids_set)]
        cites_df.to_csv(f"{self.output_dir}/cites.csv", index=False)
        print(f"   ✅ Saved {self.output_dir}/cites.csv ({len(cites_df):,} rows)")

    def run_export(self, venue_list, years):
        """
        Executes the full export workflow.
        """
        print("=" * 70)
        print("🚀 Starting Focused Graph Export")
        print("=" * 70)
        
        self.create_indices_for_export()
        
        # Step 1: Find seeds
        seed_ids = self._find_seed_papers(venue_list, years)
        if not seed_ids:
            print("❌ No seed papers found. Aborting.")
            return

        # Step 2: Find 1-hop
        all_paper_ids = self._find_1_hop_papers(seed_ids)
        if not all_paper_ids:
            print("❌ No papers found in 1-hop neighborhood. Aborting.")
            return
            
        # Step 3: Export Nodes
        all_author_ids, all_keyword_ids = self._export_nodes(all_paper_ids)
        
        # Step 4: Export Relationships
        self._export_relationships(all_paper_ids, all_author_ids, all_keyword_ids)
        
        print("\n" + "=" * 70)
        print("✅ Export completed successfully!")
        print(f"   Data saved to: {self.output_dir}")
        print("=" * 70)

if __name__ == "__main__":
    import sys
    
    # --- Configuration ---
    
    # This is the password for your EXISTING (large) database
    NEO4J_PASSWORD = "12345678"  # <-- IMPORTANT: Use your "combined" DB password
    NEO4J_URI = "neo4j://127.0.0.1:7687"
    NEO4J_USERNAME = "neo4j"
    
    # This is where the new CSVs will be saved
    OUTPUT_DIR = "data/focused_v1"
    
    # Years for the seed papers
    SEED_YEARS = [2019, 2020, 2021, 2022]
    
    # Fuzzy venue list. We use lowercase and 'contains'
    # This matches "ICLR", "ICLR 2020", "International Conference on Learning Representations"
    SEED_VENUE_LIST = [
        "cvpr",
        "computer vision and pattern recognition",
        "iclr",
        "international conference on learning representations",
        "icml",
        "international conference on machine learning"
    ]
    
    # ---------------------
    
    print("=" * 70)
    print("🚀 Neo4j Focused Graph Exporter")
    print("=" * 70)
    print(f"   Connecting to: {NEO4J_URI}")
    print(f"   Exporting to:  {OUTPUT_DIR}")
    print(f"   Seed Venues:   {', '.join(SEED_VENUE_LIST)}")
    print(f"   Seed Years:    {SEED_YEARS}")
    print("\n⚠️  This will query your LARGE database and may take time.")
    
    try:
        exporter = GraphExporter(
            uri=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            output_dir=OUTPUT_DIR
        )
        
        exporter.run_export(SEED_VENUE_LIST, SEED_YEARS)
        
    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        import traceback
        traceback.print_exc()