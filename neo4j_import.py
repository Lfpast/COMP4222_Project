from py2neo import Graph
import pandas as pd
import os
from tqdm import tqdm
import argparse

class Neo4jGraphBuilder:
    def __init__(self, uri, username, password, data_dir, batch_size):
        self.data_dir = data_dir
        self.batch_size = batch_size
        
        try:
            self.graph = Graph(uri, auth=(username, password))
            # Test connection
            self.graph.run("RETURN 1")
            print(f"✅ Successfully connected to Neo4j at {uri}")
        except Exception as e:
            print(f"❌ Failed to connect to Neo4j: {e}")
            print("\n💡 Please ensure:")
            print("   1. Neo4j is running")
            print("   2. URI is correct (default: bolt://localhost:7687)")
            print("   3. Username and password are correct")
            raise
        
    def clear_database(self):
        """Clear existing data with batching to avoid memory issues"""
        print("\n🗑️  Clearing existing data (this may take a while for large datasets)...")
        
        # First, count total nodes
        try:
            total_nodes_result = self.graph.run("MATCH (n) RETURN count(n) as total").data()
            total_nodes = total_nodes_result[0]['total'] if total_nodes_result else 0
        except:
            total_nodes = 0
        
        if total_nodes == 0:
            print("   ℹ️  Database is already empty")
            return
        
        # Use batch deletion to avoid memory issues
        deleted = 0
        
        # Create progress bar
        pbar = tqdm(total=total_nodes, desc="Clearing database", unit="nodes")
        
        while True:
            result = self.graph.run(f"""
            MATCH (n) 
            WITH n LIMIT {self.batch_size}
            DETACH DELETE n
            RETURN count(*) as deleted
            """).data()
            
            count = result[0]['deleted']
            if count == 0:
                break
            
            deleted += count
            pbar.update(count)
        
        pbar.close()
        print(f"✅ Database cleared ({deleted:,} total nodes deleted)")
    
    def create_constraints(self):
        """Create constraints and indexes to improve performance"""
        print("\n🔧 Creating constraints and indexes...")
        
        constraints = [
            "CREATE CONSTRAINT paper_id_unique IF NOT EXISTS FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE",
            "CREATE CONSTRAINT author_id_unique IF NOT EXISTS FOR (a:Author) REQUIRE a.author_id IS UNIQUE",
            "CREATE CONSTRAINT keyword_id_unique IF NOT EXISTS FOR (k:Keyword) REQUIRE k.keyword_id IS UNIQUE"
        ]
        
        for constraint in constraints:
            try:
                self.graph.run(constraint)
                print(f"   ✅ {constraint.split()[1]}")
            except Exception as e:
                if "already exists" in str(e).lower() or "equivalent" in str(e).lower():
                    print(f"   ℹ️  Constraint already exists")
                else:
                    print(f"   ⚠️  {e}")
        
        print("✅ Constraints created")
    
    def import_nodes(self):
        """Import node data using batch operations (improved performance)"""
        print("\n📥 Importing nodes...")
        
        # Import paper nodes (batch import)
        print("\n📄 Importing papers...")
        papers_df = pd.read_csv(f"{self.data_dir}/papers.csv")
        
        # Handle NaN values
        papers_df = papers_df.fillna('')
        
        # Batch import - using UNWIND for better performance
        with tqdm(total=len(papers_df), desc="Papers", unit="paper") as pbar:
            for i in range(0, len(papers_df), self.batch_size):
                batch = papers_df.iloc[i:i+self.batch_size]
                papers_data = batch.to_dict('records')
                
                query = """
                UNWIND $papers AS paper
                CREATE (p:Paper {
                    paper_id: paper.paper_id,
                    title: paper.title,
                    abstract: paper.abstract,
                    year: paper.year,
                    venue: paper.venue,
                    n_citation: paper.n_citation
                })
                """
                self.graph.run(query, papers=papers_data)
                pbar.update(len(batch))
        
        print(f"   ✅ Imported {len(papers_df):,} papers")
        
        # Import author nodes
        print("\n👥 Importing authors...")
        authors_df = pd.read_csv(f"{self.data_dir}/authors.csv")
        authors_df = authors_df.fillna('')
        
        with tqdm(total=len(authors_df), desc="Authors", unit="author") as pbar:
            for i in range(0, len(authors_df), self.batch_size):
                batch = authors_df.iloc[i:i+self.batch_size]
                authors_data = batch.to_dict('records')
                
                query = """
                UNWIND $authors AS author
                CREATE (a:Author {
                    author_id: author.author_id,
                    name: author.name,
                    org: author.org
                })
                """
                self.graph.run(query, authors=authors_data)
                pbar.update(len(batch))
        
        print(f"   ✅ Imported {len(authors_df):,} authors")
        
        # Import keyword nodes
        print("\n🏷️  Importing keywords...")
        keywords_df = pd.read_csv(f"{self.data_dir}/keywords.csv")
        keywords_df = keywords_df.fillna('')
        
        with tqdm(total=len(keywords_df), desc="Keywords", unit="keyword") as pbar:
            for i in range(0, len(keywords_df), self.batch_size):
                batch = keywords_df.iloc[i:i+self.batch_size]
                keywords_data = batch.to_dict('records')
                
                query = """
                UNWIND $keywords AS kw
                CREATE (k:Keyword {
                    keyword_id: kw.keyword_id,
                    keyword: kw.keyword
                })
                """
                self.graph.run(query, keywords=keywords_data)
                pbar.update(len(batch))
        
        print(f"   ✅ Imported {len(keywords_df):,} keywords")
    
    def import_relationships(self):
        """Import relationship data in batches"""
        print("\n🔗 Importing relationships...")
        
        # WRITTEN_BY relationship
        print("\n✍️  Creating WRITTEN_BY relationships...")
        written_by_df = pd.read_csv(f"{self.data_dir}/written_by.csv")
        written_by_df = written_by_df.fillna('')
        
        with tqdm(total=len(written_by_df), desc="WRITTEN_BY", unit="rel") as pbar:
            for i in range(0, len(written_by_df), self.batch_size):
                batch = written_by_df.iloc[i:i+self.batch_size]
                rels_data = batch.to_dict('records')
                
                query = """
                UNWIND $rels AS rel
                MATCH (p:Paper {paper_id: rel.paper_id})
                MATCH (a:Author {author_id: rel.author_id})
                CREATE (p)-[:WRITTEN_BY {author_order: rel.author_order}]->(a)
                """
                self.graph.run(query, rels=rels_data)
                pbar.update(len(batch))
        
        print(f"   ✅ Created {len(written_by_df):,} WRITTEN_BY relationships")
        
        # CITES relationship
        print("\n📚 Creating CITES relationships...")
        cites_df = pd.read_csv(f"{self.data_dir}/cites.csv")
        cites_df = cites_df.fillna('')
        
        with tqdm(total=len(cites_df), desc="CITES", unit="rel") as pbar:
            for i in range(0, len(cites_df), self.batch_size):
                batch = cites_df.iloc[i:i+self.batch_size]
                rels_data = batch.to_dict('records')
                
                query = """
                UNWIND $rels AS rel
                MATCH (citing:Paper {paper_id: rel.citing_paper})
                MATCH (cited:Paper {paper_id: rel.cited_paper})
                CREATE (citing)-[:CITES]->(cited)
                """
                self.graph.run(query, rels=rels_data)
                pbar.update(len(batch))
        
        print(f"   ✅ Created {len(cites_df):,} CITES relationships")
        
        # HAS_KEYWORD relationship
        print("\n🏷️  Creating HAS_KEYWORD relationships...")
        has_keyword_df = pd.read_csv(f"{self.data_dir}/has_keyword.csv")
        has_keyword_df = has_keyword_df.fillna('')
        
        with tqdm(total=len(has_keyword_df), desc="HAS_KEYWORD", unit="rel") as pbar:
            for i in range(0, len(has_keyword_df), self.batch_size):
                batch = has_keyword_df.iloc[i:i+self.batch_size]
                rels_data = batch.to_dict('records')
                
                query = """
                UNWIND $rels AS rel
                MATCH (p:Paper {paper_id: rel.paper_id})
                MATCH (k:Keyword {keyword_id: rel.keyword_id})
                CREATE (p)-[:HAS_KEYWORD]->(k)
                """
                self.graph.run(query, rels=rels_data)
                pbar.update(len(batch))
        
        print(f"   ✅ Created {len(has_keyword_df):,} HAS_KEYWORD relationships")
    
    def verify_import(self):
        """Verify imported data"""
        print("\n📊 Verifying imported data...")
        
        # Count nodes
        paper_count = self.graph.run("MATCH (p:Paper) RETURN count(p) as count").data()[0]['count']
        author_count = self.graph.run("MATCH (a:Author) RETURN count(a) as count").data()[0]['count']
        keyword_count = self.graph.run("MATCH (k:Keyword) RETURN count(k) as count").data()[0]['count']
        
        print(f"   📄 Papers: {paper_count:,}")
        print(f"   👥 Authors: {author_count:,}")
        print(f"   🏷️  Keywords: {keyword_count:,}")
        
        # Count relationships
        written_by_count = self.graph.run("MATCH ()-[r:WRITTEN_BY]->() RETURN count(r) as count").data()[0]['count']
        cites_count = self.graph.run("MATCH ()-[r:CITES]->() RETURN count(r) as count").data()[0]['count']
        has_keyword_count = self.graph.run("MATCH ()-[r:HAS_KEYWORD]->() RETURN count(r) as count").data()[0]['count']
        
        print(f"\n   ✍️  WRITTEN_BY: {written_by_count:,}")
        print(f"   📚 CITES: {cites_count:,}")
        print(f"   🏷️  HAS_KEYWORD: {has_keyword_count:,}")
        
        # Sample queries
        print("\n🔍 Sample queries:")
        
        # Most cited papers
        result = self.graph.run("""
        MATCH (p:Paper)<-[:CITES]-(citing:Paper)
        RETURN p.title as title, p.year as year, count(citing) as citations
        ORDER BY citations DESC
        LIMIT 5
        """).data()
        
        print("\n   Top 5 most cited papers:")
        for i, record in enumerate(result, 1):
            print(f"   {i}. [{record['year']}] {record['title'][:60]}... ({record['citations']} citations)")
        
        # Most prolific authors
        result = self.graph.run("""
        MATCH (p:Paper)-[:WRITTEN_BY]->(a:Author)
        RETURN a.name as name, count(p) as papers
        ORDER BY papers DESC
        LIMIT 5
        """).data()
        
        print("\n   Top 5 most prolific authors:")
        for i, record in enumerate(result, 1):
            print(f"   {i}. {record['name']} ({record['papers']} papers)")
        
        # Most common keywords
        result = self.graph.run("""
        MATCH (p:Paper)-[:HAS_KEYWORD]->(k:Keyword)
        RETURN k.keyword as keyword, count(p) as papers
        ORDER BY papers DESC
        LIMIT 10
        """).data()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Neo4j Import Pipeline")
    parser.add_argument('--uri', type=str, required=True, help="Neo4j connection URI")
    parser.add_argument('--username', type=str, required=True, help="Neo4j username")
    parser.add_argument('--password', type=str, required=True, help="Neo4j password")
    parser.add_argument('--data_dir', type=str, required=True, help="Data directory path (e.g., data/processed)")
    parser.add_argument('--batch_size', type=int, required=True, help="Batch size for import operations")
    args = parser.parse_args()

    try:
        builder = Neo4jGraphBuilder(
            uri=args.uri,
            username=args.username,
            password=args.password,
            data_dir=args.data_dir,
            batch_size=args.batch_size
        )
        builder.clear_database()
        builder.create_constraints()
        builder.import_nodes()
        builder.import_relationships()
        builder.verify_import()
        
        print("\n" + "=" * 70)
        print("✅ Import completed successfully!")
        print("=" * 70)
        print(f"\n📊 Imported data from: {args.data_dir}")
        
    except Exception as e:
        print(f"\n❌ Import failed: {e}")
        exit(1)