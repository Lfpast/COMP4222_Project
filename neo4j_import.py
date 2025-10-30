from py2neo import Graph, Node, Relationship
import pandas as pd
import os
from tqdm import tqdm

class Neo4jGraphBuilder:
    def __init__(self, uri="bolt://localhost:7687", username="neo4j", password="your_password"):
        """初始化Neo4j连接"""
        try:
            self.graph = Graph(uri, auth=(username, password))
            # 测试连接
            self.graph.run("RETURN 1")
            print(f"✅ Successfully connected to Neo4j at {uri}")
        except Exception as e:
            print(f"❌ Failed to connect to Neo4j: {e}")
            print("\n💡 Please ensure:")
            print("   1. Neo4j is running")
            print("   2. URI is correct (default: bolt://localhost:7687)")
            print("   3. Username and password are correct")
            raise
        
        self.data_dir = "data/processed"
        
    def clear_database(self):
        """清空现有数据"""
        print("\n🗑️  Clearing existing data...")
        self.graph.run("MATCH (n) DETACH DELETE n")
        print("✅ Database cleared")
    
    def create_constraints(self):
        """创建约束和索引以提高性能"""
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
        """使用批量导入节点数据(提高性能)"""
        print("\n📥 Importing nodes...")
        
        # 导入论文节点 (批量导入)
        print("\n📄 Importing papers...")
        papers_df = pd.read_csv(f"{self.data_dir}/papers.csv")
        
        # 处理NaN值
        papers_df = papers_df.fillna('')
        
        # 批量导入 - 使用UNWIND提高性能
        batch_size = 5000
        for i in tqdm(range(0, len(papers_df), batch_size), desc="Papers"):
            batch = papers_df.iloc[i:i+batch_size]
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
        
        print(f"   ✅ Imported {len(papers_df)} papers")
        
        # 导入作者节点
        print("\n👥 Importing authors...")
        authors_df = pd.read_csv(f"{self.data_dir}/authors.csv")
        authors_df = authors_df.fillna('')
        
        for i in tqdm(range(0, len(authors_df), batch_size), desc="Authors"):
            batch = authors_df.iloc[i:i+batch_size]
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
        
        print(f"   ✅ Imported {len(authors_df)} authors")
        
        # 导入关键词节点
        print("\n🏷️  Importing keywords...")
        keywords_df = pd.read_csv(f"{self.data_dir}/keywords.csv")
        keywords_df = keywords_df.fillna('')
        
        for i in tqdm(range(0, len(keywords_df), batch_size), desc="Keywords"):
            batch = keywords_df.iloc[i:i+batch_size]
            keywords_data = batch.to_dict('records')
            
            query = """
            UNWIND $keywords AS kw
            CREATE (k:Keyword {
                keyword_id: kw.keyword_id,
                keyword: kw.keyword
            })
            """
            self.graph.run(query, keywords=keywords_data)
        
        print(f"   ✅ Imported {len(keywords_df)} keywords")
    
    def import_relationships(self):
        """批量导入关系数据"""
        print("\n🔗 Importing relationships...")
        batch_size = 10000
        
        # WRITTEN_BY 关系
        print("\n✍️  Creating WRITTEN_BY relationships...")
        written_by_df = pd.read_csv(f"{self.data_dir}/written_by.csv")
        written_by_df = written_by_df.fillna('')
        
        for i in tqdm(range(0, len(written_by_df), batch_size), desc="WRITTEN_BY"):
            batch = written_by_df.iloc[i:i+batch_size]
            rels_data = batch.to_dict('records')
            
            query = """
            UNWIND $rels AS rel
            MATCH (p:Paper {paper_id: rel.paper_id})
            MATCH (a:Author {author_id: rel.author_id})
            CREATE (p)-[:WRITTEN_BY {author_order: rel.author_order}]->(a)
            """
            self.graph.run(query, rels=rels_data)
        
        print(f"   ✅ Created {len(written_by_df)} WRITTEN_BY relationships")
        
        # CITES 关系
        print("\n📚 Creating CITES relationships...")
        cites_df = pd.read_csv(f"{self.data_dir}/cites.csv")
        cites_df = cites_df.fillna('')
        
        for i in tqdm(range(0, len(cites_df), batch_size), desc="CITES"):
            batch = cites_df.iloc[i:i+batch_size]
            rels_data = batch.to_dict('records')
            
            query = """
            UNWIND $rels AS rel
            MATCH (citing:Paper {paper_id: rel.citing_paper})
            MATCH (cited:Paper {paper_id: rel.cited_paper})
            CREATE (citing)-[:CITES]->(cited)
            """
            self.graph.run(query, rels=rels_data)
        
        print(f"   ✅ Created {len(cites_df)} CITES relationships")
        
        # HAS_KEYWORD 关系
        print("\n🏷️  Creating HAS_KEYWORD relationships...")
        has_keyword_df = pd.read_csv(f"{self.data_dir}/has_keyword.csv")
        has_keyword_df = has_keyword_df.fillna('')
        
        for i in tqdm(range(0, len(has_keyword_df), batch_size), desc="HAS_KEYWORD"):
            batch = has_keyword_df.iloc[i:i+batch_size]
            rels_data = batch.to_dict('records')
            
            query = """
            UNWIND $rels AS rel
            MATCH (p:Paper {paper_id: rel.paper_id})
            MATCH (k:Keyword {keyword_id: rel.keyword_id})
            CREATE (p)-[:HAS_KEYWORD]->(k)
            """
            self.graph.run(query, rels=rels_data)
        
        print(f"   ✅ Created {len(has_keyword_df)} HAS_KEYWORD relationships")
    
    def verify_import(self):
        """验证导入的数据"""
        print("\n📊 Verifying imported data...")
        
        # 统计节点数量
        paper_count = self.graph.run("MATCH (p:Paper) RETURN count(p) as count").data()[0]['count']
        author_count = self.graph.run("MATCH (a:Author) RETURN count(a) as count").data()[0]['count']
        keyword_count = self.graph.run("MATCH (k:Keyword) RETURN count(k) as count").data()[0]['count']
        
        print(f"   📄 Papers: {paper_count:,}")
        print(f"   👥 Authors: {author_count:,}")
        print(f"   🏷️  Keywords: {keyword_count:,}")
        
        # 统计关系数量
        written_by_count = self.graph.run("MATCH ()-[r:WRITTEN_BY]->() RETURN count(r) as count").data()[0]['count']
        cites_count = self.graph.run("MATCH ()-[r:CITES]->() RETURN count(r) as count").data()[0]['count']
        has_keyword_count = self.graph.run("MATCH ()-[r:HAS_KEYWORD]->() RETURN count(r) as count").data()[0]['count']
        
        print(f"\n   ✍️  WRITTEN_BY: {written_by_count:,}")
        print(f"   📚 CITES: {cites_count:,}")
        print(f"   🏷️  HAS_KEYWORD: {has_keyword_count:,}")
        
        # 示例查询
        print("\n🔍 Sample queries:")
        
        # 最多引用的论文
        result = self.graph.run("""
        MATCH (p:Paper)<-[:CITES]-(citing:Paper)
        RETURN p.title as title, p.year as year, count(citing) as citations
        ORDER BY citations DESC
        LIMIT 5
        """).data()
        
        print("\n   Top 5 most cited papers:")
        for i, record in enumerate(result, 1):
            print(f"   {i}. [{record['year']}] {record['title'][:60]}... ({record['citations']} citations)")
        
        # 最高产的作者
        result = self.graph.run("""
        MATCH (p:Paper)-[:WRITTEN_BY]->(a:Author)
        RETURN a.name as name, count(p) as papers
        ORDER BY papers DESC
        LIMIT 5
        """).data()
        
        print("\n   Top 5 most prolific authors:")
        for i, record in enumerate(result, 1):
            print(f"   {i}. {record['name']} ({record['papers']} papers)")
        
        # 最热门的关键词
        result = self.graph.run("""
        MATCH (p:Paper)-[:HAS_KEYWORD]->(k:Keyword)
        RETURN k.keyword as keyword, count(p) as papers
        ORDER BY papers DESC
        LIMIT 10
        """).data()
        
        print("\n   Top 10 most common keywords:")
        for i, record in enumerate(result, 1):
            print(f"   {i}. {record['keyword']} ({record['papers']} papers)")

if __name__ == "__main__":
    print("=" * 70)
    print("🚀 Neo4j Academic Graph Import")
    print("=" * 70)
    
    # 配置参数 - 请修改为你的Neo4j配置
    NEO4J_URI = "neo4j://127.0.0.1:7687"  # Neo4j Desktop默认端口
    NEO4J_USERNAME = "neo4j"  # 默认用户名
    NEO4J_PASSWORD = "Jackson050609"  # ⚠️ 修改为你创建数据库时设置的密码
    
    print(f"\n⚙️  Configuration:")
    print(f"   URI: {NEO4J_URI}")
    print(f"   Username: {NEO4J_USERNAME}")
    print(f"   Password: {'*' * len(NEO4J_PASSWORD)}")
    print(f"   Data directory: data/processed/")
    
    # 确认是否清空数据库
    print("\n⚠️  WARNING: This will clear all existing data in the Neo4j database!")
    response = input("Continue? (yes/no): ").strip().lower()
    
    if response != 'yes':
        print("❌ Import cancelled")
        exit(0)
    
    if response != 'yes':
        print("❌ Import cancelled")
        exit(0)
    
    try:
        # 初始化导入器
        builder = Neo4jGraphBuilder(
            uri=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD
        )
        
        # 执行导入流程
        builder.clear_database()
        builder.create_constraints()
        builder.import_nodes()
        builder.import_relationships()
        builder.verify_import()
        
        print("\n" + "=" * 70)
        print("✅ Import completed successfully!")
        print("=" * 70)
        
        print("\n📋 Next steps:")
        print("   1. Open Neo4j Browser: http://localhost:7474")
        print("   2. Run sample queries to explore the graph")
        print("   3. Train HAN model using han_model.py")
        
    except Exception as e:
        print(f"\n❌ Import failed: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Check Neo4j is running: neo4j status")
        print("   2. Verify connection settings (URI, username, password)")
        print("   3. Check CSV files exist in data/processed/")
        print("   4. Ensure sufficient memory for large dataset")
        exit(1)