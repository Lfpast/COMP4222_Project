from py2neo import Graph, Node, Relationship
import pandas as pd
import os

class Neo4jGraphBuilder:
    def __init__(self, uri="bolt://localhost:7687", username="neo4j", password="your_password"):
        self.graph = Graph(uri, auth=(username, password))
        
    def clear_database(self):
        """清空现有数据"""
        self.graph.run("MATCH (n) DETACH DELETE n")
        print("Database cleared")
    
    def create_constraints(self):
        """创建约束确保数据完整性"""
        constraints = [
            "CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE",
            "CREATE CONSTRAINT author_id IF NOT EXISTS FOR (a:Author) REQUIRE a.author_id IS UNIQUE",
            "CREATE CONSTRAINT keyword_id IF NOT EXISTS FOR (k:Keyword) REQUIRE k.keyword_id IS UNIQUE"
        ]
        
        for constraint in constraints:
            try:
                self.graph.run(constraint)
            except Exception as e:
                print(f"Constraint may already exist: {e}")
    
    def import_nodes(self):
        """导入节点数据"""
        data_dir = "data"
        
        # 导入论文节点
        papers_df = pd.read_csv(f"{data_dir}/papers.csv")
        for _, row in papers_df.iterrows():
            paper = Node("Paper", 
                        paper_id=row["paper_id"],
                        title=row["title"],
                        abstract=row.get("abstract", ""),
                        year=row.get("year", ""))
            self.graph.create(paper)
        print(f"Imported {len(papers_df)} papers")
        
        # 导入作者节点
        authors_df = pd.read_csv(f"{data_dir}/authors.csv")
        for _, row in authors_df.iterrows():
            author = Node("Author",
                         author_id=row["author_id"],
                         name=row["name"])
            self.graph.create(author)
        print(f"Imported {len(authors_df)} authors")
        
        # 导入关键词节点
        keywords_df = pd.read_csv(f"{data_dir}/keywords.csv")
        for _, row in keywords_df.iterrows():
            keyword = Node("Keyword",
                          keyword_id=row["keyword_id"],
                          keyword=row["keyword"])
            self.graph.create(keyword)
        print(f"Imported {len(keywords_df)} keywords")
    
    def import_relationships(self):
        """导入关系数据"""
        data_dir = "data"
        
        # WRITTEN_BY 关系
        written_by_df = pd.read_csv(f"{data_dir}/written_by.csv")
        for _, row in written_by_df.iterrows():
            query = """
            MATCH (p:Paper {paper_id: $paper_id}), (a:Author {author_id: $author_id})
            CREATE (p)-[:WRITTEN_BY]->(a)
            """
            self.graph.run(query, paper_id=row["paper_id"], author_id=row["author_id"])
        print(f"Created {len(written_by_df)} WRITTEN_BY relationships")
        
        # CITES 关系
        cites_df = pd.read_csv(f"{data_dir}/cites.csv")
        for _, row in cites_df.iterrows():
            query = """
            MATCH (citing:Paper {paper_id: $citing_paper}), (cited:Paper {paper_id: $cited_paper})
            CREATE (citing)-[:CITES]->(cited)
            """
            self.graph.run(query, citing_paper=row["citing_paper"], cited_paper=row["cited_paper"])
        print(f"Created {len(cites_df)} CITES relationships")
        
        # HAS_KEYWORD 关系
        has_keyword_df = pd.read_csv(f"{data_dir}/has_keyword.csv")
        for _, row in has_keyword_df.iterrows():
            query = """
            MATCH (p:Paper {paper_id: $paper_id}), (k:Keyword {keyword_id: $keyword_id})
            CREATE (p)-[:HAS_KEYWORD]->(k)
            """
            self.graph.run(query, paper_id=row["paper_id"], keyword_id=row["keyword_id"])
        print(f"Created {len(has_keyword_df)} HAS_KEYWORD relationships")
    
    def test_queries(self):
        """测试查询验证数据"""
        # 查询示例：找到所有关于GNN的论文
        result = self.graph.run("""
        MATCH (p:Paper)-[:HAS_KEYWORD]->(k:Keyword)
        WHERE k.keyword CONTAINS 'GNN'
        RETURN p.title, k.keyword
        LIMIT 5
        """)
        
        print("Sample query results:")
        for record in result:
            print(f"Paper: {record['p.title']}, Keyword: {record['k.keyword']}")

if __name__ == "__main__":
    builder = Neo4jGraphBuilder(password="your_password_here")
    builder.clear_database()
    builder.create_constraints()
    builder.import_nodes()
    builder.import_relationships()
    builder.test_queries()