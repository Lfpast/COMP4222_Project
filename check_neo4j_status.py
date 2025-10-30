"""检查Neo4j数据库状态和性能"""
from py2neo import Graph
import time

# 修改为你的配置
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "Jackson050609"    # Your password

print("=" * 70)
print("🔍 Neo4j Database Status Check")
print("=" * 70)

try:
    print("\n1️⃣ Connecting to Neo4j...")
    graph = Graph(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    print("   ✅ Connected successfully")
    
    print("\n2️⃣ Checking database statistics...")
    
    # 检查节点数量
    print("\n   📊 Node counts:")
    start = time.time()
    paper_count = graph.run("MATCH (p:Paper) RETURN count(p) as count").data()[0]['count']
    print(f"      Papers: {paper_count:,} ({time.time()-start:.2f}s)")
    
    start = time.time()
    author_count = graph.run("MATCH (a:Author) RETURN count(a) as count").data()[0]['count']
    print(f"      Authors: {author_count:,} ({time.time()-start:.2f}s)")
    
    start = time.time()
    keyword_count = graph.run("MATCH (k:Keyword) RETURN count(k) as count").data()[0]['count']
    print(f"      Keywords: {keyword_count:,} ({time.time()-start:.2f}s)")
    
    # 检查关系数量
    print("\n   🔗 Relationship counts:")
    start = time.time()
    written_by = graph.run("MATCH ()-[r:WRITTEN_BY]->() RETURN count(r) as count").data()[0]['count']
    print(f"      WRITTEN_BY: {written_by:,} ({time.time()-start:.2f}s)")
    
    start = time.time()
    cites = graph.run("MATCH ()-[r:CITES]->() RETURN count(r) as count").data()[0]['count']
    print(f"      CITES: {cites:,} ({time.time()-start:.2f}s)")
    
    start = time.time()
    has_keyword = graph.run("MATCH ()-[r:HAS_KEYWORD]->() RETURN count(r) as count").data()[0]['count']
    print(f"      HAS_KEYWORD: {has_keyword:,} ({time.time()-start:.2f}s)")
    
    # 测试简单查询
    print("\n3️⃣ Testing simple query...")
    start = time.time()
    result = graph.run("MATCH (p:Paper) RETURN p.title LIMIT 1").data()
    print(f"   ✅ Query successful ({time.time()-start:.2f}s)")
    print(f"   Sample: {result[0]['p.title'][:60]}...")
    
    # 检查索引
    print("\n4️⃣ Checking indexes and constraints...")
    indexes = graph.run("SHOW INDEXES").data()
    print(f"   Total indexes: {len(indexes)}")
    for idx in indexes[:5]:
        print(f"      - {idx.get('name', 'N/A')}: {idx.get('labelsOrTypes', 'N/A')}")
    
    print("\n" + "=" * 70)
    print("✅ Database is healthy and responsive!")
    print("=" * 70)
    
    print("\n💡 Tips for using Neo4j Browser:")
    print("   1. Be patient - first load takes 10-30 seconds")
    print("   2. Start with small queries: MATCH (p:Paper) RETURN p LIMIT 10")
    print("   3. Avoid SELECT ALL without LIMIT")
    print("   4. Use WHERE clauses to filter data")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\n💡 If connection times out:")
    print("   1. Check Neo4j Desktop shows database as 'Active'")
    print("   2. Restart the database in Neo4j Desktop")
    print("   3. Increase memory in Settings → DBMS Settings → dbms.memory.heap.max_size")
