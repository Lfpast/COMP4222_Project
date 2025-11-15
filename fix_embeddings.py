"""快速修复：为现有模型添加原始 Sentence-BERT embeddings"""
import torch
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from py2neo import Graph
from tqdm import tqdm
import sys
import io

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("="*70)
print("Adding Original Embeddings to Model")
print("="*70)

MODEL_PATH = "training/models/link_prediction_v1/han_embeddings.pth"

# 1. Load model
print(f"\nLoading model from: {MODEL_PATH}")
checkpoint = torch.load(MODEL_PATH, map_location='cpu')

if 'original_embeddings' in checkpoint:
    print("\nOriginal embeddings already exist! Nothing to do.")
    sys.exit(0)

# 2. Connect to Neo4j
print("\nConnecting to Neo4j...")
graph = Graph("neo4j://127.0.0.1:7687", auth=("neo4j", "12345678"))

# 3. Load Sentence Transformer
print("Loading Sentence Transformer...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# 4. Get IDs
id_maps = checkpoint['id_maps']
paper_ids = list(id_maps['paper'].keys())
author_ids = list(id_maps['author'].keys())
keyword_ids = list(id_maps['keyword'].keys())

print(f"\nProcessing {len(paper_ids)} papers, {len(author_ids)} authors, {len(keyword_ids)} keywords")

# 5. Load data from Neo4j
print("\nLoading papers from Neo4j...")
papers = graph.run("""
MATCH (p:Paper) WHERE p.paper_id IN $ids
RETURN p.paper_id as id, p.title as title, p.abstract as abstract
""", ids=paper_ids).data()
paper_map = {p['id']: (p['title'], p.get('abstract', '')) for p in papers}

print("Loading authors...")
authors = graph.run("""
MATCH (a:Author) WHERE a.author_id IN $ids
RETURN a.author_id as id, a.name as name
""", ids=author_ids).data()
author_map = {a['id']: a['name'] for a in authors}

print("Loading keywords...")
keywords = graph.run("""
MATCH (k:Keyword) WHERE k.keyword_id IN $ids
RETURN k.keyword_id as id, k.keyword as keyword
""", ids=keyword_ids).data()
keyword_map = {k['id']: k['keyword'] for k in keywords}

# 6. Generate embeddings in correct order
print("\nGenerating embeddings...")

print("  Papers...")
paper_texts = []
for pid in paper_ids:
    if pid in paper_map:
        title, abstract = paper_map[pid]
        text = str(title)
        if abstract:
            text += " " + str(abstract)
    else:
        text = f"Paper {pid}"
    paper_texts.append(text)
paper_emb = torch.FloatTensor(model.encode(paper_texts, show_progress_bar=True, batch_size=128))

print("  Authors...")
author_names = [author_map.get(aid, 'Unknown') for aid in author_ids]
author_emb = torch.FloatTensor(model.encode(author_names, show_progress_bar=True, batch_size=128))

print("  Keywords...")
keyword_texts = [keyword_map.get(kid, '') for kid in keyword_ids]
keyword_emb = torch.FloatTensor(model.encode(keyword_texts, show_progress_bar=True, batch_size=128))

# 7. Add to checkpoint
checkpoint['original_embeddings'] = {
    'paper': paper_emb,
    'author': author_emb,
    'keyword': keyword_emb
}

# 8. Backup and save
print("\nCreating backup...")
torch.save(torch.load(MODEL_PATH, map_location='cpu'), MODEL_PATH + ".backup")

print("Saving updated model...")
torch.save(checkpoint, MODEL_PATH)

print("\n" + "="*70)
print("SUCCESS! Model updated with original embeddings")
print("="*70)
print("\nNow run: python graph_rag_engine.py")
