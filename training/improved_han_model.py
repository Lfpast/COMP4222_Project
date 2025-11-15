"""
改进的 HAN 模型训练 - 多任务学习版本

改进点：
1. 添加语义保持损失 - 让 HAN embeddings 保留语义信息
2. 使用所有边类型 - 不只是 cites，也用 written_by 和 has_keyword
3. 对比学习 - 让结构相似的论文 embedding 也接近

这样训练出的 embeddings 可以同时用于：
- 语义搜索（因为保留了原始语义）
- 图结构任务（因为学习了引用关系）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from py2neo import Graph
from tqdm import tqdm
import os
import time
from datetime import datetime, timedelta
import sys
import io

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


class ImprovedHANModel(nn.Module):
    """改进的 HAN 模型 - 支持多任务学习"""
    
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, meta_paths):
        super(ImprovedHANModel, self).__init__()
        
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.HeteroGraphConv({
            rel: dglnn.GATConv(in_dim, hidden_dim, num_heads)
            for rel in meta_paths
        }))
        
        self.layers.append(dglnn.HeteroGraphConv({
            rel: dglnn.GATConv(hidden_dim * num_heads, hidden_dim, num_heads)
            for rel in meta_paths
        }))
        
        # 投影到输出维度（保持 384 维以匹配 Sentence-BERT）
        self.output_projection = nn.Linear(hidden_dim * num_heads, out_dim)
        
        # Batch normalization（轻量级，momentum 降低以保留更多语义信息）
        self.batch_norms = nn.ModuleDict({
            'paper': nn.BatchNorm1d(out_dim, momentum=0.1),  # 降低 momentum
            'author': nn.BatchNorm1d(out_dim, momentum=0.1),
            'keyword': nn.BatchNorm1d(out_dim, momentum=0.1)
        })
        
    def forward(self, graph, inputs, apply_batchnorm=True):
        h = inputs
        for i, layer in enumerate(self.layers):
            h = layer(graph, h)
            if i < len(self.layers) - 1:
                h = {k: F.elu(v.flatten(1)) for k, v in h.items()}
            else:
                h = {k: v.flatten(1) for k, v in h.items()}
        
        h = {k: self.output_projection(v) for k, v in h.items()}
        
        if apply_batchnorm and self.training:
            h = {k: self.batch_norms[k](v) if k in self.batch_norms else v 
                 for k, v in h.items()}
        
        return h


class DotProductPredictor(nn.Module):
    """链接预测器"""
    
    def forward(self, g, h, ntype='paper'):
        node_features = h[ntype]
        s, d = g.edges()
        s_embed = node_features[s]
        d_embed = node_features[d]
        score = (s_embed * d_embed).sum(1)
        return score


class ImprovedTrainer:
    """改进的训练器 - 多任务学习"""
    
    def __init__(self, neo4j_uri, neo4j_username, neo4j_password):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 连接 Neo4j
        print(f"\nConnecting to Neo4j...")
        self.graph = Graph(neo4j_uri, auth=(neo4j_username, neo4j_password))
        self.graph.run("RETURN 1")
        print("Connected to Neo4j")
        
        # 加载 Sentence Transformer
        print("\nLoading Sentence Transformer...")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model loaded")
        
        os.makedirs("cache", exist_ok=True)
    
    def load_data_from_neo4j(self, sample_size=None):
        """加载数据（与原版相同）"""
        print("\nLoading data from Neo4j...")
        
        # 加载论文
        if sample_size:
            papers_query = f"MATCH (p:Paper) WHERE p.title IS NOT NULL RETURN p.paper_id as paper_id, p.title as title, p.abstract as abstract LIMIT {sample_size}"
        else:
            papers_query = "MATCH (p:Paper) WHERE p.title IS NOT NULL RETURN p.paper_id as paper_id, p.title as title, p.abstract as abstract"
        
        papers_df = pd.DataFrame(self.graph.run(papers_query).data())
        print(f"  Loaded {len(papers_df)} papers")
        
        paper_ids = papers_df['paper_id'].tolist()
        
        # 加载作者
        authors_query = "MATCH (a:Author)<-[:WRITTEN_BY]-(p:Paper) WHERE p.paper_id IN $ids RETURN DISTINCT a.author_id as author_id, a.name as name"
        authors_df = pd.DataFrame(self.graph.run(authors_query, ids=paper_ids).data())
        print(f"  Loaded {len(authors_df)} authors")
        
        # 加载关键词
        keywords_query = "MATCH (k:Keyword)<-[:HAS_KEYWORD]-(p:Paper) WHERE p.paper_id IN $ids RETURN DISTINCT k.keyword_id as keyword_id, k.keyword as keyword"
        keywords_df = pd.DataFrame(self.graph.run(keywords_query, ids=paper_ids).data())
        print(f"  Loaded {len(keywords_df)} keywords")
        
        # 加载关系
        written_by_query = "MATCH (p:Paper)-[:WRITTEN_BY]->(a:Author) WHERE p.paper_id IN $ids RETURN p.paper_id as paper_id, a.author_id as author_id"
        written_by_df = pd.DataFrame(self.graph.run(written_by_query, ids=paper_ids).data())
        
        cites_query = "MATCH (p1:Paper)-[:CITES]->(p2:Paper) WHERE p1.paper_id IN $ids AND p2.paper_id IN $ids RETURN p1.paper_id as citing, p2.paper_id as cited"
        cites_df = pd.DataFrame(self.graph.run(cites_query, ids=paper_ids).data())
        
        has_keyword_query = "MATCH (p:Paper)-[:HAS_KEYWORD]->(k:Keyword) WHERE p.paper_id IN $ids RETURN p.paper_id as paper_id, k.keyword_id as keyword_id"
        has_keyword_df = pd.DataFrame(self.graph.run(has_keyword_query, ids=paper_ids).data())
        
        print(f"  WRITTEN_BY: {len(written_by_df)}, CITES: {len(cites_df)}, HAS_KEYWORD: {len(has_keyword_df)}")
        
        return {
            'papers': papers_df,
            'authors': authors_df,
            'keywords': keywords_df,
            'written_by': written_by_df,
            'cites': cites_df,
            'has_keyword': has_keyword_df
        }
    
    def prepare_node_features(self, data_dict):
        """生成节点特征"""
        print("\nPreparing node features...")
        
        # 论文特征
        paper_texts = []
        for _, row in data_dict['papers'].iterrows():
            text = str(row['title'])
            if pd.notna(row.get('abstract')):
                text += " " + str(row['abstract'])
            paper_texts.append(text)
        
        paper_emb = self.sentence_model.encode(paper_texts, show_progress_bar=True, batch_size=128)
        
        # 作者特征
        author_names = data_dict['authors']['name'].fillna('Unknown').tolist()
        author_emb = self.sentence_model.encode(author_names, show_progress_bar=True, batch_size=128)
        
        # 关键词特征
        keyword_texts = data_dict['keywords']['keyword'].fillna('').tolist()
        keyword_emb = self.sentence_model.encode(keyword_texts, show_progress_bar=True, batch_size=128)
        
        return {
            'paper': torch.FloatTensor(paper_emb),
            'author': torch.FloatTensor(author_emb),
            'keyword': torch.FloatTensor(keyword_emb)
        }
    
    def build_graph(self, data_dict):
        """构建 DGL 异构图"""
        print("\nBuilding heterogeneous graph...")
        
        # ID 映射
        paper_map = {pid: idx for idx, pid in enumerate(data_dict['papers']['paper_id'])}
        author_map = {aid: idx for idx, aid in enumerate(data_dict['authors']['author_id'])}
        keyword_map = {kid: idx for idx, kid in enumerate(data_dict['keywords']['keyword_id'])}
        
        data_dict_graph = {}
        
        # WRITTEN_BY
        wb_src = [paper_map[row['paper_id']] for _, row in data_dict['written_by'].iterrows() if row['paper_id'] in paper_map and row['author_id'] in author_map]
        wb_dst = [author_map[row['author_id']] for _, row in data_dict['written_by'].iterrows() if row['paper_id'] in paper_map and row['author_id'] in author_map]
        if wb_src:
            data_dict_graph[('paper', 'written_by', 'author')] = (torch.tensor(wb_src), torch.tensor(wb_dst))
            data_dict_graph[('author', 'writes', 'paper')] = (torch.tensor(wb_dst), torch.tensor(wb_src))
        
        # CITES
        cite_src = [paper_map[row['citing']] for _, row in data_dict['cites'].iterrows() if row['citing'] in paper_map and row['cited'] in paper_map]
        cite_dst = [paper_map[row['cited']] for _, row in data_dict['cites'].iterrows() if row['citing'] in paper_map and row['cited'] in paper_map]
        if cite_src:
            data_dict_graph[('paper', 'cites', 'paper')] = (torch.tensor(cite_src), torch.tensor(cite_dst))
            data_dict_graph[('paper', 'cited_by', 'paper')] = (torch.tensor(cite_dst), torch.tensor(cite_src))
        
        # HAS_KEYWORD
        kw_src = [paper_map[row['paper_id']] for _, row in data_dict['has_keyword'].iterrows() if row['paper_id'] in paper_map and row['keyword_id'] in keyword_map]
        kw_dst = [keyword_map[row['keyword_id']] for _, row in data_dict['has_keyword'].iterrows() if row['paper_id'] in paper_map and row['keyword_id'] in keyword_map]
        if kw_src:
            data_dict_graph[('paper', 'has_keyword', 'keyword')] = (torch.tensor(kw_src), torch.tensor(kw_dst))
            data_dict_graph[('keyword', 'belongs_to', 'paper')] = (torch.tensor(kw_dst), torch.tensor(kw_src))
        
        graph = dgl.heterograph(data_dict_graph)
        print(f"  Nodes: {dict(zip(graph.ntypes, [graph.num_nodes(t) for t in graph.ntypes]))}")
        print(f"  Edges: {graph.num_edges()}")
        
        return graph, paper_map, author_map, keyword_map
    
    def train_model(self, sample_size=None, epochs=50, lr=0.001, save_dir='models/improved_han',
                   hidden_dim=128, out_dim=384, num_heads=8, 
                   semantic_weight=0.5, link_weight=0.5):
        """
        多任务训练
        
        Args:
            semantic_weight: 语义保持损失的权重（0-1）
            link_weight: 链接预测损失的权重（0-1）
        """
        print("=" * 70)
        print("IMPROVED HAN TRAINING - Multi-Task Learning")
        print("=" * 70)
        print(f"\nLoss Weights:")
        print(f"  Semantic Preservation: {semantic_weight}")
        print(f"  Link Prediction: {link_weight}")
        
        # 1. 加载数据
        data_dict = self.load_data_from_neo4j(sample_size)
        graph, paper_map, author_map, keyword_map = self.build_graph(data_dict)
        node_features = self.prepare_node_features(data_dict)
        
        # 保存原始特征（用于语义保持损失）
        original_features = {k: v.clone() for k, v in node_features.items()}
        
        # 2. 初始化模型
        print("\nInitializing model...")
        model = ImprovedHANModel(384, hidden_dim, out_dim, num_heads, graph.etypes)
        model = model.to(self.device)
        predictor = DotProductPredictor().to(self.device)
        
        graph = graph.to(self.device)
        node_features = {k: v.to(self.device) for k, v in node_features.items()}
        original_features = {k: v.to(self.device) for k, v in original_features.items()}
        
        optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=lr)
        
        # 3. 准备链接预测数据
        cites_etype = ('paper', 'cites', 'paper')
        if cites_etype not in graph.canonical_etypes:
            print("ERROR: No cites edges found!")
            return None
        
        num_edges = graph.num_edges(cites_etype)
        edge_ids = torch.arange(num_edges, device=self.device)
        batch_size = 512  # 减小 batch size 因为有两个损失
        num_batches = (num_edges + batch_size - 1) // batch_size
        
        print(f"\nTraining Setup:")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Batches per epoch: {num_batches}")
        print(f"  Total edges for link prediction: {num_edges}")
        
        # 4. 训练循环
        print("\n" + "=" * 70)
        print("Starting Training...")
        print("=" * 70 + "\n")
        
        model.train()
        predictor.train()
        history = []
        
        for epoch in range(epochs):
            epoch_start = time.time()
            total_loss = 0
            total_semantic_loss = 0
            total_link_loss = 0
            
            shuffled_ids = edge_ids[torch.randperm(num_edges)]
            
            pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs}", ncols=100)
            
            for batch_idx in pbar:
                # 前向传播
                embeddings = model(graph, node_features)
                
                # === 损失 1: 语义保持损失 ===
                # 让 HAN embeddings 接近原始 Sentence-BERT embeddings
                semantic_loss = 0
                for ntype in ['paper', 'author', 'keyword']:
                    if ntype in embeddings and ntype in original_features:
                        # MSE Loss between HAN and original
                        semantic_loss += F.mse_loss(embeddings[ntype], original_features[ntype])
                semantic_loss = semantic_loss / 3  # 平均
                
                # === 损失 2: 链接预测损失 ===
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_edges)
                batch_eids = shuffled_ids[start_idx:end_idx]
                
                src, dst = graph.find_edges(batch_eids, etype=cites_etype)
                
                # 正样本图
                pos_graph = dgl.heterograph(
                    {cites_etype: (src, dst)},
                    num_nodes_dict={'paper': graph.num_nodes('paper')}
                ).to(self.device)
                
                pos_score = predictor(pos_graph, embeddings, ntype='paper')
                
                # 负样本
                k = 3  # 减少负样本数量
                num_pos = len(batch_eids)
                neg_src = src.repeat_interleave(k)
                neg_dst = torch.randint(0, graph.num_nodes('paper'), (num_pos * k,), device=self.device)
                
                neg_graph = dgl.heterograph(
                    {cites_etype: (neg_src, neg_dst)},
                    num_nodes_dict={'paper': graph.num_nodes('paper')}
                ).to(self.device)
                
                neg_score = predictor(neg_graph, embeddings, ntype='paper')
                
                # Margin Ranking Loss
                y = torch.ones_like(pos_score)
                neg_score_mean = neg_score.view(-1, k).mean(1)
                link_loss = F.margin_ranking_loss(pos_score, neg_score_mean, y, margin=0.2)
                
                # === 总损失 ===
                loss = semantic_weight * semantic_loss + link_weight * link_loss
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # 统计
                total_loss += loss.item()
                total_semantic_loss += semantic_loss.item()
                total_link_loss += link_loss.item()
                
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Sem': f'{semantic_loss.item():.3f}',
                    'Link': f'{link_loss.item():.3f}'
                })
            
            # Epoch 结束
            avg_loss = total_loss / num_batches
            avg_sem = total_semantic_loss / num_batches
            avg_link = total_link_loss / num_batches
            epoch_time = time.time() - epoch_start
            
            history.append({
                'total': avg_loss,
                'semantic': avg_sem,
                'link': avg_link
            })
            
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} (Semantic: {avg_sem:.3f}, Link: {avg_link:.3f}) - Time: {epoch_time:.1f}s")
        
        # 5. 保存模型
        print(f"\nSaving model to {save_dir}...")
        os.makedirs(save_dir, exist_ok=True)
        
        model.eval()
        with torch.no_grad():
            final_embeddings = model(graph, node_features, apply_batchnorm=False)
            final_embeddings = {k: v.cpu() for k, v in final_embeddings.items()}
        
        # 移动 original_features 到 CPU
        original_features_cpu = {k: v.cpu() for k, v in original_features.items()}
        
        save_data = {
            'model_state_dict': model.state_dict(),
            'embeddings': final_embeddings,  # 改进的 HAN embeddings
            'original_embeddings': original_features_cpu,  # 原始 Sentence-BERT
            'id_maps': {'paper': paper_map, 'author': author_map, 'keyword': keyword_map},
            'config': {
                'in_dim': 384,
                'hidden_dim': hidden_dim,
                'out_dim': out_dim,
                'num_heads': num_heads,
                'semantic_weight': semantic_weight,
                'link_weight': link_weight,
                'etypes': graph.etypes
            },
            'training_history': history
        }
        
        model_path = os.path.join(save_dir, 'han_embeddings.pth')
        torch.save(save_data, model_path)
        print(f"Model saved: {model_path}")
        
        print("\n" + "=" * 70)
        print("Training Completed!")
        print("=" * 70)
        print(f"Final Loss: {history[-1]['total']:.4f}")
        print(f"  Semantic: {history[-1]['semantic']:.3f}")
        print(f"  Link: {history[-1]['link']:.3f}")
        
        return model, graph, final_embeddings, save_data['id_maps']


if __name__ == "__main__":
    # 配置
    CONFIG = {
        'NEO4J_URI': "neo4j://127.0.0.1:7687",
        'NEO4J_USERNAME': "neo4j",
        'NEO4J_PASSWORD': "12345678",
        'SAMPLE_SIZE': None,  # None = 全部数据
        'EPOCHS': 30,  # 减少 epochs，因为有语义损失收敛更快
        'LEARNING_RATE': 0.0005,  # 降低学习率以保留语义
        'SAVE_DIR': 'models/improved_han_v1',
        'HIDDEN_DIM': 128,
        'OUT_DIM': 384,
        'NUM_HEADS': 8,
        'SEMANTIC_WEIGHT': 0.6,  # 60% 语义保持
        'LINK_WEIGHT': 0.4  # 40% 链接预测
    }
    
    print("\nConfiguration:")
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")
    
    try:
        trainer = ImprovedTrainer(
            neo4j_uri=CONFIG['NEO4J_URI'],
            neo4j_username=CONFIG['NEO4J_USERNAME'],
            neo4j_password=CONFIG['NEO4J_PASSWORD']
        )
        
        model, graph, embeddings, id_maps = trainer.train_model(
            sample_size=CONFIG['SAMPLE_SIZE'],
            epochs=CONFIG['EPOCHS'],
            lr=CONFIG['LEARNING_RATE'],
            save_dir=CONFIG['SAVE_DIR'],
            hidden_dim=CONFIG['HIDDEN_DIM'],
            out_dim=CONFIG['OUT_DIM'],
            num_heads=CONFIG['NUM_HEADS'],
            semantic_weight=CONFIG['SEMANTIC_WEIGHT'],
            link_weight=CONFIG['LINK_WEIGHT']
        )
        
        print("\nSuccess! Model saved.")
        print(f"\nNext steps:")
        print(f"1. Update .env MODEL_PATH to: {CONFIG['SAVE_DIR']}/han_embeddings.pth")
        print(f"2. Run: python hybrid_recommender.py")
        print(f"3. Compare the improved HAN embeddings with the old ones!")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
