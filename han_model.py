import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
from py2neo import Graph
from tqdm import tqdm
import os
import time
from datetime import datetime, timedelta
import json

class HANModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, meta_paths):
        super(HANModel, self).__init__()
        
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.HeteroGraphConv({
            rel: dglnn.GATConv(in_dim, hidden_dim, num_heads)
            for rel in meta_paths
        }))
        
        self.layers.append(dglnn.HeteroGraphConv({
            rel: dglnn.GATConv(hidden_dim * num_heads, hidden_dim, num_heads)
            for rel in meta_paths
        }))
        
        # Final projection layer to match output dimension with input (384)
        self.output_projection = nn.Linear(hidden_dim * num_heads, out_dim)
        
        # Batch normalization to stabilize embeddings and prevent collapse
        self.batch_norms = nn.ModuleDict({
            'paper': nn.BatchNorm1d(out_dim, momentum=0.9),
            'author': nn.BatchNorm1d(out_dim, momentum=0.9),
            'keyword': nn.BatchNorm1d(out_dim, momentum=0.9)
        })
        
    def forward(self, graph, inputs, apply_batchnorm=True):
        h = inputs
        for i, layer in enumerate(self.layers):
            h = layer(graph, h)
            if i < len(self.layers) - 1:  # Not the last layer
                h = {k: F.elu(v.flatten(1)) for k, v in h.items()}
            else:  # Last layer - flatten attention heads
                h = {k: v.flatten(1) for k, v in h.items()}
        
        # Project to output dimension to match Sentence-BERT (384)
        h = {k: self.output_projection(v) for k, v in h.items()}
        
        # Apply batch normalization to prevent embedding collapse
        if apply_batchnorm and self.training:
            h = {k: self.batch_norms[k](v) if k in self.batch_norms else v 
                 for k, v in h.items()}
        
        return h

class GraphEmbeddingTrainer:
    def __init__(self, neo4j_uri="neo4j://127.0.0.1:7687", 
                 neo4j_username="neo4j", 
                 neo4j_password="Jackson050609"):
        """初始化训练器,连接到Neo4j数据库"""
        # 优先使用GPU,如果不可用则降级到CPU
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"🔧 Using device: {self.device} ({torch.cuda.get_device_name(0)})")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            self.device = torch.device('cpu')
            print(f"🔧 CUDA not available, falling back to: {self.device}")
        
        # 连接Neo4j
        print(f"\n🔌 Connecting to Neo4j at {neo4j_uri}...")
        try:
            self.graph = Graph(neo4j_uri, auth=(neo4j_username, neo4j_password))
            self.graph.run("RETURN 1")
            print("✅ Connected to Neo4j successfully")
        except Exception as e:
            print(f"❌ Failed to connect to Neo4j: {e}")
            raise
        
        # 初始化文本嵌入模型
        print("\n📝 Loading sentence transformer model...")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Sentence transformer loaded")
        
        # 缓存目录
        self.cache_dir = "cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def load_data_from_neo4j(self, sample_size=None):
        """从Neo4j加载图数据"""
        print("\n📊 Loading data from Neo4j...")
        
        # 1. 加载论文节点
        print("\n📄 Loading papers...")
        if sample_size:
            papers_query = f"""
            MATCH (p:Paper)
            WHERE p.title IS NOT NULL AND p.title <> ''
            RETURN p.paper_id as paper_id, p.title as title, 
                   p.abstract as abstract, p.year as year
            LIMIT {sample_size}
            """
        else:
            papers_query = """
            MATCH (p:Paper)
            WHERE p.title IS NOT NULL AND p.title <> ''
            RETURN p.paper_id as paper_id, p.title as title, 
                   p.abstract as abstract, p.year as year
            """
        papers_data = self.graph.run(papers_query).data()
        papers_df = pd.DataFrame(papers_data)
        print(f"   ✅ Loaded {len(papers_df)} papers")
        
        # 2. 加载作者节点
        print("\n👥 Loading authors...")
        authors_query = """
        MATCH (a:Author)<-[:WRITTEN_BY]-(p:Paper)
        WHERE p.paper_id IN $paper_ids
        RETURN DISTINCT a.author_id as author_id, a.name as name
        """
        authors_data = self.graph.run(authors_query, paper_ids=papers_df['paper_id'].tolist()).data()
        authors_df = pd.DataFrame(authors_data) if authors_data else pd.DataFrame({'author_id': [], 'name': []})
        print(f"   ✅ Loaded {len(authors_df)} authors")
        
        # 3. 加载关键词节点
        print("\n🏷️  Loading keywords...")
        keywords_query = """
        MATCH (k:Keyword)<-[:HAS_KEYWORD]-(p:Paper)
        WHERE p.paper_id IN $paper_ids
        RETURN DISTINCT k.keyword_id as keyword_id, k.keyword as keyword
        """
        keywords_data = self.graph.run(keywords_query, paper_ids=papers_df['paper_id'].tolist()).data()
        keywords_df = pd.DataFrame(keywords_data) if keywords_data else pd.DataFrame({'keyword_id': [], 'keyword': []})
        print(f"   ✅ Loaded {len(keywords_df)} keywords")
        
        # 4. 加载关系
        print("\n🔗 Loading relationships...")
        
        # WRITTEN_BY 关系
        written_by_query = """
        MATCH (p:Paper)-[r:WRITTEN_BY]->(a:Author)
        WHERE p.paper_id IN $paper_ids
        RETURN p.paper_id as paper_id, a.author_id as author_id, 
               r.author_order as author_order
        """
        written_by_data = self.graph.run(written_by_query, paper_ids=papers_df['paper_id'].tolist()).data()
        written_by_df = pd.DataFrame(written_by_data) if written_by_data else pd.DataFrame({'paper_id': [], 'author_id': [], 'author_order': []})
        print(f"   ✅ Loaded {len(written_by_df)} WRITTEN_BY relationships")
        
        # CITES 关系
        cites_query = """
        MATCH (citing:Paper)-[r:CITES]->(cited:Paper)
        WHERE citing.paper_id IN $paper_ids 
          AND cited.paper_id IN $paper_ids
        RETURN citing.paper_id as citing_paper, cited.paper_id as cited_paper
        """
        cites_data = self.graph.run(cites_query, paper_ids=papers_df['paper_id'].tolist()).data()
        cites_df = pd.DataFrame(cites_data) if cites_data else pd.DataFrame({'citing_paper': [], 'cited_paper': []})
        print(f"   ✅ Loaded {len(cites_df)} CITES relationships")
        
        # HAS_KEYWORD 关系
        has_keyword_query = """
        MATCH (p:Paper)-[r:HAS_KEYWORD]->(k:Keyword)
        WHERE p.paper_id IN $paper_ids
        RETURN p.paper_id as paper_id, k.keyword_id as keyword_id
        """
        has_keyword_data = self.graph.run(has_keyword_query, paper_ids=papers_df['paper_id'].tolist()).data()
        has_keyword_df = pd.DataFrame(has_keyword_data) if has_keyword_data else pd.DataFrame({'paper_id': [], 'keyword_id': []})
        print(f"   ✅ Loaded {len(has_keyword_df)} HAS_KEYWORD relationships")
        
        return {
            'papers': papers_df,
            'authors': authors_df,
            'keywords': keywords_df,
            'written_by': written_by_df,
            'cites': cites_df,
            'has_keyword': has_keyword_df
        }
    
    def prepare_node_features(self, data_dict):
        """从Neo4j数据生成节点特征"""
        print("\n🎨 Preparing node features...")
        
        papers_df = data_dict['papers']
        authors_df = data_dict['authors']
        keywords_df = data_dict['keywords']
        
        # 为论文节点生成文本嵌入特征
        print("   📄 Encoding paper titles and abstracts...")
        paper_texts = []
        for _, row in papers_df.iterrows():
            text = str(row['title'])
            if pd.notna(row['abstract']) and row['abstract']:
                text += " " + str(row['abstract'])
            paper_texts.append(text)
        
        # 使用sentence transformer生成嵌入
        paper_embeddings = self.sentence_model.encode(
            paper_texts, 
            show_progress_bar=True,
            batch_size=128
        )
        paper_features = torch.FloatTensor(paper_embeddings)
        print(f"      Papers feature shape: {paper_features.shape}")
        
        # 为作者节点生成特征 (使用作者名的嵌入)
        print("   👥 Encoding author names...")
        author_names = authors_df['name'].fillna('Unknown').tolist()
        author_embeddings = self.sentence_model.encode(
            author_names,
            show_progress_bar=True,
            batch_size=128
        )
        author_features = torch.FloatTensor(author_embeddings)
        print(f"      Authors feature shape: {author_features.shape}")
        
        # 为关键词节点生成特征
        print("   🏷️  Encoding keywords...")
        keyword_texts = keywords_df['keyword'].fillna('').tolist()
        keyword_embeddings = self.sentence_model.encode(
            keyword_texts,
            show_progress_bar=True,
            batch_size=128
        )
        keyword_features = torch.FloatTensor(keyword_embeddings)
        print(f"      Keywords feature shape: {keyword_features.shape}")
        
        return {
            'paper': paper_features,
            'author': author_features,
            'keyword': keyword_features
        }
    def build_dgl_heterograph(self, data_dict):
        """从Neo4j数据构建DGL异质图"""
        print("\n🔨 Building DGL heterogeneous graph...")
        
        papers_df = data_dict['papers']
        authors_df = data_dict['authors']
        keywords_df = data_dict['keywords']
        written_by_df = data_dict['written_by']
        cites_df = data_dict['cites']
        has_keyword_df = data_dict['has_keyword']
        
        # 创建ID映射 (字符串ID -> 数值索引)
        print("   🔢 Creating ID mappings...")
        paper_id_map = {pid: idx for idx, pid in enumerate(papers_df['paper_id'])}
        author_id_map = {aid: idx for idx, aid in enumerate(authors_df['author_id'])}
        keyword_id_map = {kid: idx for idx, kid in enumerate(keywords_df['keyword_id'])}
        
        # 构建异质图数据字典
        data_dict_graph = {}
        
        # 1. WRITTEN_BY 关系: Paper -> Author
        print("   ✍️  Adding WRITTEN_BY edges...")
        paper_to_author_src = []
        paper_to_author_dst = []
        for _, row in written_by_df.iterrows():
            if row['paper_id'] in paper_id_map and row['author_id'] in author_id_map:
                paper_to_author_src.append(paper_id_map[row['paper_id']])
                paper_to_author_dst.append(author_id_map[row['author_id']])
        
        if paper_to_author_src:
            data_dict_graph[('paper', 'written_by', 'author')] = (
                torch.tensor(paper_to_author_src), 
                torch.tensor(paper_to_author_dst)
            )
            # 反向边: Author -> Paper
            data_dict_graph[('author', 'writes', 'paper')] = (
                torch.tensor(paper_to_author_dst),
                torch.tensor(paper_to_author_src)
            )
            print(f"      Added {len(paper_to_author_src)} WRITTEN_BY edges")
        
        # 2. CITES 关系: Paper -> Paper
        print("   📚 Adding CITES edges...")
        citing_papers = []
        cited_papers = []
        for _, row in cites_df.iterrows():
            if row['citing_paper'] in paper_id_map and row['cited_paper'] in paper_id_map:
                citing_papers.append(paper_id_map[row['citing_paper']])
                cited_papers.append(paper_id_map[row['cited_paper']])
        
        if citing_papers:
            data_dict_graph[('paper', 'cites', 'paper')] = (
                torch.tensor(citing_papers),
                torch.tensor(cited_papers)
            )
            # 反向边: Paper <- Paper (被引用)
            data_dict_graph[('paper', 'cited_by', 'paper')] = (
                torch.tensor(cited_papers),
                torch.tensor(citing_papers)
            )
            print(f"      Added {len(citing_papers)} CITES edges")
        
        # 3. HAS_KEYWORD 关系: Paper -> Keyword
        print("   🏷️  Adding HAS_KEYWORD edges...")
        paper_to_keyword_src = []
        paper_to_keyword_dst = []
        for _, row in has_keyword_df.iterrows():
            if row['paper_id'] in paper_id_map and row['keyword_id'] in keyword_id_map:
                paper_to_keyword_src.append(paper_id_map[row['paper_id']])
                paper_to_keyword_dst.append(keyword_id_map[row['keyword_id']])
        
        if paper_to_keyword_src:
            data_dict_graph[('paper', 'has_keyword', 'keyword')] = (
                torch.tensor(paper_to_keyword_src),
                torch.tensor(paper_to_keyword_dst)
            )
            # 反向边: Keyword -> Paper
            data_dict_graph[('keyword', 'belongs_to', 'paper')] = (
                torch.tensor(paper_to_keyword_dst),
                torch.tensor(paper_to_keyword_src)
            )
            print(f"      Added {len(paper_to_keyword_src)} HAS_KEYWORD edges")
        
        # 创建异质图
        print("   🎯 Creating heterogeneous graph...")
        graph = dgl.heterograph(data_dict_graph)
        
        print(f"\n✅ Graph created successfully!")
        print(f"   Node types: {graph.ntypes}")
        print(f"   Edge types: {graph.etypes}")
        print(f"   Number of nodes: {dict(zip(graph.ntypes, [graph.num_nodes(ntype) for ntype in graph.ntypes]))}")
        print(f"   Number of edges: {graph.num_edges()}")
        
        return graph, paper_id_map, author_id_map, keyword_id_map
    
    def train_model(self, sample_size=10000, epochs=100, lr=0.001, save_dir='models',
                   hidden_dim=192, out_dim=384, num_heads=4):
        """训练HAN模型
        
        Args:
            sample_size: 采样论文数量
            epochs: 训练轮数
            lr: 学习率
            save_dir: 保存目录
            hidden_dim: 隐藏层维度
            out_dim: 输出嵌入维度
            num_heads: 注意力头数
        """
        print("=" * 70)
        print("🚀 Starting HAN Model Training")
        print("=" * 70)
        
        overall_start_time = time.time()
        
        # 1. 从Neo4j加载数据
        data_dict = self.load_data_from_neo4j(sample_size=sample_size)
        
        # 2. 构建异质图
        graph, paper_id_map, author_id_map, keyword_id_map = self.build_dgl_heterograph(data_dict)
        
        # 3. 准备节点特征
        node_features = self.prepare_node_features(data_dict)
        
        in_dim = 384  # Sentence-BERT embedding dimension
        etypes = graph.etypes
        
        model = HANModel(in_dim, hidden_dim, out_dim, num_heads, etypes)
        model = model.to(self.device)
        
        # 将图和节点特征都移到设备上
        graph = graph.to(self.device)
        node_features = {k: v.to(self.device) for k, v in node_features.items()}
        
        print(f"\n🏗️  Model Architecture:")
        print(f"   Input Dim: {in_dim}")
        print(f"   Hidden Dim: {hidden_dim}")
        print(f"   Output Dim: {out_dim}")
        print(f"   Attention Heads: {num_heads}")
        print(f"   Edge Types: {len(etypes)}")
        print(f"   Model Device: {self.device}")
        print(f"   Graph Device: {self.device}")
        
        print(f"\n🔧 Trial 3 Improvements:")
        print(f"   ✅ Batch Normalization: Prevents embedding collapse")
        print(f"   ✅ Contrastive Loss: Encourages embedding diversity")
        print(f"   ✅ Increased Regularization: 100x for author, 80x for keyword")
        print(f"   ✅ Gradient Accumulation: Compensates for smaller batch size")
        print(f"   ✅ Smaller Model: 96D hidden vs 128D (25% less memory)")
        
        # 5. 定义优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # 设置批处理参数
        batch_size = 128
        gradient_accumulation_steps = 1
        print(f"\n💾 Memory-Efficient Settings:")
        print(f"   Batch Size: {batch_size} nodes per batch")
        print(f"   Gradient Accumulation Steps: {gradient_accumulation_steps}")
        
        # 7. 训练循环
        print(f"\n🎯 Training for {epochs} epochs...")
        print("=" * 70 + "\n")
        
        model.train()
        train_start_time = time.time()
        
        # 记录训练历史
        training_history = {
            'losses': [],
            'epoch_times': [],
            'checkpoints': []  # 每10步的详细记录
        }
        
        # 使用tqdm创建进度条
        pbar = tqdm(range(epochs), 
                   desc="Training", 
                   ncols=100,
                   leave=True,
                   dynamic_ncols=True,
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}')
        
        for epoch in pbar:
            epoch_start = time.time()
            
            # 清空GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            optimizer.zero_grad()
            
            try:
                # 前向传播（仅计算模型，图保持在CPU）
                embeddings = model(graph, node_features, apply_batchnorm=False)
                
                # 计算改进的损失函数 - 防止 embedding collapse
                # 激进的正则化策略：强制分散嵌入，降低节点相似度
                node_weights = {
                    'paper': 0.01,        # 论文节点
                    'author': 0.02,       # 作者节点 - 增加2倍
                    'keyword': 0.02       # 关键词节点 - 增加2倍
                }
                
                loss = 0
                num_nodes = 0
                
                for ntype, emb in embeddings.items():
                    emb = emb.to(self.device)
                    weight = node_weights.get(ntype, 0.0001)
                    
                    # 1. 方差最大化 - 确保嵌入有变异性，防止 collapse
                    embedding_variance = torch.mean(torch.var(emb, dim=0))
                    # 目标方差提高到0.01，更激进地强制分散
                    variance_penalty = F.relu(0.01 - embedding_variance)
                    loss += variance_penalty * weight * 100
                    
                    # 2. 全局对比损失 - 强制所有节点相似度降低（不采样，全计算）
                    if emb.shape[0] > 1:
                        # 对于大规模数据集，分批计算相似度以避免OOM
                        if emb.shape[0] > 5000:
                            # 采用更大的样本 (2000) 确保覆盖足够的节点对
                            sample_size = 2000
                            indices = torch.randperm(emb.shape[0])[:sample_size]
                            sampled_emb = emb[indices]
                        else:
                            # 小数据集计算全部
                            sampled_emb = emb
                        
                        normalized = F.normalize(sampled_emb, p=2, dim=1)
                        similarity_matrix = torch.mm(normalized, normalized.t())
                        
                        mask = torch.eye(similarity_matrix.shape[0], device=self.device, dtype=torch.bool)
                        masked_sim = similarity_matrix.masked_fill(mask, 0)
                        
                        # 更激进的目标：相似度必须 < 0.15（而不是0.3）
                        # 这会强制节点更分散
                        contrast_loss = F.relu(masked_sim.abs() - 0.15).mean()
                        # 大幅增加权重确保对比损失主导训练
                        loss += contrast_loss * weight * 500
                    
                    num_nodes += emb.shape[0]
                
                # 梯度累积
                loss = loss / gradient_accumulation_steps
                
                # 反向传播
                loss.backward()
                
                # 每 gradient_accumulation_steps 步进行一次参数更新
                if (epoch + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                
                epoch_time = time.time() - epoch_start
                loss_value = loss.item() * gradient_accumulation_steps  # 还原实际损失用于显示
                
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    print(f"\n⚠️  GPU OOM detected. Reducing batch in next epoch...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    # 继续训练，下一轮会尝试恢复
                    epoch_time = time.time() - epoch_start
                    loss_value = float('nan')
                else:
                    raise
            
            training_history['losses'].append(loss_value)
            training_history['epoch_times'].append(epoch_time)
            
            # 每10个epoch记录详细信息(但不打印,只记录)
            if (epoch + 1) % 10 == 0:
                checkpoint = {
                    'epoch': epoch + 1,
                    'loss': loss_value,
                    'avg_loss_last_10': np.mean([l for l in training_history['losses'][-10:] if not np.isnan(l)]),
                    'epoch_time': epoch_time,
                    'avg_epoch_time': np.mean(training_history['epoch_times'][-10:]),
                    'elapsed_time': time.time() - train_start_time
                }
                training_history['checkpoints'].append(checkpoint)
            
            # 计算ETA
            if len(training_history['epoch_times']) > 0:
                avg_epoch_time = np.mean(training_history['epoch_times'])
                remaining_epochs = epochs - (epoch + 1)
                eta_seconds = avg_epoch_time * remaining_epochs
                eta = timedelta(seconds=int(eta_seconds))
            else:
                eta = timedelta(seconds=0)
            
            # 更新进度条(在同一行)
            pbar.set_postfix({
                'Loss': f'{loss_value:.4f}' if not np.isnan(loss_value) else 'OOM',
                'Time': f'{epoch_time:.2f}s',
                'ETA': str(eta)
            })
        
        pbar.close()  # 确保进度条正确关闭
        
        total_train_time = time.time() - train_start_time
        print(f"\n✅ Training completed in {timedelta(seconds=int(total_train_time))}!")
        
        # 8. 保存模型和嵌入
        print(f"\n💾 Saving to {save_dir}/...")
        os.makedirs(save_dir, exist_ok=True)
        
        # 获取最终嵌入
        model.eval()
        with torch.no_grad():
            final_embeddings = model(graph, node_features)
            final_embeddings = {k: v.cpu() for k, v in final_embeddings.items()}
        
        # 保存模型文件
        save_data = {
            'model_state_dict': model.state_dict(),
            'embeddings': final_embeddings,
            'id_maps': {
                'paper': paper_id_map,
                'author': author_id_map,
                'keyword': keyword_id_map
            },
            'config': {
                'in_dim': in_dim,
                'hidden_dim': hidden_dim,
                'out_dim': out_dim,
                'num_heads': num_heads,
                'etypes': etypes
            }
        }
        
        model_path = os.path.join(save_dir, 'han_embeddings.pth')
        torch.save(save_data, model_path)
        print(f"   ✅ Model saved: {model_path}")
        
        # 生成训练摘要
        summary = {
            'training_config': {
                'sample_size': sample_size,
                'epochs': epochs,
                'learning_rate': lr,
                'device': str(self.device),
                'save_directory': save_dir
            },
            'model_architecture': {
                'input_dim': in_dim,
                'hidden_dim': hidden_dim,
                'output_dim': out_dim,
                'num_heads': num_heads,
                'num_edge_types': len(etypes)
            },
            'dataset_info': {
                'num_papers': len(data_dict['papers']),
                'num_authors': len(data_dict['authors']),
                'num_keywords': len(data_dict['keywords']),
                'num_written_by': len(data_dict['written_by']),
                'num_cites': len(data_dict['cites']),
                'num_has_keyword': len(data_dict['has_keyword'])
            },
            'training_statistics': {
                'total_time_seconds': total_train_time,
                'total_time_formatted': str(timedelta(seconds=int(total_train_time))),
                'average_epoch_time': np.mean(training_history['epoch_times']),
                'final_loss': training_history['losses'][-1],
                'min_loss': min(training_history['losses']),
                'max_loss': max(training_history['losses']),
                'avg_loss': np.mean(training_history['losses'])
            },
            'checkpoints': training_history['checkpoints'],
            'loss_history': training_history['losses'],
            'embedding_stats': {
                ntype: {
                    'shape': list(emb.shape),
                    'mean': float(emb.mean()),
                    'std': float(emb.std()),
                    'min': float(emb.min()),
                    'max': float(emb.max())
                }
                for ntype, emb in final_embeddings.items()
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 保存摘要文件
        summary_path = os.path.join(save_dir, 'summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"   ✅ Summary saved: {summary_path}")
        
        # 显示简要统计
        print("\n📊 Training Summary:")
        print(f"   Final Loss: {training_history['losses'][-1]:.4f}")
        print(f"   Best Loss: {min(training_history['losses']):.4f}")
        print(f"   Avg Loss: {np.mean(training_history['losses']):.4f}")
        print(f"   Total Time: {timedelta(seconds=int(total_train_time))}")
        
        return model, graph, final_embeddings, save_data['id_maps']

if __name__ == "__main__":
    import sys

    # ===================== DEBUG CONFIGURATION =====================
    DEBUG_MODE = True  # 设置为 True 可直接在此处调试参数
    # 仅在 DEBUG_MODE=True 时生效
    DEBUG_CONFIG = {
        'NEO4J_URI': "neo4j://127.0.0.1:7687",
        'NEO4J_USERNAME': "neo4j",
        'NEO4J_PASSWORD': "87654321",
        'SAMPLE_SIZE': 10000,
        'EPOCHS': 500,
        'LEARNING_RATE': 0.001,
        'SAVE_DIR': 'models/trial4',
        'HIDDEN_DIM': 128,
        'OUT_DIM': 384,
        'NUM_HEADS': 16
    }
    # ===================== END DEBUG CONFIGURATION =================

    print("=" * 70)
    print("🎓 Academic Graph HAN Model Training")
    print("=" * 70)

    if DEBUG_MODE:
        NEO4J_URI = DEBUG_CONFIG['NEO4J_URI']
        NEO4J_USERNAME = DEBUG_CONFIG['NEO4J_USERNAME']
        NEO4J_PASSWORD = DEBUG_CONFIG['NEO4J_PASSWORD']
        SAMPLE_SIZE = DEBUG_CONFIG['SAMPLE_SIZE']
        EPOCHS = DEBUG_CONFIG['EPOCHS']
        LEARNING_RATE = DEBUG_CONFIG['LEARNING_RATE']
        SAVE_DIR = DEBUG_CONFIG['SAVE_DIR']
        HIDDEN_DIM = DEBUG_CONFIG['HIDDEN_DIM']
        OUT_DIM = DEBUG_CONFIG['OUT_DIM']
        NUM_HEADS = DEBUG_CONFIG['NUM_HEADS']
    else:
        if len(sys.argv) != 11:
            print("❌ 参数数量错误！请通过 run.sh 或命令行完整传递 10 个参数：")
            print("python han_model.py <NEO4J_URI> <NEO4J_USERNAME> <NEO4J_PASSWORD> <SAMPLE_SIZE> <EPOCHS> <LEARNING_RATE> <SAVE_DIR> <HIDDEN_DIM> <OUT_DIM> <NUM_HEADS>")
            exit(1)
        NEO4J_URI = sys.argv[1]
        NEO4J_USERNAME = sys.argv[2]
        NEO4J_PASSWORD = sys.argv[3]
        SAMPLE_SIZE = int(sys.argv[4]) if sys.argv[4] != "None" else None
        EPOCHS = int(sys.argv[5])
        LEARNING_RATE = float(sys.argv[6])
        SAVE_DIR = sys.argv[7]
        HIDDEN_DIM = int(sys.argv[8])
        OUT_DIM = int(sys.argv[9])
        NUM_HEADS = int(sys.argv[10])

    print(f"\n⚙️  Configuration:")
    print(f"   Neo4j URI: {NEO4J_URI}")
    print(f"   Sample size: {SAMPLE_SIZE if SAMPLE_SIZE else 'All'} papers")
    print(f"   Model: HIDDEN={HIDDEN_DIM}, OUT={OUT_DIM}, HEADS={NUM_HEADS}")
    print(f"   Training: EPOCHS={EPOCHS}, LR={LEARNING_RATE}")
    print(f"   Save directory: {SAVE_DIR}")

    try:
        # 初始化训练器
        trainer = GraphEmbeddingTrainer(
            neo4j_uri=NEO4J_URI,
            neo4j_username=NEO4J_USERNAME,
            neo4j_password=NEO4J_PASSWORD
        )

        # 训练模型
        model, graph, embeddings, id_maps = trainer.train_model(
            sample_size=SAMPLE_SIZE,
            epochs=EPOCHS,
            lr=LEARNING_RATE,
            save_dir=SAVE_DIR,
            hidden_dim=HIDDEN_DIM,
            out_dim=OUT_DIM,
            num_heads=NUM_HEADS
        )

        print("\n" + "=" * 70)
        print("✅ All tasks completed successfully!")
        print("=" * 70)

        print(f"\n📁 Output files:")
        print(f"   • {SAVE_DIR}/han_embeddings.pth - Model and embeddings")
        print(f"   • {SAVE_DIR}/summary.json - Training summary and statistics")

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 Troubleshooting:")
        print("   1. Check Neo4j is running and accessible")
        print("   2. Verify data exists in Neo4j database")
        print("   3. Install required packages: pip install torch dgl sentence-transformers py2neo")
        print("   4. Reduce SAMPLE_SIZE if out of memory")
        exit(1)