import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer

class HANModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, meta_paths):
        super(HANModel, self).__init__()
        
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.HeteroGraphConv({
            rel: dglnn.GATConv(in_dim, hidden_dim, num_heads)
            for rel in meta_paths
        }))
        
        self.layers.append(dglnn.HeteroGraphConv({
            rel: dglnn.GATConv(hidden_dim * num_heads, out_dim, 1)  # 最后一层单头
            for rel in meta_paths
        }))
        
    def forward(self, graph, inputs):
        h = inputs
        for i, layer in enumerate(self.layers):
            h = layer(graph, h)
            if i < len(self.layers) - 1:  # 不是最后一层
                h = {k: F.elu(v.flatten(1)) for k, v in h.items()}
            else:  # 最后一层
                h = {k: v.mean(1) for k, v in h.items()}
        return h

class GraphEmbeddingTrainer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def prepare_node_features(self, graph):
        """准备节点特征"""
        node_features = {}
        
        # 为论文节点生成文本特征
        papers_df = pd.read_csv("data/papers.csv")
        paper_texts = papers_df['title'] + " " + papers_df.get('abstract', '')
        paper_embeddings = self.sentence_model.encode(paper_texts.tolist())
        
        for idx, paper_id in enumerate(papers_df['paper_id']):
            node_features[('paper', paper_id)] = torch.FloatTensor(paper_embeddings[idx])
        
        # 为作者节点使用随机初始化的特征（在实际中可以从作者论文中聚合）
        authors_df = pd.read_csv("data/authors.csv")
        for _, author_row in authors_df.iterrows():
            node_features[('author', author_row['author_id'])] = torch.randn(384)  # 匹配文本嵌入维度
        
        # 为关键词节点使用随机特征
        keywords_df = pd.read_csv("data/keywords.csv")
        for _, keyword_row in keywords_df.iterrows():
            node_features[('keyword', keyword_row['keyword_id'])] = torch.randn(384)
            
        return node_features
    
    def build_dgl_heterograph(self):
        """从Neo4j构建DGL异质图"""
        # 这里需要从Neo4j中提取数据构建DGL图
        # 简化版本：直接从CSV构建
        
        papers_df = pd.read_csv("data/papers.csv")
        authors_df = pd.read_csv("data/authors.csv")
        keywords_df = pd.read_csv("data/keywords.csv")
        written_by_df = pd.read_csv("data/written_by.csv")
        cites_df = pd.read_csv("data/cites.csv")
        has_keyword_df = pd.read_csv("data/has_keyword.csv")
        
        # 构建异质图数据字典
        data_dict = {}
        
        # 添加边：论文-作者
        src = [written_by_df.index[written_by_df['paper_id'] == pid].tolist()[0] 
               for pid in written_by_df['paper_id']]
        dst = [written_by_df.index[written_by_df['author_id'] == aid].tolist()[0] 
               for aid in written_by_df['author_id']]
        data_dict[('paper', 'written_by', 'author')] = (torch.tensor(src), torch.tensor(dst))
        
        # 添加边：论文-关键词
        src = [has_keyword_df.index[has_keyword_df['paper_id'] == pid].tolist()[0]
               for pid in has_keyword_df['paper_id']]
        dst = [has_keyword_df.index[has_keyword_df['keyword_id'] == kid].tolist()[0]
               for kid in has_keyword_df['keyword_id']]
        data_dict[('paper', 'has_keyword', 'keyword')] = (torch.tensor(src), torch.tensor(dst))
        
        # 添加边：论文-论文（引用）
        # 注意：需要确保引用的论文存在于图中
        valid_cites = cites_df[
            cites_df['cited_paper'].isin(papers_df['paper_id']) & 
            cites_df['citing_paper'].isin(papers_df['paper_id'])
        ]
        src = [valid_cites.index[valid_cites['citing_paper'] == pid].tolist()[0]
               for pid in valid_cites['citing_paper']]
        dst = [valid_cites.index[valid_cites['cited_paper'] == pid].tolist()[0]
               for pid in valid_cites['cited_paper']]
        data_dict[('paper', 'cites', 'paper')] = (torch.tensor(src), torch.tensor(dst))
        
        # 创建异质图
        graph = dgl.heterograph(data_dict)
        return graph
    
    def train_model(self):
        """训练HAN模型"""
        print("Building graph...")
        graph = self.build_dgl_heterograph()
        
        print("Preparing node features...")
        node_features = self.prepare_node_features(graph)
        
        # 定义元路径
        meta_paths = [
            [('paper', 'written_by', 'author'), ('author', 'written_by', 'paper')],  # 论文-作者-论文
            [('paper', 'cites', 'paper')],  # 论文引用论文
            [('paper', 'has_keyword', 'keyword'), ('keyword', 'has_keyword', 'paper')]  # 论文-关键词-论文
        ]
        
        # 初始化模型
        in_dim = 384  # 文本嵌入维度
        hidden_dim = 256
        out_dim = 128
        num_heads = 2
        
        model = HANModel(in_dim, hidden_dim, out_dim, num_heads, meta_paths)
        model = model.to(self.device)
        
        # 简化训练过程（实际中需要准备标签数据）
        print("Model initialized successfully!")
        print(f"Graph structure: {graph}")
        print(f"Node types: {graph.ntypes}")
        print(f"Edge types: {graph.etypes}")
        
        return model, graph, node_features

if __name__ == "__main__":
    trainer = GraphEmbeddingTrainer()
    model, graph, features = trainer.train_model()
    
    # 保存嵌入
    torch.save({
        'model_state_dict': model.state_dict(),
        'graph': graph,
        'node_features': features
    }, 'models/han_embeddings.pth')
    print("Model and embeddings saved!")