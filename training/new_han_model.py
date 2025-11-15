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
# --- NEW IMPORTS ---
import dgl.dataloading
import dgl.sampling  # We now import the correct module
# --- END NEW IMPORTS ---

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
        
        # Apply batch normalization
        # Note: We apply BN during training *before* the loss
        if apply_batchnorm and self.training:
            h = {k: self.batch_norms[k](v) if k in self.batch_norms else v 
                 for k, v in h.items()}
        
        return h

# --- NEW HELPER MODULE FOR LINK PREDICTION ---
class DotProductPredictor(nn.Module):
    """
    Computes a score for a link (edge) between two nodes
    by taking the dot product of their embeddings.
    """
    def forward(self, g, h, ntype='paper'):
        # h contains node features for all node types
        node_features = h[ntype]
        
        # Get the source and destination nodes for all edges in the graph g
        s, d = g.edges()
        
        # Get the embeddings for all source and destination nodes
        s_embed = node_features[s]
        d_embed = node_features[d]
        
        # Compute the dot product
        # (s_embed * d_embed) computes element-wise product
        # .sum(1) sums across the embedding dimension to get a single score
        score = (s_embed * d_embed).sum(1)
        return score
# --- END NEW HELPER MODULE ---


class GraphEmbeddingTrainer:
    def __init__(self, neo4j_uri="neo4j://127.0.0.1:7687", 
                 neo4j_username="neo4j", 
                 neo4j_password="Jackson050609"):
        """Initializes the trainer, connecting to the Neo4j database"""
        # (This function is unchanged, but comments converted to English)
        # Use GPU if available, fallback to CPU
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"🔧 Using device: {self.device} ({torch.cuda.get_device_name(0)})")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            self.device = torch.device('cpu')
            print(f"🔧 CUDA not available, falling back to: {self.device}")
        
        # Connect to Neo4j
        print(f"\n🔌 Connecting to Neo4j at {neo4j_uri}...")
        try:
            self.graph = Graph(neo4j_uri, auth=(neo4j_username, neo4j_password))
            self.graph.run("RETURN 1")
            print("✅ Connected to Neo4j successfully")
        except Exception as e:
            print(f"❌ Failed to connect to Neo4j: {e}")
            raise
        
        # Initialize text embedding model
        print("\n📝 Loading sentence transformer model...")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Sentence transformer loaded")
        
        # Cache directory
        self.cache_dir = "cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def load_data_from_neo4j(self, sample_size=None):
        """Loads graph data from Neo4j"""
        # (This function is unchanged, but comments converted to English)
        print("\n📊 Loading data from Neo4j...")
        
        # 1. Load papers
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
        
        # 2. Load authors
        print("\n👥 Loading authors...")
        authors_query = """
        MATCH (a:Author)<-[:WRITTEN_BY]-(p:Paper)
        WHERE p.paper_id IN $paper_ids
        RETURN DISTINCT a.author_id as author_id, a.name as name
        """
        authors_data = self.graph.run(authors_query, paper_ids=papers_df['paper_id'].tolist()).data()
        authors_df = pd.DataFrame(authors_data) if authors_data else pd.DataFrame({'author_id': [], 'name': []})
        print(f"   ✅ Loaded {len(authors_df)} authors")
        
        # 3. Load keywords
        print("\n🏷️  Loading keywords...")
        keywords_query = """
        MATCH (k:Keyword)<-[:HAS_KEYWORD]-(p:Paper)
        WHERE p.paper_id IN $paper_ids
        RETURN DISTINCT k.keyword_id as keyword_id, k.keyword as keyword
        """
        keywords_data = self.graph.run(keywords_query, paper_ids=papers_df['paper_id'].tolist()).data()
        keywords_df = pd.DataFrame(keywords_data) if keywords_data else pd.DataFrame({'keyword_id': [], 'keyword': []})
        print(f"   ✅ Loaded {len(keywords_df)} keywords")
        
        # 4. Load relationships
        print("\n🔗 Loading relationships...")
        
        # WRITTEN_BY
        written_by_query = """
        MATCH (p:Paper)-[r:WRITTEN_BY]->(a:Author)
        WHERE p.paper_id IN $paper_ids
        RETURN p.paper_id as paper_id, a.author_id as author_id, 
               r.author_order as author_order
        """
        written_by_data = self.graph.run(written_by_query, paper_ids=papers_df['paper_id'].tolist()).data()
        written_by_df = pd.DataFrame(written_by_data) if written_by_data else pd.DataFrame({'paper_id': [], 'author_id': [], 'author_order': []})
        print(f"   ✅ Loaded {len(written_by_df)} WRITTEN_BY relationships")
        
        # CITES
        cites_query = """
        MATCH (citing:Paper)-[r:CITES]->(cited:Paper)
        WHERE citing.paper_id IN $paper_ids 
          AND cited.paper_id IN $paper_ids
        RETURN citing.paper_id as citing_paper, cited.paper_id as cited_paper
        """
        cites_data = self.graph.run(cites_query, paper_ids=papers_df['paper_id'].tolist()).data()
        cites_df = pd.DataFrame(cites_data) if cites_data else pd.DataFrame({'citing_paper': [], 'cited_paper': []})
        print(f"   ✅ Loaded {len(cites_df)} CITES relationships")
        
        # HAS_KEYWORD
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
        """Generates node features from data"""
        # (This function is unchanged, but comments converted to English)
        print("\n🎨 Preparing node features...")
        
        papers_df = data_dict['papers']
        authors_df = data_dict['authors']
        keywords_df = data_dict['keywords']
        
        # Paper features
        print("   📄 Encoding paper titles and abstracts...")
        paper_texts = []
        for _, row in papers_df.iterrows():
            text = str(row['title'])
            if pd.notna(row['abstract']) and row['abstract']:
                text += " " + str(row['abstract'])
            paper_texts.append(text)
        
        paper_embeddings = self.sentence_model.encode(
            paper_texts, 
            show_progress_bar=True,
            batch_size=128
        )
        paper_features = torch.FloatTensor(paper_embeddings)
        print(f"      Papers feature shape: {paper_features.shape}")
        
        # Author features
        print("   👥 Encoding author names...")
        author_names = authors_df['name'].fillna('Unknown').tolist()
        author_embeddings = self.sentence_model.encode(
            author_names,
            show_progress_bar=True,
            batch_size=128
        )
        author_features = torch.FloatTensor(author_embeddings)
        print(f"      Authors feature shape: {author_features.shape}")
        
        # Keyword features
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
        """Builds a DGL heterogeneous graph from data"""
        # (This function is unchanged, but comments converted to English)
        print("\n🔨 Building DGL heterogeneous graph...")
        
        papers_df = data_dict['papers']
        authors_df = data_dict['authors']
        keywords_df = data_dict['keywords']
        written_by_df = data_dict['written_by']
        cites_df = data_dict['cites']
        has_keyword_df = data_dict['has_keyword']
        
        # Create ID mappings
        print("   🔢 Creating ID mappings...")
        paper_id_map = {pid: idx for idx, pid in enumerate(papers_df['paper_id'])}
        author_id_map = {aid: idx for idx, aid in enumerate(authors_df['author_id'])}
        keyword_id_map = {kid: idx for idx, kid in enumerate(keywords_df['keyword_id'])}
        
        data_dict_graph = {}
        
        # 1. WRITTEN_BY
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
            data_dict_graph[('author', 'writes', 'paper')] = (
                torch.tensor(paper_to_author_dst),
                torch.tensor(paper_to_author_src)
            )
            print(f"      Added {len(paper_to_author_src)} WRITTEN_BY edges")
        
        # 2. CITES
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
            data_dict_graph[('paper', 'cited_by', 'paper')] = (
                torch.tensor(cited_papers),
                torch.tensor(citing_papers)
            )
            print(f"      Added {len(citing_papers)} CITES edges")
        
        # 3. HAS_KEYWORD
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
            data_dict_graph[('keyword', 'belongs_to', 'paper')] = (
                torch.tensor(paper_to_keyword_dst),
                torch.tensor(paper_to_keyword_src)
            )
            print(f"      Added {len(paper_to_keyword_src)} HAS_KEYWORD edges")
        
        print("   🎯 Creating heterogeneous graph...")
        graph = dgl.heterograph(data_dict_graph)
        
        print(f"\n✅ Graph created successfully!")
        print(f"   Node types: {graph.ntypes}")
        print(f"   Edge types: {graph.etypes}")
        print(f"   Number of nodes: {dict(zip(graph.ntypes, [graph.num_nodes(ntype) for ntype in graph.ntypes]))}")
        print(f"   Number of edges: {graph.num_edges()}")
        
        return graph, paper_id_map, author_id_map, keyword_id_map
    
    
    # --- THIS IS THE MODIFIED FUNCTION ---
    def train_model(self, sample_size=None, epochs=100, lr=0.001, save_dir='models',
                   hidden_dim=192, out_dim=384, num_heads=4):
        """
        Trains the HAN model using a Link Prediction task.
        
        This function teaches the model to produce similar embeddings for
        papers that cite each other (positive pairs) and dissimilar embeddings
        for papers that are not connected (negative pairs).
        """
        print("=" * 70)
        print("🚀 Starting HAN Model Training (Link Prediction Task)")
        print("=" * 70)
        
        overall_start_time = time.time()
        
        # 1. Load data, build graph, prepare features (same as before)
        #    Note: 'sample_size' is now used by load_data_from_neo4j
        print(f"Step 1: Loading data (sample_size={sample_size})...")
        data_dict = self.load_data_from_neo4j(sample_size=sample_size)
        
        print("\nStep 2: Building DGL graph...")
        graph, paper_id_map, author_id_map, keyword_id_map = self.build_dgl_heterograph(data_dict)
        
        print("\nStep 3: Preparing node features...")
        node_features = self.prepare_node_features(data_dict)
        
        cites_etype_tuple = ('paper', 'cites', 'paper')

        # Ensure 'cites' edge type exists for link prediction
        if cites_etype_tuple not in graph.canonical_etypes:
            print(f"❌ Error: Canonical edge type {cites_etype_tuple} not found in graph.")
            print(f"   Available types: {graph.canonical_etypes}")
            print("   Please check your Neo4j data and import process.")
            return None
        
        in_dim = 384  # Sentence-BERT embedding dimension
        etypes = graph.etypes
        
        # 2. Initialize Model, Predictor, and Optimizer
        print("\nStep 4: Initializing model and optimizer...")
        model = HANModel(in_dim, hidden_dim, out_dim, num_heads, etypes)
        model = model.to(self.device)
        
        predictor = DotProductPredictor()
        predictor = predictor.to(self.device)
        
        # Move graph and features to the device
        graph = graph.to(self.device)
        node_features = {k: v.to(self.device) for k, v in node_features.items()}
        
        print(f"\n🏗️  Model Architecture:")
        print(f"   Input Dim: {in_dim}, Hidden Dim: {hidden_dim}, Output Dim: {out_dim}")
        print(f"   Attention Heads: {num_heads}, Edge Types: {len(etypes)}")
        
        # Optimizer: Pass both model and predictor parameters
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(predictor.parameters()), 
            lr=lr
        )
        
        # Loss Function: We want positive scores to be > negative scores
        # We will tell the loss function that pos_score > neg_score is the "target" (y=1)
        loss_fn = nn.MarginRankingLoss(margin=0.2)
        
        # 3. Create Dataloader for Link Prediction
        print("\nStep 5: Setting up Link Prediction Dataloader...")
        
        # --- THIS IS THE FIX ---
        # Get the total number of 'cites' edges
        num_cites_edges = graph.num_edges(cites_etype_tuple)
        
        if num_cites_edges == 0:
            print("❌ Error: No 'cites' edges found in the graph.")
            print("   Cannot perform link prediction training. Check your data filtering.")
            return None

        # Create a tensor of all edge IDs, from 0 to N-1
        paper_cites_eids = torch.arange(num_cites_edges, device=self.device)
        
        
        # --- DGL 1.1.2 Compatible Implementation ---
        # In DGL 1.1.x, we don't use EdgeDataLoader, instead we manually batch the edges
        
        print("   ✅ Using manual batching for DGL 1.1.2 compatibility")
        
        # Create batches manually
        batch_size = 1024
        num_batches = (num_cites_edges + batch_size - 1) // batch_size
        
        # Shuffle edge IDs
        shuffled_eids = paper_cites_eids[torch.randperm(num_cites_edges)]
        
        print(f"\nStep 6: Training for {epochs} epochs using {len(paper_cites_eids)} 'cites' edges...")
        print(f"   Batch size: {batch_size}, Negative samples per edge: 5, Batches per epoch: {num_batches}")
        print("=" * 70 + "\n")
        
        model.train()
        predictor.train()
        
        train_start_time = time.time()
        training_history = []
        
        for epoch in range(epochs):
            epoch_start = time.time()
            total_loss = 0
            
            # Shuffle edges at the start of each epoch
            shuffled_eids = paper_cites_eids[torch.randperm(num_cites_edges)]
            
            pbar = tqdm(range(num_batches), 
                       desc=f"Epoch {epoch+1}/{epochs}", 
                       ncols=100,
                       dynamic_ncols=True)
            
            for batch_idx in pbar:
                # Compute embeddings for this batch (creates a new computation graph)
                embeddings = model(graph, node_features)
                
                # Get batch of edge IDs
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_cites_edges)
                batch_eids = shuffled_eids[start_idx:end_idx]
                
                # Get source and destination nodes for this batch
                src, dst = graph.find_edges(batch_eids, etype=cites_etype_tuple)
                
                # Create positive graph for this batch
                positive_graph = dgl.heterograph(
                    {cites_etype_tuple: (src, dst)},
                    num_nodes_dict={'paper': graph.num_nodes('paper')}
                ).to(self.device)
                
                # Score the positive edges
                pos_score = predictor(positive_graph, embeddings, ntype='paper')
                
                # Generate negative samples
                k = 5  # Number of negative samples per positive edge
                num_pos_edges = len(batch_eids)
                
                # For DGL 1.1.2, use uniform negative sampling
                neg_src = src.repeat_interleave(k)
                neg_dst = torch.randint(0, graph.num_nodes('paper'), (num_pos_edges * k,), device=self.device)
                
                # Create negative graph
                negative_graph = dgl.heterograph(
                    {cites_etype_tuple: (neg_src, neg_dst)},
                    num_nodes_dict={'paper': graph.num_nodes('paper')}
                ).to(self.device)
                
                # Score the negative edges
                neg_score = predictor(negative_graph, embeddings, ntype='paper')
                
                # Calculate Loss
                y = torch.ones_like(pos_score)
                
                # Reshape neg_score: [batch_size * k] -> [batch_size, k]
                neg_score_k = neg_score.view(-1, k)
                # Average the scores of the k negative samples
                neg_score_mean = neg_score_k.mean(1)
                
                loss = loss_fn(pos_score, neg_score_mean, y)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
            # End of epoch
            avg_loss = total_loss / num_batches
            epoch_time = time.time() - epoch_start
            training_history.append(avg_loss)
            
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Avg Loss: {avg_loss:.4f} - "
                  f"Time: {epoch_time:.2f}s")

        total_train_time = time.time() - train_start_time
        print(f"\n✅ Training completed in {timedelta(seconds=int(total_train_time))}!")
        
        # 7. Save model and embeddings
        print(f"\nStep 7: Saving model and embeddings to {save_dir}/...")
        os.makedirs(save_dir, exist_ok=True)
        
        model.eval()
        with torch.no_grad():
            # Run a final forward pass to get embeddings *without* batchnorm update
            final_embeddings = model(graph, node_features, apply_batchnorm=False)
            final_embeddings = {k: v.cpu() for k, v in final_embeddings.items()}
        
        # Save model file
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

        # (Optional: Add back your summary.json generation here)
        
        print("\n📊 Training Summary:")
        if training_history:
            print(f"   Final Loss: {training_history[-1]:.4f}")
            print(f"   Best Loss: {min(training_history):.4f}")
            print(f"   Avg Loss: {np.mean(training_history):.4f}")
        print(f"   Total Time: {timedelta(seconds=int(total_train_time))}")
        
        return model, graph, final_embeddings, save_data['id_maps']
    # --- END OF MODIFIED FUNCTION ---

if __name__ == "__main__":
    import sys

    # ===================== DEBUG CONFIGURATION =====================
    DEBUG_MODE = True  # Set to True to use the config below
    # Only used if DEBUG_MODE=True
    DEBUG_CONFIG = {
        'NEO4J_URI': "neo4j://127.0.0.1:7687",
        'NEO4J_USERNAME': "neo4j",
        'NEO4J_PASSWORD': "12345678", # <-- Use your password for the FOCUSED DB
        'SAMPLE_SIZE': None,           # <-- Set to None to use your whole focused DB
        'EPOCHS': 50,                  # <-- Start with 50-100 epochs
        'LEARNING_RATE': 0.001,
        'SAVE_DIR': 'models/link_prediction_v1', # <-- New save dir
        'HIDDEN_DIM': 128,
        'OUT_DIM': 384,
        'NUM_HEADS': 8
    }
    # ===================== END DEBUG CONFIGURATION =================

    print("=" * 70)
    print("🎓 Academic Graph HAN Model Training (Link Prediction)")
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
            print("❌ Incorrect number of arguments! Please pass 10 arguments via command line:")
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
        # Initialize trainer
        trainer = GraphEmbeddingTrainer(
            neo4j_uri=NEO4J_URI,
            neo4j_username=NEO4J_USERNAME,
            neo4j_password=NEO4J_PASSWORD
        )

        # Train model
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

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)