"""
验证 Original Sentence-BERT Embeddings

这个脚本专门用于验证原始的 Sentence-BERT embeddings（未经 HAN 训练）
可以作为基准线来对比 HAN 训练后的效果
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import json
import os
from datetime import datetime


class OriginalEmbeddingValidator:
    """验证原始 Sentence-BERT Embeddings"""
    
    def __init__(self, model_path):
        """加载模型并提取原始 embeddings"""
        print("=" * 70)
        print("Validating ORIGINAL Sentence-BERT Embeddings (Baseline)")
        print("=" * 70)
        print(f"\nLoading from: {model_path}")
        
        self.device = torch.device('cpu')
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 提取原始 embeddings
        if 'original_embeddings' in checkpoint:
            self.embeddings = checkpoint['original_embeddings']
            print("  Found 'original_embeddings' in checkpoint")
        else:
            print("  ERROR: No 'original_embeddings' found!")
            print("  Available keys:", list(checkpoint.keys()))
            raise KeyError("'original_embeddings' not found in checkpoint")
        
        self.id_maps = checkpoint['id_maps']
        
        # 输出目录
        self.output_dir = os.path.join(os.path.dirname(model_path), 'original_validation')
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"\nNode types found:")
        for ntype, emb in self.embeddings.items():
            print(f"  • {ntype}: {emb.shape}")
        
        print(f"\nOutput directory: {self.output_dir}")
    
    def analyze_distribution(self, node_type='paper'):
        """分析嵌入的分布"""
        print(f"\n{'='*70}")
        print(f"Analyzing Distribution: {node_type}")
        print("=" * 70)
        
        if node_type not in self.embeddings:
            print(f"  ERROR: {node_type} not found")
            return None
        
        # 转换为 numpy
        emb = self.embeddings[node_type]
        if isinstance(emb, torch.Tensor):
            emb = emb.cpu().numpy()
        
        print(f"\nShape: {emb.shape}")
        print(f"Dimension: {emb.shape[1]}D")
        
        # 统计信息
        print(f"\nValue Statistics:")
        print(f"  Mean: {np.mean(emb):.6f}")
        print(f"  Std:  {np.std(emb):.6f}")
        print(f"  Min:  {np.min(emb):.6f}")
        print(f"  Max:  {np.max(emb):.6f}")
        
        # L2 范数
        norms = np.linalg.norm(emb, axis=1)
        print(f"\nL2 Norm Statistics:")
        print(f"  Mean: {np.mean(norms):.4f}")
        print(f"  Std:  {np.std(norms):.4f}")
        print(f"  Min:  {np.min(norms):.4f}")
        print(f"  Max:  {np.max(norms):.4f}")
        
        # 检查异常值
        has_nan = np.any(np.isnan(emb))
        has_inf = np.any(np.isinf(emb))
        
        if has_nan or has_inf:
            print("\n  WARNING: Contains NaN or Inf!")
        else:
            print("\n  Status: Numerically stable")
        
        # 绘制分布图
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 值分布直方图
        axes[0, 0].hist(emb.flatten(), bins=100, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Value Distribution')
        axes[0, 0].set_xlabel('Embedding Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. L2 范数分布
        axes[0, 1].hist(norms, bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].set_title('L2 Norm Distribution')
        axes[0, 1].set_xlabel('L2 Norm')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 维度方差
        dim_vars = np.var(emb, axis=0)
        axes[1, 0].plot(dim_vars, linewidth=0.8)
        axes[1, 0].set_title('Variance per Dimension')
        axes[1, 0].set_xlabel('Dimension Index')
        axes[1, 0].set_ylabel('Variance')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 维度均值
        dim_means = np.mean(emb, axis=0)
        axes[1, 1].plot(dim_means, linewidth=0.8, color='red')
        axes[1, 1].set_title('Mean per Dimension')
        axes[1, 1].set_xlabel('Dimension Index')
        axes[1, 1].set_ylabel('Mean Value')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f'{node_type}_distribution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n  Saved: {save_path}")
        plt.close()
        
        return {
            'shape': list(emb.shape),
            'mean': float(np.mean(emb)),
            'std': float(np.std(emb)),
            'norm_mean': float(np.mean(norms)),
            'norm_std': float(np.std(norms)),
            'has_nan': bool(has_nan),
            'has_inf': bool(has_inf)
        }
    
    def analyze_similarity(self, node_type='paper', n_samples=100):
        """分析相似性分布"""
        print(f"\n{'='*70}")
        print(f"Analyzing Similarity: {node_type}")
        print("=" * 70)
        
        emb = self.embeddings[node_type]
        if isinstance(emb, torch.Tensor):
            emb = emb.cpu().numpy()
        
        n_nodes = emb.shape[0]
        
        # 随机采样查询
        query_indices = np.random.choice(n_nodes, min(n_samples, n_nodes), replace=False)
        
        similarities = []
        top_k_sims = {1: [], 5: [], 10: []}
        
        print(f"\nTesting {len(query_indices)} random queries...")
        
        for idx in query_indices:
            query = emb[idx].reshape(1, -1)
            sims = cosine_similarity(query, emb)[0]
            
            # 排除自身
            sims[idx] = -2
            
            # Top-1
            max_sim = np.max(sims)
            similarities.append(max_sim)
            
            # Top-K
            for k in [1, 5, 10]:
                top_k_idx = np.argsort(sims)[-k:]
                valid_sims = sims[top_k_idx[sims[top_k_idx] > -2]]
                if len(valid_sims) > 0:
                    top_k_sims[k].append(np.mean(valid_sims))
        
        avg_sim = np.mean(similarities)
        std_sim = np.std(similarities)
        
        print(f"\nCosine Similarity Statistics:")
        print(f"  Top-1 Avg: {avg_sim:.4f} ± {std_sim:.4f}")
        print(f"  Top-5 Avg: {np.mean(top_k_sims[5]):.4f}")
        print(f"  Top-10 Avg: {np.mean(top_k_sims[10]):.4f}")
        print(f"  Range: [{np.min(similarities):.4f}, {np.max(similarities):.4f}]")
        
        # 评估
        if avg_sim > 0.4:
            print(f"\n  Status: GOOD - Sentence-BERT embeddings capture semantic similarity well")
        elif avg_sim > 0.2:
            print(f"\n  Status: OK - Moderate similarity")
        else:
            print(f"\n  WARNING: Low similarity - unexpected for Sentence-BERT")
        
        # 绘制相似度分布
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(similarities, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(avg_sim, color='red', linestyle='--', label=f'Mean: {avg_sim:.3f}')
        plt.title('Top-1 Similarity Distribution')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.boxplot([top_k_sims[1], top_k_sims[5], top_k_sims[10]], 
                   labels=['Top-1', 'Top-5', 'Top-10'])
        plt.title('Top-K Average Similarity')
        plt.ylabel('Cosine Similarity')
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f'{node_type}_similarity.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n  Saved: {save_path}")
        plt.close()
        
        return {
            'avg_similarity': float(avg_sim),
            'std_similarity': float(std_sim),
            'min_similarity': float(np.min(similarities)),
            'max_similarity': float(np.max(similarities)),
            'top_k_avg': {k: float(np.mean(v)) for k, v in top_k_sims.items()}
        }
    
    def visualize_tsne(self, node_type='paper', n_samples=1000):
        """t-SNE 降维可视化"""
        print(f"\n{'='*70}")
        print(f"t-SNE Visualization: {node_type}")
        print("=" * 70)
        
        emb = self.embeddings[node_type]
        if isinstance(emb, torch.Tensor):
            emb = emb.cpu().numpy()
        
        # 采样
        n_nodes = emb.shape[0]
        if n_nodes > n_samples:
            indices = np.random.choice(n_nodes, n_samples, replace=False)
            sampled = emb[indices]
            print(f"\nSampled {n_samples} from {n_nodes} nodes")
        else:
            sampled = emb
            print(f"\nUsing all {n_nodes} nodes")
        
        # t-SNE
        print("Running t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(sampled)-1))
        emb_2d = tsne.fit_transform(sampled)
        
        # K-means 聚类
        n_clusters = min(5, len(sampled) // 20)
        print(f"K-Means clustering (k={n_clusters})...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(sampled)
        
        # 聚类质量指标
        sil_score = silhouette_score(sampled, clusters)
        cal_score = calinski_harabasz_score(sampled, clusters)
        db_score = davies_bouldin_score(sampled, clusters)
        
        print(f"\nClustering Quality:")
        print(f"  Silhouette: {sil_score:.4f} (higher is better, -1 to 1)")
        print(f"  Calinski-Harabasz: {cal_score:.2f} (higher is better)")
        print(f"  Davies-Bouldin: {db_score:.4f} (lower is better)")
        
        # 绘图
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 聚类散点图
        scatter = axes[0].scatter(emb_2d[:, 0], emb_2d[:, 1], 
                                 c=clusters, cmap='tab10', alpha=0.6, s=20)
        axes[0].set_title(f'{node_type.capitalize()} - t-SNE with K-Means (k={n_clusters})')
        axes[0].set_xlabel('t-SNE Dim 1')
        axes[0].set_ylabel('t-SNE Dim 2')
        plt.colorbar(scatter, ax=axes[0], label='Cluster')
        
        # 密度图
        sns.kdeplot(x=emb_2d[:, 0], y=emb_2d[:, 1], 
                   cmap='Blues', fill=True, ax=axes[1], thresh=0.05)
        axes[1].set_title(f'{node_type.capitalize()} - Density')
        axes[1].set_xlabel('t-SNE Dim 1')
        axes[1].set_ylabel('t-SNE Dim 2')
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f'{node_type}_tsne.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n  Saved: {save_path}")
        plt.close()
        
        return {
            'n_clusters': n_clusters,
            'silhouette_score': float(sil_score),
            'calinski_harabasz': float(cal_score),
            'davies_bouldin': float(db_score)
        }
    
    def generate_report(self):
        """生成完整报告"""
        print(f"\n{'='*70}")
        print("Generating Complete Validation Report")
        print("=" * 70)
        
        report = {
            'validation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'embedding_type': 'Original Sentence-BERT (Baseline)',
            'results': {}
        }
        
        # 对每种节点类型进行分析
        for node_type in self.embeddings.keys():
            print(f"\n{'='*70}")
            print(f"Processing: {node_type}")
            print("=" * 70)
            
            results = {}
            
            # 1. 分布分析
            results['distribution'] = self.analyze_distribution(node_type)
            
            # 2. 相似性分析
            results['similarity'] = self.analyze_similarity(node_type, n_samples=100)
            
            # 3. t-SNE 可视化
            results['clustering'] = self.visualize_tsne(node_type, n_samples=1000)
            
            report['results'][node_type] = results
        
        # 保存报告
        report_path = os.path.join(self.output_dir, 'validation_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*70}")
        print("Validation Complete!")
        print("=" * 70)
        print(f"\nReport saved: {report_path}")
        print(f"Visualizations: {self.output_dir}/*.png")
        
        # 打印摘要
        print(f"\n{'='*70}")
        print("SUMMARY - Original Sentence-BERT Embeddings")
        print("=" * 70)
        
        for ntype, res in report['results'].items():
            print(f"\n{ntype.capitalize()}:")
            print(f"  Similarity (Top-1): {res['similarity']['avg_similarity']:.4f}")
            print(f"  Silhouette Score: {res['clustering']['silhouette_score']:.4f}")
            print(f"  Status: {'GOOD' if res['similarity']['avg_similarity'] > 0.4 else 'OK'}")
        
        return report


if __name__ == "__main__":
    import sys
    
    # 从命令行获取模型路径
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "models/link_prediction_v1/han_embeddings.pth"
    
    print("\nUsage: python validate_original_embeddings.py [model_path]")
    print(f"Using: {model_path}\n")
    
    try:
        validator = OriginalEmbeddingValidator(model_path)
        report = validator.generate_report()
        
        print("\n" + "=" * 70)
        print("SUCCESS!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
