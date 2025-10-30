import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import json
import os
from datetime import datetime

class EmbeddingValidator:
    def __init__(self, model_path='models/han_embeddings.pth'):
        """初始化验证器并加载训练好的嵌入"""
        self.device = torch.device('cpu')  # 与训练保持一致
        self.model_path = model_path
        print(f"\n{'='*70}")
        print(f"🔍 Embedding Validation")
        print(f"{'='*70}")
        print(f"📂 Loading model from: {model_path}")
        
        self.load_model(model_path)
        
        # 创建可视化输出目录
        self.viz_dir = os.path.join(os.path.dirname(model_path), 'visualizations')
        os.makedirs(self.viz_dir, exist_ok=True)
        
    def load_model(self, model_path):
        """加载训练好的模型和嵌入"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Model file not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 从checkpoint中提取数据
        self.embeddings = checkpoint['embeddings']  # 已经是字典形式: {node_type: tensor}
        self.id_maps = checkpoint['id_maps']
        self.config = checkpoint['config']
        
        print(f"✅ Model loaded successfully!")
        print(f"   Node types: {list(self.embeddings.keys())}")
        print(f"   Embedding dimension: {self.config['out_dim']}")
        
        # 加载训练摘要(如果存在)
        summary_path = os.path.join(os.path.dirname(model_path), 'summary.json')
        if os.path.exists(summary_path):
            with open(summary_path, 'r', encoding='utf-8') as f:
                self.training_summary = json.load(f)
            print(f"✅ Training summary loaded")
        else:
            self.training_summary = None
            print(f"⚠️  No training summary found")
    
    
    def visualize_embeddings(self, node_type='paper', n_samples=1000):
        """可视化嵌入空间"""
        print(f"\n{'='*70}")
        print(f"📈 Visualizing {node_type.capitalize()} Embeddings")
        print(f"{'='*70}")
        
        if node_type not in self.embeddings:
            print(f"❌ Node type '{node_type}' not found")
            return None
        
        # 获取节点嵌入
        embeddings = self.embeddings[node_type].cpu().numpy()
        print(f"   Total {node_type}s: {len(embeddings)}")
        
        # 随机采样
        if len(embeddings) > n_samples:
            indices = np.random.choice(len(embeddings), n_samples, replace=False)
            sampled_embeddings = embeddings[indices]
            print(f"   Sampled: {n_samples} for visualization")
        else:
            sampled_embeddings = embeddings
            print(f"   Using all {len(embeddings)} samples")
        
        # t-SNE降维
        print("   Running t-SNE dimensionality reduction...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(sampled_embeddings)-1))
        embeddings_2d = tsne.fit_transform(sampled_embeddings)
        
        # 聚类分析
        n_clusters = min(5, len(sampled_embeddings) // 10)  # 自适应聚类数
        print(f"   Performing K-Means clustering (k={n_clusters})...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(sampled_embeddings)
        
        # 绘制可视化
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 左图: 聚类可视化
        scatter = axes[0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                   c=clusters, cmap='viridis', alpha=0.6, s=20)
        axes[0].set_title(f'{node_type.capitalize()} Embeddings - Clustering (k={n_clusters})', fontsize=14)
        axes[0].set_xlabel('t-SNE Dimension 1')
        axes[0].set_ylabel('t-SNE Dimension 2')
        plt.colorbar(scatter, ax=axes[0], label='Cluster')
        
        # 右图: 密度图
        sns.kdeplot(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], 
                   cmap='Blues', fill=True, ax=axes[1], thresh=0.05)
        axes[1].set_title(f'{node_type.capitalize()} Embeddings - Density', fontsize=14)
        axes[1].set_xlabel('t-SNE Dimension 1')
        axes[1].set_ylabel('t-SNE Dimension 2')
        
        plt.tight_layout()
        save_path = os.path.join(self.viz_dir, f'{node_type}_embeddings.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Visualization saved: {save_path}")
        plt.close()
        
        # 计算聚类质量指标
        silhouette_avg = silhouette_score(sampled_embeddings, clusters)
        calinski = calinski_harabasz_score(sampled_embeddings, clusters)
        davies_bouldin = davies_bouldin_score(sampled_embeddings, clusters)
        
        print(f"\n📊 Clustering Quality Metrics:")
        print(f"   Silhouette Score: {silhouette_avg:.4f} (higher is better, range: -1 to 1)")
        print(f"   Calinski-Harabasz: {calinski:.2f} (higher is better)")
        print(f"   Davies-Bouldin: {davies_bouldin:.4f} (lower is better)")
        
        return {
            'silhouette_score': float(silhouette_avg),
            'calinski_harabasz_score': float(calinski),
            'davies_bouldin_score': float(davies_bouldin),
            'n_clusters': n_clusters,
            'n_samples': len(sampled_embeddings)
        }
    
    
    def evaluate_similarity(self, node_type='paper', n_pairs=100):
        """评估相似性检索质量"""
        print(f"\n{'='*70}")
        print(f"🔍 Evaluating Similarity for {node_type.capitalize()}")
        print(f"{'='*70}")
        
        if node_type not in self.embeddings:
            print(f"❌ Node type '{node_type}' not found")
            return None
        
        embeddings = self.embeddings[node_type].cpu().numpy()
        n_nodes = len(embeddings)
        
        # 随机选择查询节点
        query_indices = np.random.choice(n_nodes, min(n_pairs, n_nodes), replace=False)
        
        similarity_scores = []
        top_k_similarities = []  # 存储top-5平均相似度
        
        print(f"   Testing {len(query_indices)} random queries...")
        
        for i, query_idx in enumerate(query_indices):
            query_embedding = embeddings[query_idx].reshape(1, -1)
            
            # 计算与所有其他节点的余弦相似度
            similarities = cosine_similarity(query_embedding, embeddings)[0]
            
            # 排除自身
            similarities[query_idx] = -1
            
            # 找到最相似的节点
            most_similar_idx = np.argmax(similarities)
            similarity_scores.append(similarities[most_similar_idx])
            
            # Top-5平均相似度
            top_5_indices = np.argsort(similarities)[-5:]
            top_k_similarities.append(np.mean(similarities[top_5_indices]))
            
            if i < 5:  # 打印前5个示例
                print(f"   Query {i+1}: Top-1 similarity = {similarities[most_similar_idx]:.4f}")
        
        avg_similarity = np.mean(similarity_scores)
        avg_top5_similarity = np.mean(top_k_similarities)
        std_similarity = np.std(similarity_scores)
        
        print(f"\n📈 Similarity Statistics:")
        print(f"   Avg Top-1 Similarity: {avg_similarity:.4f} ± {std_similarity:.4f}")
        print(f"   Avg Top-5 Similarity: {avg_top5_similarity:.4f}")
        print(f"   Min Similarity: {np.min(similarity_scores):.4f}")
        print(f"   Max Similarity: {np.max(similarity_scores):.4f}")
        
        return {
            'avg_similarity': float(avg_similarity),
            'std_similarity': float(std_similarity),
            'avg_top5_similarity': float(avg_top5_similarity),
            'min_similarity': float(np.min(similarity_scores)),
            'max_similarity': float(np.max(similarity_scores))
        }
    
    def analyze_loss_trend(self):
        """分析训练损失趋势"""
        if not self.training_summary or 'loss_history' not in self.training_summary:
            print("\n⚠️  No loss history available for analysis")
            return None
        
        print(f"\n{'='*70}")
        print("📉 Analyzing Loss Trend")
        print(f"{'='*70}")
        
        losses = self.training_summary['loss_history']
        
        # 计算损失统计
        final_loss = losses[-1]
        min_loss = min(losses)
        max_loss = max(losses)
        avg_loss = np.mean(losses)
        
        # 计算损失下降率
        if len(losses) > 10:
            early_avg = np.mean(losses[:10])
            late_avg = np.mean(losses[-10:])
            improvement = (early_avg - late_avg) / early_avg * 100
        else:
            improvement = 0
        
        # 检查收敛性
        if len(losses) > 20:
            recent_losses = losses[-20:]
            loss_variance = np.var(recent_losses)
            is_converged = loss_variance < 0.01  # 阈值可调
        else:
            loss_variance = np.var(losses)
            is_converged = False
        
        print(f"   Final Loss: {final_loss:.4f}")
        print(f"   Min Loss: {min_loss:.4f}")
        print(f"   Max Loss: {max_loss:.4f}")
        print(f"   Avg Loss: {avg_loss:.4f}")
        print(f"   Improvement: {improvement:.2f}%")
        print(f"   Recent Variance: {loss_variance:.6f}")
        print(f"   Converged: {'✅ Yes' if is_converged else '❌ No'}")
        
        # 绘制损失曲线
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(losses, linewidth=1.5, color='#2E86AB')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        # 使用滑动平均平滑曲线
        window = min(10, len(losses) // 10)
        if window > 1:
            smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
            plt.plot(smoothed, linewidth=2, color='#A23B72', label=f'Smoothed (window={window})')
        plt.plot(losses, linewidth=0.5, alpha=0.3, color='#2E86AB', label='Original')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Smoothed Loss Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.viz_dir, 'loss_curve.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Loss curve saved: {save_path}")
        plt.close()
        
        return {
            'final_loss': float(final_loss),
            'min_loss': float(min_loss),
            'max_loss': float(max_loss),
            'avg_loss': float(avg_loss),
            'improvement_percent': float(improvement),
            'recent_variance': float(loss_variance),
            'is_converged': bool(is_converged)
        }
    
    def generate_evaluation_report(self):
        """生成完整的评估报告"""
        print(f"\n{'='*70}")
        print("📝 Generating Evaluation Report")
        print(f"{'='*70}")
        
        report = {
            'model_path': self.model_path,
            'evaluation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'embedding_properties': {},
            'clustering_quality': {},
            'similarity_metrics': {},
            'loss_analysis': None,
            'recommendations': []
        }
        
        # 1. 嵌入属性检查
        report['embedding_properties'] = self.check_embedding_properties()
        
        # 2. 损失趋势分析
        report['loss_analysis'] = self.analyze_loss_trend()
        
        # 3. 对每种节点类型进行评估
        for node_type in self.embeddings.keys():
            print(f"\n{'-'*70}")
            print(f"Evaluating {node_type.capitalize()} embeddings...")
            print(f"{'-'*70}")
            
            # 聚类质量
            clustering_result = self.visualize_embeddings(node_type, n_samples=1000)
            if clustering_result:
                report['clustering_quality'][node_type] = clustering_result
            
            # 相似性评估
            similarity_result = self.evaluate_similarity(node_type, n_pairs=100)
            if similarity_result:
                report['similarity_metrics'][node_type] = similarity_result
        
        # 4. 生成建议
        recommendations = self._generate_recommendations(report)
        report['recommendations'] = recommendations
        
        # 5. 保存报告
        report_path = os.path.join(os.path.dirname(self.model_path), 'evaluation_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*70}")
        print(f"✅ Evaluation report saved: {report_path}")
        print(f"{'='*70}")
        
        # 打印建议
        if recommendations:
            print(f"\n💡 Recommendations for Hyperparameter Tuning:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        
        return report
    
    def _generate_recommendations(self, report):
        """基于评估结果生成超参数调整建议"""
        recommendations = []
        
        # 基于损失分析的建议
        if report['loss_analysis']:
            loss_info = report['loss_analysis']
            
            if not loss_info['is_converged']:
                recommendations.append(
                    "Loss has not converged yet. Consider increasing EPOCHS (current shows room for improvement)."
                )
            
            if loss_info['improvement_percent'] < 10:
                recommendations.append(
                    f"Low improvement ({loss_info['improvement_percent']:.1f}%). Try:\n"
                    "      - Increase LEARNING_RATE (e.g., 0.005 or 0.01)\n"
                    "      - Increase HIDDEN_DIM or OUT_DIM for more model capacity"
                )
            
            if loss_info['final_loss'] > 5.0:
                recommendations.append(
                    "High final loss. Consider:\n"
                    "      - Increase SAMPLE_SIZE for more training data\n"
                    "      - Adjust model architecture (HIDDEN_DIM, NUM_HEADS)"
                )
        
        # 基于聚类质量的建议
        if report['clustering_quality']:
            for node_type, metrics in report['clustering_quality'].items():
                silhouette = metrics['silhouette_score']
                
                if silhouette < 0.3:
                    recommendations.append(
                        f"Low clustering quality for {node_type} (Silhouette: {silhouette:.3f}). Try:\n"
                        f"      - Increase OUT_DIM to capture more features\n"
                        f"      - Increase EPOCHS for better convergence"
                    )
        
        # 基于相似性的建议
        if report['similarity_metrics']:
            for node_type, metrics in report['similarity_metrics'].items():
                avg_sim = metrics['avg_similarity']
                
                if avg_sim < 0.5:
                    recommendations.append(
                        f"Low similarity scores for {node_type} ({avg_sim:.3f}). Consider:\n"
                        f"      - Increase NUM_HEADS for better attention\n"
                        f"      - Use larger HIDDEN_DIM"
                    )
                elif avg_sim > 0.95:
                    recommendations.append(
                        f"Very high similarity for {node_type} ({avg_sim:.3f}) - possible overfitting. Try:\n"
                        f"      - Add regularization or dropout\n"
                        f"      - Reduce model complexity"
                    )
        
        # 通用建议
        if not recommendations:
            recommendations.append(
                "Model performance looks good! Consider:\n"
                "      - Fine-tuning LEARNING_RATE for slight improvements\n"
                "      - Increasing SAMPLE_SIZE if resources allow"
            )
        
        return recommendations
    
    
    def check_embedding_properties(self):
        """检查嵌入的基本属性"""
        print(f"\n{'='*70}")
        print("🔬 Checking Embedding Properties")
        print(f"{'='*70}")
        
        properties_report = {}
        
        for node_type in self.embeddings.keys():
            embeddings = self.embeddings[node_type].cpu().numpy()
            
            print(f"\n📊 {node_type.capitalize()} Embeddings:")
            print(f"   Shape: {embeddings.shape}")
            print(f"   Mean: {np.mean(embeddings):.4f}")
            print(f"   Std: {np.std(embeddings):.4f}")
            print(f"   Min: {np.min(embeddings):.4f}")
            print(f"   Max: {np.max(embeddings):.4f}")
            
            # 检查是否有NaN或Inf
            has_nan = np.any(np.isnan(embeddings))
            has_inf = np.any(np.isinf(embeddings))
            
            if has_nan or has_inf:
                print("   ❌ Contains NaN or Inf values!")
            else:
                print("   ✅ No NaN or Inf values")
            
            # 计算L2范数分布
            norms = np.linalg.norm(embeddings, axis=1)
            print(f"   L2 Norm - Mean: {np.mean(norms):.4f}, Std: {np.std(norms):.4f}")
            
            properties_report[node_type] = {
                'shape': list(embeddings.shape),
                'mean': float(np.mean(embeddings)),
                'std': float(np.std(embeddings)),
                'min': float(np.min(embeddings)),
                'max': float(np.max(embeddings)),
                'has_nan': bool(has_nan),
                'has_inf': bool(has_inf),
                'norm_mean': float(np.mean(norms)),
                'norm_std': float(np.std(norms))
            }
        
        return properties_report



def main():
    """主函数 - 支持命令行参数"""
    import sys
    
    # 默认模型路径
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = 'models/trial1/han_embeddings.pth'
    
    print(f"\n{'='*70}")
    print("🎯 HAN Model Evaluation")
    print(f"{'='*70}")
    
    try:
        # 初始化验证器
        validator = EmbeddingValidator(model_path=model_path)
        
        # 生成完整评估报告
        report = validator.generate_evaluation_report()
        
        print(f"\n{'='*70}")
        print("✅ Evaluation completed successfully!")
        print(f"{'='*70}")
        print(f"\n📂 Output files:")
        print(f"   • {os.path.dirname(model_path)}/evaluation_report.json - Full evaluation report")
        print(f"   • {os.path.dirname(model_path)}/visualizations/*.png - Visualization plots")
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    main()
