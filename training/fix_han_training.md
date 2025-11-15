# 修复 HAN 训练的建议

## 问题
当前 HAN 只做 Link Prediction，学到的 embedding 对语义搜索无用。

## 解决方案：多任务学习

### 1. 添加语义保持损失
```python
# 除了 Link Prediction Loss，还要加：
semantic_loss = MSELoss(han_embeddings, original_embeddings)

# 总损失
total_loss = link_pred_loss + 0.5 * semantic_loss
```

这样 HAN embeddings 会：
- 保留语义信息（接近 Sentence-BERT）
- 融入图结构信息（通过 Link Prediction）

### 2. 使用所有边类型
```python
# 不只用 'cites'，也用 'written_by' 和 'has_keyword'
meta_paths = [
    ('paper', 'cites', 'paper'),
    ('paper', 'written_by', 'author'),
    ('paper', 'has_keyword', 'keyword')
]
```

### 3. 添加对比学习
```python
# 让结构相似的论文 embedding 也接近
def contrastive_loss(embeddings, graph):
    # 同一作者的论文应该相似
    # 同一关键词的论文应该相似
    pass
```

## 预期效果
修复后的 HAN embeddings 应该：
- 语义搜索分数 > 0.3（现在是 0.09）
- 能找到既语义相关又结构相关的论文
- Hybrid 方法能找到纯语义找不到的"隐藏相关"论文
