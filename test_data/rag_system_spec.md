# RAG技术规格

检索延迟P95<500ms，准确率Recall@10>90%。
生成：首Token<1s，速度>30 tokens/s。
容量：100万文档，1000 QPS，99.9%可用。
向量维度3072，混合检索Milvus+BM25。