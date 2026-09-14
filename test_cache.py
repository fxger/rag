import sys
import time
sys.path.insert(0, 'D:\\Desktop\\RAG')

from src.core.embedding.embedder import EmbeddingGenerator, normalize_query
import asyncio

print('=' * 60)
print('  智能缓存测试')
print('=' * 60)

print('\n[1] 查询标准化:')
queries = [
    '公司年收入是多少?',
    '公司年收入是多少',
    '公司年收入多少?',
    '公司年收入',
    '请问公司年收入是多少呢?',
]
for q in queries:
    normalized = normalize_query(q)
    print(f'  "{q}" -> "{normalized}"')

async def test_cache():
    gen = EmbeddingGenerator()
    
    print('\n[2] 首次查询 (缓存未命中):')
    start = time.time()
    emb1 = await gen.generate_with_cache('公司年收入是多少?')
    time1 = (time.time() - start) * 1000
    print(f'  时间: {time1:.1f}ms')
    
    print('\n[3] 精确匹配 (完全相同):')
    start = time.time()
    emb2 = await gen.generate_with_cache('公司年收入是多少?')
    time2 = (time.time() - start) * 1000
    print(f'  时间: {time2:.1f}ms')
    
    print('\n[4] 模糊匹配 (少了问号):')
    start = time.time()
    emb3 = await gen.generate_with_cache('公司年收入是多少')
    time3 = (time.time() - start) * 1000
    print(f'  时间: {time3:.1f}ms')
    
    print('\n[5] 模糊匹配 (少了是):')
    start = time.time()
    emb4 = await gen.generate_with_cache('公司年收入多少?')
    time4 = (time.time() - start) * 1000
    print(f'  时间: {time4:.1f}ms')
    
    print('\n[6] 模糊匹配 (加了请问呢):')
    start = time.time()
    emb5 = await gen.generate_with_cache('请问公司年收入是多少呢?')
    time5 = (time.time() - start) * 1000
    print(f'  时间: {time5:.1f}ms')
    
    print('\n[7] 缓存统计:')
    stats = gen.cache_stats
    print(f'  精确命中: {stats["l1_exact_hits"]}次')
    print(f'  模糊命中: {stats["l1_fuzzy_hits"]}次')
    print(f'  未命中: {stats["misses"]}次')
    print(f'  命中率: {stats["hit_rate"]*100:.1f}%')
    print(f'  模糊索引: {stats["fuzzy_index_size"]}条')
    
    print('\n' + '=' * 60)

asyncio.run(test_cache())
