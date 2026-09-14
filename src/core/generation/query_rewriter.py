"""Query rewriting and expansion for improved retrieval quality.

Provides LLM-based multi-query expansion and Hypothetical Document
Embedding (HyDE) to increase recall in semantic search.

── 接入状态：已实现，评估后【不接入】主链路（2026-09-14 决策）──────────────
本模块保留为可选扩展，**不是漏接**。

不接入的四条理由：
  1. 延迟 —— 现有检索约 15ms，而 rewrite()/hyde() 各会额外插入一次 LLM
     调用（+1~3s）；优化的是链路里最短的一段，总延迟反增。
  2. 成本 —— 每次查询的 LLM 调用从 1 次变为 2~3 次（max_tokens 500/600），
     在 DeepSeek 按量计费下账单翻倍以上。
  3. 收益 —— 混合检索的 BM25 侧（jieba 分词）本身按词匹配，已覆盖大部分
     词汇鸿沟，本模块的边际收益被削弱。
  4. 举证 —— 尚无对照实验，无法量化净收益（见 执行方案.md 的 C3）。

将来若重议接入，最小代价路线：
  · 只对短查询（< 10 字）或首轮未命中触发；
  · 重写结果进缓存，避免同一 query 重复调 LLM；
  · 加 settings.ENABLE_QUERY_REWRITE 开关，默认关闭；
  · 先做 C3 对照实验拿到基线，再决定是否开启。

已知瑕疵（接入前需修）：
  · 两段提示词为英文，中文查询下引导效果打折；
  · __init__ 读 settings.OPENAI_API_KEY，实际运行在 DeepSeek 上（靠
    base_url 兼容），命名易误导。

详见 执行方案.md · A4 章节。
"""
import logging
from typing import Optional

from openai import AsyncOpenAI

from src.config.settings import get_settings

logger = logging.getLogger(__name__)

REWRITE_SYSTEM_PROMPT = """You are a search query optimization assistant.
Given a user query, generate {num_variants} alternative phrasings that capture the same intent
but use different vocabulary, structure, or perspective.

Guidelines:
- Preserve the original semantic intent exactly
- Use synonyms and alternative phrasings
- Vary specificity levels (some more general, some more specific)
- Include both question and statement forms
- Output ONLY a JSON array of strings, no explanation"""

HYDE_SYSTEM_PROMPT = """You are a knowledgeable assistant. Given a question, write a detailed,
informative paragraph that would serve as an ideal answer. This hypothetical document will be used
for semantic search retrieval, so include relevant terminology, concepts, and technical details.

Write as if you are a domain expert answering definitively. Do not hedge or express uncertainty."""


class QueryRewriter:
    """Rewrites queries to improve retrieval recall via LLM expansion."""

    def __init__(self):
        settings = get_settings()
        self.client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL,
        )
        self.model = settings.LLM_MODEL
        self.temperature = settings.LLM_TEMPERATURE

    async def rewrite(self, query: str, num_variants: int = 3) -> list[str]:
        """Generate multiple query variants for multi-query retrieval.
        
        Uses LLM to produce semantically equivalent phrasings of the input query,
        increasing the chance of matching relevant documents.
        
        Args:
            query: Original user query to expand.
            num_variants: Number of alternative queries to generate (default: 3).
            
        Returns:
            List of query strings including the original query at index 0.
            
        Raises:
            openai.APIError: On LLM API failures.
        """
        if not query.strip():
            return [query]

        system_prompt = REWRITE_SYSTEM_PROMPT.format(num_variants=num_variants)

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                temperature=self.temperature,
                max_tokens=500,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            variants = self._parse_variants(content, num_variants)
            
            # Always include original query first
            all_queries = [query] + [v for v in variants if v != query]
            logger.info(
                "Query rewrite: '%s' -> %d variants", 
                query[:50], 
                len(all_queries) - 1,
            )
            return all_queries

        except Exception as e:
            logger.error("Query rewrite failed, falling back to original: %s", e)
            return [query]

    async def hyde(self, query: str) -> str:
        """Generate hypothetical answer for Hypothetical Document Embedding.
        
        HyDE improves retrieval by embedding a plausible answer rather than the question.
        The hypothetical answer shares vocabulary and concepts with real documents,
        making it a better retrieval target than the original query.
        
        Args:
            query: User query to generate hypothetical answer for.
            
        Returns:
            Hypothetical answer text suitable for embedding and retrieval.
            
        Raises:
            openai.APIError: On LLM API failures.
        """
        if not query.strip():
            return query

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": HYDE_SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                temperature=self.temperature,
                max_tokens=600,
            )

            hypothetical = response.choices[0].message.content.strip()
            logger.info(
                "HyDE generated %d chars for query: '%s'",
                len(hypothetical),
                query[:50],
            )
            return hypothetical

        except Exception as e:
            logger.error("HyDE generation failed, falling back to original query: %s", e)
            return query

    @staticmethod
    def _parse_variants(content: str, expected_count: int) -> list[str]:
        """Parse JSON response containing query variants.
        
        Args:
            content: Raw LLM response content.
            expected_count: Expected number of variants.
            
        Returns:
            List of variant strings.
        """
        import json

        try:
            parsed = json.loads(content)
            
            # Handle both {"queries": [...]} and [...] formats
            if isinstance(parsed, dict):
                variants = next(
                    (v for v in parsed.values() if isinstance(v, list)),
                    [],
                )
            elif isinstance(parsed, list):
                variants = parsed
            else:
                return []
            
            # Clean and validate
            return [
                v.strip() for v in variants
                if isinstance(v, str) and v.strip()
            ][:expected_count]

        except (json.JSONDecodeError, StopIteration):
            logger.warning("Failed to parse query variants JSON")
            return []
