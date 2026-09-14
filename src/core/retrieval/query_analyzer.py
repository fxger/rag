"""Query analysis for retrieval optimization with dynamic weight adjustment."""
import logging
import re
from typing import Any

import jieba
import jieba.posseg as pseg

from .models import QueryAnalysis

logger = logging.getLogger(__name__)

# Query type classification patterns
QUERY_TYPE_PATTERNS = {
    "factual": [
        r"(什么是|是什么|是什么意思|定义|含义)",
        r"(谁|哪个|哪些|多少|几个|几点|何时|何地)",
        r"(在哪|哪里|地点|位置|时间|日期)",
        r"(\?|？|吗|呢|吧)",
    ],
    "conceptual": [
        r"(为什么|原因|原理|机制|解释|说明|理解)",
        r"(如何|怎么|怎样|方法|方式|步骤)",
        r"(区别|不同|差异|相同|类似|比较)",
        r"(概念|理论|思想|观点|看法)",
    ],
    "comparative": [
        r"(比较|对比|对照|相比|比起)",
        r"(优缺点|优势|劣势|利弊|好坏)",
        r"(哪个更好|哪个更|哪个最|最佳|最优)",
        r"(vs|versus|或者|还是|与|和)",
    ],
    "navigational": [
        r"(打开|访问|进入|登录|跳转)",
        r"(链接|网址|网站|页面|文档)",
        r"(找到|搜索|查找|查看|显示)",
        r"(文档|文件|资料|报告|论文)",
    ],
}

# Entity type patterns
ENTITY_PATTERNS = {
    "person": [
        r"(先生|女士|老师|教授|博士|总|经理|主任|院长|部长|书记)",
        r"[\u4e00-\u9fa5]{2,4}(?=说|认为|指出|表示|提出)",
    ],
    "organization": [
        r"(公司|集团|企业|机构|组织|部门|团队|研究院|大学|学院)",
        r"(华为|腾讯|阿里|百度|小米|字节|京东|美团|网易)",
    ],
    "location": [
        r"(省|市|区|县|镇|村|街|路|道|巷|号|楼|层|室)",
        r"(中国|美国|日本|韩国|英国|法国|德国|俄罗斯|印度|巴西)",
        r"(北京|上海|广州|深圳|杭州|成都|武汉|南京|西安|重庆)",
    ],
    "time": [
        r"\d{4}年\d{1,2}月\d{1,2}日",
        r"\d{4}[-/]\d{1,2}[-/]\d{1,2}",
        r"(今年|去年|明天|昨天|今天|前天|后天)",
        r"(上午|下午|晚上|凌晨|中午)",
        r"\d{1,2}[时点:：]\d{1,2}",
    ],
    "number": [
        r"\d+(\.\d+)?[%％]",
        r"\d+(\.\d+)?(万|亿|千|百|十)?(元|美元|人民币|欧元|英镑)",
        r"\d+(\.\d+)?(个|件|条|项|次|名|位|台|套|批|组)",
    ],
}


class QueryAnalyzer:
    """Analyzes user queries to optimize retrieval strategy.
    
    Provides query classification, keyword extraction, entity recognition,
    query expansion, and dynamic weight calculation.
    
    Attributes:
        stop_words: Set of stop words to filter out.
        base_weights: Base retrieval weights by query type.
        adjustment_factors: Weight adjustment factors by feature.
    """

    def __init__(self):
        """Initialize the query analyzer."""
        self.stop_words = self._load_stop_words()
        jieba.initialize()
        
        # Base weights by query type
        self.base_weights = {
            "factual": {"dense": 0.6, "sparse": 0.4},
            "conceptual": {"dense": 0.8, "sparse": 0.2},
            "comparative": {"dense": 0.5, "sparse": 0.3},
            "navigational": {"dense": 0.4, "sparse": 0.4},
        }
        
        # Dynamic adjustment factors
        self.adjustment_factors = {
            "has_number": {"dense": -0.08, "sparse": 0.08},
            "has_date": {"dense": -0.1, "sparse": 0.1},
            "has_entity": {"dense": 0.05, "sparse": -0.05},
            "long_query": {"dense": 0.08, "sparse": -0.08},
            "short_query": {"dense": -0.05, "sparse": 0.05},
            "question_word": {"dense": 0.05, "sparse": -0.05},
            "high_keyword_density": {"dense": -0.05, "sparse": 0.05},
        }

    def _load_stop_words(self) -> set[str]:
        """Load stop words for filtering.
        
        Returns:
            Set of stop words.
        """
        stop_words = {
            # Chinese stop words
            "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
            "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
            "你", "会", "着", "没有", "看", "好", "自己", "这", "他", "她",
            "它", "们", "那", "些", "么", "等", "吗", "呢", "吧", "啊",
            "呀", "嗯", "哦", "哈", "嘛", "哎", "唉", "喂", "啦", "噢",
            # English stop words
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "can", "shall",
            "i", "me", "my", "we", "our", "you", "your", "he", "him",
            "his", "she", "her", "it", "its", "they", "them", "their",
            "this", "that", "these", "those", "what", "which", "who",
            "whom", "when", "where", "why", "how", "all", "each", "every",
            "both", "few", "more", "most", "other", "some", "such", "no",
            "not", "only", "own", "same", "so", "than", "too", "very",
            "and", "but", "or", "nor", "for", "at", "by", "from", "in",
            "into", "of", "on", "to", "with", "as",
        }
        return stop_words

    def _classify_query_type(self, query: str) -> str:
        """Classify the query type based on patterns.
        
        Args:
            query: Query string to classify.
            
        Returns:
            Query type: factual, conceptual, comparative, or navigational.
        """
        query_lower = query.lower()
        
        type_scores = {qtype: 0 for qtype in QUERY_TYPE_PATTERNS}
        
        for qtype, patterns in QUERY_TYPE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    type_scores[qtype] += 1
        
        max_score = max(type_scores.values())
        if max_score == 0:
            return "factual"
        
        for qtype, score in type_scores.items():
            if score == max_score:
                return qtype
        
        return "factual"

    def _extract_keywords(self, query: str) -> list[str]:
        """Extract keywords from query using jieba.
        
        Args:
            query: Query string.
            
        Returns:
            List of keywords.
        """
        tokens = list(jieba.cut(query))
        keywords = [
            token.strip() 
            for token in tokens 
            if token.strip() 
            and len(token.strip()) > 1
            and token.strip() not in self.stop_words
        ]
        return keywords

    def _extract_entities(self, query: str) -> list[dict[str, str]]:
        """Extract named entities from query.
        
        Args:
            query: Query string.
            
        Returns:
            List of entity dictionaries with 'text' and 'type' keys.
        """
        entities = []
        seen_texts = set()
        
        # Pattern-based extraction
        for entity_type, patterns in ENTITY_PATTERNS.items():
            for pattern in patterns:
                matches = re.finditer(pattern, query)
                for match in matches:
                    text = match.group()
                    if text not in seen_texts:
                        seen_texts.add(text)
                        entities.append({
                            "text": text,
                            "type": entity_type,
                        })
        
        # POS-based extraction
        words = pseg.cut(query)
        for word, flag in words:
            word = word.strip()
            if not word or len(word) < 2 or word in seen_texts:
                continue
            
            entity_type = None
            if flag.startswith("nr"):
                entity_type = "person"
            elif flag.startswith("ns"):
                entity_type = "location"
            elif flag.startswith("nt"):
                entity_type = "organization"
            elif flag.startswith("t"):
                entity_type = "time"
            
            if entity_type:
                seen_texts.add(word)
                entities.append({
                    "text": word,
                    "type": entity_type,
                })
        
        return entities

    def _expand_query(self, query: str, query_type: str) -> list[str]:
        """Generate expanded query variations.
        
        Args:
            query: Original query string.
            query_type: Classified query type.
            
        Returns:
            List of expanded query variations.
        """
        expanded = [query]
        
        keywords = self._extract_keywords(query)
        if keywords:
            expanded.append(" ".join(keywords))
            if len(keywords) > 2:
                expanded.append(" ".join(keywords[:2]))
        
        if query_type == "conceptual":
            expanded.append(f"{query} 定义 含义")
            expanded.append(f"{query} 解释 说明")
        elif query_type == "comparative":
            expanded.append(f"{query} 对比 区别")
            expanded.append(f"{query} 优缺点")
        elif query_type == "factual" and keywords:
            expanded.append(" ".join(keywords[:3]))
        
        seen = set()
        unique_expanded = []
        for eq in expanded:
            if eq not in seen:
                seen.add(eq)
                unique_expanded.append(eq)
        
        return unique_expanded

    def _extract_query_features(
        self, 
        query: str, 
        keywords: list[str], 
        entities: list[dict]
    ) -> dict[str, Any]:
        """Extract query features for dynamic weight calculation.
        
        Args:
            query: Original query string.
            keywords: Extracted keywords.
            entities: Extracted entities.
            
        Returns:
            Dictionary of query features.
        """
        query_length = len(query)
        keyword_count = len(keywords)
        entity_count = len(entities)
        keyword_density = keyword_count / max(query_length, 1)
        
        features = {
            "length": query_length,
            "keyword_count": keyword_count,
            "entity_count": entity_count,
            "keyword_density": keyword_density,
            "has_number": bool(re.search(r'\d+', query)),
            "has_date": bool(re.search(r'\d{4}[-/年]\d{1,2}[-/月]?\d{0,2}', query)),
            "has_question_word": bool(re.search(r'(什么|如何|为什么|怎样|哪个|多少|如何|怎么)', query)),
            "has_comparison": bool(re.search(r'(比较|对比|区别|vs|或者|还是)', query)),
            "high_keyword_density": keyword_density > 0.5,
        }
        
        return features

    def analyze(self, query: str) -> QueryAnalysis:
        """Analyze a user query for retrieval optimization.
        
        Args:
            query: User query string to analyze.
            
        Returns:
            QueryAnalysis object with query classification and extracted information.
        """
        if not query or not query.strip():
            return QueryAnalysis(
                original_query=query,
                query_type="factual",
                keywords=[],
                entities=[],
                features={},
                expanded_queries=[query] if query else [],
            )
        
        # Classify query type
        query_type = self._classify_query_type(query)
        
        # Extract keywords
        keywords = self._extract_keywords(query)
        
        # Extract entities
        entities = self._extract_entities(query)
        
        # Extract features
        features = self._extract_query_features(query, keywords, entities)
        
        # Generate expanded queries
        expanded_queries = self._expand_query(query, query_type)
        
        analysis = QueryAnalysis(
            original_query=query,
            query_type=query_type,
            keywords=keywords,
            entities=entities,
            features=features,
            expanded_queries=expanded_queries,
        )
        
        logger.info(
            f"Query analysis: type={query_type}, "
            f"keywords={len(keywords)}, entities={len(entities)}, "
            f"features={list(features.keys())}"
        )
        
        return analysis

    def get_retrieval_weights(self, query_type: str) -> dict[str, float]:
        """Get base retrieval weights based on query type.
        
        Args:
            query_type: Classified query type.
            
        Returns:
            Dictionary with weights for dense and sparse retrieval.
        """
        return self.base_weights.get(query_type, self.base_weights["factual"]).copy()

    def get_dynamic_weights(
        self,
        query_analysis: QueryAnalysis,
        sparse_scores: list[float] | None = None,
        sparse_boost_threshold: float = 2.0,
    ) -> dict[str, float]:
        """Calculate dynamic retrieval weights based on query features.
        
        Args:
            query_analysis: Analyzed query information.
            sparse_scores: Optional list of BM25 scores for quality estimation.
            sparse_boost_threshold: Threshold for sparse score boost.
            
        Returns:
            Dictionary with normalized weights for dense and sparse retrieval.
        """
        # 1. Get base weights from query type
        base = self.base_weights.get(
            query_analysis.query_type, 
            self.base_weights["factual"]
        )
        dense_weight = base["dense"]
        sparse_weight = base["sparse"]
        
        # 2. Apply feature-based adjustments
        features = query_analysis.features or {}
        
        if features.get("has_number"):
            adj = self.adjustment_factors["has_number"]
            dense_weight += adj["dense"]
            sparse_weight += adj["sparse"]
        
        if features.get("has_date"):
            adj = self.adjustment_factors["has_date"]
            dense_weight += adj["dense"]
            sparse_weight += adj["sparse"]
        
        if features.get("entity_count", 0) > 0:
            adj = self.adjustment_factors["has_entity"]
            dense_weight += adj["dense"]
            sparse_weight += adj["sparse"]
        
        if features.get("length", 0) > 15:
            adj = self.adjustment_factors["long_query"]
            dense_weight += adj["dense"]
            sparse_weight += adj["sparse"]
        elif features.get("length", 0) < 6:
            adj = self.adjustment_factors["short_query"]
            dense_weight += adj["dense"]
            sparse_weight += adj["sparse"]
        
        if features.get("has_question_word"):
            adj = self.adjustment_factors["question_word"]
            dense_weight += adj["dense"]
            sparse_weight += adj["sparse"]
        
        if features.get("high_keyword_density"):
            adj = self.adjustment_factors["high_keyword_density"]
            dense_weight += adj["dense"]
            sparse_weight += adj["sparse"]
        
        # 3. Apply sparse quality-based adjustment
        if sparse_scores and len(sparse_scores) > 0:
            avg_score = sum(sparse_scores) / len(sparse_scores)
            if avg_score > sparse_boost_threshold:
                # High quality sparse results -> boost sparse weight
                boost = min(0.15, (avg_score - sparse_boost_threshold) * 0.05)
                sparse_weight += boost
                dense_weight -= boost
        
        # 4. Normalize weights to sum to 1.0
        total = dense_weight + sparse_weight
        if total > 0:
            dense_weight = max(0.1, min(0.9, dense_weight / total))
            sparse_weight = 1.0 - dense_weight
        else:
            dense_weight = 0.6
            sparse_weight = 0.4
        
        weights = {
            "dense": round(dense_weight, 2),
            "sparse": round(sparse_weight, 2),
        }
        
        logger.info(
            f"Dynamic weights: dense={weights['dense']}, sparse={weights['sparse']} "
            f"(query_type={query_analysis.query_type})"
        )
        
        return weights
