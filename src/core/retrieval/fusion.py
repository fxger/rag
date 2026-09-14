"""Result fusion strategies for combining retrieval results."""
import logging
from typing import Any

from .models import SearchResult

logger = logging.getLogger(__name__)


class ResultFusion:
    """Fusion strategies for combining multiple retrieval results.
    
    Provides methods to merge and rank results from different retrieval
    sources (dense, sparse, metadata-based).
    """

    def __init__(self):
        """Initialize the result fusion module."""
        pass

    def normalize_scores(self, results: list[SearchResult]) -> list[SearchResult]:
        """Normalize scores using min-max normalization.
        
        Args:
            results: List of SearchResult objects with raw scores.
            
        Returns:
            New list of SearchResult objects with normalized scores in [0, 1].
        """
        if not results:
            return []
        
        scores = [r.score for r in results]
        min_score = min(scores)
        max_score = max(scores)
        
        # Avoid division by zero
        score_range = max_score - min_score
        if score_range == 0:
            # All scores are the same, normalize to 1.0
            return [
                SearchResult(
                    chunk_id=r.chunk_id,
                    document_id=r.document_id,
                    content=r.content,
                    score=1.0,
                    rank=r.rank,
                    metadata=r.metadata.copy(),
                    source_type=r.source_type,
                )
                for r in results
            ]
        
        normalized_results = []
        for r in results:
            normalized_score = (r.score - min_score) / score_range
            normalized_results.append(
                SearchResult(
                    chunk_id=r.chunk_id,
                    document_id=r.document_id,
                    content=r.content,
                    score=normalized_score,
                    rank=r.rank,
                    metadata=r.metadata.copy(),
                    source_type=r.source_type,
                )
            )
        
        return normalized_results

    def weighted_fusion(
        self,
        results_list: list[list[SearchResult]],
        weights: list[float],
    ) -> list[SearchResult]:
        """Merge results using weighted score fusion.
        
        Combines results from multiple sources using weighted scores.
        Results from the same chunk are merged with combined scores.
        
        Args:
            results_list: List of result lists from different sources.
            weights: Weight for each result list (should sum to 1.0).
            
        Returns:
            Merged and sorted list of SearchResult objects.
            
        Raises:
            ValueError: If results_list and weights have different lengths.
        """
        if len(results_list) != len(weights):
            raise ValueError(
                f"Length mismatch: {len(results_list)} result lists vs {len(weights)} weights"
            )
        
        if not results_list:
            return []
        
        # Normalize each result set
        normalized_results_list = [
            self.normalize_scores(results) for results in results_list
        ]
        
        # Merge results by chunk_id
        merged: dict[str, dict[str, Any]] = {}
        
        for results, weight in zip(normalized_results_list, weights):
            for result in results:
                chunk_id = result.chunk_id
                if chunk_id not in merged:
                    merged[chunk_id] = {
                        "result": result,
                        "weighted_score": 0.0,
                        "sources": set(),
                    }
                merged[chunk_id]["weighted_score"] += result.score * weight
                merged[chunk_id]["sources"].add(result.source_type)
        
        # Create merged results
        merged_results = []
        for chunk_id, data in merged.items():
            original = data["result"]
            # Track all source types
            sources = ", ".join(sorted(data["sources"]))
            metadata = original.metadata.copy()
            metadata["fusion_sources"] = sources
            
            merged_results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    document_id=original.document_id,
                    content=original.content,
                    score=data["weighted_score"],
                    rank=0,  # Will be set after sorting
                    metadata=metadata,
                    source_type=sources,
                )
            )
        
        # Sort by score descending and assign ranks
        merged_results.sort(key=lambda x: x.score, reverse=True)
        for rank, result in enumerate(merged_results, start=1):
            result.rank = rank
        
        logger.info(
            f"Weighted fusion: {sum(len(r) for r in results_list)} results -> "
            f"{len(merged_results)} unique chunks"
        )
        
        return merged_results

    def rrf_fusion(
        self,
        results_list: list[list[SearchResult]],
        k: int = 60,
    ) -> list[SearchResult]:
        """Merge results using Reciprocal Rank Fusion (RRF).
        
        RRF combines results based on their ranks rather than scores,
        making it robust to different score scales across sources.
        
        Args:
            results_list: List of result lists from different sources.
            k: Constant for RRF formula (default: 60).
            
        Returns:
            Merged and sorted list of SearchResult objects.
        """
        if not results_list:
            return []
        
        # Merge results by chunk_id using RRF scores
        merged: dict[str, dict[str, Any]] = {}
        
        for results in results_list:
            for result in results:
                chunk_id = result.chunk_id
                if chunk_id not in merged:
                    merged[chunk_id] = {
                        "result": result,
                        "rrf_score": 0.0,
                        "sources": set(),
                    }
                
                # RRF formula: 1 / (k + rank)
                rrf_contribution = 1.0 / (k + result.rank)
                merged[chunk_id]["rrf_score"] += rrf_contribution
                merged[chunk_id]["sources"].add(result.source_type)
        
        # Create merged results
        merged_results = []
        for chunk_id, data in merged.items():
            original = data["result"]
            sources = ", ".join(sorted(data["sources"]))
            metadata = original.metadata.copy()
            metadata["fusion_sources"] = sources
            metadata["rrf_score"] = data["rrf_score"]
            
            merged_results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    document_id=original.document_id,
                    content=original.content,
                    score=data["rrf_score"],
                    rank=0,  # Will be set after sorting
                    metadata=metadata,
                    source_type=sources,
                )
            )
        
        # Sort by RRF score descending and assign ranks
        merged_results.sort(key=lambda x: x.score, reverse=True)
        for rank, result in enumerate(merged_results, start=1):
            result.rank = rank
        
        logger.info(
            f"RRF fusion (k={k}): {sum(len(r) for r in results_list)} results -> "
            f"{len(merged_results)} unique chunks"
        )
        
        return merged_results
