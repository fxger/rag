"""OpenAI LLM answer generation with streaming and source citation support.

Provides grounded generation for RAG systems, instructing the model to answer
based on provided context and cite sources using bracket notation.
"""
import logging
from dataclasses import dataclass, field
from typing import AsyncGenerator, Optional

from openai import AsyncOpenAI

from src.config.settings import get_settings

logger = logging.getLogger(__name__)

RAG_SYSTEM_PROMPT = """You are a knowledgeable assistant that answers questions based on the provided context.

## Instructions:
1. Answer the question using ONLY the information in the provided context
2. Cite sources using [Source N] notation when referencing specific information
3. If the context doesn't contain enough information, say so clearly
4. Be precise, factual, and concise
5. Do not fabricate information not present in the context
6. If multiple sources support a claim, cite all relevant sources
7. Format your response in clear, readable paragraphs

## Citation Format:
- Use [Source 1], [Source 2], etc. to cite sources
- Place citations immediately after the relevant claim
- Example: "The system supports real-time processing [Source 1] with sub-second latency [Source 2]."

## Response Structure:
- Lead with a direct answer to the question
- Provide supporting details from the context
- End with any caveats or limitations if applicable"""


@dataclass
class GenerationResult:
    """Structured result from answer generation.
    
    Attributes:
        answer: Generated answer text with source citations.
        sources: List of source references used in the answer.
        model: Model identifier used for generation.
        tokens_used: Total tokens consumed (prompt + completion).
        finish_reason: Reason the generation stopped.
    """
    answer: str
    sources: list[dict] = field(default_factory=list)
    model: str = ""
    tokens_used: int = 0
    finish_reason: str = ""


class AnswerGenerator:
    """Generates answers using OpenAI chat completions with RAG grounding.
    
    Supports both synchronous and streaming generation, with source citation
    formatting for grounded responses.
    """

    def __init__(self):
        settings = get_settings()
        self.client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL,
        )
        self.model = settings.LLM_MODEL
        self.max_tokens = settings.LLM_MAX_TOKENS
        self.temperature = settings.LLM_TEMPERATURE

    def _build_messages(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
    ) -> list[dict[str, str]]:
        """Build message array for chat completion.
        
        Args:
            query: User query.
            context: Retrieved context with source citations.
            system_prompt: Optional system prompt override.
            
        Returns:
            List of message dictionaries.
        """
        return [
            {
                "role": "system",
                "content": system_prompt or RAG_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"## Context\n\n{context}\n\n## Question\n\n{query}",
            },
        ]

    async def generate(self, query: str, context: str) -> str:
        """Generate a complete answer from query and context.
        
        Args:
            query: User question to answer.
            context: Retrieved context with source citations.
            
        Returns:
            Generated answer text.
            
        Raises:
            openai.APIError: On LLM API failures.
        """
        messages = self._build_messages(query, context)

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            answer = response.choices[0].message.content.strip()
            tokens_used = response.usage.total_tokens if response.usage else 0
            
            logger.info(
                "Generated answer: %d chars, %d tokens, model=%s",
                len(answer),
                tokens_used,
                self.model,
            )
            return answer

        except Exception as e:
            logger.error("Answer generation failed: %s", e)
            raise

    async def generate_stream(
        self,
        query: str,
        context: str,
    ) -> AsyncGenerator[str, None]:
        """Generate answer with streaming token output.
        
        Args:
            query: User question to answer.
            context: Retrieved context with source citations.
            
        Yields:
            Answer text tokens as they are generated.
            
        Raises:
            openai.APIError: On LLM API failures.
        """
        messages = self._build_messages(query, context)

        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error("Streaming generation failed: %s", e)
            raise

    async def generate_with_sources(
        self,
        query: str,
        context: str,
        sources: list[dict],
    ) -> GenerationResult:
        """Generate answer and return structured result with source metadata.
        
        Args:
            query: User question to answer.
            context: Retrieved context with source citations.
            sources: Source reference metadata from context builder.
            
        Returns:
            GenerationResult with answer and source information.
            
        Raises:
            openai.APIError: On LLM API failures.
        """
        messages = self._build_messages(query, context)

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            choice = response.choices[0]
            answer = choice.message.content.strip()
            tokens_used = response.usage.total_tokens if response.usage else 0

            # Extract only sources actually cited in the answer
            cited_sources = self._extract_cited_sources(answer, sources)

            result = GenerationResult(
                answer=answer,
                sources=cited_sources,
                model=self.model,
                tokens_used=tokens_used,
                finish_reason=choice.finish_reason or "",
            )

            logger.info(
                "Generated answer with sources: %d chars, %d tokens, %d/%d sources cited",
                len(answer),
                tokens_used,
                len(cited_sources),
                len(sources),
            )
            return result

        except Exception as e:
            logger.error("Answer generation with sources failed: %s", e)
            raise

    @staticmethod
    def _extract_cited_sources(answer: str, all_sources: list[dict]) -> list[dict]:
        """Extract sources actually referenced in the generated answer.
        
        Args:
            answer: Generated answer text containing [Source N] citations.
            all_sources: Full list of source references.
            
        Returns:
            List of source references that were cited in the answer.
        """
        import re

        # Find all [Source N] patterns in answer
        citation_pattern = r"\[Source\s+(\d+)\]"
        cited_ids = set(re.findall(citation_pattern, answer))

        # Filter to only cited sources
        cited = [
            src for src in all_sources
            if str(src.get("id", "")) in cited_ids
        ]

        if not cited and all_sources:
            # If no citations found but sources exist, include all (model may not have cited)
            logger.debug("No explicit citations found, including all sources")
            return all_sources

        return cited
