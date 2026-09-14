"""CJK-aware token utilities — the single source of truth for token counting.

中文没有词间空格，这会给按空白设计的实现带来两个隐蔽的错误：

1. **按空白切分**会把整篇中文文档变成 1 个 token，使滑动窗口分块彻底失效；
2. **按 ``len(text) // 4`` 估算**会把中文低估约 4 倍（中文大约 1 字 = 1 token），
   使上下文预算形同虚设。

本模块统一这两件事：分块边界、上下文预算与截断，以及对外上报的 token 数，
全部走这里，避免多处口径互相打架（历史上 chunker / context_builder 各有一套，
导致"每块最多 512 token"的承诺并不成立）。
"""
import math
import re

#: CJK 统一表意文字、扩展 A 区、兼容表意文字
CJK_RANGES = "\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"

#: 一个 CJK 字 = 一个 token；一段连续的非 CJK、非空白字符 = 一个 token
TOKEN_PATTERN = re.compile(rf"[{CJK_RANGES}]|[^\s{CJK_RANGES}]+")

#: 句末标点（中英文）
SENTENCE_END_CHARS = "。！？；.!?"


def is_cjk(char: str) -> bool:
    """Return True if *char* is a CJK ideograph."""
    return (
        "\u4e00" <= char <= "\u9fff"
        or "\u3400" <= char <= "\u4dbf"
        or "\uf900" <= char <= "\ufaff"
    )


def token_spans(text: str) -> list[tuple[int, int]]:
    """Return ``(start, end)`` character spans for every token in *text*.

    Spans index into the original string, so callers can slice text back
    verbatim.  Re-joining token strings with a space would inject spaces
    between CJK characters and corrupt the source.
    """
    return [(m.start(), m.end()) for m in TOKEN_PATTERN.finditer(text)]


def split_tokens(text: str) -> list[str]:
    """Split *text* into CJK-aware tokens.

    Each CJK ideograph becomes its own token; a run of non-CJK,
    non-whitespace characters (Latin word, number, punctuation) forms one
    token.
    """
    return [text[start:end] for start, end in token_spans(text)]


def count_chars(text: str) -> tuple[int, int]:
    """Return ``(cjk_chars, other_non_space_chars)`` for *text*."""
    cjk = other = 0
    for ch in text:
        if ch.isspace():
            continue
        if is_cjk(ch):
            cjk += 1
        else:
            other += 1
    return cjk, other


def tokens_for_counts(cjk: int, other: int) -> int:
    """Estimate tokens from pre-counted character totals.

    Used by the sliding-window chunker, which accumulates counts while
    scanning so it never re-counts the same text.
    """
    return cjk + math.ceil(other / 4)


def estimate_tokens(text: str) -> int:
    """Estimate the token count of *text*.

    Rules:
    - whitespace is free;
    - each CJK ideograph counts as 1 token;
    - everything else (Latin letters, digits, punctuation) is approximated
      at ~4 characters per token, which is the usual BPE average.

    This is a deliberately rough approximation — but it is *the* one used
    everywhere, which matters more than its absolute accuracy.
    """
    if not text:
        return 0
    cjk, other = count_chars(text)
    return tokens_for_counts(cjk, other)
