"""
LLM-powered recommendation explanations via OpenRouter.

Takes the same data already produced by the SHAP + recommendation pipeline
(per-user profile, per-book metadata, SHAP factors) and asks an LLM to
write a short, human-readable paragraph for each recommendation.

API key:    OPENROUTER_API_KEY env var
Model:      anything OpenRouter exposes; default to a chain of free-tier
            models tried in order
Reference:  https://openrouter.ai/docs
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, List, Optional

import aiohttp


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Free models tried in order if the primary is rate-limited. OpenRouter's
# free pool is shared globally so any single model can 429 at any moment.
DEFAULT_FREE_FALLBACK = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemini-2.0-flash-exp:free",
    "qwen/qwen-2.5-72b-instruct:free",
    "meta-llama/llama-3.2-3b-instruct:free",
]
DEFAULT_MODEL = ",".join(DEFAULT_FREE_FALLBACK)

HTTP_TIMEOUT_S = 60
MAX_RETRY_PER_MODEL = 2
RETRY_AFTER_CAP_S = 10.0


class LLMConfigError(RuntimeError):
    """Raised when the API key is missing or otherwise unusable."""


def get_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise LLMConfigError(
            "OPENROUTER_API_KEY не задан. Получите ключ на https://openrouter.ai "
            "и экспортируйте: export OPENROUTER_API_KEY=sk-or-v1-..."
        )
    return key


def _build_system_prompt() -> str:
    return (
        "Ты — рекомендательный ассистент для библиотеки книг. "
        "Для каждой книги напиши КОРОТКОЕ объяснение (2–3 предложения, ~40 слов) "
        "почему она может понравиться конкретному пользователю. "
        "Опирайся на: жанры книги, профиль пользователя, и SHAP-факторы "
        "(положительный SHAP = признак подталкивает score вверх). "
        "Отвечай только валидным JSON без markdown — массив объектов с полями "
        '`book_id` и `explanation`. Пиши на русском.'
    )


def _format_user_profile(profile: Dict[str, Any]) -> str:
    bits = []
    if 'avg_rating' in profile:
        bits.append(f"средний рейтинг: {profile['avg_rating']:.1f}/10")
    if 'rating_count' in profile:
        bits.append(f"оценил книг: {profile['rating_count']}")
    if 'non_zero_ratio' in profile:
        bits.append(f"% явных оценок: {profile['non_zero_ratio']*100:.0f}%")
    if 'top_genres' in profile and profile['top_genres']:
        bits.append(f"любимые жанры: {', '.join(profile['top_genres'][:5])}")
    return "; ".join(bits) if bits else "новый пользователь без истории"


def _format_recommendation(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Trim down a recommendation to what the LLM actually needs."""
    pos = rec.get('top_positive') or []
    neg = rec.get('top_negative') or []
    return {
        'book_id': str(rec.get('book_id')),
        'title': rec.get('book_title') or 'без названия',
        'genres': (rec.get('genres') or [])[:6],
        'model_score': round(float(rec.get('score', 0.0)), 2),
        'top_positive_factors': [
            {'feature': f, 'shap': round(float(v), 2)}
            for f, v in pos[:5]
        ],
        'top_negative_factors': [
            {'feature': f, 'shap': round(float(v), 2)}
            for f, v in neg[:5]
        ],
    }


async def _call_openrouter_once(
    session: aiohttp.ClientSession,
    api_key: str,
    model: str,
    user_msg: str,
) -> Dict[str, Any]:
    """One POST to /chat/completions. Returns parsed JSON or raises.

    Special: on 429 reads Retry-After and stores it as the attribute
    `.retry_after` on the raised RuntimeError so the caller can decide.
    """
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": _build_system_prompt()},
            {"role": "user",   "content": user_msg},
        ],
        "max_tokens": 800,
        "temperature": 0.6,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "Book RecSys",
        "Content-Type": "application/json",
    }
    async with session.post(OPENROUTER_URL, headers=headers, json=body) as r:
        if r.status == 200:
            return await r.json()
        text = await r.text()
        err = RuntimeError(f"OpenRouter HTTP {r.status}: {text[:400]}")
        err.status = r.status   # type: ignore[attr-defined]
        err.retry_after = 0.0   # type: ignore[attr-defined]
        if r.status == 429:
            # Try the HTTP header first, then the JSON body's metadata
            ra = r.headers.get("Retry-After")
            try:
                err.retry_after = float(ra) if ra else 0.0   # type: ignore[attr-defined]
            except ValueError:
                err.retry_after = 0.0   # type: ignore[attr-defined]
            if not err.retry_after:    # type: ignore[attr-defined]
                try:
                    j = json.loads(text)
                    meta = (j.get("error", {}) or {}).get("metadata", {}) or {}
                    err.retry_after = float(meta.get("retry_after_seconds", 0))  # type: ignore[attr-defined]
                except Exception:
                    pass
        raise err


async def explain_recommendations(
    user_profile: Dict[str, Any],
    recommendations: List[Dict[str, Any]],
    model: Optional[str] = None,
    timeout_s: float = HTTP_TIMEOUT_S,
) -> Dict[str, Any]:
    """
    Send all recommendations to OpenRouter in a single LLM call.

    `model` may be a single slug or a comma-separated list — models are
    tried in order, with a short retry per model on 429. Returns:

        {
            "model": "...",                  # the model that actually answered
            "explanations": {book_id: text, ...},
            "raw": "...",                    # raw LLM output for debugging
            "tokens": {...},                 # usage info from OpenRouter
            "attempts": [...]                # per-model status (for logs)
        }

    Raises LLMConfigError if OPENROUTER_API_KEY is missing.
    Raises RuntimeError if every model failed.
    """
    api_key = get_api_key()  # may raise LLMConfigError

    candidates = [m.strip() for m in (model or DEFAULT_MODEL).split(",") if m.strip()]
    if not candidates:
        candidates = list(DEFAULT_FREE_FALLBACK)

    payload = {
        "user_profile": _format_user_profile(user_profile),
        "books": [_format_recommendation(r) for r in recommendations],
    }
    user_msg = (
        "Пользователь:\n"
        f"  {payload['user_profile']}\n\n"
        "Книги (JSON):\n"
        f"{json.dumps(payload['books'], ensure_ascii=False, indent=2)}\n\n"
        "Верни JSON-массив объяснений."
    )

    timeout = aiohttp.ClientTimeout(total=timeout_s)
    attempts: List[Dict[str, Any]] = []
    last_error: Optional[Exception] = None

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for slug in candidates:
            for attempt in range(MAX_RETRY_PER_MODEL):
                try:
                    data = await _call_openrouter_once(session, api_key, slug, user_msg)
                    raw_text = data["choices"][0]["message"]["content"]
                    explanations = _parse_llm_json(raw_text)
                    attempts.append({"model": slug, "status": "ok", "attempt": attempt + 1})
                    return {
                        "model": data.get("model", slug),
                        "explanations": explanations,
                        "raw": raw_text,
                        "tokens": data.get("usage", {}),
                        "attempts": attempts,
                    }
                except (KeyError, IndexError) as e:
                    attempts.append({"model": slug, "status": "bad_response",
                                      "attempt": attempt + 1, "error": str(e)})
                    last_error = RuntimeError(f"OpenRouter ответ без content для {slug}")
                    break  # don't retry shape errors
                except RuntimeError as e:
                    status = getattr(e, "status", 0)
                    retry_after = float(getattr(e, "retry_after", 0.0) or 0.0)
                    attempts.append({
                        "model": slug, "status": f"http_{status}",
                        "attempt": attempt + 1, "retry_after": retry_after,
                    })
                    last_error = e
                    # Retry only on 429 / 5xx, and only if we have attempts left
                    is_transient = status == 429 or (500 <= status < 600)
                    if is_transient and attempt + 1 < MAX_RETRY_PER_MODEL:
                        wait_s = max(0.5, min(retry_after, RETRY_AFTER_CAP_S))
                        await asyncio.sleep(wait_s)
                        continue
                    break  # move on to next candidate model
                except Exception as e:
                    attempts.append({"model": slug, "status": "exception",
                                      "attempt": attempt + 1, "error": str(e)})
                    last_error = e
                    break

    # Every candidate failed
    raise RuntimeError(
        f"Все модели вернули ошибку. Попробуйте другую модель через "
        f"?model=<slug> (например anthropic/claude-haiku-4-5 за ~$0.001/вызов). "
        f"Последняя ошибка: {last_error}"
    )


def _parse_llm_json(text: str) -> Dict[str, str]:
    """Extract {book_id: explanation} from the LLM's response.

    The model SHOULD return clean JSON, but in practice some models wrap
    it in ```json fences or add prose. We strip both.
    """
    s = text.strip()
    # Strip ```json ... ``` fences if present
    if s.startswith("```"):
        s = s.split("```", 2)[1] if "```" in s[3:] else s
        if s.lower().startswith("json"):
            s = s[4:].lstrip()
    # Find first '[' or '{'
    for start_char, end_char in (('[', ']'), ('{', '}')):
        i = s.find(start_char)
        if i == -1:
            continue
        depth = 0
        end = None
        for j, c in enumerate(s[i:], start=i):
            if c == start_char:
                depth += 1
            elif c == end_char:
                depth -= 1
                if depth == 0:
                    end = j + 1
                    break
        if end is None:
            continue
        try:
            parsed = json.loads(s[i:end])
        except json.JSONDecodeError:
            continue
        return _normalize_explanations(parsed)
    # Last resort — treat entire text as a single explanation
    return {"_global": text.strip()}


def _normalize_explanations(parsed: Any) -> Dict[str, str]:
    """Accept either a list of {book_id, explanation} dicts or a flat dict."""
    out: Dict[str, str] = {}
    if isinstance(parsed, list):
        for entry in parsed:
            if not isinstance(entry, dict):
                continue
            bid = entry.get("book_id") or entry.get("id") or entry.get("isbn")
            text = entry.get("explanation") or entry.get("text") or entry.get("reason")
            if bid and text:
                out[str(bid)] = str(text).strip()
    elif isinstance(parsed, dict):
        for k, v in parsed.items():
            if isinstance(v, str):
                out[str(k)] = v.strip()
            elif isinstance(v, dict) and "explanation" in v:
                out[str(k)] = str(v["explanation"]).strip()
    return out
