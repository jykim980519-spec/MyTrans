import json
import logging
import os

import openai
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

MODEL = "gpt-6-astra"
TEMPERATURE = 0.3
MAX_CHARS = 5000
LANGUAGES = {
    "english": "English",
    "japanese": "日本語",
    "vietnamese": "Tiếng Việt",
}

SYSTEM_PROMPT = """You are a professional translator.
Translate the user's text into English, Japanese, and Vietnamese.
Detect the source language automatically. If the source text is already in one of
the target languages, output a natural, polished version of it in that language.
Preserve the original meaning, tone, and formatting exactly: keep every line break,
blank line, list marker, and numbering in the same place.
Return ONLY a JSON object with the keys "english", "japanese", "vietnamese"."""

MSG_NO_KEY = "API 키가 설정되지 않았습니다. `.env` 또는 Secrets에 `OPEN_API_KEY`를 설정하세요."
MSG_EMPTY = "번역할 내용을 입력해주세요."
MSG_TOO_LONG = "최대 5,000자까지 입력할 수 있습니다."
MSG_AUTH = "API 키가 올바르지 않습니다."
MSG_RATE_LIMIT = "요청이 많습니다. 잠시 후 다시 시도해주세요."
MSG_API = "번역 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
MSG_PARSE = "번역 결과를 처리하지 못했습니다. 다시 시도해주세요."

logger = logging.getLogger(__name__)


class TranslationParseError(Exception):
    """API 응답이 기대한 JSON 형식이 아닐 때 발생."""


def get_api_key():
    try:
        key = st.secrets["OPEN_API_KEY"]
        if key:
            return key
    except Exception:
        pass
    load_dotenv()
    return os.getenv("OPEN_API_KEY")


def translate(text: str) -> dict:
    client = OpenAI(api_key=get_api_key())
    response = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
    )
    content = response.choices[0].message.content
    try:
        data = json.loads(content or "")
    except json.JSONDecodeError as e:
        raise TranslationParseError(f"Invalid JSON: {content!r}") from e
    if not isinstance(data, dict) or not all(
        isinstance(data.get(key), str) for key in LANGUAGES
    ):
        raise TranslationParseError(f"Missing language keys: {content!r}")
    return {key: data[key] for key in LANGUAGES}


def run_translation(text: str):
    """번역을 실행하고 (결과, 에러 메시지) 중 하나를 반환한다."""
    try:
        return translate(text), None
    except openai.AuthenticationError:
        logger.exception("Authentication failed")
        return None, MSG_AUTH
    except openai.RateLimitError:
        logger.exception("Rate limit exceeded")
        return None, MSG_RATE_LIMIT
    except (openai.APIConnectionError, openai.APIError):
        logger.exception("OpenAI API error")
        return None, MSG_API
    except TranslationParseError:
        logger.exception("Failed to parse translation")
        return None, MSG_PARSE


LANGUAGE_FLAGS = {
    "english": "🇺🇸",
    "japanese": "🇯🇵",
    "vietnamese": "🇻🇳",
}

CUSTOM_CSS = """
<style>
@import url("https://fonts.googleapis.com/css2?family=Noto+Sans:wght@400;600;700&family=Noto+Sans+KR:wght@400;700&family=Noto+Sans+JP:wght@400;700&display=swap");
.block-container {
    max-width: 1200px;
    padding-top: 2.5rem;
}
.app-subtitle {
    margin: -0.75rem 0 1.5rem;
    opacity: 0.7;
    font-size: 1.05rem;
}
.char-count {
    text-align: right;
    font-size: 0.85rem;
    opacity: 0.6;
    margin: -0.5rem 0 0.75rem;
}
.char-count.near-limit {
    color: #e5484d;
    opacity: 1;
}
.lang-label {
    font-weight: 700;
    font-size: 1.1rem;
    margin-bottom: 0.25rem;
}
.result-placeholder {
    opacity: 0.5;
    font-size: 0.95rem;
    padding: 1.5rem 0;
    text-align: center;
}
[data-testid="stCode"] pre,
[data-testid="stCode"] code {
    white-space: pre-wrap !important;
    word-break: break-word;
    overflow-wrap: anywhere;
}
</style>
"""

st.set_page_config(page_title="다국어 번역기", page_icon="🌐", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

if "result" not in st.session_state:
    st.session_state.result = None
if "error" not in st.session_state:
    st.session_state.error = None

st.title("🌐 다국어 번역기")
st.markdown(
    '<p class="app-subtitle">입력한 글을 영어 · 일본어 · 베트남어로 번역합니다</p>',
    unsafe_allow_html=True,
)

if not get_api_key():
    st.error(MSG_NO_KEY)
    st.stop()

text = st.text_area(
    "번역할 내용",
    placeholder="번역할 내용을 입력하세요...",
    max_chars=MAX_CHARS,
    key="source_text",
    height=220,
)
near_limit = " near-limit" if len(text) >= MAX_CHARS * 0.9 else ""
st.markdown(
    f'<div class="char-count{near_limit}">{len(text):,} / {MAX_CHARS:,}자</div>',
    unsafe_allow_html=True,
)

if st.button("번역하기", type="primary", width="stretch"):
    if not text.strip():
        st.warning(MSG_EMPTY)
    elif len(text) > MAX_CHARS:
        st.warning(MSG_TOO_LONG)
    else:
        with st.spinner("번역 중입니다..."):
            result, error = run_translation(text)
        st.session_state.error = error
        if result is not None:
            st.session_state.result = result

if st.session_state.error:
    st.error(st.session_state.error)

st.write("")
columns = st.columns(len(LANGUAGES), gap="medium")
for column, (key, label) in zip(columns, LANGUAGES.items()):
    with column, st.container(border=True):
        st.markdown(
            f'<div class="lang-label">{LANGUAGE_FLAGS[key]} {label}</div>',
            unsafe_allow_html=True,
        )
        if st.session_state.result:
            st.code(st.session_state.result[key], language=None, wrap_lines=True)
        else:
            st.markdown(
                '<div class="result-placeholder">번역 결과가 여기에 표시됩니다</div>',
                unsafe_allow_html=True,
            )
