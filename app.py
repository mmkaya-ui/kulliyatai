"""Külliyat AI — Streamlit chat over the hybrid index built by ``ingest.py``.

Answer modes:
  * "Kaynak cümleleri" (default): offline, no API key. Retrieves, reranks and
    returns the best source sentences verbatim with [n] citations.
  * Gemini / OpenAI / Claude (OpenRouter): LLM answer grounded in the passages,
    with citation checking. Needs the matching API key.
With the reranker on, questions whose best passage scores below the
abstention threshold get a "not found in the sources" answer.
"""

from __future__ import annotations

import os
import sys

import streamlit as st

if hasattr(sys.stdout, "reconfigure"):  # Turkish characters on Windows consoles
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from rag.answer import extractive_answer, format_extractive, is_confident, source_label  # noqa: E402
from rag.config import Settings, load_dotenv_if_available  # noqa: E402
from rag.embeddings import make_embedder  # noqa: E402
from rag.index import RAGIndex  # noqa: E402
from rag.prompts import NO_ANSWER, REFUSAL_MESSAGES, build_prompt, invalid_citations  # noqa: E402
from rag.retrieval import Reranker, Retriever  # noqa: E402

load_dotenv_if_available()
st.set_page_config(page_title="Külliyat AI", layout="wide")


def get_secret(name: str) -> str | None:
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:  # no secrets.toml
        pass
    return os.environ.get(name) or None


def check_password() -> bool:
    """True if no password is configured or the user entered the right one."""
    expected = get_secret("STREAMLIT_PASSWORD")
    if not expected:
        return True
    if st.session_state.get("password_ok"):
        return True
    pw = st.text_input("Lütfen erişim şifresini giriniz:", type="password")
    if pw and pw == expected:
        st.session_state["password_ok"] = True
        st.rerun()
    elif pw:
        st.error("Şifre yanlış")
    return False


if not check_password():
    st.stop()

settings = Settings.from_env()
for key in ("OPENAI_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY"):
    if (v := get_secret(key)) is not None:
        os.environ[key] = v

MODES = {
    "📄 Kaynak cümleleri (offline, LLM yok)": None,
    f"⚡ Gemini ({settings.llms['gemini_fast']})": "GOOGLE_API_KEY",
    f"🧠 Gemini ({settings.llms['gemini_pro']})": "GOOGLE_API_KEY",
    f"🤖 OpenAI ({settings.llms['openai_fast']})": "OPENAI_API_KEY",
    f"🧐 Claude via OpenRouter ({settings.llms['openrouter_claude']})": "OPENROUTER_API_KEY",
}

st.sidebar.title("🧠 Cevap modu")
mode = st.sidebar.radio("Mod", list(MODES), index=0, label_visibility="collapsed")
use_reranker = st.sidebar.checkbox(
    "Cross-encoder reranker (daha isabetli, CPU'da yavaş)",
    value=True,
    help="Kaynak cümleleri modu ve 'bulunamadı' eşiği reranker gerektirir.",
)
top_k = st.sidebar.slider("Kaynak sayısı", 2, 8, 4)
if st.sidebar.button("🗑️ Sohbeti temizle"):
    st.session_state.messages = []
    st.rerun()

st.title("📚 Külliyat Asistanı")


@st.cache_resource(show_spinner="Dizin yükleniyor…")
def load_index(db_dir: str) -> tuple[RAGIndex, object]:
    meta = RAGIndex.read_meta(db_dir)
    provider, model = meta["embedder"].split("/", 1)
    return RAGIndex.load(db_dir), make_embedder(provider, model)


@st.cache_resource(show_spinner="Reranker yükleniyor…")
def load_reranker(model: str) -> Reranker:
    return Reranker(model)


def make_llm(label: str):
    if "Gemini" in label:
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = settings.llms["gemini_fast"] if "⚡" in label else settings.llms["gemini_pro"]
        return ChatGoogleGenerativeAI(model=model, temperature=0, max_retries=2)
    if "OpenAI" in label:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=settings.llms["openai_fast"], temperature=0)
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.llms["openrouter_claude"],
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
    )


try:
    index, embedder = load_index(str(settings.db_dir))
except Exception as e:
    st.error(f"Dizin yüklenemedi: {e}")
    st.info("Önce `python ingest.py` (veya `make ingest`) çalıştırın.")
    st.stop()

needed_key = MODES[mode]
if needed_key and not os.environ.get(needed_key):
    st.sidebar.warning(f"{needed_key} tanımlı değil; bu mod kullanılamaz. Offline moda geçin.")
if MODES[mode] is None and not use_reranker:
    st.sidebar.info("Reranker kapalı: offline modda yalnızca bulunan pasajlar gösterilir.")

retriever = Retriever(
    index,
    embedder,
    dense_weight=settings.dense_weight,
    bm25_weight=settings.bm25_weight,
    candidate_k=settings.candidate_k,
    reranker=load_reranker(settings.rerank_model) if use_reranker else None,
)
refusal = REFUSAL_MESSAGES.get(settings.answer_language, REFUSAL_MESSAGES["tr"])


def render_sources(hits) -> str:
    lines = ["", "", "---", "**Kaynaklar:**"]
    for i, h in enumerate(hits, start=1):
        score = f" · skor {h.rerank_score:.2f}" if h.rerank_score is not None else ""
        lines.append(f"- [{i}] *{source_label(h.chunk)}*{score}")
    return "\n".join(lines)


def answer(question: str) -> tuple[str, list]:
    hits = retriever.search(question, k=top_k, mode="hybrid", rerank=use_reranker)
    if use_reranker and not is_confident(hits, settings.abstain_threshold):
        if not hits:
            return refusal, hits
        note = f"_(En iyi pasajın skoru {hits[0].rerank_score:.2f}, eşik {settings.abstain_threshold:.2f})_"
        return f"{refusal}\n\n{note}", hits
    if MODES[mode] is None:
        if not use_reranker:
            body = "\n\n".join(f"**[{i}]** {h.chunk.page_content.strip()[:500]}…" for i, h in enumerate(hits, 1))
            return "**Bulunan pasajlar:**\n\n" + body + render_sources(hits), hits
        sents = extractive_answer(question, hits, retriever.reranker)
        return format_extractive(sents) + render_sources(hits), hits
    if not os.environ.get(MODES[mode]):
        return f"{MODES[mode]} tanımlı değil.", hits
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate

    chain = PromptTemplate.from_template("{p}") | make_llm(mode) | StrOutputParser()
    raw = chain.invoke({"p": build_prompt(question, [h.chunk for h in hits], settings.answer_language)})
    if NO_ANSWER in raw:
        return refusal, hits
    bad = invalid_citations(raw, len(hits))
    warn = f"\n\n⚠️ Geçersiz kaynak numarası: {bad}" if bad else ""
    return raw.strip() + warn + render_sources(hits), hits


if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Merhaba! Belgeleri taradım. Bana istediğini sorabilirsin."}
    ]
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Bir soru sor..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Aranıyor…"):
                text, hits = answer(prompt)
            st.markdown(text)
            with st.expander("Pasajları göster"):
                for i, h in enumerate(hits, start=1):
                    st.markdown(f"**[{i}] {source_label(h.chunk)}**")
                    st.text(h.chunk.page_content)
            st.session_state.messages.append({"role": "assistant", "content": text})
        except Exception as e:
            st.error(f"Hata oluştu: {e}")
