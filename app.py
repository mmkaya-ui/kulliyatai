import os
import sys

import streamlit as st

# Fix Windows console encoding for Turkish characters
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from rag.config import DEFAULT_EMBEDDING_MODELS, Settings, load_dotenv_if_available
from rag.embeddings import make_embeddings
from rag.index import load_faiss, read_meta
from rag.prompts import NO_ANSWER, REFUSAL_MESSAGES, build_prompt
from rag.retrieval import HybridRetriever, Reranker

load_dotenv_if_available()


# --- PASSWORD PROTECTION ---
def get_secret(key_name):
    try:
        if key_name in st.secrets:
            return st.secrets[key_name]
    except Exception:
        pass
    if key_name in os.environ:
        return os.environ[key_name]
    return None


def check_password():
    """Returns `True` if the user had the correct password or password not configured."""
    correct_password = get_secret("STREAMLIT_PASSWORD") or get_secret("password")
    if not correct_password:
        return True

    def password_entered():
        if st.session_state.get("password") == correct_password:
            st.session_state["password_correct"] = True
            if "password" in st.session_state:
                del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input(
            "Lütfen erişim şifresini giriniz:",
            type="password",
            on_change=password_entered,
            key="password",
        )
        return False
    elif not st.session_state["password_correct"]:
        st.text_input(
            "Lütfen erişim şifresini giriniz:",
            type="password",
            on_change=password_entered,
            key="password",
        )
        st.error("😕 Şifre yanlış")
        return False
    else:
        return True


st.set_page_config(page_title="Külliyat AI", layout="wide")

if not check_password():
    st.stop()

# --- MAIN APP START ---
settings = Settings.from_env()
DB_PATH = settings.db_dir

# 1. SETUP KEYS
os.environ["OPENAI_API_KEY"] = get_secret("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
os.environ["GOOGLE_API_KEY"] = get_secret("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
os.environ["OPENROUTER_API_KEY"] = get_secret("OPENROUTER_API_KEY") or os.environ.get(
    "OPENROUTER_API_KEY", ""
)

# --- SIDEBAR: MODEL SELECTOR ---
st.sidebar.title("🧠 Yapay Zeka Modeli")
model_choice = st.sidebar.radio(
    "Zeka Seviyesini Seçin:",
    (
        "⚡ Hızlı (Gemini Flash)",
        "🧠 Zeki (Gemini Pro)",
        "🧐 Derin (Claude 4.5 Sonnet)",
        "🔍 Yalnızca Getirme (Offline / No LLM Key)",
    ),
    index=0,
)

use_reranker = st.sidebar.checkbox("🚀 Cross-Encoder Reranker Kullan", value=False)

st.sidebar.divider()

if st.sidebar.button("🗑️ Sohbeti Temizle"):
    st.session_state.messages = []
    st.rerun()

st.title("📚 Külliyat Asistanı")


# 2. LOAD DATABASE (FAISS) — MUST MATCH ingest.py embedding model
@st.cache_resource
def load_db():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Veritabanı klasörü bulunamadı: {DB_PATH}")

    meta = read_meta(DB_PATH)
    provider = meta.get("embedding_provider", settings.embedding_provider)
    model = meta.get(
        "embedding_model", settings.embedding_model or DEFAULT_EMBEDDING_MODELS.get(provider, "")
    )
    embeddings = make_embeddings(provider, model)
    return load_faiss(DB_PATH, embeddings)


try:
    db = load_db()
except Exception as e:
    st.error(f"Veritabanı yüklenirken hata oluştu: {e}")
    st.info("Henüz 'python ingest.py' (veya 'make ingest') komutunu çalıştırdınız mı?")
    st.stop()

# 3. DYNAMIC AI SETUP
k_val = 4
llm = None

if "Hızlı" in model_choice:
    k_val = 4
    if os.environ.get("GOOGLE_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI

        # gemini-3.5-flash or gemini-3.8-flash for modern Google GenAI API
        llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0, max_retries=2)
    elif os.environ.get("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
elif "Zeki" in model_choice:
    k_val = 6
    if os.environ.get("GOOGLE_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(model="gemini-3.1-pro-preview", temperature=0, max_retries=2)
    elif os.environ.get("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model="gpt-4o", temperature=0)
elif "Claude" in model_choice:
    k_val = 6
    if os.environ.get("OPENROUTER_API_KEY"):
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model="anthropic/claude-sonnet-4",
            openai_api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
            default_headers={"HTTP-Referer": "http://localhost:8501"},
        )


# --- UPGRADE 4: Hybrid Search (Vector + BM25) ---
@st.cache_resource
def get_hybrid_retriever(_db, enable_rerank: bool):
    reranker = Reranker(settings.rerank_model) if enable_rerank else None
    return HybridRetriever(
        vectorstore=_db,
        dense_weight=settings.dense_weight,
        bm25_weight=settings.bm25_weight,
        candidate_k=settings.candidate_k,
        reranker=reranker,
    )


retriever = get_hybrid_retriever(db, use_reranker)

# 4. CHAT INTERFACE
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Merhaba! Belgeleri taradım. Bana istediğini sorabilirsin."}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Bir soru sor..."):
    if not prompt.strip():
        st.warning("Lütfen boş bir soru sormayın.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner(f"{model_choice} ile aranıyor ve düşünülüyor..."):
            try:
                # Hybrid retrieval (vector + BM25, optional rerank)
                source_docs = retriever.retrieve(prompt, k=k_val, mode="hybrid", rerank=use_reranker)

                if llm is None:
                    # Offline / retrieval-only view when no LLM key is configured
                    response_text = "**Bulunan İlgili Kaynak Pasajları (Sorgulama Modu):**\n\n"
                    for i, doc in enumerate(source_docs, start=1):
                        src = doc.metadata.get("source", "Bilinmeyen")
                        pg = doc.metadata.get("page", "?")
                        response_text += (
                            f"**[{i}] {src} (Sayfa {pg})**\n> {doc.page_content.strip()[:400]}...\n\n"
                        )
                    full_response = response_text
                else:
                    from langchain_core.output_parsers import StrOutputParser
                    from langchain_core.prompts import PromptTemplate

                    full_prompt_text = build_prompt(prompt, source_docs, language=settings.answer_language)
                    chain_prompt = PromptTemplate.from_template("{raw_prompt}")
                    chain = chain_prompt | llm | StrOutputParser()
                    raw_answer = chain.invoke({"raw_prompt": full_prompt_text})

                    if NO_ANSWER in raw_answer:
                        full_response = REFUSAL_MESSAGES.get(settings.answer_language, REFUSAL_MESSAGES["tr"])
                    else:
                        full_response = f"{raw_answer.strip()}\n\n---\n**Kaynaklar:**"
                        if source_docs:
                            seen = set()
                            for doc in source_docs:
                                source_name = doc.metadata.get("source", "Bilinmeyen Dosya")
                                page_num = doc.metadata.get("page", "?")
                                key = f"{source_name}-{page_num}"
                                if key not in seen:
                                    seen.add(key)
                                    full_response += (
                                        f"\n- *{os.path.basename(source_name)}* (Sayfa {page_num})"
                                    )
                        else:
                            full_response += "\n- *Kaynak bulunamadı.*"

                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"Hata oluştu: {e}")
