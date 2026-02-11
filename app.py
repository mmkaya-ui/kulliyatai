import streamlit as st
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# --- PASSWORD PROTECTION ---
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        correct_password = ""
        if "password" in st.secrets:
            correct_password = st.secrets["password"]
        elif "STREAMLIT_PASSWORD" in os.environ:
            correct_password = os.environ["STREAMLIT_PASSWORD"]
            
        if st.session_state["password"] == correct_password:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input(
            "Lütfen erişim şifresini giriniz:", 
            type="password", 
            on_change=password_entered, 
            key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        st.text_input(
            "Lütfen erişim şifresini giriniz:", 
            type="password", 
            on_change=password_entered, 
            key="password"
        )
        st.error("😕 Şifre yanlış")
        return False
    else:
        return True

if not check_password():
    st.stop()

# --- MAIN APP START ---
DB_PATH = "vector_db"

st.set_page_config(page_title="Külliyat AI", layout="wide")

# 1. SETUP KEYS
def get_secret(key_name):
    try:
        if key_name in st.secrets:
            return st.secrets[key_name]
    except Exception:
        pass
    if key_name in os.environ:
        return os.environ[key_name]
    return None

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

os.environ["OPENAI_API_KEY"] = get_secret("OPENAI_API_KEY") or ""
os.environ["GOOGLE_API_KEY"] = get_secret("GOOGLE_API_KEY") or ""
os.environ["OPENROUTER_API_KEY"] = get_secret("OPENROUTER_API_KEY") or ""

if not os.environ["OPENAI_API_KEY"]:
    st.warning("⚠️ OPENAI_API_KEY bulunamadı. Bazı özellikler çalışmayabilir.")


# --- SIDEBAR: MODEL SELECTOR ---
st.sidebar.title("🧠 Yapay Zeka Modeli")
model_choice = st.sidebar.radio(
    "Zeka Seviyesini Seçin:",
    ("⚡ Hızlı (Gemini Flash)", "🧠 Zeki (Gemini Pro)", "🧐 Derin (Claude 4.5 Sonnet)"),
    index=0
)
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
    
    # --- UPGRADE: text-embedding-3-large (must match ingest.py) ---
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    return FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)

try:
    db = load_db()
except Exception as e:
    st.error(f"Veritabanı yüklenirken hata oluştu: {e}")
    st.info("Henüz 'python ingest.py' komutunu çalıştırdınız mı?")
    st.stop()

# 3. DYNAMIC AI SETUP
if "Hızlı" in model_choice:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        max_tokens=None,
        max_retries=2,
    )
    k_val = 4
elif "Zeki" in model_choice:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        temperature=0
    )
    k_val = 6
else: # Claude 4.5 Sonnet
    llm = ChatOpenAI(
        model="anthropic/claude-sonnet-4",
        openai_api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
        default_headers={"HTTP-Referer": "http://localhost:8501"}
    )
    k_val = 6

# --- UPGRADE 4: Hybrid Search (Vector + BM25) ---
# Combines semantic similarity (FAISS) with keyword matching (BM25)
faiss_retriever = db.as_retriever(search_kwargs={"k": k_val})

# Build BM25 retriever from all documents in the FAISS store
@st.cache_resource
def build_hybrid_retriever(_db, _faiss_retriever, k):
    """Create an ensemble retriever combining FAISS vector search + BM25 keyword search."""
    try:
        # Extract all documents from FAISS for BM25 indexing
        all_docs_dict = _db.docstore._dict
        all_docs = list(all_docs_dict.values())
        
        if all_docs:
            bm25_retriever = BM25Retriever.from_documents(all_docs, k=k)
            # 60% vector (meaning) + 40% keyword (exact match)
            ensemble = EnsembleRetriever(
                retrievers=[_faiss_retriever, bm25_retriever],
                weights=[0.6, 0.4]
            )
            return ensemble
    except Exception:
        pass
    
    # Fallback to FAISS-only if BM25 fails
    return _faiss_retriever

retriever = build_hybrid_retriever(db, faiss_retriever, k_val)

# Modern RAG prompt template
rag_prompt = ChatPromptTemplate.from_template(
    """Sen bir külliyat asistanısın. Aşağıdaki bağlam bilgilerini kullanarak soruyu detaylı ve doğru şekilde yanıtla.
Cevabını Türkçe ver. Eğer bağlamda yeterli bilgi yoksa, bunu belirt.

Bağlam:
{context}

Soru: {question}

Detaylı Cevap:"""
)

def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

# 4. CHAT INTERFACE
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Merhaba! Külliyatı taradım. Bana istediğini sorabilirsin."}]

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
        with st.spinner(f"{model_choice} ile düşünülüyor..."):
            try:
                # Hybrid retrieval (vector + BM25)
                source_docs = retriever.invoke(prompt)
                context = format_docs(source_docs)
                
                # Build and invoke the chain
                chain = rag_prompt | llm | StrOutputParser()
                answer = chain.invoke({"context": context, "question": prompt})
                
                full_response = f"{answer}\n\n---\n**Kaynaklar:**"
                if source_docs:
                    seen = set()
                    for doc in source_docs:
                        source_name = doc.metadata.get('source', 'Bilinmeyen Dosya')
                        page_num = doc.metadata.get('page', 'Bilinmeyen Sayfa')
                        key = f"{source_name}-{page_num}"
                        if key not in seen:
                            seen.add(key)
                            full_response += f"\n- *{os.path.basename(source_name)}* (Sayfa {page_num})"
                else:
                    full_response += "\n- *Kaynak bulunamadı.*"

                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"Hata oluştu: {e}")