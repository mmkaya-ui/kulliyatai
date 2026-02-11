import streamlit as st
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- PASSWORD PROTECTION ---
# This block stops the app from loading unless the user enters the correct password.
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        # Check secrets first, then environment variable
        correct_password = ""
        if "password" in st.secrets:
            correct_password = st.secrets["password"]
        elif "STREAMLIT_PASSWORD" in os.environ:
            correct_password = os.environ["STREAMLIT_PASSWORD"]
            
        if st.session_state["password"] == correct_password:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Lütfen erişim şifresini giriniz:", 
            type="password", 
            on_change=password_entered, 
            key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password incorrect, show input again.
        st.text_input(
            "Lütfen erişim şifresini giriniz:", 
            type="password", 
            on_change=password_entered, 
            key="password"
        )
        st.error("😕 Şifre yanlış")
        return False
    else:
        # Password correct.
        return True

if not check_password():
    st.stop()  # Do not run any code below this line if password is wrong

# --- MAIN APP START ---
DB_PATH = "vector_db"

st.set_page_config(page_title="Külliyat AI", layout="wide")

# 1. SETUP KEYS
# Try to load from Streamlit secrets, then fallback to os.environ for Vercel/local .env
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
    # Load env vars if locally using .env
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

# Add Clear History Button
if st.sidebar.button("🗑️ Sohbeti Temizle"):
    st.session_state.messages = []
    st.rerun()

st.title("📚 Külliyat Asistanı")

# 2. LOAD DATABASE (FAISS)
@st.cache_resource
def load_db():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Veritabanı klasörü bulunamadı: {DB_PATH}")
        
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
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
        model="gemini-2.5-flash-preview-04-17",
        temperature=0,
        max_tokens=None,
        max_retries=2,
    )
    k_val = 3
elif "Zeki" in model_choice:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro-preview-05-06",
        temperature=0
    )
    k_val = 5
else: # Claude 3.5
    llm = ChatOpenAI(
        model="anthropic/claude-sonnet-4",
        openai_api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
        default_headers={"HTTP-Referer": "http://localhost:8501"}
    )
    k_val = 5

retriever = db.as_retriever(search_kwargs={"k": k_val})

# Modern RAG prompt template
rag_prompt = ChatPromptTemplate.from_template(
    """Aşağıdaki bağlam bilgilerini kullanarak soruyu yanıtla.
Eğer bağlamda cevap bulamıyorsan, bilmediğini söyle. Cevabını Türkçe ver.

Bağlam:
{context}

Soru: {question}

Cevap:"""
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 4. CHAT INTERFACE
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Merhaba! Külliyatı taradım. Bana istediğini sorabilirsin."}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Bir soru sor..."):
    # Edge Case: Empty or whitespace-only prompt
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
                # Retrieve relevant documents
                source_docs = retriever.invoke(prompt)
                context = format_docs(source_docs)
                
                # Build and invoke the chain
                chain = rag_prompt | llm | StrOutputParser()
                answer = chain.invoke({"context": context, "question": prompt})
                
                full_response = f"{answer}\n\n---\n**Kaynaklar:**"
                if source_docs:
                    for doc in source_docs:
                        source_name = doc.metadata.get('source', 'Bilinmeyen Dosya')
                        page_num = doc.metadata.get('page', 'Bilinmeyen Sayfa')
                        full_response += f"\n- *{os.path.basename(source_name)}* (Sayfa {page_num})"
                else:
                    full_response += "\n- *Kaynak bulunamadı.*"

                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"Hata oluştu: {e}")