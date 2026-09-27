import hashlib
import io
import os
import tempfile
 
import streamlit as st
from dotenv import load_dotenv
from elevenlabs import ElevenLabs
from elevenlabs.core import ApiError
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
 
load_dotenv()
 
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = "EXAVITQu4vr4xnSDxMaL"
ELEVENLABS_STT_MODEL = "scribe_v1"
ELEVENLABS_TTS_MODEL = "eleven_v3"
GEMINI_MODEL = "gemini-3.5-flash-lite"
 
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K = 4
 
LANG_NAMES = {"eng": "English", "hin": "Hindi", "guj": "Gujarati"}
 
st.set_page_config(page_title="Voice RAG (EN / HI / GU)", page_icon="\U0001F3A4", layout="centered")
 
 
# Vector Store
 
@st.cache_resource(show_spinner=False)
def load_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
 
 
def build_vectorstore(pdf_files) -> tuple[FAISS | None, int]:
    """Load each uploaded PDF with PyPDFLoader, split into chunks with
    RecursiveCharacterTextSplitter, and embed them into a FAISS index."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    all_pages = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for f in pdf_files:
            tmp_path = os.path.join(tmpdir, f.name)
            with open(tmp_path, "wb") as out:
                out.write(f.getvalue())
            all_pages.extend(PyPDFLoader(tmp_path).load())
 
        if not all_pages:
            return None, 0
 
        chunks = splitter.split_documents(all_pages)
        vectorstore = FAISS.from_documents(chunks, load_embeddings())
        return vectorstore, len(chunks)
 
 
# ElevenLabs
 
@st.cache_resource(show_spinner=False)
def get_elevenlabs_client() -> ElevenLabs:
    return ElevenLabs(api_key=ELEVENLABS_API_KEY)
 
 
def elevenlabs_speech_to_text(audio_bytes: bytes) -> tuple[str, str]:
    client = get_elevenlabs_client()
    result = client.speech_to_text.convert(
        file=io.BytesIO(audio_bytes),
        model_id=ELEVENLABS_STT_MODEL,
    )
    return (result.text or "").strip(), result.language_code or "eng"
 
 
def elevenlabs_text_to_speech(text: str) -> bytes:
    client = get_elevenlabs_client()
    audio = client.text_to_speech.convert(
        text=text,
        voice_id=ELEVENLABS_VOICE_ID,
        model_id=ELEVENLABS_TTS_MODEL,
        output_format="mp3_44100_128",
    )
    if isinstance(audio, (bytes, bytearray)):
        return bytes(audio)
    return b"".join(audio)
 
 
# Gemini LLM
 
TRANSLATE_PROMPT = ChatPromptTemplate.from_template(
    "Translate the following {lang} text to English. "
    "Reply with ONLY the translation, no notes or quotes.\n\n"
    "Text: {text}"
)
 
ANSWER_PROMPT = ChatPromptTemplate.from_template(
    "You are a helpful assistant answering questions using only the context below. "
    "If the context does not contain the answer, say so honestly rather than guessing.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Write your answer in {lang}. Keep it clear and natural to read aloud."
)
 
 
def get_llm(api_key: str) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=api_key, temperature=0.3)
 
 
def gemini_translate_to_english(llm: ChatGoogleGenerativeAI, text: str, source_lang_name: str) -> str:
    chain = TRANSLATE_PROMPT | llm | StrOutputParser()
    return chain.invoke({"lang": source_lang_name, "text": text}).strip()
 
 
def gemini_answer(llm: ChatGoogleGenerativeAI, question: str, docs: list, answer_lang_name: str) -> str:
    context = "\n\n---\n\n".join(d.page_content for d in docs) if docs else "(no relevant context found)"
    chain = ANSWER_PROMPT | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question, "lang": answer_lang_name}).strip()
 
 
# Streamlit UI
 
def run_pipeline(audio_bytes: bytes, gemini_key: str, vectorstore: FAISS) -> dict | None:
    """Runs STT -> (translate) -> retrieve -> answer -> TTS, reporting progress
    into a live st.status block. Returns a result dict, or None if nothing
    could be transcribed."""
    with st.status("Listening...", expanded=True) as status_box:
        status_box.write("Transcribing your question...")
        question_text, lang_code = elevenlabs_speech_to_text(audio_bytes)
        lang_name = LANG_NAMES.get(lang_code, lang_code)
 
        if not question_text:
            status_box.update(label="Didn't catch that", state="error")
            return None
 
        status_box.write(f"You asked ({lang_name}): {question_text}")
 
        llm = get_llm(gemini_key)
 
        english_query = question_text
        if lang_code != "eng":
            status_box.write(f"Translating your {lang_name} question to English for search...")
            english_query = gemini_translate_to_english(llm, question_text, lang_name)
 
        status_box.write("Searching your documents...")
        retrieved_docs = vectorstore.similarity_search(english_query, k=TOP_K)
 
        status_box.write(f"Writing the answer in {lang_name}...")
        answer_text = gemini_answer(llm, question_text, retrieved_docs, lang_name)
 
        status_box.write("Generating speech...")
        answer_audio = elevenlabs_text_to_speech(answer_text)
 
        status_box.update(label="Done", state="complete", expanded=False)
 
    return {
        "question": question_text,
        "lang_name": lang_name,
        "english_query": english_query if lang_code != "eng" else None,
        "docs": retrieved_docs,
        "answer_text": answer_text,
        "answer_audio": answer_audio,
    }
 
st.title("\U0001F3A4 Voice RAG \u2014 English / Hindi / Gujarati")
st.caption("Tap the mic, ask your question, tap again to stop. The answer plays back automatically.")
 
if not ELEVENLABS_API_KEY:
    st.error("ELEVENLABS_API_KEY is missing from your .env file. Add it and restart the app.")
 
with st.sidebar:
    st.subheader("1. Gemini API key")
    gemini_key = st.text_input(
        "Paste your Gemini API key",
        type="password",
        help="Used only for this browser session. Never written to disk or logged.",
    )
    st.session_state["gemini_key"] = gemini_key
    key_ready = bool(gemini_key.strip())
    st.caption("\u2705 Key entered" if key_ready else "\u26A0\uFE0F Required before you can ask a question")
 
    st.divider()
    st.subheader("2. Your documents")
    pdf_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
    if st.button("Upload", disabled=not pdf_files):
        with st.spinner("Reading your documents..."):
            vectorstore, num_chunks = build_vectorstore(pdf_files)
        if vectorstore is None:
            st.error("Could not read any text from those PDFs.")
        else:
            st.session_state["vectorstore"] = vectorstore
            st.success(f"Ready \u2014 loaded {len(pdf_files)} file(s).")
 
index_ready = "vectorstore" in st.session_state
 
st.divider()
st.subheader("3. Ask your question")
 
audio_value = st.audio_input("Tap the mic to ask, tap again to stop")
 
audio_hash = hashlib.md5(audio_value.getvalue()).hexdigest() if audio_value is not None else None
is_new_recording = audio_hash is not None and audio_hash != st.session_state.get("processed_audio_hash")
 
if is_new_recording:
    if not key_ready:
        st.info("Enter your Gemini API key in the sidebar \u2014 your question will process automatically once it's there.")
    elif not index_ready:
        st.info("Upload PDFs in the sidebar first \u2014 your question will process automatically once they're ready.")
    else:
        st.session_state["processed_audio_hash"] = audio_hash
        try:
            result = run_pipeline(audio_value.getvalue(), st.session_state["gemini_key"], st.session_state["vectorstore"])
            st.session_state["last_answer"] = result
            st.session_state["autoplay_pending"] = result is not None
        except ApiError as e:
            st.session_state["last_answer"] = None
            st.error(f"ElevenLabs API error: {e.status_code} \u2014 {e.body}")
        except Exception as e:
            st.session_state["last_answer"] = None
            st.error(f"Something went wrong: {e}")
 
if st.session_state.get("last_answer"):
    ans = st.session_state["last_answer"]
    st.markdown(f"**You asked ({ans['lang_name']}):** {ans['question']}")
    if ans["english_query"]:
        st.caption(f"Translated for search: {ans['english_query']}")
 
    with st.expander("Retrieved context"):
        for i, doc in enumerate(ans["docs"], 1):
            st.markdown(f"**Chunk {i}**")
            st.write(doc.page_content)
 
    st.markdown(f"**Answer ({ans['lang_name']}):**")
    st.write(ans["answer_text"])
 
    autoplay = st.session_state.pop("autoplay_pending", False)
    st.audio(ans["answer_audio"], format="audio/mp3", autoplay=autoplay)