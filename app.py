import os
from collections import defaultdict

import fitz  # PyMuPDF
import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

st.set_page_config(page_title="PaperPilot | Research Assistant", page_icon="📚", layout="wide")

st.markdown("""
<style>
    .main {background: #f7f9fc;}
    .block-container {padding-top: 2rem; max-width: 1200px;}
    .hero {padding: 1.4rem 1.6rem; border-radius: 18px; background: linear-gradient(120deg,#102a43,#245b91); color: white;}
    .hero h1 {color:white; margin:0;}
    .hero p {color:#e7f0fa; margin:.4rem 0 0 0;}
    div[data-testid="stMetric"] {background:white; padding:12px; border-radius:12px; border:1px solid #e6ebf2;}
</style>
""", unsafe_allow_html=True)

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "paper_texts" not in st.session_state:
    st.session_state.paper_texts = {}
if "paper_stats" not in st.session_state:
    st.session_state.paper_stats = {}

st.markdown('<div class="hero"><h1>📚 PaperPilot</h1><p>Your AI-powered research paper summarizer, analyzer and comparison assistant.</p></div>', unsafe_allow_html=True)
st.write("Upload research-paper PDFs, extract their key contributions, and ask questions grounded in the uploaded documents.")

def extract_pdf(uploaded_file):
    """Extract searchable text from each PDF page."""
    pages = []
    with fitz.open(stream=uploaded_file.getvalue(), filetype="pdf") as pdf:
        for page_number, page in enumerate(pdf, start=1):
            page_text = page.get_text("text").strip()
            if page_text:
                pages.append(Document(
                    page_content=page_text,
                    metadata={"source": uploaded_file.name, "page": page_number}
                ))
    return pages

def make_llm(api_key):
    return ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        google_api_key=api_key,
        temperature=0.2,
    )

import time

def invoke_text(llm, prompt):
    for attempt in range(4):
        try:
            response = llm.invoke(prompt)
            return (
                response.content
                if isinstance(response.content, str)
                else str(response.content)
            )

        except Exception as e:
            error = str(e)

            if "503" in error or "UNAVAILABLE" in error:
                if attempt < 3:
                    wait_time = 5 * (2 ** attempt)
                    time.sleep(wait_time)
                    continue

            raise

def build_index(uploaded_files, api_key):
    all_pages = []
    paper_texts = {}
    paper_stats = {}
    for uploaded_file in uploaded_files:
        pages = extract_pdf(uploaded_file)
        full_text = "\n\n".join(p.page_content for p in pages)
        if not full_text.strip():
            st.warning(f"No selectable text found in **{uploaded_file.name}**. It may be a scanned PDF; OCR is not included in this mini project.")
            continue
        paper_texts[uploaded_file.name] = full_text
        paper_stats[uploaded_file.name] = {
            "pages": len(pages),
            "words": len(full_text.split())
        }
        all_pages.extend(pages)

    if not all_pages:
        raise ValueError("No readable PDF text was found. Please upload text-based PDFs.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=180,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(all_pages)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=api_key,
    )
    store = FAISS.from_documents(chunks, embeddings)
    return store, paper_texts, paper_stats, len(chunks)

def paper_context(paper_name, paper_text, max_chars=26000):
    # Keep prompts bounded for reliability; this is a small educational prototype.
    return paper_text[:max_chars]

def summarize_paper(llm, paper_name, paper_text):
    context = paper_context(paper_name, paper_text)
    prompt = f"""
You are PaperPilot, a careful academic research assistant.
Analyze the research paper below. Use only information present in the paper.
If a detail is not stated, write "Not clearly stated in the provided text".
Return clear headings:
1. Simple Summary (5-7 sentences)
2. Research Problem / Objective
3. Methodology (models, algorithms, techniques and workflow)
4. Dataset (name, size, source and features, if stated)
5. Results (metrics and numerical values exactly as reported)
6. Main Findings
7. Limitations
8. Future Work
9. Important Keywords

Paper filename: {paper_name}

PAPER TEXT:
{context}
"""
    return invoke_text(llm, prompt)

def retrieve_context(question, k=7):
    docs = st.session_state.vector_store.similarity_search(question, k=k)
    return "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'Unknown')}, page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in docs
    )

with st.sidebar:
    st.header("⚙️ Setup")
    api_key = st.text_input(
        "Gemini API key",
        value=os.getenv("GOOGLE_API_KEY", os.getenv("GEMINI_API_KEY", "")),
        type="password",
        help="Create a key in Google AI Studio. Your key is used for this session and is not displayed.",
    )
    st.caption("The app uses Gemini for answers and embeddings, and stores the FAISS index in session memory.")
    st.divider()
    st.markdown("**Workflow**")
    st.markdown("1. Add your API key\n2. Upload PDF papers\n3. Click Process Papers\n4. Explore summaries and ask questions")

left, right = st.columns([1.2, 1])
with left:
    st.subheader("1. Upload research papers")
    uploads = st.file_uploader(
        "Choose one or more research-paper PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Text-based PDFs work best. Scanned image-only PDFs need OCR before processing.",
    )
    process = st.button("🚀 Process Papers", type="primary", use_container_width=True, disabled=not (uploads and api_key))

with right:
    st.subheader("2. What PaperPilot can do")
    st.markdown("""
    - **Summarize** each paper in simple language
    - **Extract** methodology, datasets and results
    - **Answer** questions using retrieved paper passages
    - **Compare** papers side by side
    """)

if process:
    try:
        with st.spinner("Reading PDFs, creating embeddings and building the FAISS index..."):
            store, texts, stats, chunk_count = build_index(uploads, api_key)
            st.session_state.vector_store = store
            st.session_state.paper_texts = texts
            st.session_state.paper_stats = stats
            st.session_state.chunk_count = chunk_count
            st.session_state.api_key = api_key
        st.success(f"Processed {len(texts)} paper(s) and indexed {chunk_count} text chunks.")
    except Exception as exc:
        st.error(f"Could not process the papers: {exc}")
        st.info("Check your API key, internet connection, package installation, and PDF readability.")

if st.session_state.paper_texts:
    st.divider()
    st.subheader("📊 Uploaded paper library")
    cols = st.columns(min(4, max(1, len(st.session_state.paper_stats))))
    for idx, (name, info) in enumerate(st.session_state.paper_stats.items()):
        with cols[idx % len(cols)]:
            st.metric(label=name[:28] + ("…" if len(name) > 28 else ""), value=f"{info['pages']} pages", delta=f"{info['words']:,} words")

    tab1, tab2, tab3 = st.tabs(["📝 Paper Summary", "💬 Ask PaperPilot", "⚖️ Compare Papers"])

    with tab1:
        st.markdown("Generate a structured analysis for one uploaded paper.")
        selected_paper = st.selectbox("Select a paper", list(st.session_state.paper_texts.keys()), key="summary_paper")
        if st.button("Generate Summary", key="generate_summary"):
            try:
                llm = make_llm(st.session_state.api_key)
                with st.spinner("Analyzing the paper..."):
                    result = summarize_paper(llm, selected_paper, st.session_state.paper_texts[selected_paper])
                st.session_state["summary_" + selected_paper] = result
            except Exception as exc:
                st.error(f"Summary generation failed: {exc}")
        saved_summary = st.session_state.get("summary_" + selected_paper)
        if saved_summary:
            st.markdown(saved_summary)
            st.download_button(
                "⬇️ Download summary",
                data=saved_summary,
                file_name=f"{selected_paper.rsplit('.', 1)[0]}_summary.txt",
                mime="text/plain",
            )

    with tab2:
        st.markdown("Ask questions across all processed papers. PaperPilot retrieves relevant passages before answering.")
        question = st.text_input("Your question", placeholder="Example: Which dataset was used and what accuracy was achieved?")
        if st.button("Ask Question", type="primary", key="ask_question", disabled=not question.strip()):
            try:
                with st.spinner("Searching papers and preparing an answer..."):
                    context = retrieve_context(question, k=8)
                    llm = make_llm(st.session_state.api_key)
                    answer = invoke_text(llm, f"""
You are PaperPilot, an academic assistant. Answer the question using ONLY the context below.
If the context does not contain the answer, say that the uploaded papers do not provide enough information.
Be precise with numerical results. Mention the source filename and page when possible.
Do not invent citations or facts.

QUESTION:
{question}

RETRIEVED PAPER PASSAGES:
{context}
""")
                st.markdown("### Answer")
                st.write(answer)
                with st.expander("View retrieved source passages"):
                    st.text(context)
            except Exception as exc:
                st.error(f"Question answering failed: {exc}")

    with tab3:
        st.markdown("Compare the methodology, datasets, metrics and findings of two uploaded papers.")
        names = list(st.session_state.paper_texts.keys())
        if len(names) < 2:
            st.info("Upload and process at least two papers to use comparison.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                paper_a = st.selectbox("First paper", names, index=0, key="compare_a")
            with c2:
                paper_b = st.selectbox("Second paper", names, index=1, key="compare_b")
            comparison_focus = st.text_input(
                "Comparison focus (optional)",
                placeholder="Example: model architecture, dataset size and accuracy",
            )
            if st.button("Compare Papers", type="primary", disabled=(paper_a == paper_b)):
                try:
                    llm = make_llm(st.session_state.api_key)
                    prompt = f"""
Compare these two research papers using only the supplied text.
Create a Markdown table with rows for research objective, methodology, algorithms/models,
dataset, evaluation metrics, key results, strengths and limitations.
After the table, explain the major similarities, differences, and which paper performs better
for each reported metric. Do not claim one is better if the metrics are not comparable.
Write "Not stated" when information is missing. Preserve reported numbers accurately.
Extra focus: {comparison_focus or "overall comparison"}.

PAPER A: {paper_a}
{paper_context(paper_a, st.session_state.paper_texts[paper_a], 18000)}

PAPER B: {paper_b}
{paper_context(paper_b, st.session_state.paper_texts[paper_b], 18000)}
"""
                    with st.spinner("Comparing both papers..."):
                        comparison = invoke_text(llm, prompt)
                    st.markdown(comparison)
                    st.download_button("⬇️ Download comparison", comparison, "paper_comparison.md", "text/markdown")
                except Exception as exc:
                    st.error(f"Comparison failed: {exc}")

st.divider()
st.caption("PaperPilot Mini Project • Educational prototype • Always verify AI-extracted claims against the original paper.")
