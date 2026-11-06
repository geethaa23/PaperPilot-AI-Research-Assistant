# PaperPilot – Research Paper Assistant

A beginner-friendly mini project built with **Python, Streamlit, PyMuPDF, LangChain, FAISS and Google Gemini**.

## 1. What this project does

- Upload one or more research-paper PDF files.
- Extract PDF text page by page using PyMuPDF.
- Split the extracted text into overlapping chunks.
- Create Gemini embeddings and store them in a FAISS vector index.
- Generate a structured summary with methodology, dataset, results, findings, limitations and future work.
- Ask questions across all uploaded papers using retrieval-augmented generation (RAG).
- Compare two papers in a table.

## 2. Requirements

- Windows 10/11, macOS or Linux
- Python 3.10 or 3.11 recommended
- Internet connection
- Gemini API key from Google AI Studio
- Text-based research paper PDFs (image-only/scanned PDFs need OCR; OCR is not included)

## 3. Installation (Windows)

Open Command Prompt in this project folder and run:

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `py -3.11` is unavailable, use `python -m venv .venv` with an installed supported Python version.

## 4. Get your Gemini API key

1. Open https://aistudio.google.com/apikey
2. Sign in with your Google account.
3. Create an API key.
4. Copy the key.

You can paste the key into the app's sidebar when it runs. Alternatively, copy `.env.example` to `.env` and replace the placeholder:

```env
GOOGLE_API_KEY=your_actual_key
```

Never upload your `.env` file to GitHub or share your API key.

## 5. Run the project

```bash
streamlit run app.py
```

The application normally opens at:

`http://localhost:8501`

## 6. How to use

1. Enter your Gemini API key in the left sidebar.
2. Upload one or more research paper PDFs.
3. Click **Process Papers** and wait for indexing to finish.
4. Open **Paper Summary**, choose a paper and click **Generate Summary**.
5. Open **Ask PaperPilot**, type a question and click **Ask Question**.
6. Upload at least two papers to use **Compare Papers**.

## 7. Example questions

- What is the main objective of this research paper?
- Explain the methodology in simple words.
- Which dataset was used?
- What algorithms or models were implemented?
- What accuracy or performance metrics were reported?
- What are the limitations of this paper?
- Compare the datasets and methodologies used in both papers.
- Which paper reports better accuracy? Are the evaluation metrics comparable?

## 8. Project folder structure

```text
PaperPilot/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python libraries
├── .env.example           # Example API-key configuration
├── .gitignore             # Files excluded from Git
└── README.md              # Setup and usage instructions
```

## 9. How the project works

```text
Upload PDF research papers
          |
          v
Extract text using PyMuPDF
          |
          v
Split text into smaller chunks
          |
          v
Generate embeddings using Gemini
          |
          v
Store vectors in FAISS
          |
          v
User asks a question
          |
          v
FAISS retrieves relevant chunks
          |
          v
Gemini generates a grounded answer
          |
          v
Display answer with source filename/page
```

This is a **Retrieval-Augmented Generation (RAG)** workflow. FAISS finds relevant text; Gemini writes the response using that retrieved context.

## 10. Important notes and limitations

- This is an educational prototype, not a validated scientific-review system.
- AI can misunderstand tables, equations, charts or complex research claims. Verify extracted results in the original PDF.
- PyMuPDF extracts selectable text. Scanned PDFs require an OCR step such as Tesseract, which is not part of this version.
- The summary and comparison prompts use a bounded portion of each PDF to keep the mini project manageable. Very long papers may need a map-reduce summarization upgrade.
- Gemini API usage may be subject to quotas, rate limits and charges depending on the account and current Google terms.
- The FAISS index is held in the Streamlit session in this starter version. It is rebuilt when papers are processed again.

## 11. Common errors

**`streamlit is not recognized`**

Activate the virtual environment and run:

```bash
python -m streamlit run app.py
```

**Gemini API key error**

Check that the key is valid, has no extra spaces, and is entered in the sidebar. Check your internet connection and API quota.

**No readable text found**

The PDF may be scanned or image-only. Use a text-based PDF or add OCR before processing.

**FAISS installation error**

Use Python 3.10 or 3.11 in a fresh virtual environment, then retry:

```bash
python -m pip install --upgrade pip
pip install faiss-cpu
```

## 12. Short project abstract

PaperPilot is an AI-powered research paper assistant designed to simplify the process of understanding and comparing academic publications. The system accepts multiple PDF research papers and extracts their text using PyMuPDF. The extracted content is divided into smaller chunks, converted into numerical embeddings using Google's Gemini embedding model, and stored in a FAISS vector database. LangChain manages the document processing and retrieval workflow. When a user asks a question, FAISS retrieves the most relevant passages and Gemini generates an answer based on those passages. The application also provides structured summaries and comparisons of research objectives, methodologies, datasets, evaluation metrics and results. The project demonstrates the practical use of Generative AI and Retrieval-Augmented Generation for academic research assistance.

## 13. Future enhancements

- OCR support for scanned research papers.
- Export summaries to PDF or Word.
- Persistent FAISS storage so users can reopen an old paper library.
- Citation highlighting with page-level references.
- Support for DOCX research documents.
- A dashboard showing paper topics, models and datasets.
