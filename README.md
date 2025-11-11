# RAG Document Q&A

A small Streamlit-based Retrieval-Augmented-Generation (RAG) demo for asking questions over uploaded documents.

## What this project contains

- `app.py` — Streamlit web app. Upload PDFs / text files, index them with FAISS, and run a retrieval+LLM chain.
- `requirements.txt` — Python package pins (use a venv / conda env for installs).
- `research_papers/` — (optional) location for keeping documents locally.

## Requirements

- Python 3.11+ (project was developed with Python 3.13 in a Conda env).
- `pip` available (or use Conda).
- A GROQ API key for `ChatGroq` usage. Set it in a `.env` file as shown below.

## Quick setup (recommended: virtual env)

PowerShell (preferred):

```powershell
# Create venv and activate
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

Conda (alternative):

```powershell
# Create and activate a conda env
conda create -p rag-qa python==3.12 -y
conda activate rag-qa

# Install from requirements
pip install -r requirements.txt
```

If you cannot install into the Conda base environment due to permissions, prefer using a venv (above) or use `pip install --user -r requirements.txt`.

## Environment variables

Create a `.env` file in the project root with at least:

```
GROQ_API_KEY=sk-...
```

The app will stop with an error if `GROQ_API_KEY` is not set.

## Run the app

From PowerShell (after activating the env):

```powershell
streamlit run "e:\Projects\RAG Document Q&A\app.py"
```

If you prefer to use the Conda Python executable directly (example from this environment):

```powershell
D:\Anaconda\python.exe -m streamlit run "e:\Projects\RAG Document Q&A\app.py"
```

## Troubleshooting

- If you still see import errors in the editor (Pylance), reload the VS Code window or restart the Python language server: Command Palette -> `Developer: Reload Window` or `Python: Restart Language Server`.
- If packages were installed with `--user`, ensure VS Code is using the same Python interpreter (select Interpreter from the status bar).
- If FAISS is giving platform errors, you can try `faiss-cpu` or use a different vectorstore supported by LangChain.

## Next steps

- Run the app and paste any runtime traceback here if it fails; I will fix the code accordingly.

