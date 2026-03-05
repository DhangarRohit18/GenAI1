# NoteVault AI: Setup & Run Guide

Follow these steps to set up NoteVault AI on your local machine.

## Step 1: Install Ollama
Download and install Ollama from [ollama.com](https://ollama.com/).
After installation, open your terminal and run:
```bash
ollama run mistral
```
This ensures the model is downloaded and ready for offline use.

## Step 2: System Dependencies
### Windows
You may need to install **Microsoft C++ Build Tools** for FAISS and other libraries.
Ensure **OpenCV** dependencies are met (standard on most Windows dev environments).

## Step 3: Python Environment
We recommend using a virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

## Step 4: PaddleOCR Requirements
PaddleOCR requires `paddlepaddle`. If you have a GPU, install `paddlepaddle-gpu`. For CPU only:
```bash
pip install paddlepaddle
```

## Step 5: Launching the System
```bash
streamlit run app.py
```

## Step 6: Critical - Pre-download Models (Before going Offline)
The first time you run NoteVault AI, it will download several models:
1. **Ollama**: `mistral` (approx 4GB)
2. **PaddleOCR**: Detection and Recognition models (approx 100MB)
3. **Sentence-Transformers**: `all-MiniLM-L6-v2` (approx 80MB)

**Run the pipeline once with a sample PDF while connected to the internet to ensure all models are cached locally for the offline demo.**

## How to use for the Demo:
1. **Upload**: Select the handwritten PDF. Wait for the success message.
2. **Chat**: Ask specific questions like "What are the three steps of normalization?"
3. **Verify**: Use the source citations to show the judges exact page references.
4. **Summary**: Click "Generate Study Summary" for a quick overview.
5. **Bonus**: Export a PDF report to show the "Study Report" feature.
