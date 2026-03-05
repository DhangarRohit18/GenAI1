# NoteVault AI — Gen AI Hackathon 2026

NoteVault AI is a **100% offline**, personal study assistant designed to convert handwritten notes into a searchable knowledge base. It uses custom "from-scratch" mathematical components and local neural models to ensure privacy and speed.

## 🚀 Key Features
- **Offline OCR:** Extracts text from handwritten or typed PDF notes using PaddleOCR.
- **Visual Pipeline:** Real-time tracking of processing stages (PDF → Image → OCR → Vector Index).
- **From-Scratch Vector Engine:** Uses TF-IDF and Cosine Similarity implemented in NumPy for semantic search.
- **Local LLM Integration:** Powered by **Mistral 7B via Ollama** (with TinyLlama fallback) for answering study questions.
- **Study Report Generation:** Export your chat history and knowledge insights into a professional PDF report.
- **Custom CNN Training:** A demo of a neural network built with PyTorch to show digit recognition capabilities.

## 🛠 Tech Stack
- **Frontend:** Streamlit
- **OCR:** PaddleOCR (Mobile v5 Models)
- **Math/Vector:** NumPy (TF-IDF, Cosine Similarity)
- **LLM:** Ollama (Mistral / Llama3)
- **PDF Processing:** PyMuPDF (fitz), OpenCV, ReportLab

## 📦 Installation & Setup

### 1. Requirements
Ensure you have Python 3.10+ installed.

```bash
pip install -r requirements.txt
```

### 2. Ollama (Optional but Recommended)
For the best experience, install Ollama:
1. Download from [ollama.com](https://ollama.com/download).
2. Run `ollama pull mistral`.

### 3. Run the Application
```bash
streamlit run app.py
```

## 📝 Usage
1. Upload a PDF of your handwritten notes.
2. Wait for the 7-stage pipeline to complete.
3. Start asking questions in the chat!
4. Download your study report at the end of the session.

---
*Created for GenHack 2026*
