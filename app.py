import streamlit as st
import os
import requests

# Skip the slow HuggingFace connectivity check every launch
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

from rag_engine_scratch import NoteVaultScratchEngine as NoteVaultEngine
from report_generator import generate_pdf_report
import time

def _check_ollama():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        if r.status_code == 200:
            models = [m["name"].split(":")[0] for m in r.json().get("models", [])]
            return True, models
        return False, []
    except Exception:
        return False, []

# Page Config
st.set_page_config(page_title="NoteVault AI", page_icon="📝", layout="wide")

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
    }
    .stTextInput>div>div>input {
        background-color: #262730;
        color: white;
    }
    .source-box {
        background-color: #1e1e26;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 5px;
        border-left: 5px solid #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize Session State
if 'engine' not in st.session_state:
    with st.spinner("Preparing NoteVault Intelligence..."):
        st.session_state.engine = NoteVaultEngine()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processed' not in st.session_state:
    st.session_state.processed = False

# Sidebar
with st.sidebar:
    st.title("⚙️ NoteVault Settings")
    st.divider()

    # --- Ollama Status ---
    ollama_running, ollama_models = _check_ollama()
    if ollama_running:
        st.success("🟢 Ollama: Running")
        if ollama_models:
            model_choice = st.selectbox(
                "🤖 Ollama Model",
                ollama_models,
                help="Select which locally-pulled Ollama model to use."
            )
        else:
            st.warning("No models pulled yet. Run: ollama pull mistral")
            model_choice = "mistral"
    else:
        st.warning("🔴 Ollama: Not running")
        st.caption("Fallback: TinyLlama (transformers) will be used.")
        st.code("ollama pull mistral", language="bash")
        model_choice = "mistral"

    # Wire model selection to engine
    if st.session_state.get("engine") and st.session_state.engine.llm:
        st.session_state.engine.llm.set_model(model_choice)

    st.divider()
    if st.session_state.get("processed"):
        st.success("✅ PDF Loaded")
        st.metric("Pages Processed", len(st.session_state.engine.processed_pages))
        st.metric("Questions Asked", len(st.session_state.chat_history))
        if st.session_state.engine.llm:
            st.caption(f"LLM: {st.session_state.engine.llm.backend_info}")
    else:
        st.info("Upload a PDF to begin.")
    st.divider()
    if st.button("🔄 Reset Session", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.processed = False
        st.session_state.engine = NoteVaultEngine()
        st.rerun()

# Layout
col1, col2 = st.columns([1, 2])

with col1:
    st.title("NoteVault AI")
    st.subheader("Your Personal Offline Study Assistant")
    
    uploaded_file = st.file_uploader("Upload Handwritten Notes (PDF)", type="pdf")
    
    if uploaded_file and not st.session_state.processed:
        import time

        # ── Pipeline Stage Display ─────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🔄 Pipeline Progress")

        STAGES = [
            ("📄", "Stage 1", "PDF → Image Conversion",   "Splitting PDF pages into high-resolution images"),
            ("🔍", "Stage 2", "Image Preprocessing",       "Enhancing contrast & correcting page tilt with OpenCV"),
            ("✍️", "Stage 3", "Handwriting OCR",           "Extracting text from handwritten pages via PaddleOCR"),
            ("✂️", "Stage 4", "Semantic Chunking",         "Breaking extracted text into searchable knowledge chunks"),
            ("🧮", "Stage 5", "TF-IDF Model Training",     "Fitting custom TF-IDF vocabulary on your specific notes"),
            ("📐", "Stage 6", "Vector Indexing",            "Building NumPy cosine-similarity matrix for fast search"),
            ("✅", "Stage 7", "Ready",                      "Your notes are indexed and ready for questions"),
        ]

        # Render stage placeholders
        stage_placeholders = []
        for icon, label, title, desc in STAGES:
            col_icon, col_text = st.columns([1, 8])
            with col_icon:
                p_icon = st.empty()
                p_icon.markdown(f"<div style='font-size:28px;text-align:center'>⏳</div>", unsafe_allow_html=True)
            with col_text:
                p_title = st.empty()
                p_title.markdown(f"**{label}: {title}** — <span style='color:#888'>{desc}</span>", unsafe_allow_html=True)
            stage_placeholders.append((p_icon, p_title, icon, label, title, desc))

        progress_bar = st.progress(0, text="Waiting to start...")
        result_area  = st.empty()
        pipeline_failed = False

        def mark_stage(idx, running=True):
            p_icon, p_title, icon, label, title, desc = stage_placeholders[idx]
            if running:
                p_icon.markdown(f"<div style='font-size:28px;text-align:center'>🔄</div>", unsafe_allow_html=True)
                p_title.markdown(f"**{label}: {title}** — <span style='color:#f0a500'>*Running…*</span>", unsafe_allow_html=True)
            else:
                p_icon.markdown(f"<div style='font-size:28px;text-align:center'>{icon}</div>", unsafe_allow_html=True)
                p_title.markdown(f"**{label}: {title}** — <span style='color:#4CAF50'>✔ Done</span>", unsafe_allow_html=True)
            pct = int(((idx + (0 if running else 1)) / len(STAGES)) * 100)
            progress_bar.progress(pct, text=f"{label}: {title}…" if running else f"✅ {label} complete")

        def mark_stage_error(idx, error):
            import traceback
            p_icon, p_title, icon, label, title, desc = stage_placeholders[idx]
            p_icon.markdown("<div style='font-size:28px;text-align:center'>❌</div>", unsafe_allow_html=True)
            p_title.markdown(
                f"**{label}: {title}** — <span style='color:#ff4444'>❌ FAILED</span>",
                unsafe_allow_html=True
            )
            progress_bar.progress(
                int(((idx) / len(STAGES)) * 100),
                text=f"❌ {label} failed — see error below"
            )
            # Show a detailed error card
            st.markdown(
                f"""<div style='background:#2a0a0a;border-left:5px solid #ff4444;
                    padding:12px;border-radius:6px;margin:6px 0'>
                    <b style='color:#ff6666'>⚠ Error in {label}: {title}</b><br>
                    <code style='color:#ffaaaa'>{type(error).__name__}: {str(error)}</code>
                </div>""",
                unsafe_allow_html=True
            )
            with st.expander(f"📋 Full Traceback — {label}: {title}", expanded=True):
                st.code(traceback.format_exc(), language="python")

        # ── Stage 1: Save PDF ─────────────────────────────────────────────
        mark_stage(0, running=True)
        try:
            if not os.path.exists("data"):
                os.makedirs("data")
            temp_path = "data/input_notes.pdf"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            mark_stage(0, running=False)
        except Exception as e:
            mark_stage_error(0, e)
            pipeline_failed = True

        # ── Stage 2: PDF → Images ─────────────────────────────────────────
        image_paths = []
        if not pipeline_failed:
            mark_stage(1, running=True)
            try:
                from preprocessing import pdf_to_images
                image_paths = pdf_to_images(temp_path)
                mark_stage(1, running=False)
            except Exception as e:
                mark_stage_error(1, e)
                pipeline_failed = True

        # ── Stage 3: OCR ──────────────────────────────────────────────────
        all_raw_texts = []
        if not pipeline_failed:
            mark_stage(2, running=True)
            try:
                st.session_state.engine.load_ocr()
                ocr_errors = []
                for i, img_path in enumerate(image_paths):
                    progress_bar.progress(
                        int((2 / len(STAGES)) * 100) + int((i / max(len(image_paths), 1)) * (100 // len(STAGES))),
                        text=f"OCR: reading page {i+1} of {len(image_paths)}…"
                    )
                    try:
                        text, _ = st.session_state.engine.ocr_manager.extract_text(img_path)
                    except Exception as page_err:
                        text = ""
                        ocr_errors.append(f"Page {i+1}: {type(page_err).__name__}: {page_err}")
                    all_raw_texts.append({"page": i + 1, "text": text})
                    st.session_state.engine.processed_pages.append({"page": i + 1, "text": text})

                if ocr_errors:
                    st.warning(f"⚠ OCR had issues on {len(ocr_errors)} page(s) — partial text extracted:")
                    with st.expander("📋 Per-Page OCR Errors"):
                        for err_msg in ocr_errors:
                            st.code(err_msg, language="text")

                mark_stage(2, running=False)
            except Exception as e:
                mark_stage_error(2, e)
                pipeline_failed = True

        # ── Stage 4: Semantic Chunking ────────────────────────────────────
        all_chunks, all_page_nums = [], []
        if not pipeline_failed:
            mark_stage(3, running=True)
            try:
                for entry in all_raw_texts:
                    chunks = [c.strip() for c in entry["text"].split(".") if len(c.strip()) > 10]
                    all_chunks.extend(chunks)
                    all_page_nums.extend([entry["page"]] * len(chunks))
                if not all_chunks:
                    raise ValueError("No text chunks produced. The PDF may be blank or OCR extracted nothing.")
                mark_stage(3, running=False)
            except Exception as e:
                mark_stage_error(3, e)
                pipeline_failed = True

        # ── Stage 5: TF-IDF Training ──────────────────────────────────────
        if not pipeline_failed:
            mark_stage(4, running=True)
            try:
                st.session_state.engine.vector_engine.tfidf.fit_transform(all_chunks)
                mark_stage(4, running=False)
            except Exception as e:
                mark_stage_error(4, e)
                pipeline_failed = True

        # ── Stage 6: Vector Indexing ──────────────────────────────────────
        if not pipeline_failed:
            mark_stage(5, running=True)
            try:
                for i in range(len(all_chunks)):
                    st.session_state.engine.vector_engine.metadata.append({
                        "text": all_chunks[i],
                        "page": all_page_nums[i]
                    })
                mark_stage(5, running=False)
            except Exception as e:
                mark_stage_error(5, e)
                pipeline_failed = True

        # ── Stage 7: Final Status ─────────────────────────────────────────
        if not pipeline_failed:
            mark_stage(6, running=False)
            st.session_state.processed = True
            progress_bar.progress(100, text="✅ All stages complete!")
            result_area.success(
                f"🎉 **Ready!** Indexed **{len(all_chunks)} knowledge chunks** "
                f"from **{len(image_paths)} pages** into your from-scratch vector engine."
            )
        else:
            progress_bar.progress(0, text="❌ Pipeline stopped due to error above")
            result_area.error(
                "Pipeline failed. Fix the error shown above then click **Reset Session** in the sidebar and re-upload the PDF."
            )
            
    if st.session_state.processed:
        if st.button("🧠 Run Custom CNN Training Demo", use_container_width=True):
            with st.spinner("Training Custom CNN from Scratch with PyTorch..."):
                import subprocess
                result = subprocess.run(
                    ["python", "train_demo.py"],
                    capture_output=True, text=True, cwd="e:/GenHack"
                )
                st.success("Custom CNN Model Training Complete!")
                if result.stdout:
                    st.code(result.stdout)

        if st.button("📋 Generate Study Summary", use_container_width=True):
            with st.spinner("Generating summary via local LLM..."):
                full_text = " ".join([p['text'] for p in st.session_state.engine.processed_pages])
                if full_text.strip():
                    summary = st.session_state.engine.get_summary(full_text)
                    st.info(summary)
                else:
                    st.warning("No text extracted yet.")

        if st.button("📄 Export Study Report", use_container_width=True):
            if not st.session_state.chat_history:
                st.warning("Ask some questions first before exporting the report.")
            else:
                report_path = generate_pdf_report(st.session_state.chat_history)
                with open(report_path, "rb") as f:
                    st.download_button(
                        "⬇️ Download PDF Report", f,
                        file_name="NoteVault_Study_Report.pdf",
                        mime="application/pdf"
                    )

with col2:
    st.header("Chat Interface")
    
    # Display Chat History
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])
        with st.chat_message("assistant"):
            st.write(chat["answer"])
            if chat["sources"]:
                with st.expander("View Sources"):
                    for s in chat["sources"]:
                        st.markdown(f"<div class='source-box'><b>Page {s['page']}</b><br>{s['text']}</div>", unsafe_allow_html=True)

    # User Input
    if prompt := st.chat_input("Ask a question about your notes..."):
        if not st.session_state.processed:
            st.warning("Please upload a PDF first.")
        else:
            with st.chat_message("user"):
                st.write(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner(f"Thinking with {model_choice}..."):
                    # Ensure LLM is loaded with the currently selected model
                    st.session_state.engine.load_llm(model_id=model_choice)
                    answer, sources = st.session_state.engine.ask(prompt)
                    st.write(answer)
                    if sources:
                        with st.expander("View Sources"):
                            for s in sources:
                                st.markdown(f"<div class='source-box'><b>Page {s['page']}</b><br>{s['text']}</div>", unsafe_allow_html=True)
            
            # Save to history
            st.session_state.chat_history.append({
                "question": prompt,
                "answer": answer,
                "sources": sources
            })
