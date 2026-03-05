import streamlit as st
import os
import requests
import time
import datetime
import base64
from rag_engine_scratch import NoteVaultScratchEngine as NoteVaultEngine
from report_generator import generate_pdf_report, generate_transcription_pdf

# --- CONFIG & THEME ---
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
if not os.path.exists("data"):
    try:
        os.makedirs("data")
    except Exception:
        pass
        
st.set_page_config(page_title="NoteVault AI", page_icon="📝", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS (Production Grade) ---
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Permanent Dark Mode Layout Colors */
    :root {{
        --bg-color: #111827;           /* Deep Midnight */
        --sidebar-bg: #1F2937;        /* Dark Slate */
        --text-color: #F9FAFB;        /* Off-White */
        --text-secondary: #9CA3AF;     /* Gray */
        --card-bg: #1F2937;           /* Slightly Lighter Slate */
        --border-color: #374151;       /* Subtle Border */
    }}

    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
        background-color: var(--bg-color) !important;
        color: var(--text-color);
    }}
    
    [data-testid="stAppViewContainer"] {{
        background-color: var(--bg-color) !important;
    }}
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {{
        background-color: var(--sidebar-bg) !important;
        border-right: 1px solid var(--border-color);
    }}
    
    .sidebar-brand {{
        font-size: 1.5rem;
        font-weight: 700;
        color: #4A6CF7;
        margin-bottom: 2rem;
        padding: 0 1rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }}
    
    /* Responsive App Container */
    [data-testid="stAppViewContainer"] {{
        background-color: var(--bg-color) !important;
        padding-bottom: 2rem;
    }}
    
    .main-container {{
        max-width: 1300px;
        margin: 0 auto;
        padding: 0.75rem 1.5rem;
    }}
    
    /* Horizontal Workspace Flow */
    .workspace-split {{
        display: flex;
        gap: 1.5rem;
        align-items: flex-start;
    }}
    
    .workspace-sidebar {{
        flex: 0 0 350px;
        position: sticky;
        top: 1rem;
    }}
    
    .workspace-main {{
        flex: 1;
    }}
    
    /* Stats Grid (Responsive) */
    .stats-container {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 1.5rem;
        margin: 1.5rem 0;
    }}
    
    .stats-card {{
        background: var(--card-bg);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid var(--border-color);
        text-align: left;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s, box-shadow 0.2s;
    }}
    
    .stats-card:hover {{
        transform: translateY(-4px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }}
    
    .stats-value {{
        font-size: 2rem;
        font-weight: 800;
        color: #4A6CF7;
        margin-bottom: 0.25rem;
    }}
    
    .stats-label {{
        font-size: 0.875rem;
        font-weight: 500;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.025em;
    }}
    
    /* Chat Interface (Compatibility & Wrapping) */
    .chat-container {{
        display: flex;
        flex-direction: column;
        gap: 1.25rem;
        padding: 1rem 0;
        max-width: 900px;
        margin: 0 auto;
    }}
    
    .chat-bubble {{
        padding: 1rem 1.5rem;
        border-radius: 1.25rem;
        line-height: 1.6;
        font-size: 1rem;
        max-width: 85%;
        word-wrap: break-word;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }}
    
    .user-msg {{
        align-self: flex-end;
        background-color: #4A6CF7;
        color: white !important;
        border-bottom-right-radius: 4px;
    }}
    
    .ai-msg {{
        align-self: flex-start;
        background-color: var(--card-bg);
        color: var(--text-color) !important;
        border: 1px solid var(--border-color);
        border-bottom-left-radius: 4px;
    }}
    
    /* Media Queries for different viewports */
    @media (max-width: 992px) {{
        .chat-bubble {{ max-width: 95%; }}
        .stats-container {{ grid-template-columns: 1fr 1fr; }}
    }}
    
    @media (max-width: 768px) {{
        .stats-container {{ grid-template-columns: 1fr; }}
        .sidebar-brand {{ font-size: 1.2rem; }}
        .main-container {{ padding: 1rem; }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: 10px !important;
        }}
        .stTabs [data-baseweb="tab"] {{
            padding: 8px 12px !important;
            font-size: 14px !important;
        }}
    }}
    
    /* Premium Tab Styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 24px;
        background-color: transparent !important;
        padding: 0 1rem;
        border-bottom: 2px solid var(--border-color);
        margin-bottom: 2rem;
    }}

    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        white-space: pre;
        background-color: transparent !important;
        border: none !important;
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
        font-size: 16px !important;
        transition: all 0.3s ease !important;
    }}

    .stTabs [data-baseweb="tab"]:hover {{
        color: #4A6CF7 !important;
    }}

    .stTabs [aria-selected="true"] {{
        color: #4A6CF7 !important;
        border-bottom: 3px solid #4A6CF7 !important;
        font-weight: 700 !important;
    }}
    
    .user-msg {{
        align-self: flex-end;
        background-color: #4A6CF7;
        color: #FFFFFF !important;
        border-bottom-right-radius: 4px;
    }}
    
    .ai-msg {{
        align-self: flex-start;
        background-color: var(--card-bg);
        color: var(--text-color) !important;
        border: 1px solid var(--border-color);
        border-bottom-left-radius: 4px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }}
    
    .confidence-badge {{
        display: inline-block;
        padding: 2px 8px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }}
    
    .conf-high {{ background: #D1FAE5; color: #065F46; }}
    .conf-med {{ background: #FEF3C7; color: #92400E; }}
    .conf-low {{ background: #FEE2E2; color: #991B1B; }}
    
    /* Flashcards Compatibility */
    .flashcard-main {{
        background: var(--card-bg);
        min-height: 280px;
        width: 100%;
        max-width: 550px;
        margin: 1.5rem auto;
        perspective: 1000px;
        cursor: pointer;
        border-radius: 24px;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        border: 1px solid var(--border-color);
        transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }}
    
    .flashcard-main:hover {{
        transform: scale(1.02);
    }}
    
    .flashcard-term {{
        font-size: 1.5rem;
        font-weight: 700;
        color: #4A6CF7;
    }}
    
    .flashcard-def {{
        font-size: 1.1rem;
        color: var(--text-color);
        line-height: 1.6;
    }}
    
    /* Custom Buttons */
    .stButton>button {{
        border-radius: 8px !important;
        font-weight: 500 !important;
        transition: all 0.2s !important;
    }}

    /* Streamlit Widget Overrides for Dark Mode */
    div[data-testid="stExpander"] {{
        background-color: var(--card-bg) !important;
        border: 1px solid var(--border-color) !important;
    }}
    
    /* Fix for status boxes visibility */
    div[data-testid="stNotification"] p {{
        color: #111827 !important;
    }}
    
    /* Hide default streamlit elements */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    </style>
    """, unsafe_allow_html=True)

# --- UTILS ---
def _check_ollama():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        if r.status_code == 200:
            models = [m["name"].split(":")[0] for m in r.json().get("models", [])]
            return True, models
        return False, []
    except Exception:
        return False, []

# --- INITIALIZATION ---
try:
    if 'engine' not in st.session_state:
        st.session_state.engine = NoteVaultEngine()
except Exception as e:
    st.error(f"Critical System Error: Failed to initialize AI Engine. {e}")
    st.info("Check if 'data' directory is writable and all dependencies are installed.")
    st.stop()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'quiz_data' not in st.session_state:
    st.session_state.quiz_data = []
if 'quiz_results' not in st.session_state:
    st.session_state.quiz_results = {}
if 'flashcards' not in st.session_state:
    st.session_state.flashcards = []
if 'page' not in st.session_state:
    st.session_state.page = "Dashboard"
if 'fc_index' not in st.session_state:
    st.session_state.fc_index = 0
if 'fc_flipped' not in st.session_state:
    st.session_state.fc_flipped = False
if 'view_page' not in st.session_state:
    st.session_state.view_page = 1
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True
if 'active_filename' not in st.session_state:
    st.session_state.active_filename = None

# --- SIDEBAR GLOBAL ACTIONS ---
with st.sidebar:
    st.markdown('<div class="sidebar-brand"><span>📝</span> NoteVault AI</div>', unsafe_allow_html=True)
    st.info("System optimized for handwritten notes. Use the tabs to navigate your study vault.")
    
    st.divider()
    
    # Global Status Indicator
    ollama_running, _ = _check_ollama()
    if ollama_running:
        st.write("🟢 **Core Engine:** Online")
    else:
        st.write("🔴 **Core Engine:** Local Only")
    
    st.divider()
    
    # Quick Actions
    if st.button("🗑 Reset Session", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# --- MAIN APP INTERFACE (TAB BASED) ---
tabs = st.tabs(["📊 Dashboard", "📂 Vault", "💬 Study AI", "🎯 Practice", "📄 Session"])

with tabs[0]:
    c_main, c_side = st.columns([3, 1.2])
    
    with c_main:
        st.title("Hi, Scholar! 👋")
        st.markdown("Your neural vault is active and ready.")
        
        # Stats Grid - More Compact
        st.markdown('<div class="stats-container">', unsafe_allow_html=True)
        s1, s2, s3 = st.columns(3)
        with s1:
            st.markdown(f'<div class="stats-card"><div class="stats-value">{len(st.session_state.engine.processed_pages)}</div><div class="stats-label">Pages</div></div>', unsafe_allow_html=True)
        with s2:
            st.markdown(f'<div class="stats-card"><div class="stats-value">{len(st.session_state.chat_history)}</div><div class="stats-label">Queries</div></div>', unsafe_allow_html=True)
        with s3:
            st.markdown(f'<div class="stats-card"><div class="stats-value">{len(st.session_state.flashcards)}</div><div class="stats-label">Cards</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.session_state.processed:
            st.success(f"**Live Document:** {st.session_state.active_filename}")
        else:
            st.warning("No notes uploaded yet. Check the **Vault** tab.")
            
    with c_side:
        st.markdown("### ⚡ Quick Actions")
        if st.button("💬 Jump to Chat", use_container_width=True):
            # We'd need state triggers for tab switching, 
            # for now let's use labels
            st.info("Switch to the 'Study AI' tab above.")
        if st.button("🎯 Take a Quiz", use_container_width=True):
            st.info("Switch to the 'Practice' tab above.")

with tabs[1]:
    v_col1, v_col2 = st.columns([1, 1.5])
    
    with v_col1:
        st.subheader("🛠 Vault Control")
        
        # Reset Logic
        if st.session_state.get('reset_vault'):
            st.session_state.processed = False
            st.session_state.active_filename = None
            st.session_state.chat_history = []
            st.session_state.reset_vault = False
            st.rerun()

        if not st.session_state.processed:
            st.markdown("Select a handwritten PDF to begin analysis.")
            # Use a more prominent label
            uploaded_file = st.file_uploader("Drop PDF here", type=["pdf"], key="pdf_main_uploader")
            
            if uploaded_file is not None:
                st.session_state.active_filename = uploaded_file.name
                
                # Immediate processing button
                if st.button("🚀 Process & Index Document", type="primary", use_container_width=True):
                    try:
                        with st.status("🛠 Pipeline: Building Knowledge Vault...", expanded=True) as status:
                            st.write("💾 Saving file contents...")
                            temp_path = "data/input_notes.pdf"
                            with open(temp_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            
                            st.write("📸 Page Generation...")
                            from preprocessing import pdf_to_images
                            pdf_to_images(temp_path)
                            
                            st.write("👁️ Running OCR (this may take a minute)...")
                            st.session_state.engine.process_pdf(temp_path)
                            
                            st.session_state.processed = True
                            status.update(label="✅ Ready!", state="complete", expanded=False)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Analysis Failed: {str(e)}")
                        st.info("Ensure the PDF is not password protected and PaddleOCR models are accessible.")
        else:
            st.success(f"**Indexed:** {st.session_state.active_filename}")
            if st.button("📁 Load Different File", use_container_width=True):
                st.session_state.reset_vault = True
                st.rerun()
        
        st.divider()
        if st.session_state.processed:
            st.markdown("#### 📖 Page Selector")
            num_pages = len(st.session_state.engine.processed_pages)
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                st.session_state.view_page = st.number_input("Page", 1, num_pages, st.session_state.view_page)
            with p_col2:
                zoom = st.slider("Zoom", 400, 1600, 900)

    with v_col2:
        st.subheader("🖼 Document Preview")
        if st.session_state.processed:
            img_path = f"data/temp_images/page_{st.session_state.view_page}.png"
            if os.path.exists(img_path):
                st.image(img_path, width=zoom)
            else:
                st.info("Select a page to view preview.")
        else:
            st.info("Upload and process notes to see preview here.")

with tabs[2]:
    st.subheader("💬 Interactive Learning Assistant")
    if not st.session_state.processed:
        st.error("Please upload notes first.")
    else:
        for chat in st.session_state.chat_history:
            st.markdown(f'<div class="chat-bubble user-msg">{chat["question"]}</div>', unsafe_allow_html=True)
            conf = chat.get('confidence', 0)
            conf_class = 'conf-high' if conf > 75 else ('conf-med' if conf > 40 else 'conf-low')
            st.markdown(f"""
                <div class="chat-bubble ai-msg">
                    <span class="confidence-badge {conf_class}">Confidence: {conf}%</span><br>
                    {chat['answer']}
                </div>
            """, unsafe_allow_html=True)
            if chat.get('sources'):
                with st.expander("📚 Sources"):
                    for s in chat['sources']:
                        st.markdown(f"**Page {s['page']}**: {s['text']}")

        if prompt := st.chat_input("Ask anything about your notes..."):
            st.session_state.engine.load_llm()
            answer, sources, confidence, ref_text = st.session_state.engine.ask(prompt)
            st.session_state.chat_history.append({
                "question": prompt, "answer": answer, "sources": sources,
                "confidence": confidence, "referenced_text": ref_text
            })
            st.rerun()

with tabs[3]:
    pt1, pt2 = st.tabs(["🎯 Quiz", "🗂 Flashcards"])
    
    with pt1:
        st.subheader("🎯 Test Your Knowledge")
        if not st.session_state.processed:
            st.error("Upload notes first.")
        else:
            if st.button("✨ Generate Quiz", key="gen_q", type="primary"):
                with st.spinner("Generating..."):
                    raw = st.session_state.engine.generate_quiz()
                    # (Parser logic condensed)
                    questions = []
                    for b in raw.split("---"):
                        lines = [l.strip() for l in b.strip().split("\n") if l.strip()]
                        if len(lines) >= 6:
                            q_item = {"q": "", "options": {}, "correct": ""}
                            for l in lines:
                                if l.startswith("Q:"): q_item["q"] = l.replace("Q:", "").strip()
                                elif l.startswith("A)"): q_item["options"]["A"] = l.replace("A)", "").strip()
                                elif l.startswith("B)"): q_item["options"]["B"] = l.replace("B)", "").strip()
                                elif l.startswith("C)"): q_item["options"]["C"] = l.replace("C)", "").strip()
                                elif l.startswith("D)"): q_item["options"]["D"] = l.replace("D)", "").strip()
                                elif l.startswith("Correct:"): q_item["correct"] = l.replace("Correct:", "").strip().upper()[:1]
                            if q_item["q"] and q_item["correct"]: questions.append(q_item)
                    st.session_state.quiz_data = questions
                    st.session_state.quiz_results = {}
                    st.rerun()
            
            if st.session_state.quiz_data:
                for i, q in enumerate(st.session_state.quiz_data):
                    st.markdown(f"**Q{i+1}: {q['q']}**")
                    cols = st.columns(2)
                    for idx, char in enumerate(["A", "B", "C", "D"]):
                        if cols[idx % 2].button(f"{char}) {q['options'].get(char, '...')}", key=f"q{i}_{char}_t"):
                            st.session_state.quiz_results[i] = char
                            st.rerun()
                    if i in st.session_state.quiz_results:
                        if st.session_state.quiz_results[i] == q['correct']: st.success("Correct!")
                        else: st.error(f"Correct answer: {q['correct']}")
                    st.divider()

    with pt2:
        st.subheader("🗂 Active Recall Flashcards")
        if not st.session_state.processed:
            st.error("Upload notes first.")
        else:
            if not st.session_state.flashcards:
                if st.button("✨ Generate Cards", key="gen_f"):
                    st.session_state.flashcards = st.session_state.engine.extract_flashcards()
                    st.rerun()
            
            if st.session_state.flashcards:
                card = st.session_state.flashcards[st.session_state.fc_index]
                st.markdown(f"""
                    <div class="flashcard-main">
                        <div class="{'flashcard-term' if not st.session_state.fc_flipped else 'flashcard-def'}">
                            {card['front'] if not st.session_state.fc_flipped else card['back']}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                c1, c2, c3 = st.columns([1, 1, 1])
                if c2.button("Flip Card", use_container_width=True, key="flip_t"):
                    st.session_state.fc_flipped = not st.session_state.fc_flipped
                    st.rerun()
                if c1.button("⬅️ Prev", key="prev_f") and st.session_state.fc_index > 0:
                    st.session_state.fc_index -= 1
                    st.session_state.fc_flipped = False
                    st.rerun()
                if c3.button("Next ➡️", key="next_f") and st.session_state.fc_index < len(st.session_state.flashcards) - 1:
                    st.session_state.fc_index += 1
                    st.session_state.fc_flipped = False
                    st.rerun()

with tabs[4]:
    st.subheader("⚙️ Session Controls")
    c1, c2 = st.columns(2)
    with c1:
        st.write("### 📄 Reporting")
        if st.button("📥 Download PDF Study Report", use_container_width=True):
            path = generate_pdf_report(st.session_state.chat_history)
            with open(path, "rb") as f: st.download_button("Click to Download", f, "NoteVault_Report.pdf")
    with c2:
        st.write("### 🧠 Preferences")
        running, models = _check_ollama()
        if running:
            choice = st.selectbox("LLM Model", models if models else ["mistral"])
            st.session_state.engine.load_llm(choice)
        else:
            st.warning("Ollama not running. Fallback active.")
