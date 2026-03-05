from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
import datetime
import os

def generate_pdf_report(history, output_path="data/study_report.pdf"):
    if not os.path.exists("data"):
        os.makedirs("data")
    doc = SimpleDocTemplate(output_path, pagesize=letter, topMargin=50, bottomMargin=50)
    styles = getSampleStyleSheet()
    PRIMARY = colors.HexColor("#4A6CF7")
    TEXT_MAIN = colors.HexColor("#111827")
    TEXT_SEC = colors.HexColor("#6B7280")
    BG_REF = colors.HexColor("#F9FAFB")
    title_style = ParagraphStyle('Title', parent=styles['Title'], textColor=PRIMARY, fontSize=24, spaceAfter=20, fontName='Helvetica-Bold')
    quest_style = ParagraphStyle('Quest', parent=styles['Heading3'], textColor=TEXT_MAIN, spaceBefore=20, fontName='Helvetica-Bold')
    ans_style = ParagraphStyle('Ans', parent=styles['Normal'], textColor=TEXT_MAIN, spaceAfter=12, leading=14)
    meta_style = ParagraphStyle('Meta', parent=styles['Italic'], fontSize=9, textColor=TEXT_SEC)
    ref_style = ParagraphStyle('Ref', parent=styles['Normal'], fontSize=9, leftIndent=25, rightIndent=25, 
                               backColor=BG_REF, borderPadding=10, borderStrokeColor=colors.lightgrey, 
                               borderStrokeWidth=0.5, textColor=TEXT_SEC)
    story = []
    story.append(Paragraph("NoteVault AI: Knowledge Insights", title_style))
    story.append(Paragraph(f"Study Session Analysis &bull; {datetime.datetime.now().strftime('%B %d, %Y')}", meta_style))
    story.append(Spacer(1, 25))
    story.append(HRFlowable(width="100%", thickness=1.5, color=PRIMARY))
    story.append(Spacer(1, 15))
    if not history:
        story.append(Paragraph("No interactive data found for this session.", styles['Normal']))
    else:
        for i, item in enumerate(history):
            story.append(Paragraph(f"{i+1}. {item['question']}", quest_style))
            story.append(Paragraph(f"<b>AI Insight:</b> {item['answer']}", ans_style))
            source_pages = list(set([str(s['page']) for s in item.get('sources', [])]))
            conf = item.get('confidence', 0)
            meta_info = (f"<b>Reliability Score:</b> {conf}% &nbsp;&nbsp; | &nbsp;&nbsp; "
                         f"<b>Source Pages:</b> {', '.join(source_pages) if source_pages else 'N/A'}")
            story.append(Paragraph(meta_info, meta_style))
            if item.get('referenced_text'):
                story.append(Spacer(1, 8))
                story.append(Paragraph(f'<i>&ldquo;{item["referenced_text"]}&rdquo;</i>', ref_style))
            story.append(Spacer(1, 10))
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
    doc.build(story)
    return output_path

def generate_transcription_pdf(processed_pages, output_path="data/transcribed_notes.pdf"):
    if not os.path.exists("data"):
        os.makedirs("data")
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("NoteVault AI: Transcribed Notes", styles['Title']))
    story.append(Paragraph(f"Extracted on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 20))
    for page in processed_pages:
        story.append(Paragraph(f"--- Page {page['page']} ---", styles['Heading2']))
        story.append(Spacer(1, 10))
        text = page['text'].replace('\n', '<br/>')
        if not text.strip():
            text = "<i>[No text extracted from this page]</i>"
        story.append(Paragraph(text, styles['Normal']))
        story.append(Spacer(1, 20))
    doc.build(story)
    return output_path
