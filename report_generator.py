from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
import datetime
import os

def generate_pdf_report(history, output_path="data/study_report.pdf"):
    """
    history: list of {question, answer, sources, confidence, referenced_text}
    """
    if not os.path.exists("data"):
        os.makedirs("data")

    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Custom Styles
    quest_style = ParagraphStyle('Quest', parent=styles['Heading3'], textColor=colors.navy)
    ans_style = ParagraphStyle('Ans', parent=styles['Normal'], spaceAfter=10)
    meta_style = ParagraphStyle('Meta', parent=styles['Italic'], fontSize=9, textColor=colors.grey)
    ref_style = ParagraphStyle('Ref', parent=styles['Normal'], fontSize=9, leftIndent=20, borderPadding=5, 
                               backColor=colors.whitesmoke, borderStrokeColor=colors.lightgrey, borderStrokeWidth=0.5)

    story = []

    # Title Section
    story.append(Paragraph("NoteVault AI: Study Session Report", styles['Title']))
    story.append(Paragraph(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.black))
    story.append(Spacer(1, 20))

    if not history:
        story.append(Paragraph("No questions asked in this session.", styles['Normal']))
    else:
        for i, item in enumerate(history):
            # Question
            story.append(Paragraph(f"Question {i+1}: {item['question']}", quest_style))
            story.append(Spacer(1, 5))
            
            # Answer
            story.append(Paragraph(f"<b>Answer:</b> {item['answer']}", ans_style))
            
            # Confidence & Source Meta
            source_pages = list(set([str(s['page']) for s in item.get('sources', [])]))
            meta_info = (f"<b>Confidence:</b> {item.get('confidence', 0)}% &nbsp; | &nbsp; "
                         f"<b>Source Pages:</b> {', '.join(source_pages) if source_pages else 'N/A'}")
            story.append(Paragraph(meta_info, meta_style))
            story.append(Spacer(1, 10))
            
            # Referenced Text
            if item.get('referenced_text'):
                story.append(Paragraph("<i>Key Referenced Text:</i>", meta_style))
                story.append(Paragraph(f'"{item["referenced_text"]}"', ref_style))
            
            story.append(Spacer(1, 20))
            story.append(HRFlowable(width="80%", thickness=0.5, color=colors.lightgrey, dash=(2,2)))
            story.append(Spacer(1, 20))

    doc.build(story)
    return output_path

def generate_transcription_pdf(processed_pages, output_path="data/transcribed_notes.pdf"):
    """
    Creates a formatted PDF of all extracted text from the notes.
    processed_pages: list of {page, text}
    """
    if not os.path.exists("data"):
        os.makedirs("data")

    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph("NoteVault AI: Transcribed Notes", styles['Title']))
    story.append(Paragraph(f"Extracted on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 20))

    for page in processed_pages:
        # Page Header
        story.append(Paragraph(f"--- Page {page['page']} ---", styles['Heading2']))
        story.append(Spacer(1, 10))
        
        # Extracted Text
        text = page['text'].replace('\n', '<br/>')  # Basic formatting
        if not text.strip():
            text = "<i>[No text extracted from this page]</i>"
        
        story.append(Paragraph(text, styles['Normal']))
        story.append(Spacer(1, 20))

    doc.build(story)
    return output_path
