from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
import datetime
import os

def generate_pdf_report(history, output_path="data/study_report.pdf"):
    """
    history: list of {question, answer, sources}
    """
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "NoteVault AI: Study Session Report")
    c.setFont("Helvetica", 10)
    c.drawString(50, height - 70, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    y = height - 110
    for i, item in enumerate(history):
        if y < 100: # New page
            c.showPage()
            y = height - 50
        
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, f"Q{i+1}: {item['question']}")
        y -= 20
        
        c.setFont("Helvetica", 10)
        # Handle long text (simple wrap)
        text_obj = c.beginText(50, y)
        text_obj.textLines(item['answer'][:500] + "...") # Truncate for report simplicity
        c.drawText(text_obj)
        
        y -= 60
        c.setFont("Helvetica-Oblique", 9)
        source_pages = list(set([str(s['page']) for s in item['sources']]))
        c.drawString(50, y, f"Sources: Pages {', '.join(source_pages)}")
        y -= 40
        
    c.save()
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
