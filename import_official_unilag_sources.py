import html as html_lib
import os
import re
import tempfile
from pathlib import Path

import requests
from pypdf import PdfReader

from app import create_app
from models import db, Document, FAQ
from utils import generate_faqs_from_text, rebuild_local_index


PROGRAMMES_URL = "https://admissions.unilag.edu.ng/programmes.html"
HANDBOOK_ITEM_URL = "https://ir.unilag.edu.ng/items/99034da5-7643-4033-b499-173b9a8cb02a/full"
COURSE_PAGES = [
    "https://dlilearn.unilag.edu.ng/course/info.php?id=587",  # MAT203 - Abstract Algebra 1
    "https://dlilearn.unilag.edu.ng/course/info.php?id=65",   # ENG121 - English Grammar Usage, Lexis and Structure
    "https://dlilearn.unilag.edu.ng/course/info.php?id=21",   # FBA311 - Bus Info. Tech.
    "https://dlilearn.unilag.edu.ng/course/info.php?id=255",  # SOC111 - Introduction to Sociology
    "https://dlilearn.unilag.edu.ng/course/info.php?id=276",  # SOC212 - Fundamentals of Social Statistics
    "https://dlilearn.unilag.edu.ng/course/info.php?id=207",  # BUS120 - Introduction to Management
    "https://dlilearn.unilag.edu.ng/course/info.php?id=574",  # ASE450 - Curriculum in English
    "https://dlilearn.unilag.edu.ng/course/info.php?id=290",  # PAD423 - Office Practice and Administration
    "https://dlilearn.unilag.edu.ng/course/info.php?id=203",  # BUS251 - Principles and Practices of Management
    "https://dlilearn.unilag.edu.ng/course/info.php?id=337",  # ACC320 - Management Accounting 1
    "https://dlilearn.unilag.edu.ng/course/info.php?id=35",   # ENG211 - Phonetics & Phonology
]


def _clean_html_to_text(raw_html):
    text = re.sub(r"(?is)<(script|style).*?>.*?</\\1>", " ", raw_html)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</(p|div|li|tr|h[1-6])>", "\n", text)
    text = re.sub(r"(?i)<li[^>]*>", "\n- ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html_lib.unescape(text)
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines).strip()


def _fetch(url, timeout=45):
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp


def _store_document(title, content, category="official", source_url=None):
    if source_url:
        content = f"Source: {source_url}\n\n{content}"
    existing = Document.query.filter(Document.title == title).first()
    if existing:
        existing.content = content
        existing.category = category
        doc = existing
    else:
        doc = Document(title=title, content=content, category=category, is_active=True)
        db.session.add(doc)
        db.session.flush()
    return doc


def _extract_pdf_text_limited(pdf_path, max_pages=25):
    reader = PdfReader(pdf_path)
    parts = []
    for page in reader.pages[:max_pages]:
        parts.append(page.extract_text() or "")
    return "\n".join(parts).strip()


def _extract_course_outline_text(raw_html):
    title_match = re.search(r"<title>(.*?)</title>", raw_html, re.IGNORECASE | re.DOTALL)
    title = html_lib.unescape(title_match.group(1)).strip() if title_match else "UNILAG Course Outline"

    summary_match = re.search(
        r'<div class="summary">.*?<div class="no-overflow">(.*?)</div>.*?<div class="teachers">(.*?)</div>',
        raw_html,
        re.IGNORECASE | re.DOTALL,
    )
    if summary_match:
        summary_html = summary_match.group(1)
        teacher_html = summary_match.group(2)
        summary_text = _clean_html_to_text(summary_html)
        teacher_text = _clean_html_to_text(teacher_html)
        return f"{title}\n\n{summary_text}\n\n{teacher_text}".strip()

    return _clean_html_to_text(raw_html)


def _generate_and_store_faqs(source_text, category, limit=30):
    created = 0
    faqs = generate_faqs_from_text(source_text, max_faqs=limit, category=category)
    for item in faqs:
        question = (item.get("question") or "").strip()
        answer = (item.get("answer") or "").strip()
        if not question or not answer:
            continue
        with db.session.no_autoflush:
            exists = FAQ.query.filter(FAQ.question.ilike(question)).first()
            if exists:
                if (exists.answer or "").strip() != answer or (exists.category or "") != category:
                    exists.answer = answer
                    exists.category = category
                continue
            db.session.add(FAQ(question=question, answer=answer, category=category, is_active=True))
            created += 1
    return created


def import_programmes_page():
    response = _fetch(PROGRAMMES_URL)
    text = _clean_html_to_text(response.text)

    # Break out the main faculty lists into a compact catalog document.
    lines = text.splitlines()
    faculty_sections = []
    current = None
    current_lines = []
    for line in lines:
        line_upper = line.upper()
        if line_upper.startswith("FACULTY OF ") or line_upper.startswith("COLLEGE OF ") or line_upper.startswith("SCHOOL OF "):
            if current and current_lines:
                faculty_sections.append((current, "\n".join(current_lines).strip()))
            current = line.strip()
            current_lines = [line]
        elif current:
            current_lines.append(line)
    if current and current_lines:
        faculty_sections.append((current, "\n".join(current_lines).strip()))

    doc = _store_document(
        "UNILAG Undergraduate Programmes (Official)",
        text,
        category="official-programmes",
        source_url=PROGRAMMES_URL,
    )

    # Add focused faculty documents for better retrieval.
    for faculty, section_text in faculty_sections:
        _store_document(
            f"UNILAG {faculty.title()}",
            section_text,
            category="official-programmes",
            source_url=PROGRAMMES_URL,
        )
        _generate_and_store_faqs(section_text, category="official-programmes", limit=8)

    return doc, len(faculty_sections)


def import_handbook_pdfs():
    response = _fetch(HANDBOOK_ITEM_URL)
    pdf_urls = re.findall(r'https://ir\.unilag\.edu\.ng/bitstreams/[^"]+/download', response.text)
    pdf_urls = list(dict.fromkeys(pdf_urls))
    imported = []

    for pdf_url in pdf_urls[:3]:
        try:
            pdf_resp = _fetch(pdf_url, timeout=120)
            filename = pdf_url.rstrip("/").split("/")[-2] + ".pdf"
            safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "_", filename)
            with tempfile.TemporaryDirectory() as tmpdir:
                pdf_path = os.path.join(tmpdir, safe_name)
                with open(pdf_path, "wb") as f:
                    f.write(pdf_resp.content)
                text = _extract_pdf_text_limited(pdf_path, max_pages=25)
                if not text.strip():
                    continue
                title = f"UNILAG Handbook PDF {safe_name}"
                _store_document(
                    title,
                    text,
                    category="official-handbook",
                    source_url=pdf_url,
                )
                faq_count = _generate_and_store_faqs(text, category="official-handbook", limit=25)
                imported.append((title, faq_count))
        except Exception:
            continue

    return imported


def import_course_pages():
    imported = []
    for url in COURSE_PAGES:
        try:
            resp = _fetch(url)
            text = _extract_course_outline_text(resp.text)
            if not text.strip():
                continue
            title_match = re.search(r"<title>(.*?)</title>", resp.text, re.IGNORECASE | re.DOTALL)
            title = html_lib.unescape(title_match.group(1)).strip() if title_match else url
            title = re.sub(r"\s+", " ", title)
            doc = _store_document(
                f"UNILAG DLI {title}",
                text,
                category="official-course-outline",
                source_url=url,
            )
            faq_count = _generate_and_store_faqs(text, category="official-course-outline", limit=12)
            imported.append((doc.title, faq_count))
        except Exception:
            continue
    return imported


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Import official UNILAG source material.")
    parser.add_argument("--include-pdfs", action="store_true", help="Also attempt to import handbook PDF files.")
    args = parser.parse_args()

    app = create_app()
    with app.app_context():
        prog_doc, faculty_count = import_programmes_page()
        course_imports = import_course_pages()
        handbook_imports = import_handbook_pdfs() if args.include_pdfs else []

        db.session.commit()
        rebuild_local_index()

        print(f"Imported official programmes doc: {prog_doc.title}")
        print(f"Faculty sections added: {faculty_count}")
        print(f"Handbook PDFs imported: {len(handbook_imports)}")
        for title, faq_count in handbook_imports:
            print(f"  - {title}: {faq_count} FAQs")
        print(f"Course pages imported: {len(course_imports)}")
        for title, faq_count in course_imports:
            print(f"  - {title}: {faq_count} FAQs")


if __name__ == "__main__":
    main()
