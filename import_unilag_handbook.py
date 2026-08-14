from app import create_app
from models import db, Document, FAQ
from utils import extract_text_from_file, generate_faqs_from_text, rebuild_local_index, upsert_embedding


HANDOUT_PATH = "unilag_handbook_2026_2027.md"
TITLE = "UNILAG Student Handbook 2026/2027"
CATEGORY = "handbook"

CURATED_FAQS = [
    ("What courses are offered here?", "Course offerings depend on the faculty, department, and academic session. Check the official course catalogue, department page, or student portal for the current list. If you tell me your faculty, department, or level, I can narrow it down.", "academics"),
    ("How do I find the course catalog?", "Use the official department page, faculty notice board, or student portal for the current course list and registration guidance. For exact course codes and prerequisites, always check the department's approved catalog.", "academics"),
    ("What is the hostel application window?", "The hostel application window runs from Monday, August 17, 2026 to Monday, August 31, 2026 based on the current calendar provided.", "hostel"),
    ("What are the hostel rules?", "Hostel bed spaces are non-transferable. Squatting, subletting, illegal appliances, cooking inside bedrooms, unauthorized modifications, and noise violations are prohibited. Most halls also enforce a night curfew.", "hostel"),
    ("How much is the hostel fee?", "General hostel bed spaces cost about N80,000 per session. Premium options such as Sodeinde Hall, Moremi Hall Extension, and Landmark Student Hostel cost more.", "hostel"),
    ("What are the library opening hours?", "During term time the library opens Monday to Saturday from 8:00 AM to 10:00 PM. During vacation periods it opens Monday to Saturday from 8:00 AM to 6:00 PM. Faculty and departmental libraries usually open Monday to Friday from 8:00 AM to 4:00 PM.", "library"),
    ("What are the admission requirements?", "You need a minimum UTME score of 200, at least five credits including English Language and Mathematics, first-choice admission selection, and you must be at least 16 years old by September 30, 2026.", "admissions"),
    ("When is the O-Level upload deadline?", "The O-Level upload deadline is August 19, 2026. Candidates must upload results on both JAMB CAPS and the UNILAG portal.", "admissions"),
    ("What are the undergraduate fees?", "Acceptance charges are approximately N45,000 for fresh students. Faculty of Arts baseline fees are roughly N51,500, while science and lab-based courses may go up to N222,500.", "fees"),
    ("How do I pay school fees?", "Payment is electronic only through Remita on the student portal. Cash payment is not accepted.", "fees"),
    ("What is the grading scale?", "The university uses a 5.0 CGPA system. A is 5.00, B is 4.00, C is 3.00, D is 2.00, E is 1.00, and F is 0.00.", "academics"),
    ("What is the attendance requirement?", "Students must attend at least 65% of lectures for a course to be eligible to sit for the exam.", "academics"),
    ("What are the exam rules?", "Phones, smartwatches, and programmable calculators are banned in exam halls. Late arrival beyond 30 minutes may prevent entry, and students may not leave during the first hour or final 15 minutes.", "academics"),
    ("Who is eligible for hostel balloting?", "Freshmen and final-year students are given priority during hostel balloting.", "hostel"),
    ("When must hostel payment be completed?", "Once a bed space is allocated, payment usually must be completed within 3 to 5 days or the space will be forfeited.", "hostel"),
    ("What student welfare support is available?", "Student Affairs coordinates work-study support, installment payment options, and welfare programs for students facing hardship or disability-related challenges.", "student support"),
    ("What are the course areas in the Faculty of Engineering?", "Engineering students usually study mathematics, physics, design, coding, circuits, structures, laboratory work, and SIWES preparation. The exact catalog depends on the department.", "academics"),
    ("What are the course areas in the Faculty of Science?", "Science students usually study laboratory practicals, research methods, calculations, report writing, and scientific analysis. Exact courses vary by department.", "academics"),
    ("What are the course areas in the Faculty of Social Sciences?", "Social Sciences students usually study research methods, statistics, writing, analysis, and field-based learning. Exact courses vary by department.", "academics"),
    ("What are the course areas in the Faculty of Management Sciences?", "Management Sciences students usually study finance, management, records, presentation skills, ethics, and applied business studies.", "academics"),
    ("What are the course areas in the College of Medicine?", "Medicine students usually study anatomy, clinical posting, laboratory training, patient care, and professional conduct. Exact course requirements vary by program.", "academics"),
    ("What are the course areas in the Faculty of Arts and Education?", "Arts and Education students usually study reading, writing, lesson planning, seminars, teaching practice, and creative expression.", "academics"),
    ("What are the engineering faculty rules?", "Engineering students should follow lab safety rules, complete practical sessions, and confirm SIWES and equipment requirements with the department.", "academics"),
    ("What are the science faculty rules?", "Science students should follow strict lab safety rules, attend practical sessions, and complete lab reports and field work when required.", "academics"),
    ("What are the medicine faculty rules?", "Medical students should maintain professionalism, punctuality, and strong attendance during lectures, practicals, and clinical postings.", "academics"),
]


def main():
    app = create_app()
    with app.app_context():
        with open(HANDOUT_PATH, "r", encoding="utf-8") as f:
            content = f.read().strip()

        if not content:
            raise RuntimeError("Handbook file is empty")

        existing = Document.query.filter(Document.title == TITLE).first()
        if existing:
            existing.content = content
            existing.category = CATEGORY
            doc = existing
        else:
            doc = Document(title=TITLE, content=content, category=CATEGORY, is_active=True)
            db.session.add(doc)

        db.session.commit()
        upsert_embedding("document", doc.id, f"{doc.title}\n{doc.content}")

        curated_added = 0
        curated_updated = 0
        for question, answer, category in CURATED_FAQS:
            exists = FAQ.query.filter(FAQ.question.ilike(question)).first()
            if exists:
                if (exists.answer or "").strip() != answer.strip() or (exists.category or "") != category:
                    exists.answer = answer
                    exists.category = category
                    curated_updated += 1
                continue
            db.session.add(FAQ(question=question, answer=answer, category=category, is_active=True))
            curated_added += 1

        # Generate a modest FAQ set from the handbook for local retrieval.
        faqs = generate_faqs_from_text(content, max_faqs=40, category="handbook")
        added = 0
        for item in faqs:
            question = (item.get("question") or "").strip()
            answer = (item.get("answer") or "").strip()
            if not question or not answer:
                continue
            exists = FAQ.query.filter(FAQ.question.ilike(question)).first()
            if exists:
                continue
            faq = FAQ(question=question, answer=answer, category=item.get("category", "handbook"), is_active=True)
            db.session.add(faq)
            added += 1

        db.session.commit()
        rebuild_local_index()

        print(f"Saved handbook document: {doc.id}")
        print(f"Added curated FAQ rows: {curated_added}")
        print(f"Updated curated FAQ rows: {curated_updated}")
        print(f"Added FAQ rows: {added}")


if __name__ == "__main__":
    main()
