from openai import OpenAI
import re
import os
import json
import difflib
from flask import current_app
from flask_login import current_user
from datetime import datetime
from models import db, FAQ, Policy, Document, Calendar, KnowledgeBaseEntry, Embedding
from sqlalchemy import or_

client = None
last_openai_error = None
local_index = {
    "built_at": None,
    "docs": [],
    "vocab": {},
    "idf": {},
}

_QUERY_EXPANSIONS = {
    "fee": ["tuition", "payment", "school fees", "charges"],
    "fees": ["tuition", "payment", "school fees", "charges"],
    "tuition": ["fees", "payment", "school fees"],
    "id": ["student id", "id card", "identity card", "student card"],
    "student id": ["student id card", "id card", "identity card"],
    "registration": ["course registration", "enrollment", "register"],
    "register": ["course registration", "enrollment", "add/drop"],
    "admission": ["admissions", "entry requirements", "entry"],
    "hostel": ["accommodation", "residence", "housing"],
    "library": ["library hours", "borrowing", "study space"],
    "scholarship": ["financial aid", "bursary", "sponsorship"],
    "result": ["results", "grades", "transcript"],
    "exam": ["exams", "assessment", "test"],
    "attendance": ["class attendance", "presence", "lecture attendance"],
    "clearance": ["clearance letter", "graduation clearance", "registration clearance"],
    "transcript": ["result slip", "academic transcript", "grades"],
    "complaint": ["report issue", "student support", "help desk"],
    "portal": ["student portal", "login", "account"],
}

def init_openai(config):
    global client
    global last_openai_error
    api_key = config.get('OPENAI_API_KEY')
    if api_key:
        api_key = api_key.strip().strip('"').strip("'")

    if api_key and api_key != 'your_openai_key_here' and api_key.startswith('sk-'):
        try:
            # Preferred initialization for modern OpenAI SDK
            client = OpenAI(api_key=api_key)
            print(f"[DEBUG] OpenAI client initialized successfully with key: {api_key[:20]}...")
            last_openai_error = None
        except TypeError as e:
            # Handle older HSOD or httpx compatibility where `proxies` may be present
            err = str(e).lower()
            if 'proxies' in err or 'http_client' in err:
                try:
                    import httpx
                    # Force no environment proxy use for httpx to avoid proxies keyword mismatch
                    http_client = httpx.Client(trust_env=False)
                    client = OpenAI(api_key=api_key, http_client=http_client)
                    print(f"[DEBUG] OpenAI client initialized in fallback mode with custom http_client: {api_key[:20]}...")
                    last_openai_error = None
                except Exception as inner_e:
                    print(f"[DEBUG] Fallback OpenAI init failed: {type(inner_e).__name__}: {inner_e}")
                    try:
                        from flask import current_app
                        current_app.logger.error(f"Fallback OpenAI init failed: {type(inner_e).__name__}: {inner_e}")
                    except:
                        pass
                    client = None
                    last_openai_error = str(inner_e)
            else:
                print(f"[DEBUG] Failed to initialize OpenAI client: {type(e).__name__}: {e}")
                try:
                    from flask import current_app
                    current_app.logger.error(f"Failed to initialize OpenAI client: {type(e).__name__}: {e}")
                except:
                    pass
                client = None
                last_openai_error = str(e)
        except Exception as e:
            print(f"[DEBUG] Failed to initialize OpenAI client: {type(e).__name__}: {e}")
            try:
                from flask import current_app
                current_app.logger.error(f"Failed to initialize OpenAI client: {type(e).__name__}: {e}")
            except:
                pass
            client = None
            last_openai_error = str(e)
    else:
        print(f"[DEBUG] OpenAI API key not set or invalid: {bool(api_key)}")
        client = None  # Fallback no raise
        last_openai_error = "OpenAI API key not set or invalid"


EMBEDDING_MODEL = "text-embedding-3-small"


def _truncate(text, max_len=2000):
    text = (text or "").strip()
    return text[:max_len]


def _tokenize(text):
    text = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
    text = re.sub(r"\s+", " ", text).strip()
    tokens = set()
    for t in text.split():
        if len(t) <= 2:
            continue
        tokens.add(t)
        if t.endswith('ies') and len(t) > 3:
            tokens.add(t[:-3] + 'y')
        elif t.endswith('es') and len(t) > 3:
            tokens.add(t[:-2])
            tokens.add(t[:-1])
        elif t.endswith('s') and not t.endswith('ss') and len(t) > 3:
            tokens.add(t[:-1])
        elif t.endswith('ing') and len(t) > 5:
            tokens.add(t[:-3])
        elif t.endswith('ed') and len(t) > 4:
            tokens.add(t[:-2])
    return list(tokens)


def _best_snippet(text, query, max_len=800):
    """Pick the most relevant paragraph from a long document."""
    if not text:
        return ""
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if not paragraphs:
        return text[:max_len]

    q_tokens = set(_tokenize(query))
    if not q_tokens:
        return paragraphs[0][:max_len]

    best = None
    best_score = -1
    for p in paragraphs:
        p_tokens = _tokenize(p)
        if not p_tokens:
            continue
        overlap = len(q_tokens.intersection(p_tokens))
        # favor paragraphs that include query terms
        score = overlap + (0.1 * len(p_tokens))
        if score > best_score:
            best_score = score
            best = p
    if best:
        return best[:max_len]
    return paragraphs[0][:max_len]


def _sectioned_snippet(text, query, max_len=800):
    """Split by headings and pick best section by query overlap."""
    if not text:
        return ""
    # Detect headings by common patterns
    lines = [ln.rstrip() for ln in text.splitlines()]
    sections = []
    current_title = None
    current_body = []
    for ln in lines:
        stripped = ln.strip()
        if not stripped:
            continue
        is_heading = (
            stripped.isupper() or
            stripped.endswith(":") or
            re.match(r"^\d+(\.\d+)*\s+.+", stripped)
        )
        if is_heading and len(stripped.split()) <= 12:
            if current_title or current_body:
                sections.append((current_title or "", "\n".join(current_body)))
            current_title = stripped.strip(":")
            current_body = []
        else:
            current_body.append(stripped)
    if current_title or current_body:
        sections.append((current_title or "", "\n".join(current_body)))

    if not sections:
        return _best_snippet(text, query, max_len=max_len)

    q_tokens = set(_tokenize(query))
    best = None
    best_score = -1
    for title, body in sections:
        combined = f"{title}\n{body}"
        tokens = set(_tokenize(combined))
        overlap = len(q_tokens.intersection(tokens))
        # Boost heading matches
        head_boost = 2 if any(t in _tokenize(title) for t in q_tokens) else 0
        score = overlap + head_boost
        if score > best_score:
            best_score = score
            best = combined
    if not best or best_score <= 0:
        return _best_snippet(text, query, max_len=max_len)
    return best[:max_len]


def _unique_preserve_order(items):
    seen = set()
    ordered = []
    for item in items:
        key = item.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(item)
    return ordered


def _build_query_variants(query, history=None):
    """Expand a user question into close semantic variants."""
    history = history or []
    base = (query or "").strip()
    variants = [base]

    q_lower = base.lower()
    for key, synonyms in _QUERY_EXPANSIONS.items():
        if key in q_lower:
            for synonym in synonyms:
                variants.append(q_lower.replace(key, synonym))

    tokens = [t for t in re.split(r"\s+", q_lower) if t]
    for i, token in enumerate(tokens):
        if token in _QUERY_EXPANSIONS:
            for synonym in _QUERY_EXPANSIONS[token]:
                rebuilt = tokens[:]
                rebuilt[i] = synonym
                variants.append(" ".join(rebuilt))

    if history:
        last_user = ""
        last_bot = ""
        for turn in reversed(history):
            role = turn.get("role")
            content = (turn.get("content") or "").strip()
            if not content:
                continue
            if role == "assistant" and not last_bot:
                last_bot = content
            elif role == "user" and not last_user:
                last_user = content
            if last_user and last_bot:
                break
        follow_up_markers = (
            len(tokens) <= 5
            or q_lower.startswith(("what about", "how about", "and ", "also ", "that ", "it ", "they "))
            or any(word in q_lower for word in ("that", "those", "it", "they", "them", "this", "these"))
        )
        if follow_up_markers:
            if last_bot:
                variants.append(f"{last_bot} {base}")
            if last_user:
                variants.append(f"{last_user} {base}")

    return _unique_preserve_order(variants[:8])


def _knowledge_item_to_record(entry):
    if not entry:
        return None
    text = " ".join([
        entry.question or "",
        entry.title or "",
        entry.answer or "",
        entry.tags or "",
        entry.aliases or "",
        entry.related_questions or "",
    ]).strip()
    return {
        "type": "KNOWLEDGE",
        "snippet": entry.question or entry.title,
        "full_text": entry.answer,
        "category": entry.category,
        "title": entry.title,
        "tags": entry.tags,
        "aliases": entry.aliases,
        "priority": float(entry.priority or 0),
        "text": text,
    }


def _knowledge_source_text(entry):
    parts = [
        entry.title or "",
        entry.question or "",
        entry.answer or "",
        entry.category or "",
        entry.tags or "",
        entry.aliases or "",
    ]
    return "\n".join(part for part in parts if part)


def seed_common_knowledge_base():
    """Populate the richer knowledge table with safe, common student questions."""
    entries = [
        {
            "title": "Student ID Card",
            "question": "How do I get my student ID card?",
            "answer": "Apply through the registry or student affairs office after registration, upload any required photo or clearance details, and collect the card once it is issued. If your school uses a portal, check there first for the exact ID-card process.",
            "category": "records",
            "tags": "student id,id card,identity card,student card",
            "aliases": "id card, identity card, student card",
            "related_questions": "How long does it take to get my ID card?; What do I do if I lose my ID card?",
        },
        {
            "title": "Portal Login",
            "question": "How do I log in to the student portal?",
            "answer": "Use your school email, matric number, or student ID on the official portal. If the password does not work, reset it with the portal recovery link or contact ICT support.",
            "category": "portal",
            "tags": "portal,login,password,ict",
            "aliases": "student portal, sign in, log in",
            "related_questions": "How do I reset my portal password?; Why is my portal account locked?",
        },
        {
            "title": "Portal Password Reset",
            "question": "How do I reset my portal password?",
            "answer": "Use the password reset or forgot-password option on the portal, then follow the email or SMS verification steps. If that fails, contact ICT or the help desk.",
            "category": "portal",
            "tags": "password reset,portal,login",
            "aliases": "forgot password, reset password",
            "related_questions": "How do I log in to the student portal?; Why is my account locked?",
        },
        {
            "title": "Course Registration",
            "question": "How do I register for courses?",
            "answer": "Log in to the student portal, open course registration, select the courses for your level and semester, check prerequisites, and submit before the deadline. Always confirm the final course load with your department.",
            "category": "registration",
            "tags": "registration,courses,add/drop,semester",
            "aliases": "course registration, register courses, enroll courses",
            "related_questions": "When does course registration open?; What happens if I miss the registration deadline?",
        },
        {
            "title": "Late Registration",
            "question": "What happens if I register late?",
            "answer": "Late registration can attract a penalty, require department approval, or limit the courses you can take. If you are already late, contact your department or registry immediately.",
            "category": "registration",
            "tags": "late registration,penalty,deadline",
            "aliases": "register late, missed registration",
            "related_questions": "When is the registration deadline?; How do I get approval for late registration?",
        },
        {
            "title": "Fee Payment",
            "question": "How do I pay my school fees?",
            "answer": "Use the official student portal or the school-approved payment channel, confirm the payment reference, and keep your receipt. If payment does not reflect, contact bursary or finance with proof of payment.",
            "category": "fees",
            "tags": "fees,tuition,payment,bursary,finance",
            "aliases": "school fees, tuition payment, pay fees",
            "related_questions": "What is the fee payment deadline?; How do I confirm payment was successful?",
        },
        {
            "title": "Fee Deadline",
            "question": "When is the fee payment deadline?",
            "answer": "The deadline is usually listed in the academic calendar or payment notice. If you cannot find the exact date, check the portal or ask the finance office for the current session deadline.",
            "category": "fees",
            "tags": "fee deadline,tuition deadline,payment deadline",
            "aliases": "fees deadline, payment date",
            "related_questions": "What happens if I pay late?; How do I pay my school fees?",
        },
        {
            "title": "Scholarships",
            "question": "What scholarships or financial aid are available?",
            "answer": "Scholarships and bursaries are usually announced by the university, government, or external sponsors. Check the portal, notice board, or student affairs office for eligibility, deadlines, and application steps.",
            "category": "financial aid",
            "tags": "scholarship,bursary,financial aid,grant",
            "aliases": "scholarships, bursaries, aid",
            "related_questions": "How do I apply for a scholarship?; What documents do I need for financial aid?",
        },
        {
            "title": "Hostel Accommodation",
            "question": "How do I apply for hostel accommodation?",
            "answer": "Apply through the housing or student affairs office and follow the hostel allocation instructions on the portal. Availability is usually limited, so apply early and keep your payment and clearance documents ready.",
            "category": "housing",
            "tags": "hostel,accommodation,housing,residence",
            "aliases": "hostel allocation, residence hall",
            "related_questions": "How much is hostel accommodation?; Can I change my hostel room?",
        },
        {
            "title": "Library Hours",
            "question": "What are the library opening hours?",
            "answer": "Library hours depend on the campus policy and session timetable. Check the library notice board, portal, or the official handbook for the current opening and closing hours.",
            "category": "library",
            "tags": "library hours,library opening time,library",
            "aliases": "library time, opening hours",
            "related_questions": "How do I borrow books from the library?; Can I access the library online?",
        },
        {
            "title": "Transcript Request",
            "question": "How do I request my transcript?",
            "answer": "Request transcripts through the registry or records office, complete the required form, and pay any transcript fee if applicable. Some schools require clearance before releasing transcripts.",
            "category": "records",
            "tags": "transcript,records,certificate,results",
            "aliases": "academic transcript, result slip",
            "related_questions": "How do I get my results?; What documents are required for transcript processing?",
        },
        {
            "title": "Exam Timetable",
            "question": "Where do I find the exam timetable?",
            "answer": "The exam timetable is usually posted on the portal, department notice board, or academic calendar. If your timetable changes, the department or exam office will usually announce the update.",
            "category": "exams",
            "tags": "exam timetable,exam schedule,exams",
            "aliases": "exam schedule, test timetable",
            "related_questions": "What should I bring to the exam hall?; Can I resit a missed exam?",
        },
        {
            "title": "Attendance Requirement",
            "question": "What is the attendance requirement?",
            "answer": "Most schools require students to attend a minimum percentage of lectures and practicals before sitting exams. Check your handbook or department rules for the exact threshold.",
            "category": "attendance",
            "tags": "attendance,lectures,classes,minimum attendance",
            "aliases": "class attendance, lecture attendance",
            "related_questions": "What happens if my attendance is low?; How do I get an attendance excuse?",
        },
        {
            "title": "Dress Code",
            "question": "What is the dress code policy?",
            "answer": "Students are expected to dress decently and responsibly while on campus. Offensive, revealing, or unsafe clothing may be restricted depending on the school policy and the event location.",
            "category": "student life",
            "tags": "dress code,appearance,conduct",
            "aliases": "dress policy, clothing policy",
            "related_questions": "Is there a dress code for exams?; What happens if I break the dress code?",
        },
        {
            "title": "Deferment",
            "question": "How do I apply for deferment of admission?",
            "answer": "Write to the admissions office or registry, state the reason for deferment, and attach any supporting documents. Approval usually depends on the university rules and the reason given.",
            "category": "admissions",
            "tags": "deferment,admission deferment,postpone admission",
            "aliases": "defer admission, postpone admission",
            "related_questions": "How long can admission deferment last?; What documents support deferment?",
        },
        {
            "title": "Admission Requirements",
            "question": "What are the admission requirements?",
            "answer": "Admission requirements usually depend on your program. Check the official admission guide for the minimum grades, entrance exams, subject combinations, and any additional departmental requirements.",
            "category": "admissions",
            "tags": "admission requirements,entry requirements,program admission",
            "aliases": "entry requirements, program requirements",
            "related_questions": "How do I apply for admission?; What subjects are required for my course?",
        },
        {
            "title": "Academic Calendar",
            "question": "Where can I find the academic calendar?",
            "answer": "The academic calendar is usually published by the registrar or academic office on the portal and notice boards. It normally includes registration dates, lecture dates, exams, and holidays.",
            "category": "calendar",
            "tags": "calendar,academic calendar,semester dates",
            "aliases": "school calendar, session calendar",
            "related_questions": "When does the semester start?; When are exams scheduled?",
        },
        {
            "title": "Results",
            "question": "How do I check my results?",
            "answer": "Results are usually available through the student portal, department office, or exam office after grading is complete. If your result is missing, contact your department or course adviser.",
            "category": "records",
            "tags": "results,grades,portal,transcript",
            "aliases": "exam result, grade report",
            "related_questions": "What if my result has an error?; How do I request a recheck?",
        },
        {
            "title": "Clearance",
            "question": "How do I get clearance for graduation or final year?",
            "answer": "Clearance is usually handled by the department, bursary, library, and registry. Complete all required forms, settle any outstanding fees, and confirm that your records are updated before the final sign-off.",
            "category": "graduation",
            "tags": "clearance,graduation,final year,sign off",
            "aliases": "graduation clearance, final clearance",
            "related_questions": "What documents are needed for clearance?; How long does clearance take?",
        },
        {
            "title": "Medical Excuse",
            "question": "How do I submit a medical excuse?",
            "answer": "Submit your medical note or report to the department or student affairs office as soon as possible. Keep the original copy and ask whether you also need to upload it to the portal.",
            "category": "student support",
            "tags": "medical excuse,sick note,absence",
            "aliases": "medical note, sick leave",
            "related_questions": "Can a medical excuse cover missed exams?; Who approves medical excuses?",
        },
        {
            "title": "Counseling Support",
            "question": "Where can I get counseling or student support?",
            "answer": "Student counseling is usually available through student affairs, the health center, or a dedicated counseling unit. If you need urgent help, contact the school support office right away.",
            "category": "wellbeing",
            "tags": "counseling,student support,wellbeing",
            "aliases": "student counseling, support office",
            "related_questions": "How do I book a counseling session?; Is student counseling confidential?",
        },
        {
            "title": "Complaint Process",
            "question": "How do I report a complaint or issue?",
            "answer": "Report the issue to the relevant office, department, or student affairs desk with clear details and any supporting evidence. If the issue is urgent, escalate it through the school help desk or official complaint channel.",
            "category": "support",
            "tags": "complaint,issue,help desk,report",
            "aliases": "report issue, student complaint",
            "related_questions": "How long do complaints take to resolve?; Who handles student complaints?",
        },
    ]

    existing = {
        _normalize(row.question)
        for row in db.session.query(KnowledgeBaseEntry).all()
    }
    added = 0
    for row in entries:
        if _normalize(row["question"]) in existing:
            continue
        entry = KnowledgeBaseEntry(
            title=row["title"],
            question=row["question"],
            answer=row["answer"],
            category=row.get("category", "general"),
            tags=row.get("tags"),
            aliases=row.get("aliases"),
            source_type="guidance",
            audience="student",
            priority=1,
            related_questions=row.get("related_questions"),
        )
        db.session.add(entry)
        db.session.flush()
        upsert_embedding("knowledge", entry.id, _knowledge_source_text(entry))
        added += 1
    if added:
        db.session.commit()
    return added


def rebuild_local_index():
    """Build a lightweight local TF-IDF index for semantic-ish search."""
    global local_index
    docs = []

    faqs = db.session.query(FAQ).filter_by(is_active=True).limit(400).all()
    for f in faqs:
        docs.append({
            "type": "FAQ",
            "snippet": f.question,
            "full_text": f.answer,
            "category": f.category,
        })

    policies = db.session.query(Policy).filter_by(is_active=True).limit(200).all()
    for p in policies:
        docs.append({
            "type": "POLICY",
            "snippet": p.title,
            "full_text": p.content,
            "category": p.category,
        })

    documents = db.session.query(Document).filter_by(is_active=True).limit(200).all()
    for d in documents:
        docs.append({
            "type": "DOC",
            "snippet": d.title,
            "full_text": d.content,
            "category": d.category,
        })

    knowledge_entries = db.session.query(KnowledgeBaseEntry).filter_by(is_active=True).limit(300).all()
    for k in knowledge_entries:
        docs.append({
            "type": "KNOWLEDGE",
            "snippet": k.question or k.title,
            "full_text": " ".join([
                k.answer or "",
                k.tags or "",
                k.aliases or "",
                k.related_questions or "",
            ]),
            "category": k.category,
        })

    events = db.session.query(Calendar).filter_by(is_active=True).limit(200).all()
    for c in events:
        docs.append({
            "type": "CALENDAR",
            "snippet": c.event_name,
            "full_text": f"{c.event_date}: {c.description or ''}",
            "category": "calendar",
        })

    # Build vocab and DF
    df = {}
    doc_tokens = []
    for d in docs:
        toks = _tokenize(d["snippet"] + " " + d["full_text"])
        doc_tokens.append(toks)
        for t in set(toks):
            df[t] = df.get(t, 0) + 1

    # Limit vocab size for speed
    vocab = {t: i for i, (t, _) in enumerate(sorted(df.items(), key=lambda x: x[1], reverse=True)[:5000])}
    N = max(1, len(docs))
    idf = {}
    for t in vocab.keys():
        idf[t] = (1.0 + (N / (1 + df.get(t, 0))))

    vectors = []
    norms = []
    for toks in doc_tokens:
        tf = {}
        for t in toks:
            if t in vocab:
                tf[t] = tf.get(t, 0) + 1
        vec = {}
        norm = 0.0
        for t, cnt in tf.items():
            val = (cnt / max(1, len(toks))) * idf.get(t, 1.0)
            vec[t] = val
            norm += val * val
        vectors.append(vec)
        norms.append(norm ** 0.5)

    local_index = {
        "built_at": datetime.utcnow(),
        "docs": docs,
        "vocab": vocab,
        "idf": idf,
        "vectors": vectors,
        "norms": norms,
    }
    return True


def local_vector_search(query, limit=3):
    """Use local TF-IDF vectors to find semantically similar items."""
    if not local_index.get("docs"):
        rebuild_local_index()
    docs = local_index.get("docs", [])
    if not docs:
        return []

    q_tokens = _tokenize(query)
    if not q_tokens:
        return []

    q_tf = {}
    for t in q_tokens:
        if t in local_index["vocab"]:
            q_tf[t] = q_tf.get(t, 0) + 1
    if not q_tf:
        return []

    q_vec = {}
    q_norm = 0.0
    for t, cnt in q_tf.items():
        val = (cnt / max(1, len(q_tokens))) * local_index["idf"].get(t, 1.0)
        q_vec[t] = val
        q_norm += val * val
    q_norm = q_norm ** 0.5 or 1.0

    scored = []
    for i, dvec in enumerate(local_index["vectors"]):
        denom = (local_index["norms"][i] or 1.0) * q_norm
        if denom == 0:
            continue
        dot = 0.0
        for t, qv in q_vec.items():
            dv = dvec.get(t)
            if dv:
                dot += qv * dv
        score = dot / denom
        if score > 0.1:
            scored.append((score, i))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for score, idx in scored[:limit]:
        item = dict(docs[idx])
        item["priority"] = 1.8 + score
        results.append(item)
    return results


def _bm25_rank(query, docs, k1=1.5, b=0.75, limit=3):
    # docs: list of dicts {type,snippet,full_text,category}
    if not docs:
        return []
    q_tokens = _tokenize(query)
    if not q_tokens:
        return []

    doc_tokens = [(_tokenize(d.get("snippet", "") + " " + d.get("full_text", ""))) for d in docs]
    doc_lens = [len(toks) for toks in doc_tokens]
    avgdl = (sum(doc_lens) / max(1, len(doc_lens))) or 1.0

    # Document frequencies
    df = {}
    for toks in doc_tokens:
        for t in set(toks):
            df[t] = df.get(t, 0) + 1

    scores = []
    for i, toks in enumerate(doc_tokens):
        if not toks:
            scores.append(0.0)
            continue
        tf = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        score = 0.0
        for q in q_tokens:
            if q not in tf:
                continue
            n_q = df.get(q, 0)
            idf = max(0.0, ((len(doc_tokens) - n_q + 0.5) / (n_q + 0.5)))
            denom = tf[q] + k1 * (1 - b + b * (doc_lens[i] / avgdl))
            score += idf * (tf[q] * (k1 + 1) / max(1e-6, denom))
        scores.append(score)

    ranked = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)
    results = []
    for idx, score in ranked[:limit]:
        if score <= 0:
            continue
        item = dict(docs[idx])
        item["priority"] = 1.1 + score
        results.append(item)
    return results


def _bm25_score(query, text):
    q_tokens = set(_tokenize(query))
    doc_tokens = _tokenize(text)
    if not q_tokens or not doc_tokens:
        return 0.0
    
    tf = {}
    for t in doc_tokens:
        tf[t] = tf.get(t, 0) + 1
    
    score = 0.0
    for q in q_tokens:
        if q in tf:
            score += tf[q]
    return score / max(1, len(doc_tokens))


def _get_embedding(text):
    global last_openai_error
    if not client:
        return None
    text = _truncate(text, 2000)
    if not text:
        return None
    try:
        resp = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
            encoding_format="float"
        )
        return resp.data[0].embedding
    except Exception as exc:
        last_openai_error = str(exc)
        try:
            current_app.logger.warning(f"Embedding generation failed: {exc}")
        except Exception:
            pass
        return None


def upsert_embedding(content_type, content_id, text):
    try:
        emb = _get_embedding(text)
        if emb is None:
            return False
        payload = json.dumps(emb)
        existing = Embedding.query.filter_by(content_type=content_type, content_id=content_id).first()
        if existing:
            existing.text = _truncate(text, 4000)
            existing.vector = payload
        else:
            existing = Embedding(
                content_type=content_type,
                content_id=content_id,
                text=_truncate(text, 4000),
                vector=payload
            )
            db.session.add(existing)
        db.session.commit()
        return True
    except Exception as exc:
        try:
            current_app.logger.warning(f"Embedding upsert failed: {exc}")
        except Exception:
            pass
        db.session.rollback()
        return False


def delete_embedding(content_type, content_id):
    emb = Embedding.query.filter_by(content_type=content_type, content_id=content_id).first()
    if emb:
        db.session.delete(emb)
        db.session.commit()


def _cosine_similarity(a, b):
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(len(a)):
        va = a[i]
        vb = b[i]
        dot += va * vb
        na += va * va
        nb += vb * vb
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / ((na ** 0.5) * (nb ** 0.5))


def semantic_search(query, limit=3):
    q_emb = _get_embedding(query)
    if q_emb is None:
        return []

    # Cap rows for performance
    rows = Embedding.query.order_by(Embedding.updated_at.desc()).limit(600).all()
    scored = []
    for row in rows:
        try:
            vec = json.loads(row.vector)
        except Exception:
            continue
        score = _cosine_similarity(q_emb, vec)
        if score > 0.2:
            scored.append((score, row.content_type, row.content_id))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:limit]
    if not top:
        return []

    results = []
    for score, ctype, cid in top:
        if ctype == "faq":
            faq = FAQ.query.get(cid)
            if faq and faq.is_active:
                results.append({
                    'type': 'FAQ',
                    'snippet': faq.question,
                    'full_text': faq.answer,
                    'category': faq.category,
                    'priority': 4 + score,
                })
        elif ctype == "policy":
            pol = Policy.query.get(cid)
            if pol and pol.is_active:
                results.append({
                    'type': 'POLICY',
                    'snippet': pol.title,
                    'full_text': _sectioned_snippet(pol.content, query),
                    'category': pol.category,
                    'priority': 3 + score,
                })
        elif ctype == "document":
            doc = Document.query.get(cid)
            if doc and doc.is_active:
                results.append({
                    'type': 'DOC',
                    'snippet': doc.title,
                    'full_text': _sectioned_snippet(doc.content, query),
                    'category': doc.category,
                    'priority': 2.5 + score,
                })
        elif ctype == "knowledge":
            entry = KnowledgeBaseEntry.query.get(cid)
            if entry and entry.is_active:
                results.append({
                    'type': 'KNOWLEDGE',
                    'snippet': entry.question or entry.title,
                    'full_text': entry.answer,
                    'category': entry.category,
                    'priority': 3.2 + score,
                })
    return results


def _normalize(text):
    text = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
    return re.sub(r"\s+", " ", text).strip()


_STOP_WORDS = frozenset({
    "what", "where", "when", "why", "how", "who", "which",
    "are", "is", "was", "were", "be", "been", "being",
    "the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "at", "by", "from",
    "do", "does", "did", "can", "could", "should", "would", "will", "may", "might",
    "my", "your", "our", "their", "this", "that", "these", "those",
})


def _is_relevant_fallback(query, item):
    q_tokens = set(_tokenize(query))
    if not q_tokens:
        return True

    meaningful = {t for t in q_tokens if t not in _STOP_WORDS}
    if not meaningful:
        return True

    snippet = (item.get('snippet') or '').lower()
    full_text = (item.get('full_text') or '').lower()

    snippet_tokens = set(_tokenize(snippet))
    combined_tokens = set(_tokenize(snippet + ' ' + full_text))

    snippet_overlap = meaningful.intersection(snippet_tokens)
    combined_overlap = meaningful.intersection(combined_tokens)

    snippet_score = len(snippet_overlap) * 3
    full_text_only_score = len(combined_overlap - snippet_overlap)
    total_score = snippet_score + full_text_only_score

    if len(meaningful) <= 1:
        return total_score >= 2

    return total_score >= 3


def _compute_priority(query, item):
    """Compute a relevance score between the query and a KB item."""
    q_norm = _normalize(query)
    snippet = _normalize(item.get('snippet') or "")
    full_text = _normalize(item.get('full_text') or "")
    combined = (snippet + " " + full_text).strip()
    q_tokens = set(_tokenize(query))
    
    score = 0.0
    
    if not q_norm or not combined:
        return 0.0
    
    q_lower = query.lower().strip()
    snippet_lower = (item.get('snippet') or "").lower().strip()
    full_lower = (item.get('full_text') or "").lower().strip()
    
    if q_lower == snippet_lower or q_lower == full_lower:
        score += 50.0
    elif snippet_lower.startswith(q_lower) or full_lower.startswith(q_lower):
        score += 40.0
    elif q_lower.startswith(snippet_lower) or q_lower.startswith(full_lower):
        score += 30.0
    
    q_set = set(q_norm.split())
    c_set = set(combined.split())
    if q_set and c_set:
        jaccard = len(q_set & c_set) / max(1, len(q_set | c_set))
        score += jaccard * 20
    
    meaningful = {t for t in q_tokens if t not in {
        "what", "where", "when", "why", "how", "who", "which",
        "are", "is", "was", "were", "be", "been", "being",
        "the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "at", "by", "from",
        "do", "does", "did", "can", "could", "should", "would", "will", "may", "might",
        "my", "your", "our", "their", "this", "that", "these", "those",
    }}
    if meaningful:
        c_tokens = set(_tokenize(combined))
        overlap = len(meaningful.intersection(c_tokens))
        score += (overlap / max(1, len(meaningful))) * 20
        if overlap > 0:
            score += 5
    
    # BM25-like score for term frequency quality
    bm25 = _bm25_score(query, combined)
    score += bm25 * 10
    
    item_type = item.get('type', '')
    if item_type == 'FAQ':
        score += 2
    elif item_type == 'POLICY':
        score += 2
    elif item_type == 'DOC':
        score += 1
    elif item_type == 'CALENDAR':
        score += 1
    
    return score


def search_kb_items(query, limit=3, history=None):
    """Return structured KB matches across all knowledge tables."""
    search_queries = _build_query_variants(query, history=history)
    base_query = (query or "").strip()
    q_tokens = set(_tokenize(base_query))
    meaningful_tokens = {t for t in q_tokens if t not in _STOP_WORDS}

    def _add_result(results, seen, item):
        snippet = (item.get("snippet") or "").strip().lower()
        key = (item.get("type"), snippet)
        if not snippet or key in seen:
            return
        seen.add(key)
        results.append(item)

    def _collect_for_search(search_query, results, seen):
        tokens = set(_tokenize(search_query))
        keywords = [f"%{kw}%" for kw in re.split(r"\s+", search_query.lower()) if kw]

        if keywords:
            faqs = db.session.query(FAQ).filter(
                or_(*(FAQ.question.ilike(kw) | FAQ.answer.ilike(kw) for kw in keywords))
            ).all()
            for faq in faqs:
                item = {
                    "type": "FAQ",
                    "snippet": faq.question,
                    "full_text": faq.answer,
                    "category": faq.category,
                }
                item["priority"] = _compute_priority(base_query or search_query, item)
                _add_result(results, seen, item)

            policies = db.session.query(Policy).filter(
                or_(*(Policy.content.ilike(kw) | Policy.title.ilike(kw) for kw in keywords))
            ).all()
            for pol in policies:
                snippet_text = _sectioned_snippet(pol.content, search_query)
                item = {
                    "type": "POLICY",
                    "snippet": pol.title,
                    "full_text": snippet_text,
                    "category": pol.category,
                }
                item["priority"] = _compute_priority(base_query or search_query, item)
                _add_result(results, seen, item)

            docs = db.session.query(Document).filter(
                or_(*(Document.content.ilike(kw) | Document.title.ilike(kw) for kw in keywords))
            ).all()
            for doc in docs:
                snippet_text = _sectioned_snippet(doc.content, search_query)
                item = {
                    "type": "DOC",
                    "snippet": doc.title,
                    "full_text": snippet_text,
                    "category": doc.category,
                }
                item["priority"] = _compute_priority(base_query or search_query, item)
                _add_result(results, seen, item)

            events = db.session.query(Calendar).filter(
                or_(*(Calendar.event_name.ilike(kw) | Calendar.description.ilike(kw) for kw in keywords))
            ).all()
            for event in events:
                item = {
                    "type": "CALENDAR",
                    "snippet": event.event_name,
                    "full_text": f"{event.event_date}: {event.description or ''}",
                    "category": "calendar",
                }
                item["priority"] = _compute_priority(base_query or search_query, item)
                _add_result(results, seen, item)

            knowledge_entries = db.session.query(KnowledgeBaseEntry).filter(
                or_(
                    *(KnowledgeBaseEntry.question.ilike(kw) |
                      KnowledgeBaseEntry.answer.ilike(kw) |
                      KnowledgeBaseEntry.title.ilike(kw) |
                      KnowledgeBaseEntry.tags.ilike(kw) |
                      KnowledgeBaseEntry.aliases.ilike(kw) |
                      KnowledgeBaseEntry.related_questions.ilike(kw)
                      for kw in keywords)
                )
            ).all()
            for entry in knowledge_entries:
                item = _knowledge_item_to_record(entry)
                if not item:
                    continue
                item["priority"] = _compute_priority(base_query or search_query, item) + 3
                _add_result(results, seen, item)

        if tokens:
            results.extend([hit for hit in local_vector_search(search_query, limit=limit * 2)
                            if (hit.get("type"), (hit.get("snippet") or "").strip().lower()) not in seen])

    results = []
    seen = set()
    for search_query in search_queries:
        _collect_for_search(search_query, results, seen)
        if len(results) >= max(limit * 4, 12):
            break

    if not results:
        results.extend(local_vector_search(base_query, limit=limit))

    # Keep only broad-table matches that still touch the user's topic.
    if q_tokens:
        filtered = []
        for r in results:
            text = (r.get("snippet") or "") + " " + (r.get("full_text") or "")
            tokens = set(_tokenize(text))
            required = meaningful_tokens if meaningful_tokens else q_tokens
            if tokens.intersection(required):
                filtered.append(r)
        if filtered:
            results = filtered

    # If keyword matches are thin, use fuzzy similarity and BM25 to broaden coverage.
    def _normalize(text):
        text = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
        return re.sub(r"\s+", " ", text).strip()

    def _score(a, b):
        a_n = _normalize(a)
        b_n = _normalize(b)
        if not a_n or not b_n:
            return 0.0
        seq = difflib.SequenceMatcher(None, a_n, b_n).ratio()
        a_tokens = set(a_n.split())
        b_tokens = set(b_n.split())
        jaccard = len(a_tokens & b_tokens) / max(1, len(a_tokens | b_tokens))
        return max(seq, jaccard)

    if len(results) < limit:
        query_norm = _normalize(base_query)
        seen_pairs = {(r["type"], (r.get("snippet") or "").strip().lower()) for r in results}

        faq_rows = db.session.query(FAQ).filter_by(is_active=True).limit(250).all()
        for faq in faq_rows:
            score = _score(query_norm, faq.question)
            if score >= 0.45 and ("FAQ", faq.question.strip().lower()) not in seen_pairs:
                item = {
                    "type": "FAQ",
                    "snippet": faq.question,
                    "full_text": faq.answer,
                    "category": faq.category,
                    "priority": 2 + score,
                }
                _add_result(results, seen_pairs, item)

        knowledge_rows = db.session.query(KnowledgeBaseEntry).filter_by(is_active=True).limit(300).all()
        for entry in knowledge_rows:
            score = max(
                _score(query_norm, entry.question),
                _score(query_norm, entry.title),
                _score(query_norm, entry.aliases or ""),
                _score(query_norm, entry.tags or ""),
            )
            if score >= 0.4 and ("KNOWLEDGE", (entry.question or entry.title or "").strip().lower()) not in seen_pairs:
                item = _knowledge_item_to_record(entry)
                if not item:
                    continue
                item["priority"] = 2.5 + score
                _add_result(results, seen_pairs, item)

        policies_all = db.session.query(Policy).filter_by(is_active=True).limit(100).all()
        for pol in policies_all:
            score = _score(query_norm, pol.title)
            if score >= 0.5 and ("POLICY", pol.title.strip().lower()) not in seen_pairs:
                item = {
                    "type": "POLICY",
                    "snippet": pol.title,
                    "full_text": pol.content,
                    "category": pol.category,
                    "priority": 1.5 + score,
                }
                _add_result(results, seen_pairs, item)

        docs_all = db.session.query(Document).filter_by(is_active=True).limit(100).all()
        for doc in docs_all:
            score = _score(query_norm, doc.title)
            if score >= 0.5 and ("DOC", doc.title.strip().lower()) not in seen_pairs:
                item = {
                    "type": "DOC",
                    "snippet": doc.title,
                    "full_text": doc.content,
                    "category": doc.category,
                    "priority": 1.2 + score,
                }
                _add_result(results, seen_pairs, item)

    if len(results) < limit:
        pool = []
        pool.extend([{
            "type": "FAQ",
            "snippet": f.question,
            "full_text": f.answer,
            "category": f.category,
        } for f in db.session.query(FAQ).filter_by(is_active=True).limit(300).all()])
        pool.extend([{
            "type": "KNOWLEDGE",
            "snippet": k.question or k.title,
            "full_text": k.answer,
            "category": k.category,
        } for k in db.session.query(KnowledgeBaseEntry).filter_by(is_active=True).limit(300).all()])
        pool.extend([{
            "type": "POLICY",
            "snippet": p.title,
            "full_text": p.content,
            "category": p.category,
        } for p in db.session.query(Policy).filter_by(is_active=True).limit(200).all()])
        pool.extend([{
            "type": "DOC",
            "snippet": d.title,
            "full_text": d.content,
            "category": d.category,
        } for d in db.session.query(Document).filter_by(is_active=True).limit(200).all()])
        pool.extend([{
            "type": "CALENDAR",
            "snippet": c.event_name,
            "full_text": f"{c.event_date}: {c.description or ''}",
            "category": "calendar",
        } for c in db.session.query(Calendar).filter_by(is_active=True).limit(200).all()])

        bm25_hits = _bm25_rank(base_query, pool, limit=limit)
        for hit in bm25_hits:
            _add_result(results, seen, hit)

    # Final scoring and dedupe.
    deduped = []
    seen_final = set()
    for r in results:
        text = (r.get("snippet") or "") + " " + (r.get("full_text") or "")
        token_overlap = len(set(_tokenize(text)).intersection(meaningful_tokens or q_tokens))
        if q_tokens and token_overlap == 0 and r.get("type") not in {"KNOWLEDGE", "FAQ"}:
            continue
        key = (r.get("type"), (r.get("snippet") or "").strip().lower())
        if key in seen_final:
            continue
        seen_final.add(key)
        if "priority" not in r or r.get("priority") is None:
            r["priority"] = _compute_priority(base_query, r)
        deduped.append(r)

    deduped = sorted(deduped, key=lambda x: x.get("priority", 0), reverse=True)
    return deduped[:limit]


def search_kb(query, limit=3, history=None):
    """Retrieve relevant context using ORM keyword search."""
    results = search_kb_items(query, limit=limit, history=history)
    context_parts = []
    for i, r in enumerate(results, 1):
        rtype = r.get('type', 'INFO')
        snippet = (r.get('snippet') or "").strip()
        full_text = (r.get('full_text') or "").strip()
        max_len = 800
        if len(full_text) > max_len:
            full_text = full_text[:max_len] + "..."
        context_parts.append(f"[{i}] {rtype}: {snippet}\n{full_text}")
    context = "\n\n".join(context_parts)
    sources = [r['snippet'] for r in results]
    return context, sources

def detect_intent(query):
    """Use OpenAI to detect intent."""
    if not client:
        return 'general', {}
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{
            "role": "user",
            "content": f"Classify intent for university query: '{query}'. Possible intents: admissions, registration, fees, deadlines, policy, facilities, general. Respond with ONLY one word: the intent."
        }],
        max_tokens=10
    )
    intent = response.choices[0].message.content.strip().lower()
    return intent, {'query': query}

def get_all_faqs(limit=10):
    """Get all FAQs for listing common questions."""
    faqs = db.session.query(FAQ).limit(200).all()
    cleaned = []
    for faq in faqs:
        q = (faq.question or "").strip()
        if not q:
            continue
        q_lower = q.lower()
        if "table of contents" in q_lower:
            continue
        if "how does it relate to" in q_lower or "what does the document say about" in q_lower:
            continue
        cleaned.append({'question': q, 'category': faq.category})
        if len(cleaned) >= limit:
            break
    return cleaned

def is_greeting(query):
    """Check if the query is a greeting."""
    greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening', 'greetings', 'howdy', 'hola', 'bonjour']
    query_lower = query.lower().strip()
    return any(greeting in query_lower for greeting in greetings)


def is_school_query(query):
    """Heuristic to detect if a question is about the university/school."""
    q = (query or "").lower()
    keywords = [
        "university", "college", "campus", "admission", "admissions", "enrollment",
        "registration", "tuition", "fees", "scholarship", "scholarships",
        "course", "courses", "program", "programs", "department", "faculty",
        "semester", "term", "academic", "exam", "exams", "result", "results",
        "hostel", "housing", "accommodation", "library", "handbook", "policy",
        "policies", "calendar", "deadline", "deadlines", "portal", "matric",
        "student", "students", "lecturer", "lecture", "timetable", "class",
    ]
    return any(k in q for k in keywords)


def detect_school_intent(query):
    """Use AI to classify if a query is school-related; fallback to heuristic."""
    if not client:
        return is_school_query(query)
    try:
        prompt = f"""Is the following user question about a university/school or student services?
Respond with ONLY 'school' or 'general'.

Question: {query}
"""
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=5
        )
        label = resp.choices[0].message.content.strip().lower()
        return label == "school"
    except Exception:
        return is_school_query(query)

def _strip_question_echo(query, answer):
    """Remove echoed question text from the start of AI answers."""
    if not answer:
        return answer

    def _norm(s):
        return re.sub(r"\s+", " ", (s or "").strip().lower())

    q_norm = _norm(query)
    if not q_norm:
        return answer.strip()

    lines = [ln.strip() for ln in answer.splitlines() if ln.strip()]
    if not lines:
        return answer.strip()

    first = lines[0]
    first_norm = _norm(first)

    # Remove explicit "Question: ..." or "Q: ..." lines
    if first_norm in (f"question: {q_norm}", f"q: {q_norm}", q_norm):
        lines = lines[1:]
    # Remove in-line echo like: "<question>? <answer>"
    elif first_norm.startswith(q_norm) and len(first_norm) > len(q_norm) + 2:
        trimmed = first[len(query):].lstrip(" :-")
        if trimmed:
            lines[0] = trimmed
        else:
            lines = lines[1:]

    # Strip trailing "Sources:" lines if model adds them
    cleaned = []
    for ln in lines:
        ln_norm = _norm(ln)
        if ln_norm.startswith("sources:") or ln_norm.startswith("source:"):
            continue
        cleaned.append(ln)

    cleaned_text = "\n".join(cleaned).strip()
    return cleaned_text if cleaned_text else answer.strip()

def _is_admin_view():
    try:
        return current_user.is_authenticated and getattr(current_user, "is_admin", lambda: False)()
    except Exception:
        return False


def _local_kb_fallback(query, limit=5):
    """Return the best local KB answer without calling OpenAI."""
    items = search_kb_items(query, limit=limit)
    for item in items:
        answer = _format_kb_answer(query, item)
        if answer:
            snippet = (item.get('snippet') or '').strip()
            sources = [snippet] if snippet else []
            return answer, sources
    return None, []


def _is_openai_capacity_error(exc):
    """Detect quota/rate-limit failures that should permanently disable AI replies."""
    text = f"{type(exc).__name__}: {exc}".lower()
    signals = [
        "insufficient_quota",
        "quota",
        "rate limit",
        "too many requests",
        "billing",
        "429",
    ]
    return any(signal in text for signal in signals)


def _suggest_related_questions(items, query, limit=4):
    """Extract short follow-up questions from the matched KB items."""
    related = []
    q_lower = (query or "").lower()
    for item in items or []:
        if item.get("type") in {"FAQ", "KNOWLEDGE"} and item.get("snippet"):
            related.append(item["snippet"])
        extra = item.get("related_questions")
        if extra:
            if isinstance(extra, str):
                parts = re.split(r"[;\n•]+", extra)
            elif isinstance(extra, (list, tuple)):
                parts = list(extra)
            else:
                parts = []
            related.extend(parts)

    for key, expansions in _QUERY_EXPANSIONS.items():
        if key in q_lower:
            related.extend(expansions[:2])

    generic = [
        "What office handles this?",
        "What documents do I need?",
        "When is the deadline?",
        "How do I apply?",
    ]
    if not related:
        related.extend(generic)

    cleaned = []
    for item in _unique_preserve_order(related):
        text = item.strip().strip("•- ")
        if not text:
            continue
        if text.endswith("?"):
            cleaned.append(text)
        elif len(text.split()) <= 12:
            cleaned.append(f"{text}?")
        else:
            cleaned.append(text)
        if len(cleaned) >= limit:
            break
    return cleaned


def _compose_local_response(query, items):
    if not items:
        return "Sorry, I don't have that info in the knowledge base. Please contact admin or check the handbook.", []

    top = items[0]
    answer = _format_kb_answer(query, top)
    if not answer:
        answer = (top.get("full_text") or "").strip()
    if not answer:
        return "Sorry, I don't have that info in the knowledge base. Please contact admin or check the handbook.", []

    related = _suggest_related_questions(items, query)
    if related:
        answer = f"{answer}\n\nYou may also want to ask:\n- " + "\n- ".join(related)

    sources = [item["snippet"] for item in items if item.get("snippet")]
    return answer, sources[:5]

def _format_kb_answer(query, item):
    """Format a KB item into a clean answer without repeating the question."""
    if not item:
        return ""

    body = (item.get('full_text') or '').strip()
    if not body:
        return ""

    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    short = " ".join(lines[:4])[:600] if lines else body[:600]

    # Only show a source label when it's not just the FAQ question itself
    prefix = ""
    if item.get('type') in ("POLICY", "DOC", "CALENDAR") and item.get('snippet'):
        prefix = f"From {item['snippet']}:\n\n"

    # Build key points only when there's enough content, and avoid duplicates
    bullets = []
    if len(lines) >= 4 or len(body) > 300:
        candidates = [ln for ln in lines if len(ln) > 30]
        if not candidates and len(lines) >= 2:
            candidates = lines[:3]
        seen = set()
        for ln in candidates:
            cleaned = ln.strip()
            if not cleaned or cleaned in seen or cleaned in short:
                continue
            seen.add(cleaned)
            bullets.append(cleaned)
            if len(bullets) >= 4:
                break

    answer = f"{prefix}{short}"
    if bullets:
        answer += "\n\nKey points:\n- " + "\n- ".join(bullets)
    return answer

def generate_rag_response(query, history=None):
    """Full RAG pipeline with greeting detection and FAQ listing."""
    global client
    global last_openai_error
    config = current_app.config
    
    history = history or []
    
    # Handle greetings specially
    if is_greeting(query):
        faqs = get_all_faqs()
        if faqs:
            faq_list = "\n".join([f"• {faq['question']}" for faq in faqs[:8]])
            welcome_msg = f"""Hello! 👋 Welcome to the University Information Assistant!

I'm here to help you with information about admissions, policies, deadlines, and more. Here are some common questions students ask:

{faq_list}

What would you like to know about?"""
        else:
            welcome_msg = "Hello! 👋 Welcome to the University Information Assistant! How can I help you today?"
        
        return welcome_msg, []
    
    search_items = search_kb_items(query, limit=5, history=history)
    context, sources = search_kb(query, limit=5, history=history)
    local_answer, local_sources = _compose_local_response(query, search_items)
    q_lower = (query or "").lower()
    admin_hint = ""
    if _is_admin_view() and is_school_query(query) and any(k in q_lower for k in ["tuition", "fee", "fees", "cost", "payment"]):
        admin_hint = "\n\n(Admin tip: add the official fees/tuition info in the Admin dashboard under FAQs or Policies.)"

    if search_items and context.strip():
        if not config['OPENAI_API_KEY'] or not client:
            return local_answer + admin_hint, local_sources

        conv_messages = []
        for turn in history[-6:]:
            role = turn.get('role', 'user')
            content = turn.get('content', '')
            if role == 'user':
                conv_messages.append({"role": "user", "content": content})
            else:
                conv_messages.append({"role": "assistant", "content": content})

        system_prompt = (
            "You are a helpful university information assistant. "
            "Cross-check the provided knowledge base sources and answer naturally. "
            "If the question is only partially covered, give the best safe answer you can based on related information, and say what is uncertain. "
            "Do not invent official policy, dates, or fees. "
            "If useful, end with a short 'Related questions:' list containing 2-4 follow-up questions."
        )

        try:
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(conv_messages)
            messages.append({
                "role": "user",
                "content": f"Knowledge base context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
            })
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.2,
                max_tokens=500
            )
            answer = response.choices[0].message.content.strip()
            answer = _strip_question_echo(query, answer)
            if not answer:
                answer = local_answer or "Sorry, I don't have that info in the knowledge base. Please contact admin or check the handbook."
            related = _suggest_related_questions(search_items, query)
            if related and "related questions:" not in answer.lower():
                answer = f"{answer}\n\nRelated questions:\n- " + "\n- ".join(related)
            if admin_hint:
                answer = f"{answer}{admin_hint}"
            return answer, sources or local_sources
        except Exception as e:
            last_openai_error = str(e)
            current_app.logger.error(f"RAG error: {str(e)}")
            if _is_openai_capacity_error(e):
                client = None
            if local_answer:
                return local_answer + admin_hint, local_sources
            return (
                "I couldn't access the AI service right now. Please try again later or ask a school-specific question."
                + admin_hint,
                sources,
            )

    if not config['OPENAI_API_KEY'] or not client:
        if local_answer:
            return local_answer + admin_hint, local_sources
        if is_school_query(query):
            return (
                "I don't have that in the school knowledge base yet. Please check the official handbook or contact admin for accurate details."
                + admin_hint,
                [],
            )
        return (
            "I can help with general questions, but I'm currently in offline mode. Please add more details and I'll do my best."
            + admin_hint,
            [],
        )

    try:
        general_prompt = (
            "You are a helpful assistant. Answer the user's question clearly and safely. "
            "Do not repeat the question. If you are unsure, say so briefly."
            f"\n\nQuestion: {query}\n\nAnswer:"
        )
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": general_prompt}],
            temperature=0.4,
            max_tokens=400
        )
        answer = response.choices[0].message.content.strip()
        answer = _strip_question_echo(query, answer)
        if admin_hint:
            answer = f"{answer}{admin_hint}"
        return answer, []
    except Exception as e:
        last_openai_error = str(e)
        current_app.logger.error(f"General AI error: {str(e)}")
        if _is_openai_capacity_error(e):
            client = None
        if local_answer:
            return local_answer + admin_hint, local_sources
        return (
            "I couldn't access the AI service right now. Please try again later or ask a school-specific question."
            + admin_hint,
            [],
        )

def get_fallback(query):
    return f"For '{query}', please check the student handbook, admin dashboard, or contact university support for latest info."


def extract_text_from_file(file_path, filename):
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
        except Exception as exc:
            raise RuntimeError("PDF support requires pypdf") from exc
        reader = PdfReader(file_path)
        parts = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        return "\n".join(parts).strip()
    if ext == ".docx":
        try:
            import docx
        except Exception as exc:
            raise RuntimeError("DOCX support requires python-docx") from exc
        doc = docx.Document(file_path)
        parts = [p.text for p in doc.paragraphs if p.text]
        return "\n".join(parts).strip()
    raise ValueError("Unsupported file type. Please upload .pdf, .docx, or .txt")


def generate_faqs_from_text(text, max_faqs=8, category="general"):
    global last_openai_error
    if not client:
        last_openai_error = "OpenAI client not initialized"
        return _fallback_generate_faqs(text, max_faqs=max_faqs, category=category)

    safe_text = (text or "").strip()
    if not safe_text:
        return []

    # Keep the prompt small enough for the model
    safe_text = safe_text[:12000]

    prompt = f"""
You are an assistant that creates FAQs from university documents.
Return ONLY valid JSON in this format:
[
  {{"question": "...", "answer": "..."}},
  ...
]
Generate up to {max_faqs} concise, high-quality FAQs.

Document:
{safe_text}
""".strip()

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=800,
        )
    except Exception as exc:
        last_openai_error = str(exc)
        return _fallback_generate_faqs(text, max_faqs=max_faqs, category=category)

    raw = response.choices[0].message.content.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Basic fallback parsing if JSON fails
        faqs = []
        blocks = re.split(r"\n{2,}", raw)
        for block in blocks:
            q_match = re.search(r"Q[:\-]\s*(.+)", block, re.IGNORECASE)
            a_match = re.search(r"A[:\-]\s*(.+)", block, re.IGNORECASE)
            if q_match and a_match:
                faqs.append({"question": q_match.group(1).strip(), "answer": a_match.group(1).strip()})
        data = faqs

    cleaned = []
    for item in data[:max_faqs]:
        question = (item.get("question") or "").strip()
        answer = (item.get("answer") or "").strip()
        if question and answer:
            cleaned.append({"question": question, "answer": answer, "category": category})
    return cleaned


def _fallback_generate_faqs(text, max_faqs=8, category="general"):
    """Local FAQ generator when OpenAI is unavailable."""
    text = (text or "").strip()
    if not text:
        return []

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    faqs = []

    # Try Q/A formatted lines
    i = 0
    while i < len(lines) - 1 and len(faqs) < max_faqs:
        q_match = re.match(r"^(q|question)[:\-]\s*(.+)$", lines[i], re.IGNORECASE)
        a_match = re.match(r"^(a|answer)[:\-]\s*(.+)$", lines[i + 1], re.IGNORECASE)
        if q_match and a_match:
            faqs.append({
                "question": q_match.group(2).strip(),
                "answer": a_match.group(2).strip(),
                "category": category
            })
            i += 2
            continue
        i += 1

    if len(faqs) >= max_faqs:
        return faqs[:max_faqs]

    # Use heading + paragraph heuristic (keyword-aware)
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    for p in paragraphs:
        if len(faqs) >= max_faqs:
            break
        p_lines = [ln.strip() for ln in p.splitlines() if ln.strip()]
        if not p_lines:
            continue
        heading = p_lines[0]
        body = " ".join(p_lines[1:]).strip()
        if heading.endswith(":"):
            heading = heading[:-1].strip()
        if body:
            # Pull a keyword from the body for a smarter question
            keywords = _tokenize(body)
            key = keywords[0] if keywords else heading
            if heading.lower() in ["science", "engineering", "arts", "social sciences", "management sciences"]:
                question = f"What are the subject requirements for {heading}?"
            else:
                question = f"What is {heading}?"
            faqs.append({"question": question, "answer": body[:600], "category": category})

    if len(faqs) >= max_faqs:
        return faqs[:max_faqs]

    # Fallback: create FAQs from top paragraphs (keyword summary)
    for p in paragraphs:
        if len(faqs) >= max_faqs:
            break
        words = p.split()
        if len(words) < 6:
            continue
        keywords = _tokenize(p)
        key = " ".join(keywords[:4]) if keywords else "this topic"
        question = f"What does the document say about {key}?"
        faqs.append({"question": question, "answer": p[:600], "category": category})

    return faqs[:max_faqs]
