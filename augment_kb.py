import argparse
import re
from collections import OrderedDict

from app import create_app
from models import db, FAQ, KnowledgeBaseEntry
from utils import rebuild_local_index


QUESTION_OPENERS = [
    "How do I {topic}?",
    "How can I {topic}?",
    "What is the process to {topic}?",
    "What do I need to {topic}?",
    "Where do I go to {topic}?",
    "Where can I find information on {topic}?",
    "Can you explain how to {topic}?",
    "Please tell me how to {topic}.",
    "I need help with {topic}.",
    "I want to know how to {topic}.",
]


CATEGORY_VARIANTS = {
    "fees": [
        "How much are the fees?",
        "When is the payment deadline?",
        "How do I pay school fees?",
        "Can I pay in instalments?",
        "What happens if I pay late?",
    ],
    "registration": [
        "How do I register for courses?",
        "When does registration open?",
        "What if I miss the registration deadline?",
        "How do I add or drop a course?",
        "How many courses can I take?",
    ],
    "courses": [
        "What courses are offered here?",
        "Which programs are available?",
        "What is the course catalogue?",
        "What classes can I take?",
        "How do I choose my courses?",
    ],
    "academics": [
        "How do I check my timetable?",
        "What is the grading scale?",
        "When are exams held?",
        "How do I request a transcript?",
    ],
    "hostel": [
        "How do I apply for hostel accommodation?",
        "How much is hostel accommodation?",
        "How do I pay my hostel fee?",
        "What items are allowed in the dormitory?",
    ],
    "library": [
        "What are the library opening hours?",
        "How do I borrow books?",
        "What are the library rules?",
    ],
    "admissions": [
        "How do I apply for admission?",
        "What are the admission requirements?",
        "When is the application deadline?",
    ],
}


def _norm(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())).strip()


def _topic_from_question(question):
    q = _norm(question)
    prefixes = [
        "how do i ",
        "how can i ",
        "what is the ",
        "what are the ",
        "what is ",
        "what are ",
        "where can i ",
        "where do i ",
        "can i ",
        "tell me about ",
        "please tell me about ",
        "i want to know about ",
        "i need help with ",
    ]
    for prefix in prefixes:
        if q.startswith(prefix):
            return q[len(prefix):].strip(" ?.")
    return q.strip(" ?.")


def _unique_keep_order(items):
    return list(OrderedDict.fromkeys(item for item in items if item))


def build_variants(question, category=None, aliases=None, tags=None, related=None, limit=24):
    topic = _topic_from_question(question)
    variant_sources = [topic]
    if aliases:
        variant_sources.extend([x.strip() for x in re.split(r"[,;|]", aliases) if x.strip()])
    if tags:
        variant_sources.extend([x.strip() for x in re.split(r"[,;|]", tags) if x.strip()])
    if related:
        variant_sources.extend([x.strip() for x in re.split(r"[;\n]", related) if x.strip()])

    topics = _unique_keep_order(variant_sources)
    variants = [question]

    for t in topics[:6]:
        for opener in QUESTION_OPENERS:
            variants.append(opener.format(topic=t))

    if category and category.lower() in CATEGORY_VARIANTS:
        variants.extend(CATEGORY_VARIANTS[category.lower()])

    # Small wording shifts for variety.
    variants.extend([
        f"Could you help me with {topic}?",
        f"I was wondering about {topic}.",
        f"Can you give me details about {topic}?",
        f"Where do I get information about {topic}?",
        f"What should I know about {topic}?",
    ])

    cleaned = []
    seen = set()
    for variant in variants:
        text = re.sub(r"\s+", " ", variant).strip(" ?.")
        key = _norm(text)
        if not key or key in seen:
            continue
        seen.add(key)
        cleaned.append(text + ("?" if not text.endswith("?") else ""))
        if len(cleaned) >= limit:
            break
    return cleaned


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic university Q&A paraphrases.")
    parser.add_argument("--count", type=int, default=500, help="Maximum synthetic questions to create.")
    parser.add_argument("--per-source", type=int, default=20, help="Maximum variants per source item.")
    parser.add_argument("--write-db", action="store_true", help="Insert generated questions into the FAQ table.")
    args = parser.parse_args()

    app = create_app()
    total_added = 0

    with app.app_context():
        sources = []
        sources.extend(FAQ.query.filter_by(is_active=True).all())
        sources.extend(KnowledgeBaseEntry.query.filter_by(is_active=True).all())

        existing = {_norm(row.question) for row in FAQ.query.all()}
        generated = []

        for row in sources:
            if total_added >= args.count:
                break
            if hasattr(row, "question"):
                question = row.question or ""
            else:
                continue

            category = getattr(row, "category", "general") or "general"
            aliases = getattr(row, "aliases", None)
            tags = getattr(row, "tags", None)
            related = getattr(row, "related_questions", None)
            answer = getattr(row, "answer", None) or ""

            for variant in build_variants(question, category=category, aliases=aliases, tags=tags, related=related, limit=args.per_source):
                if total_added >= args.count:
                    break
                key = _norm(variant)
                if key in existing:
                    continue
                existing.add(key)
                generated.append((variant, answer, category))
                total_added += 1

        if args.write_db and generated:
            for question, answer, category in generated:
                db.session.add(FAQ(question=question, answer=answer, category=category, is_active=True))
            db.session.commit()
            rebuild_local_index()

        print(f"Generated {len(generated)} synthetic questions.")
        if args.write_db:
            print("Inserted synthetic questions into FAQ.")


if __name__ == "__main__":
    main()
