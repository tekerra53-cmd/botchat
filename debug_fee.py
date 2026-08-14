import sys
sys.path.insert(0, r'C:\Users\hp\Desktop\555 university_chatbot')

from app import create_app
from models import FAQ, Policy
from utils import search_kb_items, _tokenize, _compute_priority

app = create_app()

with app.app_context():
    query = "What is the fee structure?"
    print(f"Query: {query}")
    
    # Check keyword search
    from sqlalchemy import or_
    import re
    keywords = [f"%{kw}%" for kw in re.split(r'\s+', query.lower()) if kw]
    faqs = FAQ.query.filter(
        or_(*(FAQ.question.ilike(kw) | FAQ.answer.ilike(kw) for kw in keywords))
    ).all()
    print(f"\nKeyword search found {len(faqs)} FAQs:")
    for faq in faqs:
        item = {'type': 'FAQ', 'snippet': faq.question, 'full_text': faq.answer}
        p = _compute_priority(query, item)
        print(f"  [{faq.id}] {faq.question[:60]}... priority={p:.2f}")
    
    policies = Policy.query.filter(
        or_(*(Policy.content.ilike(kw) | Policy.title.ilike(kw) for kw in keywords))
    ).all()
    print(f"\nKeyword search found {len(policies)} policies:")
    for pol in policies:
        item = {'type': 'POLICY', 'snippet': pol.title, 'full_text': pol.content}
        p = _compute_priority(query, item)
        print(f"  [{pol.id}] {pol.title[:60]}... priority={p:.2f}")
    
    print(f"\nAll search_kb_items results (limit=10):")
    items = search_kb_items(query, limit=10)
    for i, item in enumerate(items):
        print(f"  {i+1}. [{item.get('type')}] {item.get('snippet', '')[:60]}... priority={item.get('priority', 0):.2f}")
