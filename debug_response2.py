import sys
import os
sys.path.insert(0, r'C:\Users\hp\Desktop\555 university_chatbot')

from app import create_app
from utils import search_kb_items, search_kb, generate_rag_response

app = create_app()

with app.app_context():
    queries = [
        "What is the library hours?",
        "Where is the cafeteria?",
        "How do I get a student ID?",
        "Can I get a scholarship?",
        "What sports are available?",
        "How do I print my results?",
    ]
    
    for q in queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        print('='*60)
        
        items = search_kb_items(q, limit=3)
        print(f"Items found: {len(items)}")
        for i, item in enumerate(items):
            print(f"\nItem {i+1}:")
            print(f"  Type: {item.get('type')}")
            print(f"  Snippet: {item.get('snippet', '')[:100]}")
            print(f"  Priority: {item.get('priority')}")
        
        context, sources = search_kb(q)
        print(f"\nContext length: {len(context)}")
        print(f"Sources: {sources}")
        
        response, resp_sources = generate_rag_response(q)
        print(f"\nResponse:\n{response[:300]}")
