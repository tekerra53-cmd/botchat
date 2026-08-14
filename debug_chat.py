import sys
import os
sys.path.insert(0, r'C:\Users\hp\Desktop\555 university_chatbot')

from app import create_app
from utils import search_kb_items, search_kb, generate_rag_response

app = create_app()

with app.app_context():
    queries = [
        "What are the admission requirements for Computer Science?",
        "What is the minimum attendance requirement?",
        "What is the dress code policy?",
        "How can I apply for deferment of admission?",
        "What is the fee structure?",
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
            print(f"  Full text (first 150): {item.get('full_text', '')[:150]}")
        
        context, sources = search_kb(q)
        print(f"\nContext length: {len(context)}")
        print(f"Sources: {sources}")
        print(f"\nContext preview:\n{context[:500]}")
