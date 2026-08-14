import sys
import os
sys.path.insert(0, r'C:\Users\hp\Desktop\555 university_chatbot')

from app import create_app
from utils import generate_rag_response

app = create_app()

with app.app_context():
    queries = [
        "What are the admission requirements for Computer Science?",
        "What is the minimum attendance requirement?",
        "What is the dress code policy?",
        "How can I apply for deferment of admission?",
        "What is the fee structure?",
        "Hello",
    ]
    
    for q in queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        print('='*60)
        
        response, sources = generate_rag_response(q)
        print(f"Response:\n{response}")
        print(f"\nSources: {sources}")
