from app import create_app
app = create_app()
with app.app_context():
    from models import db, FAQ, Document
    # Test FAQ
    faq = FAQ(question="Test question?", answer="Test answer", category="test", created_by=1)
    db.session.add(faq)
    # Test Document
    doc = Document(title="Test Doc", content="Test content", category="test", created_by=1)
    db.session.add(doc)
    db.session.commit()
    print('Added test FAQ & Document. Refresh /admin to test delete!')

