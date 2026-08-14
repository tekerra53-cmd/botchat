from app import create_app
app = create_app()
with app.app_context():
    from models import db, Document, FAQ, Policy, Calendar
    db.session.query(Document).delete()
    db.session.query(FAQ).delete()
    db.session.query(Policy).delete()
    db.session.query(Calendar).delete()
    db.session.commit()
    print('Cleared all test data: Documents, FAQs, Policies, Events!')

