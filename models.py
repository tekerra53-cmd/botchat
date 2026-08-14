from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from sqlalchemy import UniqueConstraint

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), default='user')  # admin, moderator, user
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def get_id(self):
        return str(self.id)

    def is_admin(self):
        return self.role == 'admin'

    def is_moderator(self):
        return self.role in ['admin', 'moderator']

    def __repr__(self):
        return f'<User {self.username}>'


class ChatSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)  # Allow anonymous
    session_id = db.Column(db.String(100), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_activity = db.Column(db.DateTime, default=datetime.utcnow)

    messages = db.relationship('ChatMessage', backref='session', lazy=True, cascade='all, delete-orphan')


class ChatMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('chat_session.id'), nullable=False)
    message_type = db.Column(db.String(20), nullable=False)  # 'user' or 'bot'
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    sources = db.Column(db.Text)  # JSON string of sources used


class UserFeedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    session_id = db.Column(db.Integer, db.ForeignKey('chat_session.id'), nullable=True)
    rating = db.Column(db.Integer)  # 1-5 stars
    feedback_text = db.Column(db.Text)
    message_id = db.Column(db.Integer, db.ForeignKey('chat_message.id'), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class FAQ(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    question = db.Column(db.Text, nullable=False)
    answer = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(100), default='general')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = db.Column(db.Integer, db.ForeignKey('user.id'))
    is_active = db.Column(db.Boolean, default=True)
    view_count = db.Column(db.Integer, default=0)

class Policy(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(100), default='policy')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = db.Column(db.Integer, db.ForeignKey('user.id'))
    is_active = db.Column(db.Boolean, default=True)
    view_count = db.Column(db.Integer, default=0)

class Calendar(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    event_name = db.Column(db.String(200), nullable=False)
    event_date = db.Column(db.Date, nullable=False)
    description = db.Column(db.Text)
    location = db.Column(db.String(200))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = db.Column(db.Integer, db.ForeignKey('user.id'))
    is_active = db.Column(db.Boolean, default=True)

class Document(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(100), default='handbook')
    file_path = db.Column(db.String(500))  # For uploaded files
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = db.Column(db.Integer, db.ForeignKey('user.id'))
    is_active = db.Column(db.Boolean, default=True)
    download_count = db.Column(db.Integer, default=0)


class KnowledgeBaseEntry(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    question = db.Column(db.Text, nullable=False)
    answer = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(100), default='general')
    tags = db.Column(db.Text)
    aliases = db.Column(db.Text)
    source_type = db.Column(db.String(50), default='guidance')
    audience = db.Column(db.String(50), default='student')
    priority = db.Column(db.Integer, default=0)
    is_active = db.Column(db.Boolean, default=True)
    view_count = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = db.Column(db.Integer, db.ForeignKey('user.id'))
    related_questions = db.Column(db.Text)


class SystemLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    action = db.Column(db.String(100), nullable=False)  # login, create, update, delete
    resource_type = db.Column(db.String(50), nullable=False)  # faq, policy, user, etc.
    resource_id = db.Column(db.Integer)
    details = db.Column(db.Text)
    ip_address = db.Column(db.String(45))
    user_agent = db.Column(db.Text)
    content = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(100), default='handbook')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user = db.relationship('User', backref='system_logs')


class Embedding(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content_type = db.Column(db.String(20), nullable=False)  # faq, policy, document
    content_id = db.Column(db.Integer, nullable=False)
    text = db.Column(db.Text, nullable=False)
    vector = db.Column(db.Text, nullable=False)  # JSON list of floats
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint('content_type', 'content_id', name='uq_embedding_content'),
    )
