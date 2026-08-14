import os
from dotenv import load_dotenv

load_dotenv()


def clean_api_key(raw_key):
    if not raw_key:
        return None
    key = raw_key.strip().strip('"').strip("'")
    return key if key else None


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'hardcoded-fallback-for-dev-only'
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'instance', 'uploads')
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB
    SHOW_AI_WARNING = False

    database_url = os.environ.get('DATABASE_URL')
    mysql_uri = (
        f"mysql+pymysql://{os.environ.get('MYSQL_USER')}:{os.environ.get('MYSQL_PASSWORD')}"
        f"@{os.environ.get('MYSQL_HOST')}/{os.environ.get('MYSQL_DB')}"
    ) if all(os.environ.get(k) for k in ['MYSQL_USER', 'MYSQL_PASSWORD', 'MYSQL_HOST', 'MYSQL_DB']) else None

    if database_url:
        SQLALCHEMY_DATABASE_URI = database_url
    elif mysql_uri:
        SQLALCHEMY_DATABASE_URI = mysql_uri
    elif os.environ.get('VERCEL'):
        SQLALCHEMY_DATABASE_URI = 'sqlite:////tmp/chatbot.db'
    else:
        SQLALCHEMY_DATABASE_URI = 'sqlite:///chatbot.db'

    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        'connect_args': {
            'check_same_thread': False,
            'timeout': 30,
        }
    }
    OPENAI_API_KEY = clean_api_key(os.environ.get('OPENAI_API_KEY'))
    
    # Chat settings
    MAX_CONTEXT_LENGTH = 4000
