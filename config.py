import os
from dotenv import load_dotenv
from urllib.parse import unquote

load_dotenv()

try:
    import pymysql
    pymysql.install_as_MySQLdb()
except Exception:
    pass


def clean_api_key(raw_key):
    if not raw_key:
        return None
    key = raw_key.strip().strip('"').strip("'")
    return key if key else None


def normalize_sqlite_url(raw_url, base_dir, on_vercel=False):
    if not raw_url or not raw_url.startswith('sqlite:///'):
        return raw_url

    path = raw_url[len('sqlite:///'):]
    if path.startswith('/') or ':' in path[:3]:
        return raw_url

    filename = os.path.basename(unquote(path)) or 'chatbot.db'
    if on_vercel:
        db_path = os.path.join('/tmp', filename).replace('\\', '/')
    else:
        db_path = os.path.join(base_dir, 'instance', filename).replace('\\', '/')
    return f'sqlite:///{db_path}'


def _is_mysql_url(raw_url):
    if not raw_url:
        return False
    return raw_url.startswith('mysql://') or raw_url.startswith('mysql+pymysql://') or raw_url.startswith('mysql+mysqlconnector://')


def _normalize_database_url(raw_url, base_dir, on_vercel=False):
    if not raw_url:
        return None
    if raw_url.startswith('mysql://') and not raw_url.startswith('mysql+pymysql://'):
        raw_url = raw_url.replace('mysql://', 'mysql+pymysql://', 1)
    if raw_url.startswith('mysql+pymysql://'):
        return raw_url
    if raw_url.startswith('sqlite:///'):
        return normalize_sqlite_url(raw_url, base_dir, on_vercel=on_vercel)
    return raw_url


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'hardcoded-fallback-for-dev-only'
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB
    SHOW_AI_WARNING = False

    database_url = os.environ.get('DATABASE_URL')
    mysql_uri = (
        f"mysql+pymysql://{os.environ.get('MYSQL_USER')}:{os.environ.get('MYSQL_PASSWORD')}"
        f"@{os.environ.get('MYSQL_HOST')}/{os.environ.get('MYSQL_DB')}"
    ) if all(os.environ.get(k) for k in ['MYSQL_USER', 'MYSQL_PASSWORD', 'MYSQL_HOST', 'MYSQL_DB']) else None

    if database_url:
        SQLALCHEMY_DATABASE_URI = _normalize_database_url(database_url, BASE_DIR, on_vercel=bool(os.environ.get('VERCEL')))
    elif mysql_uri:
        SQLALCHEMY_DATABASE_URI = mysql_uri
    elif os.environ.get('VERCEL'):
        SQLALCHEMY_DATABASE_URI = 'sqlite:////tmp/chatbot.db'
    else:
        local_db_path = os.path.join(BASE_DIR, 'instance', 'chatbot.db').replace('\\', '/')
        SQLALCHEMY_DATABASE_URI = f'sqlite:///{local_db_path}'

    print(f'[CONFIG] DATABASE_URL={database_url!r}')
    print(f'[CONFIG] SQLALCHEMY_DATABASE_URI={SQLALCHEMY_DATABASE_URI!r}')

    if os.environ.get('UPLOAD_FOLDER'):
        UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER')
    elif os.environ.get('VERCEL'):
        UPLOAD_FOLDER = '/tmp/uploads'
    else:
        UPLOAD_FOLDER = os.path.join(BASE_DIR, 'instance', 'uploads')

    SQLALCHEMY_TRACK_MODIFICATIONS = False
    if SQLALCHEMY_DATABASE_URI.startswith('sqlite'):
        SQLALCHEMY_ENGINE_OPTIONS = {
            'connect_args': {
                'check_same_thread': False,
                'timeout': 30,
            }
        }
    else:
        SQLALCHEMY_ENGINE_OPTIONS = {
            'pool_size': 5,
            'max_overflow': 10,
            'pool_timeout': 30,
            'pool_recycle': 1800,
        }
    OPENAI_API_KEY = clean_api_key(os.environ.get('OPENAI_API_KEY'))
    
    # Chat settings
    MAX_CONTEXT_LENGTH = 4000
