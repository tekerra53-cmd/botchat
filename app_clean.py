from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session, abort
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from config import Config
from models import db, User, FAQ, Policy, Document, Calendar, ChatSession, ChatMessage, UserFeedback, SystemLog
from utils import (
    init_openai,
    generate_rag_response,
    extract_text_from_file,
    generate_faqs_from_text,
    upsert_embedding,
    delete_embedding,
    rebuild_local_index,
)
from werkzeug.utils import secure_filename
from sqlalchemy.exc import OperationalError
from datetime import datetime, timedelta
import os
import uuid
import json
import time
import threading
import utils

login_manager = LoginManager()


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    db.init_app(app)
    login_manager.init_app(app)
    setattr(login_manager, 'login_view', 'login')  # type: ignore[assignment]

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    try:
        init_openai(app.config)
        import utils
        if utils.client:
            app.logger.info('OpenAI client initialized successfully')
        else:
            app.logger.warning('OpenAI client not initialized - using fallbacks')
    except Exception as e:
        app.logger.warning(f'OpenAI initialization error: {e}')

    with app.app_context():
        db.create_all()

        # Ensure default admin
        admin_user = User.query.filter_by(username='admin').first()
        if not admin_user:
            admin_user = User(username='admin', email='admin@university.edu', role='admin')  # type: ignore[call-arg]
            admin_user.set_password('admin123')  # Change this!
            db.session.add(admin_user)
            db.session.commit()
        else:
            if admin_user.role != 'admin':
                admin_user.role = 'admin'
            if not admin_user.email:
                admin_user.email = 'admin@university.edu'
            db.session.commit()

    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    def log_action(
        user_id,
        action,
        resource_type,
        resource_id=None,
        details=None,
        content=None,
        category=None,
    ):
        if content is None:
            if details:
                content = details
            else:
                content = f"{action} {resource_type}"
                if resource_id is not None:
                    content += f" #{resource_id}"
        if category is None:
            category = 'handbook'

        log = SystemLog(
            user_id=user_id,  # type: ignore[call-arg]
            action=action,  # type: ignore[call-arg]
            resource_type=resource_type,  # type: ignore[call-arg]
            resource_id=resource_id,  # type: ignore[call-arg]
            details=details,  # type: ignore[call-arg]
            ip_address=request.remote_addr,  # type: ignore[call-arg]
            user_agent=request.headers.get('User-Agent'),  # type: ignore[call-arg]
            content=content,  # type: ignore[call-arg]
            category=category,  # type: ignore[call-arg]
        )
        db.session.add(log)

        # SQLite can lock on concurrent writes; retry with backoff.
        retries = 4
        backoff = 0.25
        for attempt in range(1, retries + 1):
            try:
                db.session.commit()
                break
            except OperationalError as exc:
                db.session.rollback()
                if 'database is locked' in str(exc).lower() and attempt < retries:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                app.logger.error(f"Database write failed after {attempt} attempts: {exc}")
                raise

    @app.route('/')
    def index():
        session_id = session.get('chat_session_id')
        if not session_id:
            session_id = str(uuid.uuid4())
            session['chat_session_id'] = session_id
            chat_session = ChatSession(session_id=session_id)  # type: ignore[call-arg]
            db.session.add(chat_session)
            db.session.commit()
        return render_template('index.html')

    @app.route('/api/chat', methods=['POST'])
    def chat():
        data = request.json
        query = data.get('message', '').strip() if data else ''  # type: ignore[union-attr]
        if not query:
            return jsonify({'response': 'Please ask a question.'})

        session_id = session.get('chat_session_id')
        if not session_id:
            session_id = str(uuid.uuid4())
            session['chat_session_id'] = session_id
            chat_session = ChatSession(session_id=session_id)  # type: ignore[call-arg]
            db.session.add(chat_session)
            db.session.commit()

        chat_session = ChatSession.query.filter_by(session_id=session_id).first()
        if not chat_session:
            chat_session = ChatSession(session_id=session_id)  # type: ignore[call-arg]
            db.session.add(chat_session)
            db.session.commit()

        chat_session.last_activity = datetime.utcnow()
        db.session.commit()

        user_msg = ChatMessage(
            session_id=chat_session.id,  # type: ignore[call-arg]
            message_type='user',  # type: ignore[call-arg]
            content=query  # type: ignore[call-arg]
        )
        db.session.add(user_msg)

        response, sources = generate_rag_response(query)

        bot_msg = ChatMessage(
            session_id=chat_session.id,  # type: ignore[call-arg]
            message_type='bot',  # type: ignore[call-arg]
            content=response,  # type: ignore[call-arg]
            sources=json.dumps(sources) if sources else None  # type: ignore[call-arg]
        )
        db.session.add(bot_msg)
        db.session.commit()

        return jsonify({'response': response, 'sources': sources, 'message_id': bot_msg.id})

    @app.route('/api/feedback', methods=['POST'])
    def submit_feedback():
        data = request.json
        message_id = data.get('message_id') if data else None  # type: ignore[union-attr]
        rating = data.get('rating') if data else None  # type: ignore[union-attr]
        feedback_text = data.get('feedback') if data else None  # type: ignore[union-attr]

        if not message_id or not rating:
            return jsonify({'error': 'Missing required fields'}), 400

        session_id = session.get('chat_session_id')
        chat_session = ChatSession.query.filter_by(session_id=session_id).first() if session_id else None

        feedback = UserFeedback(
            user_id=current_user.id if current_user.is_authenticated else None,  # type: ignore[call-arg]
            session_id=chat_session.id if chat_session else None,  # type: ignore[call-arg]
            rating=int(rating),  # type: ignore[call-arg]
            feedback_text=feedback_text,  # type: ignore[call-arg]
            message_id=message_id  # type: ignore[call-arg]
        )
        db.session.add(feedback)
        db.session.commit()

        return jsonify({'success': True})

    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if request.method == 'POST':
            user = User.query.filter_by(username=request.form['username']).first()
            if user and user.check_password(request.form['password']) and user.is_active:
                login_user(user)
                user.last_login = datetime.utcnow()
                db.session.commit()
                log_action(user.id, 'login', 'user', user.id, f'User {user.username} logged in')
                next_page = request.args.get('next')
                return redirect(next_page) if next_page else redirect(url_for('admin'))
            flash('Invalid credentials or account inactive')
        return render_template('login.html')

    @app.route('/register', methods=['GET', 'POST'])
    def register():
        if current_user.is_authenticated:
            return redirect(url_for('admin'))

        if request.method == 'POST':
            username = request.form['username'].strip()
            email = request.form['email'].strip()
            password = request.form['password']

            if User.query.filter_by(username=username).first():
                flash('Username already exists')
                return render_template('register.html')
            if User.query.filter_by(email=email).first():
                flash('Email already registered')
                return render_template('register.html')

            user = User(username=username, email=email, role='user')
            user.set_password(password)
            db.session.add(user)
            db.session.commit()

            log_action(user.id, 'register', 'user', user.id, f'User {username} registered')
            flash('Registration successful! Please login.')
            return redirect(url_for('login'))

        return render_template('register.html')

    @app.route('/logout')
    @login_required
    def logout():
        log_action(current_user.id, 'logout', 'user', current_user.id, f'User {current_user.username} logged out')
        logout_user()
        return redirect(url_for('index'))

    def ensure_admin_access():
        if not current_user.is_moderator():
            if User.query.filter(User.role.in_(['admin', 'moderator'])).count() == 0:
                current_user.role = 'admin'
                db.session.commit()
                log_action(current_user.id, 'promote', 'user', current_user.id, 'Auto-promoted first user to admin')
                flash('No admin existed; your account has been promoted to admin for initial setup.')
            else:
                flash('Access denied: administrator or moderator role required.')
                return redirect(url_for('index'))
        return None

    def admin_context():
        faqs = FAQ.query.filter_by(is_active=True).order_by(FAQ.created_at.desc()).all()
        policies = Policy.query.filter_by(is_active=True).order_by(Policy.created_at.desc()).all()
        calendars = Calendar.query.filter_by(is_active=True).order_by(Calendar.event_date).all()
        documents = Document.query.filter_by(is_active=True).order_by(Document.created_at.desc()).all()

        total_sessions = ChatSession.query.count()
        total_messages = ChatMessage.query.count()
        avg_rating = db.session.query(db.func.avg(UserFeedback.rating)).scalar() or 0
        ai_available = bool(getattr(utils, 'client', None))
        ai_error = getattr(utils, 'last_openai_error', None)
        show_ai_warning = bool(app.config.get('SHOW_AI_WARNING', True))

        return dict(
            faqs=faqs,
            policies=policies,
            calendars=calendars,
            documents=documents,
            total_faqs=len(faqs),
            total_policies=len(policies),
            total_events=len(calendars),
            total_docs=len(documents),
            total_sessions=total_sessions,
            total_messages=total_messages,
            avg_rating=round(avg_rating, 1),
            ai_available=ai_available,
            ai_error=ai_error,
            show_ai_warning=show_ai_warning,
        )

    @app.route('/admin', methods=['GET', 'POST'])
    @login_required
    def admin():
        denied = ensure_admin_access()
        if denied:
            return denied

        if request.method == 'POST':
            action = request.form.get('action', '')

            if action == 'add_faq':
                faq = FAQ(
                    question=request.form['question'],  # type: ignore[call-arg]
                    answer=request.form['answer'],  # type: ignore[call-arg]
                    category=request.form.get('category', 'general'),  # type: ignore[call-arg]
                    created_by=current_user.id  # type: ignore[call-arg]
                )
                db.session.add(faq)
                db.session.commit()
                upsert_embedding('faq', faq.id, f"Q: {faq.question}\nA: {faq.answer}")
                log_action(current_user.id, 'create', 'faq', faq.id, f'Created FAQ: {faq.question[:50]}...')
                flash('FAQ added successfully!')

            elif action == 'edit_faq':
                faq = FAQ.query.get_or_404(request.form['faq_id'])
                if not current_user.is_admin() and faq.created_by != current_user.id:
                    abort(403)
                faq.question = request.form['question']
                faq.answer = request.form['answer']
                faq.category = request.form.get('category', 'general')
                faq.updated_at = datetime.utcnow()
                db.session.commit()
                upsert_embedding('faq', faq.id, f"Q: {faq.question}\nA: {faq.answer}")
                log_action(current_user.id, 'update', 'faq', faq.id, f'Updated FAQ: {faq.question[:50]}...')
                flash('FAQ updated successfully!')

            elif action == 'delete_faq':
                faq = FAQ.query.get_or_404(request.form['faq_id'])
                if not current_user.is_admin() and faq.created_by != current_user.id:
                    abort(403)
                db.session.delete(faq)
                db.session.commit()
                delete_embedding('faq', faq.id)
                log_action(current_user.id, 'delete', 'faq', faq.id, f'Deleted FAQ: {faq.question[:50]}...')
                flash('FAQ deleted successfully!')

            elif action == 'add_policy':
                policy = Policy(
                    title=request.form['title'],  # type: ignore[call-arg]
                    content=request.form['content'],  # type: ignore[call-arg]
                    category=request.form.get('category', 'policy'),  # type: ignore[call-arg]
                    created_by=current_user.id  # type: ignore[call-arg]
                )
                db.session.add(policy)
                db.session.commit()
                upsert_embedding('policy', policy.id, f"{policy.title}\n{policy.content}")
                log_action(current_user.id, 'create', 'policy', policy.id, f'Created policy: {policy.title}')
                flash('Policy added successfully!')

            elif action == 'edit_policy':
                policy = Policy.query.get_or_404(request.form['policy_id'])
                if not current_user.is_admin() and policy.created_by != current_user.id:
                    abort(403)
                policy.title = request.form['title']
                policy.content = request.form['content']
                policy.category = request.form.get('category', 'policy')
                policy.updated_at = datetime.utcnow()
                db.session.commit()
                upsert_embedding('policy', policy.id, f"{policy.title}\n{policy.content}")
                log_action(current_user.id, 'update', 'policy', policy.id, f'Updated policy: {policy.title}')
                flash('Policy updated successfully!')

            elif action == 'delete_policy':
                policy = Policy.query.get_or_404(request.form['policy_id'])
                if not current_user.is_admin() and policy.created_by != current_user.id:
                    abort(403)
                db.session.delete(policy)
                db.session.commit()
                delete_embedding('policy', policy.id)
                log_action(current_user.id, 'delete', 'policy', policy.id, f'Deleted policy: {policy.title}')
                flash('Policy deleted successfully!')

            elif action == 'add_calendar':
                from datetime import date
                cal = Calendar(
                    event_name=request.form['event_name'],  # type: ignore[call-arg]
                    event_date=date.fromisoformat(request.form['event_date']),  # type: ignore[call-arg]
                    description=request.form.get('description'),  # type: ignore[call-arg]
                    location=request.form.get('location'),  # type: ignore[call-arg]
                    created_by=current_user.id  # type: ignore[call-arg]
                )
                db.session.add(cal)
                db.session.commit()
                log_action(current_user.id, 'create', 'calendar', cal.id, f'Created event: {cal.event_name}')
                flash('Event added successfully!')

            elif action == 'edit_calendar':
                cal = Calendar.query.get_or_404(request.form['calendar_id'])
                if not current_user.is_admin() and cal.created_by != current_user.id:
                    abort(403)
                cal.event_name = request.form['event_name']
                cal.event_date = request.form['event_date']
                cal.description = request.form.get('description')
                cal.location = request.form.get('location')
                cal.updated_at = datetime.utcnow()
                db.session.commit()
                log_action(current_user.id, 'update', 'calendar', cal.id, f'Updated event: {cal.event_name}')
                flash('Event updated successfully!')

            elif action == 'delete_calendar':
                cal = Calendar.query.get_or_404(request.form['calendar_id'])
                if not current_user.is_admin() and cal.created_by != current_user.id:
                    abort(403)
                db.session.delete(cal)
                db.session.commit()
                log_action(current_user.id, 'delete', 'calendar', cal.id, f'Deleted event: {cal.event_name}')
                flash('Event deleted successfully!')

            elif action == 'add_doc':
                title = request.form['title'].strip()
                category = request.form.get('category', 'handbook')
                content = request.form.get('content', '').strip()
                uploaded = request.files.get('file')
                file_path = None

                if uploaded and uploaded.filename:
                    filename = secure_filename(uploaded.filename)
                    if not filename:
                        flash('Invalid file name.')
                        return redirect(url_for('admin'))
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}_{filename}")
                    uploaded.save(file_path)
                    try:
                        content = extract_text_from_file(file_path, filename)
                    except Exception as exc:
                        try:
                            if os.path.exists(file_path):
                                os.remove(file_path)
                        except Exception:
                            pass
                        flash(f'File processing failed: {exc}')
                        return redirect(url_for('admin'))

                if not content:
                    flash('Please provide content or upload a file.')
                    return redirect(url_for('admin'))

                doc = Document(
                    title=title,  # type: ignore[call-arg]
                    content=content,  # type: ignore[call-arg]
                    category=category,  # type: ignore[call-arg]
                    created_by=current_user.id,  # type: ignore[call-arg]
                    file_path=file_path,  # type: ignore[call-arg]
                )
                db.session.add(doc)
                db.session.commit()
                upsert_embedding('document', doc.id, f"{doc.title}\n{doc.content}")
                log_action(current_user.id, 'create', 'document', doc.id, f'Created document: {doc.title}')

                if request.form.get('generate_faqs') == '1':
                    try:
                        max_faqs = int(request.form.get('faq_count', '8') or 8)
                    except ValueError:
                        max_faqs = 8

                    def _faq_worker(doc_id, user_id, max_faqs, category):
                        with app.app_context():
                            doc_row = Document.query.get(doc_id)
                            if not doc_row or not doc_row.content:
                                return
                            faqs = generate_faqs_from_text(doc_row.content, max_faqs=max_faqs, category=category)
                            added = 0
                            for item in faqs:
                                question = item['question'].strip()
                                answer = item['answer'].strip()
                                if not question or not answer:
                                    continue
                                exists = FAQ.query.filter(FAQ.question.ilike(question)).first()
                                if exists:
                                    continue
                                faq = FAQ(
                                    question=question,  # type: ignore[call-arg]
                                    answer=answer,  # type: ignore[call-arg]
                                    category=item.get('category', 'general'),  # type: ignore[call-arg]
                                    created_by=user_id  # type: ignore[call-arg]
                                )
                                db.session.add(faq)
                                added += 1
                            if added:
                                db.session.commit()
                                created = FAQ.query.filter(FAQ.created_by == user_id).order_by(FAQ.id.desc()).limit(added).all()
                                for row in created:
                                    upsert_embedding('faq', row.id, f"Q: {row.question}\nA: {row.answer}")

                    threading.Thread(
                        target=_faq_worker,
                        args=(doc.id, current_user.id, max_faqs, category),
                        daemon=True,
                    ).start()
                    flash('Document added. FAQ generation started in background.')
                else:
                    flash('Document added successfully!')

            elif action == 'edit_doc':
                doc = Document.query.get_or_404(request.form['doc_id'])
                if not current_user.is_admin() and doc.created_by != current_user.id:
                    abort(403)
                doc.title = request.form['title']
                doc.content = request.form['content']
                doc.category = request.form.get('category', 'handbook')
                doc.updated_at = datetime.utcnow()
                db.session.commit()
                upsert_embedding('document', doc.id, f"{doc.title}\n{doc.content}")
                log_action(current_user.id, 'update', 'document', doc.id, f'Updated document: {doc.title}')
                flash('Document updated successfully!')

            elif action == 'delete_doc':
                doc = Document.query.get_or_404(request.form['doc_id'])
                if not current_user.is_admin() and doc.created_by != current_user.id:
                    abort(403)
                # Clean up uploaded file if exists
                if doc.file_path and os.path.exists(doc.file_path):
                    try:
                        os.remove(doc.file_path)
                    except OSError:
                        pass
                db.session.delete(doc)
                db.session.commit()
                delete_embedding('document', doc.id)
                log_action(current_user.id, 'delete', 'document', doc.id, f'Deleted document: {doc.title}')
                flash('Document deleted successfully!')

        return render_template('admin_create.html', **admin_context())

    @app.route('/admin/documents')
    @login_required
    def admin_documents():
        denied = ensure_admin_access()
        if denied:
            return denied
        return render_template('admin_documents.html', **admin_context())

    @app.route('/admin/records')
    @login_required
    def admin_records():
        denied = ensure_admin_access()
        if denied:
            return denied
        return render_template('admin_records.html', **admin_context())

    @app.route('/admin/users', methods=['GET', 'POST'])
    @login_required
    def manage_users():
        if not current_user.is_admin():
            abort(403)

        if request.method == 'POST':
            action = request.form.get('action')

            if action == 'create_user':
                username = request.form['username'].strip()
                email = request.form['email'].strip()
                password = request.form['password']
                role = request.form.get('role', 'user')

                if User.query.filter_by(username=username).first():
                    flash('Username already exists')
                    return redirect(url_for('manage_users'))

                user = User(username=username, email=email, role=role)  # type: ignore[call-arg]
                user.set_password(password)
                db.session.add(user)
                db.session.commit()
                log_action(current_user.id, 'create', 'user', user.id, f'Created user: {username}')
                flash('User created successfully!')

            elif action == 'update_user':
                user = User.query.get_or_404(request.form['user_id'])
                user.email = request.form['email']
                user.role = request.form.get('role', 'user')
                user.is_active = 'is_active' in request.form
                db.session.commit()
                log_action(current_user.id, 'update', 'user', user.id, f'Updated user: {user.username}')
                flash('User updated successfully!')

            elif action == 'delete_user':
                user = User.query.get_or_404(request.form['user_id'])
                if user.id == current_user.id:
                    flash('Cannot delete your own account!')
                    return redirect(url_for('manage_users'))
                db.session.delete(user)
                db.session.commit()
                log_action(current_user.id, 'delete', 'user', user.id, f'Deleted user: {user.username}')
                flash('User deleted successfully!')

        users = User.query.order_by(User.created_at.desc()).all()
        return render_template('users.html', users=users)

    @app.route('/admin/analytics')
    @login_required
    def analytics():
        if not current_user.is_moderator():
            abort(403)

        total_sessions = ChatSession.query.count()
        total_messages = ChatMessage.query.count()
        user_messages = ChatMessage.query.filter_by(message_type='user').count()
        bot_messages = ChatMessage.query.filter_by(message_type='bot').count()

        feedback_count = UserFeedback.query.count()
        avg_rating = db.session.query(db.func.avg(UserFeedback.rating)).scalar() or 0

        faq_views = db.session.query(db.func.sum(FAQ.view_count)).scalar() or 0
        policy_views = db.session.query(db.func.sum(Policy.view_count)).scalar() or 0
        doc_downloads = db.session.query(db.func.sum(Document.download_count)).scalar() or 0

        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        recent_sessions = ChatSession.query.filter(ChatSession.created_at >= thirty_days_ago).count()
        recent_feedback = UserFeedback.query.filter(UserFeedback.created_at >= thirty_days_ago).count()

        popular_faqs = FAQ.query.filter_by(is_active=True).order_by(FAQ.view_count.desc()).limit(5).all()
        recent_logs = SystemLog.query.order_by(SystemLog.created_at.desc()).limit(8).all()

        days = 14
        start_date = (datetime.utcnow() - timedelta(days=days - 1)).date()
        start_dt = datetime.combine(start_date, datetime.min.time())

        def _daily_counts(model, field):
            rows = (
                db.session.query(db.func.date(field), db.func.count(model.id))
                .filter(field >= start_dt)
                .group_by(db.func.date(field))
                .all()
            )
            counts = {str(row[0]): int(row[1]) for row in rows}
            labels = []
            series = []
            for i in range(days):
                day = start_date + timedelta(days=i)
                labels.append(day.strftime('%b %d'))
                series.append(counts.get(day.isoformat(), 0))
            return labels, series

        activity_labels, session_series = _daily_counts(ChatSession, ChatSession.created_at)
        _, message_series = _daily_counts(ChatMessage, ChatMessage.timestamp)
        _, feedback_series = _daily_counts(UserFeedback, UserFeedback.created_at)

        return render_template(
            'analytics.html',
            total_sessions=total_sessions,
            total_messages=total_messages,
            user_messages=user_messages,
            bot_messages=bot_messages,
            feedback_count=feedback_count,
            avg_rating=round(avg_rating, 1),
            faq_views=faq_views,
            policy_views=policy_views,
            doc_downloads=doc_downloads,
            recent_sessions=recent_sessions,
            recent_feedback=recent_feedback,
            popular_faqs=popular_faqs,
            recent_logs=recent_logs,
            activity_labels=activity_labels,
            session_series=session_series,
            message_series=message_series,
            feedback_series=feedback_series,
        )

    @app.route('/admin/logs')
    @login_required
    def view_logs():
        if not current_user.is_admin():
            abort(403)

        page = request.args.get('page', 1, type=int)
        logs = SystemLog.query.order_by(SystemLog.created_at.desc()).paginate(page=page, per_page=20)
        action_icons = {
            'login': 'sign-in-alt',
            'logout': 'sign-out-alt',
            'create': 'plus',
            'update': 'edit',
            'delete': 'trash',
            'register': 'user-plus',
            'promote': 'user-shield',
        }
        return render_template('logs.html', logs=logs, action_icons=action_icons)

    @app.route('/admin/export')
    @login_required
    def export_data():
        if not current_user.is_admin():
            abort(403)
        logs = SystemLog.query.order_by(SystemLog.created_at.desc()).all()
        lines = ['timestamp,user,action,resource_type,resource_id,details,ip_address']
        for log in logs:
            user = log.user.username if hasattr(log, 'user') and log.user else ''
            details = (log.details or '').replace('\n', ' ').replace(',', ';')
            lines.append(
                f"{log.created_at},{user},{log.action},{log.resource_type},{log.resource_id or ''},{details},{log.ip_address or ''}"
            )
        csv_data = '\n'.join(lines)
        return app.response_class(csv_data, mimetype='text/csv', headers={'Content-Disposition': 'attachment;filename=system_logs.csv'})

    @app.route('/admin/preview_doc', methods=['POST'])
    @login_required
    def preview_doc():
        if not current_user.is_moderator():
            abort(403)
        uploaded = request.files.get('file')
        if not uploaded or not uploaded.filename:
            return jsonify({'error': 'No file uploaded'}), 400
        filename = secure_filename(uploaded.filename)
        if not filename:
            return jsonify({'error': 'Invalid file name'}), 400
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"preview_{uuid.uuid4()}_{filename}")
        uploaded.save(temp_path)
        try:
            text = extract_text_from_file(temp_path, filename)
        except Exception as exc:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass
            return jsonify({'error': f'File processing failed: {exc}'}), 400
        finally:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass

        preview = (text or '').strip()
        if len(preview) > 3000:
            preview = preview[:3000] + '\n... (truncated)'
        return jsonify({'preview': preview})

    @app.route('/admin/retry_ai', methods=['POST'])
    @login_required
    def retry_ai():
        if not current_user.is_moderator():
            abort(403)
        try:
            init_openai(app.config)
        except Exception:
            pass
        ai_available = bool(getattr(utils, 'client', None))  # type: ignore[name-defined]
        ai_error = getattr(utils, 'last_openai_error', None)  # type: ignore[name-defined]
        return jsonify({'ai_available': ai_available, 'ai_error': ai_error})


    @app.route('/admin/rebuild_index', methods=['POST'])
    @login_required
    def rebuild_index():
        if not current_user.is_moderator():
            abort(403)
        try:
            rebuild_local_index()
        except Exception as exc:
            return jsonify({'success': False, 'error': str(exc)}), 500
        return jsonify({'success': True})

    @app.route('/admin/export_analytics')
    @login_required
    def admin_export_analytics():
        if not current_user.is_moderator():
            abort(403)

        data = {
            'timestamp': datetime.utcnow().isoformat(),
            'total_sessions': ChatSession.query.count(),
            'total_messages': ChatMessage.query.count(),
            'user_messages': ChatMessage.query.filter_by(message_type='user').count(),
            'bot_messages': ChatMessage.query.filter_by(message_type='bot').count(),
            'feedback_count': UserFeedback.query.count(),
            'avg_rating': db.session.query(db.func.avg(UserFeedback.rating)).scalar() or 0,
            'faq_views': db.session.query(db.func.sum(FAQ.view_count)).scalar() or 0,
            'policy_views': db.session.query(db.func.sum(Policy.view_count)).scalar() or 0,
            'doc_downloads': db.session.query(db.func.sum(Document.download_count)).scalar() or 0,
            'recent_sessions_30d': ChatSession.query.filter(ChatSession.created_at >= datetime.utcnow() - timedelta(days=30)).count(),
            'recent_feedback_30d': UserFeedback.query.filter(UserFeedback.created_at >= datetime.utcnow() - timedelta(days=30)).count(),
        }

        lines = ['metric,value']
        for key, value in data.items():
            lines.append(f"{key},{value}")

        csv_data = '\n'.join(lines)
        return app.response_class(csv_data, mimetype='text/csv', headers={'Content-Disposition': 'attachment;filename=analytics.csv'})

    return app
