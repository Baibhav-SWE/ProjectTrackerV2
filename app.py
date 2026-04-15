from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_file
from datetime import datetime, timedelta
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
import os
import secrets
import plotly
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
import html
import numpy as np
import re
from urllib.parse import urlparse
from pathlib import Path
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import OperationFailure
from bson import ObjectId
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail as SendGridMail, Email, To, Content

# Load only `.env` beside this file (not `.env.example` or any other name; not cwd-dependent).
_load_dotenv_path = Path(__file__).resolve().parent / '.env'
load_dotenv(_load_dotenv_path)

# Initialize OpenAI client with API key (optional)
openai_api_key = os.getenv('OPENAI_API_KEY')
openai_client = None
if openai_api_key:
    from openai import OpenAI
    openai_client = OpenAI(api_key=openai_api_key)
else:
    print("Warning: OpenAI API key not found. LLM chatbot feature will be disabled.")

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here')
_sg_key = (os.getenv('SENDGRID_API_KEY') or '').strip()
if _sg_key.lower().startswith('bearer '):
    _sg_key = _sg_key[7:].strip()
if (len(_sg_key) >= 2) and ((_sg_key[0] == _sg_key[-1]) and _sg_key[0] in '"\''):
    _sg_key = _sg_key[1:-1].strip()
app.config['SENDGRID_API_KEY'] = _sg_key
app.config['EMAIL_FROM'] = (os.getenv('EMAIL_FROM') or '').strip()
# Correct absolute URLs in password-reset emails when not using request context defaults
if os.getenv('SERVER_NAME'):
    app.config['SERVER_NAME'] = os.getenv('SERVER_NAME').strip()
if os.getenv('PREFERRED_URL_SCHEME'):
    app.config['PREFERRED_URL_SCHEME'] = os.getenv('PREFERRED_URL_SCHEME').strip()

# MongoDB connection
MONGODB_URI = os.getenv('MONGODB_URI')
mongo_client = None
mongo_db = None

if MONGODB_URI:
    try:
        print(f"Attempting to connect to MongoDB...")
        mongo_client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000, tlsAllowInvalidCertificates=True)
        mongo_client.server_info()  # Force connection
        print("Successfully connected to MongoDB")
        # Get database name from URI or use default
        db_name = MONGODB_URI.split('/')[-1].split('?')[0] if '/' in MONGODB_URI else 'project_tracker'
        mongo_db = mongo_client[db_name]
    except Exception as e:
        print(f"Error connecting to MongoDB: {str(e)}")
        mongo_client = None
        mongo_db = None
else:
    print("Warning: MONGODB_URI not found in environment variables.")

# Helper function to get collections
def get_users_collection():
    return mongo_db['users'] if mongo_db is not None else None

def get_samples_collection():
    return mongo_db['samples'] if mongo_db is not None else None

def get_experiments_collection():
    return mongo_db['experiments'] if mongo_db is not None else None

def get_prefixes_collection():
    return mongo_db['prefixes'] if mongo_db is not None else None

def get_trash_collection():
    return mongo_db['trash'] if mongo_db is not None else None

def get_plots_collection():
    return mongo_db['plots'] if mongo_db is not None else None


def get_compare_pre_post_collections():
    """Return (`pre_data`, `post_data`) collection handles on the same cluster.

    Order: ``MONGODB_COMPARE_DB`` if set → database from ``MONGODB_URI`` → then
    ``AWI_users`` if those collections have documents (common Atlas layout).

    If your URI points at e.g. ``project_tracker`` but pre/post live under
    ``AWI_users``, either set ``MONGODB_COMPARE_DB=AWI_users`` or rely on the
    automatic fallback when the URI database has no rows in those collections.
    """
    if mongo_client is None:
        return None, None

    override = os.getenv('MONGODB_COMPARE_DB', '').strip()

    def cols(db):
        if db is None:
            return None, None
        return db['pre_data'], db['post_data']

    def either_has_docs(pre, post):
        try:
            return pre.find_one({}) is not None or post.find_one({}) is not None
        except Exception:
            return False

    candidates = []
    if override:
        candidates.append(mongo_client[override])
    if mongo_db is not None:
        candidates.append(mongo_db)
    for fb in ('AWI_users',):
        candidates.append(mongo_client[fb])

    seen = set()
    ordered_dbs = []
    for db in candidates:
        if db is None:
            continue
        n = db.name
        if n in seen:
            continue
        seen.add(n)
        ordered_dbs.append(db)

    for db in ordered_dbs:
        pre, post = cols(db)
        if pre is None or post is None:
            continue
        if either_has_docs(pre, post):
            if mongo_db is not None and db.name != mongo_db.name and not override:
                print(
                    f"Compare tab: using database {db.name!r} for pre_data/post_data "
                    f"(URI default is {mongo_db.name!r}). Set MONGODB_COMPARE_DB={db.name} in .env."
                )
            return pre, post

    db = mongo_client[override] if override else mongo_db
    if db is None:
        return None, None
    return db['pre_data'], db['post_data']


# Admin required decorator
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('is_admin', False):
            flash('This operation requires admin privileges', 'error')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# MongoDB index already present but with different options (e.g. sparse vs not) — safe to skip
_INDEX_CONFLICT_CODES = frozenset({85, 86})  # IndexOptionsConflict, IndexKeySpecsConflict


def _try_create_index(collection, keys, **kwargs):
    try:
        collection.create_index(keys, **kwargs)
    except OperationFailure as e:
        if e.code in _INDEX_CONFLICT_CODES:
            return
        print(f"Note: Could not create index {keys!r}: {e}")
    except Exception as e:
        print(f"Note: Could not create index {keys!r}: {e}")


# Initialize database with admin user
def init_db():
    if mongo_db is None:
        print("Warning: MongoDB not connected. Cannot initialize database.")
        return
    
    users = get_users_collection()
    if users is None:
        return
    
    _try_create_index(users, 'username', unique=True, sparse=True)
    _try_create_index(users, 'email', unique=True, sparse=True)
    
    samples = get_samples_collection()
    if samples is not None:
        _try_create_index(samples, 'id', unique=True, sparse=True)
    
    prefixes = get_prefixes_collection()
    if prefixes is not None:
        _try_create_index(prefixes, 'prefix', unique=True, sparse=True)
    
    # Check if admin user exists
    try:
        existing_admin = users.find_one({'username': 'admin'})
        if not existing_admin:
            print("No admin user found, creating admin user...")
            admin_user = {
                'username': 'admin',
                'email': 'admin@example.com',
                'password': generate_password_hash('admin123', method='pbkdf2:sha256'),
                'is_admin': True,
                'is_active': True,
                'created_at': datetime.utcnow(),
                'last_login': None,
                'reset_token': None,
                'reset_token_expiry': None,
                'notification_preferences': {
                    'email_notifications': True,
                    'system_notifications': True
                },
                'dashboard_preferences': {
                    'recent_activity': True,
                    'saved_queries': []
                }
            }
            users.insert_one(admin_user)
            print("Created admin user with username: admin, password: admin123")
        else:
            print("Admin user already exists")
    except Exception as e:
        print(f"Error during admin user setup: {e}")

# Initialize on startup
init_db()

# Routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        login_id = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        
        users = get_users_collection()
        if users is None:
            flash('Database not connected', 'error')
            return redirect(url_for('login'))
        
        if not login_id:
            flash('Please enter your username or email.', 'error')
            return redirect(url_for('login'))
        
        user = users.find_one({'$or': [{'username': login_id}, {'email': login_id}]})
        if not user and '@' in login_id:
            user = users.find_one(
                {'email': {'$regex': f'^{re.escape(login_id)}$', '$options': 'i'}}
            )
        
        if user and check_password_hash(user['password'], password):
            if not user.get('is_active', True):
                flash('Your account has been deactivated. Please contact an administrator.', 'error')
                return redirect(url_for('login'))
            
            session['user_id'] = str(user['_id'])
            session['username'] = user['username']
            session['is_admin'] = user.get('is_admin', False)
            
            # Update last login
            users.update_one({'_id': user['_id']}, {'$set': {'last_login': datetime.utcnow()}})
            
            return redirect(url_for('index'))
        else:
            flash('Invalid username/email or password', 'error')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    samples = get_samples_collection()
    if samples is None:
        flash('Database not connected', 'error')
        return render_template('index.html', samples=[])
    
    all_samples = list(samples.find().sort([('company_name', 1)]))
    # Sort by company name and sequence number
    all_samples.sort(key=lambda x: (
        x.get('company_name', '').lower(),
        int(x.get('id', '0-0-0').split('-')[-1]) if x.get('id', '0-0-0').split('-')[-1].isdigit() else float('inf')
    ))
    return render_template('index.html', samples=all_samples)

@app.route('/add', methods=['GET', 'POST'])
@login_required
def add_sample():
    prefixes = get_prefixes_collection()
    all_prefixes = list(prefixes.find().sort('full_form', 1)) if prefixes is not None else []

    if request.method == 'POST':
        samples = get_samples_collection()
        if samples is None:
            flash('Database not connected', 'error')
            return render_template('add.html', prefixes=all_prefixes)
        
        company_name = request.form['company_prefix']
        erb_number = request.form['ERB']
        sample_id_part = request.form['sample_id']
        
        if not sample_id_part:
            flash('Sample ID is required!', 'error')
            return render_template('add.html', prefixes=all_prefixes)
        
        full_sample_id = f"{company_name}-Ex{erb_number}-{sample_id_part}"
        
        # Check if sample ID already exists
        existing_sample = samples.find_one({'id': full_sample_id})
        if existing_sample:
            flash('Sample ID already exists! Please choose a different ID.', 'error')
            return render_template('add.html', prefixes=all_prefixes)
        
        # Handle image upload
        sample_image = None
        if 'sample_image' in request.files:
            file = request.files['sample_image']
            if file and file.filename:
                allowed_extensions = {'jpg', 'jpeg', 'png'}
                if '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions:
                    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
                    file.save(os.path.join('static', 'sample_images', filename))
                    sample_image = f"sample_images/{filename}"

        cleaning = 'Y' if request.form.get('cleaning') == 'on' else 'N'
        coating = 'Y' if request.form.get('coating') == 'on' else 'N'
        annealing = 'Y' if request.form.get('annealing') == 'on' else 'N'
        done = 'Y' if all([cleaning == 'Y', coating == 'Y', annealing == 'Y']) else 'N'
        
        new_sample = {
            'id': full_sample_id,
            'company_name': company_name,
            'ERB': erb_number,
            'ERB_description': request.form.get('ERB_description'),
            'date': request.form['date'],
            'time': request.form['time'],
            'am_pm': request.form['am_pm'],
            'recipe_front': request.form['recipe_front'],
            'recipe_back': request.form['recipe_back'],
            'glass_type': request.form['glass_type'],
            'length': int(request.form['length']),
            'thickness': int(request.form['thickness']),
            'height': int(request.form['height']),
            'cleaning': cleaning,
            'coating': coating,
            'annealing': annealing,
            'done': done,
            'sample_image': sample_image,
            'image_description': request.form.get('image_description'),
            'created_at': datetime.utcnow()
        }
        samples.insert_one(new_sample)

        # Create experiment if any experiment data is provided
        experiments = get_experiments_collection()
        if experiments is not None and any(request.form.get(field) for field in ['transmittance', 'reflectance', 'absorbance', 'plqy', 'sem', 'edx', 'xrd']):
            experiment = {
                'sample_id': full_sample_id,
                'transmittance': request.form.get('transmittance'),
                'reflectance': request.form.get('reflectance'),
                'absorbance': request.form.get('absorbance'),
                'plqy': request.form.get('plqy'),
                'sem': request.form.get('sem'),
                'edx': request.form.get('edx'),
                'xrd': request.form.get('xrd'),
                'created_at': datetime.utcnow()
            }
            experiments.insert_one(experiment)

        flash('Sample added successfully!', 'success')
        return redirect(url_for('index'))
    return render_template('add.html', prefixes=all_prefixes)

@app.route('/edit/<string:id>', methods=['GET', 'POST'])
@login_required
def edit_sample(id):
    samples = get_samples_collection()
    prefixes = get_prefixes_collection()
    
    if samples is None:
        flash('Database not connected', 'error')
        return redirect(url_for('index'))
    
    sample = samples.find_one({'id': id})
    if not sample:
        flash('Sample not found', 'error')
        return redirect(url_for('index'))
    
    all_prefixes = list(prefixes.find().sort('full_form', 1)) if prefixes is not None else []

    if request.method == 'POST':
        # Handle image upload
        if 'sample_image' in request.files:
            file = request.files['sample_image']
            if file and file.filename:
                allowed_extensions = {'jpg', 'jpeg', 'png'}
                if '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions:
                    # Delete old image if it exists
                    if sample.get('sample_image'):
                        old_image_path = os.path.join('static', sample['sample_image'])
                        if os.path.exists(old_image_path):
                            os.remove(old_image_path)
                    
                    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
                    file.save(os.path.join('static', 'sample_images', filename))
                    sample['sample_image'] = f"sample_images/{filename}"

        cleaning = 'Y' if request.form.get('cleaning') == 'on' else 'N'
        coating = 'Y' if request.form.get('coating') == 'on' else 'N'
        annealing = 'Y' if request.form.get('annealing') == 'on' else 'N'
        done = 'Y' if all([cleaning == 'Y', coating == 'Y', annealing == 'Y']) else 'N'

        update_data = {
            'company_name': request.form['company_prefix'],
            'ERB': request.form['ERB'],
            'ERB_description': request.form.get('ERB_description'),
            'date': request.form['date'],
            'time': request.form['time'],
            'am_pm': request.form['am_pm'],
            'recipe_front': request.form['recipe_front'],
            'recipe_back': request.form['recipe_back'],
            'glass_type': request.form['glass_type'],
            'length': int(request.form['length']),
            'thickness': int(request.form['thickness']),
            'height': int(request.form['height']),
            'cleaning': cleaning,
            'coating': coating,
            'annealing': annealing,
            'done': done,
            'image_description': request.form.get('image_description'),
            'sample_image': sample.get('sample_image')
        }
        
        samples.update_one({'id': id}, {'$set': update_data})
        flash('Sample updated successfully!', 'success')
        return redirect(url_for('index'))
    
    return render_template('edit.html', sample=sample, prefixes=all_prefixes)

@app.route('/delete/<string:id>')
@login_required
def delete_sample(id):
    samples = get_samples_collection()
    experiments = get_experiments_collection()
    trash = get_trash_collection()
    plots = get_plots_collection()
    
    if samples is None:
        flash('Database not connected', 'error')
        return redirect(url_for('index'))
    if trash is None:
        flash('Temp Files is not available (MongoDB). Sample was not deleted.', 'error')
        return redirect(url_for('index'))
    
    try:
        sample = samples.find_one({'id': id})
        if not sample:
            flash('Sample not found', 'error')
            return redirect(url_for('index'))
        
        experiment = experiments.find_one({'sample_id': id}) if experiments is not None else None
        
        trash_record = {
            'sample': sample,
            'experiment': experiment,
            'deleted_at': datetime.utcnow(),
            'deleted_by': session.get('username'),
        }
        trash.insert_one(trash_record)
        
        samples.delete_one({'id': id})
        if experiments is not None:
            experiments.delete_one({'sample_id': id})
        if plots is not None:
            plots.delete_many({'sample_id': id})
        
        flash('Record moved to Temp Files successfully!', 'success')
        
    except Exception as e:
        flash(f'Error deleting record: {str(e)}', 'error')
        
    return redirect(url_for('index'))

@app.route('/experiments')
@login_required
def experiments():
    experiments_col = get_experiments_collection()
    if experiments_col is None:
        return render_template('experiments.html', experiments=[])
    
    all_experiments = list(experiments_col.find())
    return render_template('experiments.html', experiments=all_experiments)

@app.route('/add_experiment/<string:sample_id>', methods=['GET', 'POST'])
@login_required
def add_experiment(sample_id):
    samples = get_samples_collection()
    experiments = get_experiments_collection()
    
    if samples is None:
        flash('Database not connected', 'error')
        return redirect(url_for('experiments'))
    
    sample = samples.find_one({'id': sample_id})
    if not sample:
        flash('Sample not found', 'error')
        return redirect(url_for('experiments'))
    
    if request.method == 'POST':
        def process_data(file_data):
            if not file_data:
                return None
            try:
                content = file_data.read().decode('utf-8')
                lines = content.strip().split('\n')
                data = []
                for line in lines:
                    values = line.strip().split(',')
                    if len(values) >= 2:
                        try:
                            x = float(values[0])
                            y = float(values[1])
                            data.append([x, y])
                        except ValueError:
                            continue
                return json.dumps(data)
            except Exception as e:
                print(f"Error processing data: {str(e)}")
                return None

        experiment = {
            'sample_id': sample_id,
            'transmittance': process_data(request.files.get('transmittance_file')),
            'reflectance': process_data(request.files.get('reflectance_file')),
            'absorbance': process_data(request.files.get('absorbance_file')),
            'plqy': process_data(request.files.get('plqy_file')),
            'sem': request.form.get('sem'),
            'edx': request.form.get('edx'),
            'xrd': request.form.get('xrd'),
            'created_at': datetime.utcnow()
        }
        
        if experiments is not None:
            experiments.insert_one(experiment)
        
        flash('Experiment added successfully!', 'success')
        return redirect(url_for('experiments'))
        
    return render_template('add_experiment.html', sample=sample)

@app.route('/edit_experiment/<string:id>', methods=['GET', 'POST'])
@login_required
def edit_experiment(id):
    experiments = get_experiments_collection()
    if experiments is None:
        flash('Database not connected', 'error')
        return redirect(url_for('experiments'))
    
    experiment = experiments.find_one({'sample_id': id})
    if not experiment:
        flash('Experiment not found', 'error')
        return redirect(url_for('experiments'))
    
    if request.method == 'POST':
        update_data = {
            'transmittance': request.form['transmittance'],
            'reflectance': request.form['reflectance'],
            'absorbance': request.form['absorbance'],
            'plqy': request.form['plqy'],
            'sem': request.form['sem'],
            'edx': request.form['edx'],
            'xrd': request.form['xrd']
        }
        experiments.update_one({'sample_id': id}, {'$set': update_data})
        flash('Experiment updated successfully!', 'success')
        return redirect(url_for('experiments'))
    
    return render_template('edit_experiment.html', experiment=experiment)

@app.route('/combined_view')
@login_required
def combined_view():
    samples = get_samples_collection()
    experiments = get_experiments_collection()
    
    if samples is None:
        return render_template('combined_view.html', results=[])
    
    all_samples = list(samples.find().sort([('company_name', 1)]))
    
    results = []
    for sample in all_samples:
        experiment = experiments.find_one({'sample_id': sample['id']}) if experiments is not None else None
        results.append((sample, experiment))
    
    # Sort by company name and sequence number
    results.sort(key=lambda x: (
        x[0].get('company_name', '').lower(),
        int(x[0].get('id', '0-0-0').split('-')[-1]) if x[0].get('id', '0-0-0').split('-')[-1].isdigit() else float('inf')
    ))
    
    return render_template('combined_view.html', results=results)

@app.route('/prefix_table', methods=['GET', 'POST'])
@login_required
def prefix_table():
    prefixes = get_prefixes_collection()
    
    if prefixes is None:
        flash('Database not connected', 'error')
        return render_template('prefix_table.html', prefixes=[])
    
    try:
        if request.method == 'POST':
            prefix = request.form.get('prefix')
            full_form = request.form.get('full_form')
            
            if prefix and full_form:
                existing_prefix = prefixes.find_one({'prefix': prefix})
                if existing_prefix:
                    flash('Prefix already exists!', 'error')
                else:
                    prefixes.insert_one({'prefix': prefix, 'full_form': full_form})
                    flash('Prefix added successfully!', 'success')
                
        all_prefixes = list(prefixes.find().sort('prefix', 1))
        return render_template('prefix_table.html', prefixes=all_prefixes)
    
    except Exception as e:
        print(f"Error in prefix_table: {str(e)}")
        flash('An error occurred while loading the prefix table.', 'error')
        return render_template('prefix_table.html', prefixes=[])

@app.route('/delete_prefix/<string:prefix>')
@login_required
def delete_prefix(prefix):
    prefixes = get_prefixes_collection()
    if prefixes is not None:
        try:
            prefixes.delete_one({'prefix': prefix})
            flash('Prefix deleted successfully!', 'success')
        except Exception as e:
            flash('Error deleting prefix!', 'error')
    return redirect(url_for('prefix_table'))

@app.route('/register', methods=['POST'])
def register():
    username = request.form.get('username')
    email = request.form.get('email')
    password = request.form.get('password')
    confirm_password = request.form.get('confirm_password')

    if not username or not email or not password or not confirm_password:
        flash('All fields are required.', 'error')
        return redirect(url_for('login'))

    if password != confirm_password:
        flash('Passwords do not match.', 'error')
        return redirect(url_for('login'))

    users = get_users_collection()
    if users is None:
        flash('Database not connected', 'error')
        return redirect(url_for('login'))

    existing_user = users.find_one({'$or': [{'username': username}, {'email': email}]})
    if existing_user:
        if existing_user.get('username') == username:
            flash('Username already exists.', 'error')
        else:
            flash('Email already exists.', 'error')
        return redirect(url_for('login'))

    hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
    
    new_user = {
        'username': username,
        'email': email,
        'password': hashed_password,
        'is_admin': False,
        'is_active': True,
        'created_at': datetime.utcnow(),
        'last_login': None,
        'reset_token': None,
        'reset_token_expiry': None,
        'notification_preferences': {
            'email_notifications': True,
            'system_notifications': True
        },
        'dashboard_preferences': {
            'recent_activity': True,
            'saved_queries': []
        }
    }
    
    users.insert_one(new_user)
    flash('Registration successful! You can now log in.', 'success')
    return redirect(url_for('login'))

@app.route('/trash')
@login_required
def view_trash():
    trash = get_trash_collection()
    if trash is None:
        return render_template('trash.html', trash_records=[])
    
    trash_records = list(trash.find().sort('deleted_at', -1))
    # Template expects deleted_at / deleted_by on the sample row; they live on the trash root doc
    formatted_records = []
    for record in trash_records:
        sample = record.get('sample')
        if not sample:
            continue
        sample = dict(sample)
        sample['deleted_at'] = record.get('deleted_at')
        sample['deleted_by'] = record.get('deleted_by')
        formatted_records.append((sample, record.get('experiment')))
    
    return render_template('trash.html', trash_records=formatted_records)

@app.route('/restore/<string:id>')
@login_required
def restore_from_trash(id):
    trash = get_trash_collection()
    samples = get_samples_collection()
    experiments = get_experiments_collection()
    
    if trash is None or samples is None:
        flash('Database not connected', 'error')
        return redirect(url_for('view_trash'))
    
    try:
        trash_record = trash.find_one({'sample.id': id})
        if not trash_record:
            flash('Record not found in Temp Files', 'error')
            return redirect(url_for('view_trash'))
        
        # Check if sample already exists
        if samples.find_one({'id': id}):
            flash(f'A sample with ID {id} already exists!', 'error')
            return redirect(url_for('view_trash'))
        
        trash_oid = trash_record.get('_id')
        
        # Restore sample (strip display-only keys if present)
        sample_data = trash_record.get('sample')
        if sample_data:
            sample_data = dict(sample_data)
            sample_data.pop('deleted_at', None)
            sample_data.pop('deleted_by', None)
            samples.insert_one(sample_data)
        
        # Restore experiment if exists
        experiment_data = trash_record.get('experiment')
        if experiment_data and experiments is not None:
            experiments.insert_one(dict(experiment_data))
        
        if trash_oid is not None:
            trash.delete_one({'_id': trash_oid})
        else:
            trash.delete_one({'sample.id': id})
        flash('Record restored successfully!', 'success')
        
    except Exception as e:
        flash(f'Error restoring record: {str(e)}', 'error')
        
    return redirect(url_for('view_trash'))

def _find_user_by_email(users, email):
    """Exact match first, then case-insensitive match on stored email."""
    if not email:
        return None
    user = users.find_one({'email': email})
    if user:
        return user
    return users.find_one({'email': {'$regex': f'^{re.escape(email)}$', '$options': 'i'}})


@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = (request.form.get('email') or '').strip()
        users = get_users_collection()

        if users is None:
            flash('Database not connected', 'error')
            return redirect(url_for('forgot_password'))

        if not email:
            flash('Please enter your email address.', 'error')
            return redirect(url_for('forgot_password'))

        api_key = app.config.get('SENDGRID_API_KEY') or ''
        email_from = app.config.get('EMAIL_FROM') or ''
        if not api_key or not email_from:
            flash(
                'Password reset email is not configured (missing SENDGRID_API_KEY or EMAIL_FROM). '
                'Contact your administrator.',
                'error',
            )
            return redirect(url_for('forgot_password'))

        user = _find_user_by_email(users, email)

        if not user:
            flash(
                'If that email is registered, you will receive a link to reset your password shortly.',
                'success',
            )
            return redirect(url_for('login'))

        token = secrets.token_urlsafe(32)
        users.update_one(
            {'_id': user['_id']},
            {'$set': {
                'reset_token': token,
                'reset_token_expiry': datetime.utcnow() + timedelta(hours=1),
            }},
        )

        reset_link = url_for('reset_password', token=token, _external=True)
        to_address = user.get('email') or email

        try:
            sg = SendGridAPIClient(api_key)
            message = SendGridMail(
                from_email=Email(email_from),
                to_emails=To(to_address),
                subject='Password Reset Request - Project Tracker',
                html_content=Content(
                    'text/html',
                    f'''
                        <html>
                            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                                <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                                    <h2 style="color: #ff1825;">Password Reset Request</h2>
                                    <p>Hello,</p>
                                    <p>We received a request to reset your password. Click the button below:</p>
                                    <div style="text-align: center; margin: 30px 0;">
                                        <a href="{html.escape(reset_link)}"
                                           style="background-color: #ff1825; color: white; padding: 12px 24px;
                                                  text-decoration: none; border-radius: 4px; font-weight: bold;">
                                            Reset Password
                                        </a>
                                    </div>
                                    <p>This link will expire in 1 hour.</p>
                                    <p style="font-size: 12px; color: #666;">If you did not request this, you can ignore this email.</p>
                                </div>
                            </body>
                        </html>
                        ''',
                ),
            )
            response = sg.send(message)
            status = getattr(response, 'status_code', None)
            if status is not None and status not in (200, 201, 202):
                raise RuntimeError(f'SendGrid HTTP {status}: {getattr(response, "body", "")}')

            flash(
                'If that email is registered, you will receive a link to reset your password shortly.',
                'success',
            )
            return redirect(url_for('login'))

        except Exception as e:
            err = str(e)
            print(f"Error sending email: {err}")
            users.update_one(
                {'_id': user['_id']},
                {'$set': {'reset_token': None, 'reset_token_expiry': None}},
            )
            hint = ''
            if '401' in err or 'Unauthorized' in err:
                hint = (
                    'HTTP 401 means SendGrid rejected your API key (wrong, revoked, or typo in .env)—not EMAIL_FROM or SERVER_NAME. '
                    'Create a new key with Mail Send in the SendGrid dashboard and restart the app. '
                )
            flash(
                'Could not send the reset email. '
                + hint
                + 'Otherwise verify EMAIL_FROM is a verified sender and, if reset links are wrong, set SERVER_NAME and PREFERRED_URL_SCHEME. '
                f'Details: {err[:180]}',
                'error',
            )
            return redirect(url_for('forgot_password'))

    return render_template('forgot_password.html')

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    users = get_users_collection()
    if users is None:
        flash('Database not connected', 'error')
        return redirect(url_for('forgot_password'))
    
    user = users.find_one({'reset_token': token})
    
    if not user or user.get('reset_token_expiry', datetime.min) < datetime.utcnow():
        flash('Invalid or expired password reset link.', 'error')
        return redirect(url_for('forgot_password'))
    
    if request.method == 'POST':
        password = request.form.get('password') or ''
        confirm_password = request.form.get('confirm_password') or ''

        if not password:
            flash('Please choose a password.', 'error')
            return render_template('reset_password.html', token=token)

        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return render_template('reset_password.html', token=token)

        users.update_one(
            {'_id': user['_id']},
            {'$set': {
                'password': generate_password_hash(password, method='pbkdf2:sha256'),
                'reset_token': None,
                'reset_token_expiry': None
            }}
        )
        
        flash('Your password has been reset successfully.', 'success')
        return redirect(url_for('login'))
    
    return render_template('reset_password.html', token=token)

@app.route('/plots', methods=['GET', 'POST'])
@login_required
def plots():
    samples = get_samples_collection()
    experiments = get_experiments_collection()
    plots_col = get_plots_collection()
    
    if request.method == 'POST':
        sample_id = request.form.get('sample_id')
        sharepoint_link = request.form.get('sharepoint_link')
        
        if sample_id and sharepoint_link:
            if samples is not None and not samples.find_one({'id': sample_id}):
                flash('Sample ID not found! Please enter a valid sample ID.', 'error')
            elif plots_col and plots_col.find_one({'sample_id': sample_id}):
                flash('A plot entry already exists for this sample ID!', 'error')
            elif plots_col:
                plots_col.insert_one({
                    'sample_id': sample_id,
                    'sharepoint_link': sharepoint_link,
                    'created_at': datetime.utcnow(),
                    'created_by': session.get('username')
                })
                flash('Plot entry added successfully!', 'success')
        else:
            flash('Both Sample ID and SharePoint Link are required!', 'error')
    
    # Get all experiments with samples
    plot_data = {
        'transmittance': [],
        'reflectance': [],
        'absorbance': [],
        'plqy': [],
        'sem': [],
        'edx': [],
        'xrd': []
    }
    
    if samples is not None and experiments is not None:
        all_samples = {s['id']: s for s in samples.find()}
        for exp in experiments.find():
            sample = all_samples.get(exp.get('sample_id'))
            if sample:
                for measurement_type in plot_data.keys():
                    data = exp.get(measurement_type)
                    if data:
                        plot_data[measurement_type].append({
                            'id': sample['id'],
                            'data': data,
                            'recipe_front': sample.get('recipe_front'),
                            'recipe_back': sample.get('recipe_back'),
                            'glass_type': sample.get('glass_type')
                        })
    
    # Get plots entries
    plots_entries = []
    if plots_col is not None and samples is not None:
        for plot in plots_col.find().sort('created_at', -1):
            sample = samples.find_one({'id': plot.get('sample_id')})
            plots_entries.append((plot, sample))
    
    return render_template('plots.html', plot_data=json.dumps(plot_data), plots_entries=plots_entries)

@app.route('/delete_plot/<string:plot_id>')
@login_required
def delete_plot(plot_id):
    plots_col = get_plots_collection()
    if plots_col is not None:
        try:
            plots_col.delete_one({'_id': ObjectId(plot_id)})
            flash('Plot entry deleted successfully!', 'success')
        except Exception as e:
            flash('Error deleting plot entry!', 'error')
    return redirect(url_for('plots'))

@app.route('/reset_admin', methods=['GET'])
def reset_admin():
    users = get_users_collection()
    if users is None:
        return 'Database not connected'
    
    try:
        admin = users.find_one({'username': 'admin'})
        if admin:
            users.update_one(
                {'_id': admin['_id']},
                {'$set': {'password': generate_password_hash('admin123', method='pbkdf2:sha256')}}
            )
            return 'Admin password reset successfully to "admin123"'
        else:
            users.insert_one({
                'username': 'admin',
                'email': 'admin@example.com',
                'password': generate_password_hash('admin123', method='pbkdf2:sha256'),
                'is_admin': True,
                'is_active': True,
                'created_at': datetime.utcnow()
            })
            return 'New admin user created with password "admin123"'
    except Exception as e:
        return f'Error: {str(e)}'

# Admin routes
def _user_display_name(user):
    return user.get('username') or user.get('email') or str(user.get('_id', 'user'))


def _get_user_by_id(users, user_id):
    try:
        oid = ObjectId(user_id)
    except Exception:
        return None
    return users.find_one({'_id': oid})


@app.route('/admin/users')
@login_required
@admin_required
def admin_users():
    users = get_users_collection()
    if users is None:
        return render_template('admin/users.html', users=[])
    
    all_users = list(users.find())
    return render_template('admin/users.html', users=all_users)

@app.route('/admin/users/<string:user_id>/toggle_admin', methods=['POST'])
@login_required
@admin_required
def toggle_admin_status(user_id):
    users = get_users_collection()
    if users is None:
        flash('Database not connected', 'error')
        return redirect(url_for('admin_users'))
    
    user = _get_user_by_id(users, user_id)
    if not user:
        flash('User not found', 'error')
        return redirect(url_for('admin_users'))
    
    if str(user['_id']) == session['user_id']:
        flash('You cannot modify your own admin status', 'error')
        return redirect(url_for('admin_users'))
    
    new_status = not user.get('is_admin', False)
    users.update_one({'_id': user['_id']}, {'$set': {'is_admin': new_status}})
    flash(f'Admin status {"granted" if new_status else "revoked"} for {_user_display_name(user)}', 'success')
    return redirect(url_for('admin_users'))

@app.route('/admin/users/<string:user_id>/toggle_active', methods=['POST'])
@login_required
@admin_required
def toggle_user_active(user_id):
    users = get_users_collection()
    if users is None:
        flash('Database not connected', 'error')
        return redirect(url_for('admin_users'))
    
    user = _get_user_by_id(users, user_id)
    if not user:
        flash('User not found', 'error')
        return redirect(url_for('admin_users'))
    
    if str(user['_id']) == session['user_id']:
        flash('You cannot deactivate your own account', 'error')
        return redirect(url_for('admin_users'))
    
    new_status = not user.get('is_active', True)
    users.update_one({'_id': user['_id']}, {'$set': {'is_active': new_status}})
    flash(f'User {_user_display_name(user)} has been {"activated" if new_status else "deactivated"}', 'success')
    return redirect(url_for('admin_users'))

@app.route('/admin/users/<string:user_id>/delete', methods=['POST'])
@login_required
@admin_required
def delete_user(user_id):
    users = get_users_collection()
    if users is None:
        flash('Database not connected', 'error')
        return redirect(url_for('admin_users'))
    
    user = _get_user_by_id(users, user_id)
    if not user:
        flash('User not found', 'error')
        return redirect(url_for('admin_users'))
    
    if str(user['_id']) == session['user_id']:
        flash('You cannot delete your own account', 'error')
        return redirect(url_for('admin_users'))
    
    users.delete_one({'_id': user['_id']})
    flash(f'User {_user_display_name(user)} has been deleted', 'success')
    return redirect(url_for('admin_users'))

# Chatbot helpers (MongoDB-backed)
CHATBOT_COLUMN_ALIASES = {
    'id': 'id',
    'sample id': 'id',
    'company': 'company_name',
    'company name': 'company_name',
    'company_name': 'company_name',
    'erb': 'erb',
    'erb number': 'erb',
    'erb description': 'erb_description',
    'date': 'date',
    'time': 'time',
    'recipe front': 'recipe_front',
    'recipe back': 'recipe_back',
    'glass': 'glass_type',
    'glass type': 'glass_type',
    'dimensions': 'dimensions',
    'dimension': 'dimensions',
    'cleaning': 'cleaning',
    'coating': 'coating',
    'annealing': 'annealing',
    'done': 'done',
    'transmittance': 'transmittance',
    'reflectance': 'reflectance',
    'absorbance': 'absorbance',
    'plqy': 'plqy',
    'sem': 'sem',
    'edx': 'edx',
    'xrd': 'xrd'
}


def normalize_status_value(raw_value):
    normalized = str(raw_value).strip().lower()
    if normalized in {'y', 'yes', 'true', '1'}:
        return 'Y'
    if normalized in {'n', 'no', 'false', '0'}:
        return 'N'
    return None


def parse_selected_columns(query):
    lowered = query.lower()
    selected_columns = []

    column_match = re.search(r'(?:show|display|list)\s+(.+?)\s+columns?', lowered)
    if column_match:
        chunk = column_match.group(1)
        chunk = chunk.replace(' and ', ',')
        for part in [p.strip() for p in chunk.split(',') if p.strip()]:
            mapped = CHATBOT_COLUMN_ALIASES.get(part)
            if mapped and mapped not in selected_columns:
                selected_columns.append(mapped)

    return selected_columns or None


def parse_chatbot_filters(query):
    filters = {}
    match_notes = []

    id_match = re.search(r"\bid\s*(?:=|is)?\s*['\"]?([a-zA-Z0-9\-]+)['\"]?", query, re.IGNORECASE)
    if id_match:
        sample_id = id_match.group(1).strip()
        filters['id'] = {'$regex': f'^{re.escape(sample_id)}$', '$options': 'i'}
        match_notes.append(f"id={sample_id}")

    erb_match = re.search(r"\berb(?:\s*number)?\s*(?:=|is)?\s*['\"]?([a-zA-Z0-9\-]+)['\"]?", query, re.IGNORECASE)
    if erb_match:
        erb_value = erb_match.group(1).strip()
        filters['ERB'] = {'$regex': f'^{re.escape(erb_value)}$', '$options': 'i'}
        match_notes.append(f"ERB={erb_value}")

    company_match = re.search(r"\bfrom\s+([a-zA-Z0-9 _\-&]+?)(?:\s+(?:where|with|and)\b|$)", query, re.IGNORECASE)
    if not company_match:
        company_match = re.search(r"\bcompany(?:\s*name)?\s*(?:=|is)\s*['\"]?([a-zA-Z0-9 _\-&]+?)['\"]?(?:\s|$)", query, re.IGNORECASE)
    if company_match:
        company_name = company_match.group(1).strip()
        if company_name:
            filters['company_name'] = {'$regex': re.escape(company_name), '$options': 'i'}
            match_notes.append(f"company={company_name}")

    for field in ['cleaning', 'coating', 'annealing', 'done']:
        status_match = re.search(rf"\b{field}\s*(?:=|is)?\s*['\"]?([a-zA-Z]+)['\"]?", query, re.IGNORECASE)
        if status_match:
            normalized = normalize_status_value(status_match.group(1))
            if normalized:
                filters[field] = normalized
                match_notes.append(f"{field}={normalized}")

    return filters, match_notes


def run_chatbot_query(query):
    samples = get_samples_collection()
    experiments = get_experiments_collection()

    if samples is None:
        return None, None, None, 'Database not connected. Please check your MongoDB configuration.'

    selected_columns = parse_selected_columns(query)
    sample_filters, applied_filters = parse_chatbot_filters(query)
    sample_docs = list(samples.find(sample_filters).sort([('company_name', 1), ('id', 1)]).limit(200))

    if not sample_docs:
        return [], selected_columns, query, 'No matching records found for this query.'

    experiment_map = {}
    if experiments is not None:
        sample_ids = [doc.get('id') for doc in sample_docs if doc.get('id')]
        for exp_doc in experiments.find({'sample_id': {'$in': sample_ids}}):
            experiment_map[exp_doc.get('sample_id')] = exp_doc

    results = [(sample_doc, experiment_map.get(sample_doc.get('id'))) for sample_doc in sample_docs]

    response_prefix = f"Found {len(results)} record(s)"
    if applied_filters:
        response_prefix += f" using filters: {', '.join(applied_filters)}."
    else:
        response_prefix += "."

    if selected_columns:
        response_prefix += f" Showing selected columns: {', '.join(selected_columns)}."

    return results, selected_columns, response_prefix, None


@app.route('/chatbot', methods=['GET', 'POST'])
@login_required
def chatbot():
    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        if not query:
            return render_template(
                'chatbot.html',
                results=None,
                query=query,
                error='Please enter a query.',
                response=None,
                selected_columns=None
            )

        results, selected_columns, response, error = run_chatbot_query(query)
        return render_template(
            'chatbot.html',
            results=results,
            query=query,
            error=error,
            response=response,
            selected_columns=selected_columns
        )

    return render_template('chatbot.html', results=None, query=None, error=None, response=None, selected_columns=None)

@app.route('/chatbot_new', methods=['GET', 'POST'])
@login_required
def chatbot_new():
    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        if not query:
            return render_template(
                'chatbot_new.html',
                results=None,
                query=query,
                error='Please enter a query.',
                response=None,
                selected_columns=None
            )

        results, selected_columns, response, error = run_chatbot_query(query)
        return render_template(
            'chatbot_new.html',
            results=results,
            query=query,
            error=error,
            response=response,
            selected_columns=selected_columns
        )

    return render_template('chatbot_new.html', results=None, query=None, error=None, response=None, selected_columns=None)

@app.route('/chatbot_llm', methods=['GET', 'POST'])
@login_required
def chatbot_llm():
    return redirect(url_for('chatbot_new'))

def _pre_post_entry_label(doc):
    if not doc:
        return 'Unnamed'
    for key in ('design_name', 'name', 'title', 'filename', 'file_name'):
        v = doc.get(key)
        if v is not None and str(v).strip():
            return str(v).strip()
    oid = doc.get('_id')
    return str(oid) if oid is not None else 'Unnamed'


def _coerce_pair_lists(x, y):
    if x is None or y is None:
        return [], []
    try:
        xa = [float(v) for v in x]
        ya = [float(v) for v in y]
    except (TypeError, ValueError):
        return [], []
    if len(xa) != len(ya) or not xa:
        return [], []
    return xa, ya


def _xy_from_raw(raw):
    """Build x,y lists from JSON string, [[wl,val],...], {wavelength,values}, or list of dicts."""
    if raw is None:
        return [], []
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return [], []
        try:
            return _xy_from_raw(json.loads(s))
        except json.JSONDecodeError:
            return [], []
    if isinstance(raw, dict):
        for kx, ky in (
            ('wavelength', 'value'),
            ('wavelength', 'values'),
            ('x', 'y'),
            ('X', 'Y'),
            ('lambda', 'y'),
        ):
            if kx in raw and ky in raw:
                return _coerce_pair_lists(raw[kx], raw[ky])
        return [], []
    if isinstance(raw, (list, tuple)):
        if not raw:
            return [], []
        first = raw[0]
        if isinstance(first, (list, tuple)) and len(first) >= 2:
            try:
                return [float(p[0]) for p in raw], [float(p[1]) for p in raw]
            except (TypeError, ValueError, IndexError):
                return [], []
        if isinstance(first, dict):
            xs, ys = [], []
            for p in raw:
                if not isinstance(p, dict):
                    continue
                xv = p.get('wavelength') or p.get('x') or p.get('X') or p.get('lambda')
                yv = p.get('value') or p.get('y') or p.get('Y') or p.get('values')
                if xv is None or yv is None:
                    continue
                try:
                    xs.append(float(xv))
                    ys.append(float(yv))
                except (TypeError, ValueError):
                    continue
            return (xs, ys) if xs else ([], [])
    return [], []


def _series_from_doc(doc, *keys):
    """Read series from top-level fields or from nested ``data`` (Atlas pre_data/post_data)."""
    sources = [doc]
    nested = doc.get('data')
    if isinstance(nested, dict):
        sources.append(nested)
    for src in sources:
        for k in keys:
            if k not in src:
                continue
            val = src[k]
            if val in (None, '', []):
                continue
            x, y = _xy_from_raw(val)
            if x:
                return x, y
    return [], []


def _spectra_from_wavelength_tra_map(d):
    """Parse ``{\"380\": [T, R, A], ...}`` — key = wavelength (nm), value[0]=T, [1]=R, [2]=A."""
    if not isinstance(d, dict):
        return None
    skip = frozenset({
        'transmittance', 'reflectance', 'absorbance', 'design_name', 'name', 'title',
        '_id', 'filename', 'file_name', 'data', 'created_at', 'updated_at', 'id',
    })
    points = []
    for k, v in d.items():
        if k in skip:
            continue
        try:
            wl = float(k)
        except (TypeError, ValueError):
            continue
        if not isinstance(v, (list, tuple)) or len(v) < 3:
            continue
        try:
            t, r, a = float(v[0]), float(v[1]), float(v[2])
        except (TypeError, ValueError, IndexError):
            continue
        points.append((wl, t, r, a))
    if not points:
        return None
    points.sort(key=lambda p: p[0])
    xs = [p[0] for p in points]
    return (
        (xs, [p[1] for p in points]),
        (xs, [p[2] for p in points]),
        (xs, [p[3] for p in points]),
    )


def _tra_series_from_doc(doc):
    """If ``data`` (or root) is a wavelength → [T,R,A] map, return three (x,y) series."""
    for src in (doc.get('data'), doc):
        if not isinstance(src, dict):
            continue
        triple = _spectra_from_wavelength_tra_map(src)
        if triple:
            return triple
    return None


def _resolve_tra_series(doc):
    """Three (x,y) pairs for T, R, A from either wavelength map or legacy fields."""
    triple = _tra_series_from_doc(doc)
    if triple:
        return triple[0], triple[1], triple[2]
    return (
        _series_from_doc(doc, *_T_KEYS),
        _series_from_doc(doc, *_R_KEYS),
        _series_from_doc(doc, *_A_KEYS),
    )


def _avg_optical_band(x, y, lo=400.0, hi=1200.0):
    vals = []
    for xi, yi in zip(x, y):
        try:
            xf, yf = float(xi), float(yi)
        except (TypeError, ValueError):
            continue
        if lo <= xf <= hi and yf > 0 and yf == yf:
            vals.append(yf)
    return sum(vals) / len(vals) if vals else 0.0


def _pct_gain(pre_v, post_v):
    if pre_v and pre_v > 0:
        return ((post_v - pre_v) / pre_v) * 100.0
    return 0.0


def _dual_trace_plot_json(x1, y1, x2, y2, title, yaxis_title, name_a='Pre', name_b='Post'):
    fig = go.Figure()
    if x1 and y1:
        fig.add_trace(go.Scatter(x=x1, y=y1, mode='lines', name=name_a))
    if x2 and y2:
        fig.add_trace(go.Scatter(x=x2, y=y2, mode='lines', name=name_b))
    fig.update_layout(
        title=title,
        xaxis_title='Wavelength (nm)',
        yaxis_title=yaxis_title,
        template='plotly_white',
        height=420,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    return json.dumps(fig.to_plotly_json())


def _combined_compare_plot_json(
    pre_t, pre_r, pre_a, post_t, post_r, post_a,
    pre_label, post_label,
    pre_avg_t, post_avg_t, pre_avg_r, post_avg_r, pre_avg_a, post_avg_a,
    gain_t, gain_r, gain_a,
):
    """Single figure matching View Plots: all T/R/A pre+post series, shared axes, stats annotation."""
    colors = ['#0066FF', '#FF3333', '#33CC33', '#FFD700', '#9933FF', '#FF8000']
    pl = html.escape(str(pre_label))
    ql = html.escape(str(post_label))
    stats_html = (
        f'<b>Pre ({pl}):</b> '
        f'T: {pre_avg_t:.2f}% | R: {pre_avg_r:.2f}% | A: {pre_avg_a:.3f}<br>'
        f'<b>Post ({ql}):</b> '
        f'T: {post_avg_t:.2f}% | R: {post_avg_r:.2f}% | A: {post_avg_a:.3f}<br>'
        f'<b>Gains:</b> T: {gain_t:+.2f}% | R: {gain_r:+.2f}% | A: {gain_a:+.2f}%'
    )

    series_specs = [
        (pre_t, f'Transmittance ({pre_label})', colors[0]),
        (post_t, f'Transmittance ({post_label})', colors[1]),
        (pre_r, f'Reflectance ({pre_label})', colors[2]),
        (post_r, f'Reflectance ({post_label})', colors[3]),
        (pre_a, f'Absorbance ({pre_label})', colors[4]),
        (post_a, f'Absorbance ({post_label})', colors[5]),
    ]

    fig = go.Figure()
    for (xs, ys), name, color in series_specs:
        if not xs or not ys:
            continue
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode='lines',
                type='scatter',
                name=name,
                line=dict(color=color, width=3, shape='spline', smoothing=1.2),
                hovertemplate=(
                    '<b>%{fullData.name}</b><br>Wavelength: %{x:.0f} nm<br>'
                    'Value: %{y:.4f}<br><extra></extra>'
                ),
            )
        )

    fig.update_layout(
        title=dict(
            text='TRA vs Wavelength',
            font=dict(size=20, color='#333'),
            x=0.5,
            y=0.98,
        ),
        xaxis=dict(
            title=dict(text='Wavelength (nm)', font=dict(size=14, color='#333'), standoff=20),
            showgrid=True,
            gridcolor='#E5E5E5',
            gridwidth=1,
            zeroline=False,
            tickfont=dict(size=12, color='#333'),
            range=[300, 1000],
        ),
        yaxis=dict(
            title=dict(text='TRA (%)', font=dict(size=14, color='#333'), standoff=20),
            showgrid=True,
            gridcolor='#E5E5E5',
            gridwidth=1,
            zeroline=True,
            zerolinecolor='#E5E5E5',
            tickfont=dict(size=12, color='#333'),
            range=[-25, 100],
            dtick=25,
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=1100,
        margin=dict(l=80, r=80, t=60, b=280),
        showlegend=True,
        legend=dict(
            x=0.98,
            y=1,
            xanchor='right',
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='white',
            font=dict(size=12, color='#333'),
            itemwidth=30,
            itemsizing='constant',
        ),
        annotations=[
            dict(
                xref='paper',
                yref='paper',
                x=0.5,
                y=-0.14,
                xanchor='center',
                yanchor='top',
                text=f'<b style="font-size: 16px; color: #333;">Statistics Summary</b><br>{stats_html}',
                showarrow=False,
                font=dict(family='monospace', size=13, color='#333'),
                align='center',
                bgcolor='white',
                bordercolor='#333',
                borderwidth=1,
                borderpad=10,
                width=820,
            )
        ],
        hovermode='closest',
        hoverdistance=10,
    )
    return json.dumps(fig.to_plotly_json())


_T_KEYS = (
    'transmittance', 'Transmittance', 'TRA', 'tra', 'T', 't',
    'pre_transmittance', 'post_transmittance',
)
_R_KEYS = (
    'reflectance', 'Reflectance', 'R', 'r',
    'pre_reflectance', 'post_reflectance',
)
_A_KEYS = (
    'absorbance', 'Absorbance', 'A', 'a',
    'pre_absorbance', 'post_absorbance',
)


@app.route('/compare', methods=['GET', 'POST'])
@login_required
def compare():
    def render_compare(pre_files, post_files, **extra):
        ctx = {
            'pre_data_files': pre_files,
            'post_data_files': post_files,
            'show_selection': True,
            'error': extra.pop('error', False),
            'selected_pre_file': extra.pop('selected_pre_file', None),
            'selected_post_file': extra.pop('selected_post_file', None),
        }
        ctx.update(extra)
        return render_template('compare.html', **ctx)

    label_projection = {
        'design_name': 1, 'name': 1, 'title': 1, 'filename': 1, 'file_name': 1,
    }

    try:
        if mongo_client is None:
            flash('MongoDB is not connected.', 'error')
            return render_compare([], [], error=False)

        pre_col, post_col = get_compare_pre_post_collections()
        if pre_col is None or post_col is None:
            flash('Could not open pre_data / post_data collections.', 'error')
            return render_compare([], [], error=False)

        pre_data_files = [
            {'_id': str(d['_id']), 'label': _pre_post_entry_label(d)}
            for d in pre_col.find({}, label_projection)
        ]
        post_data_files = [
            {'_id': str(d['_id']), 'label': _pre_post_entry_label(d)}
            for d in post_col.find({}, label_projection)
        ]

        selected_pre = request.form.get('pre_file_id') if request.method == 'POST' else None
        selected_post = request.form.get('post_file_id') if request.method == 'POST' else None

        if request.method == 'GET' or not (selected_pre and selected_post):
            return render_compare(
                pre_data_files, post_data_files,
                selected_pre_file=selected_pre, selected_post_file=selected_post,
            )

        if selected_pre == selected_post:
            flash('Choose different documents for pre and post.', 'error')
            return render_compare(
                pre_data_files, post_data_files,
                selected_pre_file=selected_pre, selected_post_file=selected_post,
            )

        try:
            oid_pre = ObjectId(selected_pre)
            oid_post = ObjectId(selected_post)
        except Exception:
            flash('Invalid document id.', 'error')
            return render_compare(
                pre_data_files, post_data_files,
                selected_pre_file=selected_pre, selected_post_file=selected_post,
            )

        pre_doc = pre_col.find_one({'_id': oid_pre})
        post_doc = post_col.find_one({'_id': oid_post})
        if not pre_doc or not post_doc:
            flash('Pre or post document was not found.', 'error')
            return render_compare(
                pre_data_files, post_data_files,
                selected_pre_file=selected_pre, selected_post_file=selected_post,
            )

        pre_label = _pre_post_entry_label(pre_doc)
        post_label = _pre_post_entry_label(post_doc)

        pre_t, pre_r, pre_a = _resolve_tra_series(pre_doc)
        post_t, post_r, post_a = _resolve_tra_series(post_doc)

        if not (pre_t[0] or post_t[0] or pre_r[0] or post_r[0] or pre_a[0] or post_a[0]):
            flash(
                'No transmittance, reflectance, or absorbance data found. '
                'Expected a data object mapping wavelength to [T,R,A] (e.g. "380": [T,R,A]) '
                'or separate transmittance / reflectance / absorbance fields.',
                'error',
            )
            return render_compare(
                pre_data_files, post_data_files,
                selected_pre_file=selected_pre, selected_post_file=selected_post,
            )

        pre_avg_transmittance = _avg_optical_band(pre_t[0], pre_t[1])
        post_avg_transmittance = _avg_optical_band(post_t[0], post_t[1])
        pre_avg_reflectance = _avg_optical_band(pre_r[0], pre_r[1])
        post_avg_reflectance = _avg_optical_band(post_r[0], post_r[1])
        pre_avg_absorbance = _avg_optical_band(pre_a[0], pre_a[1])
        post_avg_absorbance = _avg_optical_band(post_a[0], post_a[1])
        tg = _pct_gain(pre_avg_transmittance, post_avg_transmittance)
        rg = _pct_gain(pre_avg_reflectance, post_avg_reflectance)
        ag = _pct_gain(pre_avg_absorbance, post_avg_absorbance)

        combined_compare_plot = _combined_compare_plot_json(
            pre_t, pre_r, pre_a, post_t, post_r, post_a,
            pre_label, post_label,
            pre_avg_transmittance, post_avg_transmittance,
            pre_avg_reflectance, post_avg_reflectance,
            pre_avg_absorbance, post_avg_absorbance,
            tg, rg, ag,
        )

        return render_compare(
            pre_data_files, post_data_files,
            selected_pre_file=selected_pre, selected_post_file=selected_post,
            combined_compare_plot=combined_compare_plot,
            pre_avg_transmittance=pre_avg_transmittance,
            post_avg_transmittance=post_avg_transmittance,
            pre_avg_reflectance=pre_avg_reflectance,
            post_avg_reflectance=post_avg_reflectance,
            pre_avg_absorbance=pre_avg_absorbance,
            post_avg_absorbance=post_avg_absorbance,
            transmittance_gain=tg,
            reflectance_gain=rg,
            absorbance_gain=ag,
        )

    except Exception as e:
        print(f"Unexpected error in compare route: {str(e)}")
        flash(f"An unexpected error occurred: {str(e)}", 'error')
        return render_compare([], [], error=True)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5111))
    app.run(host='0.0.0.0', port=port, debug=True)
