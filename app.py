from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_file
from datetime import datetime, timedelta
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
import os
from flask_mail import Mail
import secrets
import plotly
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
import numpy as np
import re
from urllib.parse import urlparse
from dotenv import load_dotenv
from pymongo import MongoClient
from bson import ObjectId
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail as SendGridMail, Email, To, Content

# Load environment variables from .env file
load_dotenv()

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
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.getenv('EMAIL_ADDRESS')
app.config['MAIL_PASSWORD'] = os.getenv('EMAIL_PASSWORD')
app.config['SENDGRID_API_KEY'] = os.getenv('SENDGRID_API_KEY')
app.config['EMAIL_FROM'] = os.getenv('EMAIL_FROM')
mail = Mail(app)

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
    return mongo_db['users'] if mongo_db else None

def get_samples_collection():
    return mongo_db['samples'] if mongo_db else None

def get_experiments_collection():
    return mongo_db['experiments'] if mongo_db else None

def get_prefixes_collection():
    return mongo_db['prefixes'] if mongo_db else None

def get_trash_collection():
    return mongo_db['trash'] if mongo_db else None

def get_plots_collection():
    return mongo_db['plots'] if mongo_db else None

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

# Initialize database with admin user
def init_db():
    if mongo_db is None:
        print("Warning: MongoDB not connected. Cannot initialize database.")
        return
    
    users = get_users_collection()
    if users is None:
        return
    
    # Create indexes
    users.create_index('username', unique=True)
    users.create_index('email', unique=True)
    
    samples = get_samples_collection()
    if samples:
        samples.create_index('id', unique=True)
    
    prefixes = get_prefixes_collection()
    if prefixes:
        prefixes.create_index('prefix', unique=True)
    
    # Check if admin user exists
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

# Initialize on startup
init_db()

# Routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        users = get_users_collection()
        if users is None:
            flash('Database not connected', 'error')
            return redirect(url_for('login'))
        
        user = users.find_one({'username': username})
        
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
            flash('Invalid username or password', 'error')
    
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
    all_prefixes = list(prefixes.find().sort('full_form', 1)) if prefixes else []

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
        if experiments and any(request.form.get(field) for field in ['transmittance', 'reflectance', 'absorbance', 'plqy', 'sem', 'edx', 'xrd']):
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
    
    all_prefixes = list(prefixes.find().sort('full_form', 1)) if prefixes else []

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
    
    try:
        sample = samples.find_one({'id': id})
        if not sample:
            flash('Sample not found', 'error')
            return redirect(url_for('index'))
        
        experiment = experiments.find_one({'sample_id': id}) if experiments else None
        
        # Move to trash
        if trash:
            trash_record = {
                'sample': sample,
                'experiment': experiment,
                'deleted_at': datetime.utcnow(),
                'deleted_by': session.get('username')
            }
            trash.insert_one(trash_record)
        
        # Delete from main collections
        samples.delete_one({'id': id})
        if experiments:
            experiments.delete_one({'sample_id': id})
        if plots:
            plots.delete_many({'sample_id': id})
        
        flash('Record moved to trash successfully!', 'success')
        
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
        
        if experiments:
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
        experiment = experiments.find_one({'sample_id': sample['id']}) if experiments else None
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
    if prefixes:
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
    # Convert to format expected by template
    formatted_records = []
    for record in trash_records:
        formatted_records.append((record.get('sample'), record.get('experiment')))
    
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
            flash('Record not found in trash', 'error')
            return redirect(url_for('view_trash'))
        
        # Check if sample already exists
        if samples.find_one({'id': id}):
            flash(f'A sample with ID {id} already exists!', 'error')
            return redirect(url_for('view_trash'))
        
        # Restore sample
        sample_data = trash_record.get('sample')
        if sample_data:
            samples.insert_one(sample_data)
        
        # Restore experiment if exists
        experiment_data = trash_record.get('experiment')
        if experiment_data and experiments:
            experiments.insert_one(experiment_data)
        
        # Delete from trash
        trash.delete_one({'sample.id': id})
        flash('Record restored successfully!', 'success')
        
    except Exception as e:
        flash(f'Error restoring record: {str(e)}', 'error')
        
    return redirect(url_for('view_trash'))

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')
        users = get_users_collection()
        
        if users is None:
            flash('Database not connected', 'error')
            return redirect(url_for('forgot_password'))
        
        user = users.find_one({'email': email})
        
        if user:
            token = secrets.token_urlsafe(32)
            users.update_one(
                {'_id': user['_id']},
                {'$set': {
                    'reset_token': token,
                    'reset_token_expiry': datetime.utcnow() + timedelta(hours=1)
                }}
            )
            
            reset_link = url_for('reset_password', token=token, _external=True)
            
            try:
                sg = SendGridAPIClient(app.config['SENDGRID_API_KEY'])
                message = SendGridMail(
                    from_email=Email(app.config['EMAIL_FROM']),
                    to_emails=To(email),
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
                                        <a href="{reset_link}" 
                                           style="background-color: #ff1825; color: white; padding: 12px 24px; 
                                                  text-decoration: none; border-radius: 4px; font-weight: bold;">
                                            Reset Password
                                        </a>
                                    </div>
                                    <p>This link will expire in 1 hour.</p>
                                </div>
                            </body>
                        </html>
                        '''
                    )
                )
                sg.send(message)
                flash('Password reset instructions have been sent to your email.', 'success')
                return redirect(url_for('login'))
                
            except Exception as e:
                print(f"Error sending email: {str(e)}")
                flash('Error sending password reset email. Please try again later.', 'error')
                return redirect(url_for('forgot_password'))
        
        flash('Email address not found.', 'error')
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
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return render_template('reset_password.html')
        
        users.update_one(
            {'_id': user['_id']},
            {'$set': {
                'password': generate_password_hash(password),
                'reset_token': None,
                'reset_token_expiry': None
            }}
        )
        
        flash('Your password has been reset successfully.', 'success')
        return redirect(url_for('login'))
    
    return render_template('reset_password.html')

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
            if samples and not samples.find_one({'id': sample_id}):
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
    
    if samples and experiments:
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
    if plots_col and samples:
        for plot in plots_col.find().sort('created_at', -1):
            sample = samples.find_one({'id': plot.get('sample_id')})
            plots_entries.append((plot, sample))
    
    return render_template('plots.html', plot_data=json.dumps(plot_data), plots_entries=plots_entries)

@app.route('/delete_plot/<string:plot_id>')
@login_required
def delete_plot(plot_id):
    plots_col = get_plots_collection()
    if plots_col:
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
    
    user = users.find_one({'_id': ObjectId(user_id)})
    if not user:
        flash('User not found', 'error')
        return redirect(url_for('admin_users'))
    
    if str(user['_id']) == session['user_id']:
        flash('You cannot modify your own admin status', 'error')
        return redirect(url_for('admin_users'))
    
    new_status = not user.get('is_admin', False)
    users.update_one({'_id': user['_id']}, {'$set': {'is_admin': new_status}})
    flash(f'Admin status {"granted" if new_status else "revoked"} for {user["username"]}', 'success')
    return redirect(url_for('admin_users'))

@app.route('/admin/users/<string:user_id>/toggle_active', methods=['POST'])
@login_required
@admin_required
def toggle_user_active(user_id):
    users = get_users_collection()
    if users is None:
        flash('Database not connected', 'error')
        return redirect(url_for('admin_users'))
    
    user = users.find_one({'_id': ObjectId(user_id)})
    if not user:
        flash('User not found', 'error')
        return redirect(url_for('admin_users'))
    
    if str(user['_id']) == session['user_id']:
        flash('You cannot deactivate your own account', 'error')
        return redirect(url_for('admin_users'))
    
    new_status = not user.get('is_active', True)
    users.update_one({'_id': user['_id']}, {'$set': {'is_active': new_status}})
    flash(f'User {user["username"]} has been {"activated" if new_status else "deactivated"}', 'success')
    return redirect(url_for('admin_users'))

@app.route('/admin/users/<string:user_id>/delete', methods=['POST'])
@login_required
@admin_required
def delete_user(user_id):
    users = get_users_collection()
    if users is None:
        flash('Database not connected', 'error')
        return redirect(url_for('admin_users'))
    
    user = users.find_one({'_id': ObjectId(user_id)})
    if not user:
        flash('User not found', 'error')
        return redirect(url_for('admin_users'))
    
    if str(user['_id']) == session['user_id']:
        flash('You cannot delete your own account', 'error')
        return redirect(url_for('admin_users'))
    
    users.delete_one({'_id': user['_id']})
    flash(f'User {user["username"]} has been deleted', 'success')
    return redirect(url_for('admin_users'))

# Chatbot routes (simplified)
@app.route('/chatbot', methods=['GET', 'POST'])
@login_required
def chatbot():
    return render_template('chatbot.html', results=None, query=None, error=None, response="Chatbot feature coming soon with MongoDB.", selected_columns=None)

@app.route('/chatbot_new', methods=['GET', 'POST'])
@login_required
def chatbot_new():
    return render_template('chatbot_new.html', results=None, query=None, error=None, response="Chatbot feature coming soon with MongoDB.", selected_columns=None)

@app.route('/chatbot_llm', methods=['GET', 'POST'])
@login_required
def chatbot_llm():
    if openai_client is None:
        flash('LLM Chatbot is not available. OpenAI API key is not configured.', 'error')
    return render_template('chatbot_llm.html')

@app.route('/compare', methods=['GET', 'POST'])
@login_required
def compare():
    try:
        if mongo_db is None:
            raise Exception("MongoDB connection is not available")
            
        pre_data_files = list(mongo_db.pre_data.find({}, {'design_name': 1}))
        post_data_files = list(mongo_db.post_data.find({}, {'design_name': 1}))
        
        for doc in pre_data_files + post_data_files:
            doc['_id'] = str(doc['_id'])

        if request.method == 'GET' or not (request.form.get('pre_file_id') and request.form.get('post_file_id')):
            return render_template('compare.html',
                                pre_data_files=pre_data_files,
                                post_data_files=post_data_files,
                                show_selection=True)

        # Rest of compare logic...
        return render_template('compare.html',
            pre_data_files=pre_data_files,
            post_data_files=post_data_files,
            show_selection=True,
            error=False
        )
            
    except Exception as e:
        print(f"Unexpected error in compare route: {str(e)}")
        flash(f"An unexpected error occurred: {str(e)}", 'error')
        return render_template('compare.html', error=True)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5111))
    app.run(host='0.0.0.0', port=port, debug=True)
