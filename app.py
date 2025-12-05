from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from config import Config, DevelopmentConfig, ProductionConfig, ReplitConfig
import os
from flask_mail import Mail, Message
import secrets
import plotly
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
import numpy as np
import re
from urllib.parse import urlparse
from openai import OpenAI
from dotenv import load_dotenv
from pymongo import MongoClient
from bson import ObjectId  # Add this import at the top with other imports
from flask_migrate import Migrate  # Add this import
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Email, To, Content

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client with API key (optional - LLM chatbot will be disabled without it)
openai_api_key = os.getenv('OPENAI_API_KEY')
client = None
if openai_api_key:
    client = OpenAI(api_key=openai_api_key)
else:
    print("Warning: OpenAI API key not found. LLM chatbot feature will be disabled.")

app = Flask(__name__)
# Database configuration
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)  # Force reload of environment variables

# Use Replit PostgreSQL connection if available
if 'REPLIT_DB_URL' in os.environ:
    # Use Replit's database URL if available
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('REPLIT_DB_URL')
elif 'DATABASE_URL' in os.environ:
    # Parse the DATABASE_URL to add SSL mode if not present
    db_url = os.environ['DATABASE_URL']
    parsed_url = urlparse(db_url)
    
    # If SSL mode is not specified in the URL, add it
    if 'sslmode=' not in db_url:
        if '?' in db_url:
            db_url += '&sslmode=require'
        else:
            db_url += '?sslmode=require'
    
    app.config['SQLALCHEMY_DATABASE_URI'] = db_url
else:
    # Use local development database as fallback
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///project_tracker.db'

# Add database configuration options
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Only add PostgreSQL-specific options if using PostgreSQL
if 'postgresql' in app.config.get('SQLALCHEMY_DATABASE_URI', ''):
    app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
        'pool_size': 10,
        'pool_recycle': 3600,
        'pool_pre_ping': True,
        'pool_use_lifo': True,
        'connect_args': {
            'connect_timeout': 10,
            'keepalives': 1,
            'keepalives_idle': 30,
            'keepalives_interval': 10,
            'keepalives_count': 5
        }
    }
else:
    # SQLite-friendly options
    app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
        'pool_pre_ping': True
    }

app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here')  # Change this to a secure secret key
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.getenv('EMAIL_ADDRESS')
app.config['MAIL_PASSWORD'] = os.getenv('EMAIL_PASSWORD')
app.config['SENDGRID_API_KEY'] = os.getenv('SENDGRID_API_KEY')
app.config['EMAIL_FROM'] = os.getenv('EMAIL_FROM')
db = SQLAlchemy(app)
migrate = Migrate(app, db)  # Initialize Flask-Migrate
mail = Mail(app)

# Import Supabase config for additional features
supabase = None
try:
    from supabase_config import supabase
except ImportError:
    print("Supabase configuration not found, continuing without Supabase features")
    pass

# Add this decorator definition at the top of your file, after imports
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('is_admin', False):
            flash('This operation requires admin privileges', 'error')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function

# Add User model for authentication
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(256), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, nullable=True)
    reset_token = db.Column(db.String(100), unique=True, nullable=True)
    reset_token_expiry = db.Column(db.DateTime, nullable=True)
    
    # User preferences
    notification_preferences = db.Column(db.JSON, default=dict)
    dashboard_preferences = db.Column(db.JSON, default=dict)
    
    def __repr__(self):
        return f'<User {self.username}>'

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Define the Sample model
class Sample(db.Model):
    __tablename__ = 'sample'
    id = db.Column(db.String(100), primary_key=True)
    company_name = db.Column(db.String(200), nullable=False)
    ERB = db.Column(db.Text, nullable=True)
    ERB_description = db.Column(db.Text, nullable=True)
    date = db.Column(db.String(10), nullable=False)
    time = db.Column(db.String(8), nullable=False)
    am_pm = db.Column(db.String(2), nullable=False)
    recipe_front = db.Column(db.String(200), nullable=False)
    recipe_back = db.Column(db.String(200), nullable=False)
    glass_type = db.Column(db.String(100), nullable=False)
    length = db.Column(db.Integer, nullable=False)
    thickness = db.Column(db.Integer, nullable=False)
    height = db.Column(db.Integer, nullable=False)
    cleaning = db.Column(db.String(1), default='N')
    coating = db.Column(db.String(1), default='N')
    annealing = db.Column(db.String(1), default='N')
    done = db.Column(db.String(1), default='N')
    sample_image = db.Column(db.String(500), nullable=True)
    image_description = db.Column(db.Text, nullable=True)
    # Add relationship to Experiment
    experiment = db.relationship('Experiment', backref='sample', uselist=False)

# Define the Experiment model
class Experiment(db.Model):
    __tablename__ = 'experiment'
    id = db.Column(db.String(100), db.ForeignKey('sample.id'), primary_key=True)
    transmittance = db.Column(db.String(500))
    reflectance = db.Column(db.String(500))
    absorbance = db.Column(db.String(500))
    plqy = db.Column(db.String(500))
    sem = db.Column(db.String(500))
    edx = db.Column(db.String(500))
    xrd = db.Column(db.String(500))

# Add this with your other models
class Prefix(db.Model):
    __tablename__ = 'prefix'
    prefix = db.Column(db.String(10), primary_key=True)
    full_form = db.Column(db.String(200), nullable=False)

# Modify the trash models
class SampleTrash(db.Model):
    __tablename__ = 'sample_trash'
    id = db.Column(db.String, primary_key=True)
    company_name = db.Column(db.String(200), nullable=False)
    ERB = db.Column(db.Text, nullable=True)
    ERB_description = db.Column(db.Text, nullable=True)
    date = db.Column(db.String(10), nullable=False)
    time = db.Column(db.String(8), nullable=False)
    am_pm = db.Column(db.String(2), nullable=False)
    recipe_front = db.Column(db.String(200), nullable=False)
    recipe_back = db.Column(db.String(200), nullable=False)
    glass_type = db.Column(db.String(100), nullable=False)
    length = db.Column(db.Integer, nullable=False)
    thickness = db.Column(db.Integer, nullable=False)
    height = db.Column(db.Integer, nullable=False)
    cleaning = db.Column(db.String(1), default='N')
    coating = db.Column(db.String(1), default='N')
    annealing = db.Column(db.String(1), default='N')
    done = db.Column(db.String(1), default='N')
    sample_image = db.Column(db.String(500), nullable=True)
    image_description = db.Column(db.Text, nullable=True)
    deleted_at = db.Column(db.DateTime, default=datetime.utcnow)
    deleted_by = db.Column(db.String(80))
    # Add relationship to ExperimentTrash
    experiment = db.relationship('ExperimentTrash', backref='sample_trash', uselist=False)

class ExperimentTrash(db.Model):
    __tablename__ = 'experiment_trash'
    id = db.Column(db.String(100), db.ForeignKey('sample_trash.id'), primary_key=True)
    transmittance = db.Column(db.String(500))
    reflectance = db.Column(db.String(500))
    absorbance = db.Column(db.String(500))
    plqy = db.Column(db.String(500))
    sem = db.Column(db.String(500))
    edx = db.Column(db.String(500))
    xrd = db.Column(db.String(500))
    deleted_at = db.Column(db.DateTime, default=datetime.utcnow)
    deleted_by = db.Column(db.String(80))

# Add these routes at the beginning of your app
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            if not user.is_active:
                flash('Your account has been deactivated. Please contact an administrator.', 'error')
                return redirect(url_for('login'))
            
            session['user_id'] = user.id
            session['username'] = user.username
            session['is_admin'] = user.is_admin
            
            # Update last login
            user.last_login = datetime.utcnow()
            db.session.commit()
            
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# Modify your existing routes to require login
@app.route('/')
@login_required
def index():
    # Use func.lower() for case-insensitive sorting of company names
    samples = Sample.query\
        .order_by(db.func.lower(Sample.company_name).asc())\
        .all()
    # Additional Python-level sorting for sequence numbers
    samples.sort(key=lambda x: (
        x.company_name.lower(),
        int(x.id.split('-')[-1]) if x.id.split('-')[-1].isdigit() else float('inf')
    ))
    return render_template('index.html', samples=samples)

@app.route('/add', methods=['GET', 'POST'])
@login_required
def add_sample():
    # Get all prefixes for the dropdown
    prefixes = Prefix.query.order_by(Prefix.full_form).all()

    if request.method == 'POST':
        # Get form data
        company_name = request.form['company_prefix']
        erb_number = request.form['ERB']
        sample_id_part = request.form['sample_id']  # Get manual sample ID part
        
        # Validate sample ID part
        if not sample_id_part:
            flash('Sample ID is required!', 'error')
            return render_template('add.html', prefixes=prefixes)
        
        # Generate the full ID in the format PREFIX-ExERB-SampleID
        full_sample_id = f"{company_name}-Ex{erb_number}-{sample_id_part}"
        
        # Check if sample ID already exists
        existing_sample = Sample.query.get(full_sample_id)
        if existing_sample:
            flash('Sample ID already exists! Please choose a different ID.', 'error')
            return render_template('add.html', prefixes=prefixes)
        
        # Handle image upload
        sample_image = None
        if 'sample_image' in request.files:
            file = request.files['sample_image']
            if file and file.filename:
                # Check if the file extension is allowed
                allowed_extensions = {'jpg', 'jpeg', 'png'}
                if '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions:
                    # Create a unique filename
                    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
                    # Save the file
                    file.save(os.path.join('static', 'sample_images', filename))
                    sample_image = f"sample_images/{filename}"

        # Get process status values
        cleaning = 'Y' if request.form.get('cleaning') == 'on' else 'N'
        coating = 'Y' if request.form.get('coating') == 'on' else 'N'
        annealing = 'Y' if request.form.get('annealing') == 'on' else 'N'
        # Set done to 'Y' only if all processes are 'Y'
        done = 'Y' if all([cleaning == 'Y', coating == 'Y', annealing == 'Y']) else 'N'
        
        # Create new sample
        new_sample = Sample(
            id=full_sample_id,  # Use the full generated ID
            company_name=company_name,
            ERB=erb_number,
            ERB_description=request.form.get('ERB_description'),
            date=request.form['date'],
            time=request.form['time'],
            am_pm=request.form['am_pm'],
            recipe_front=request.form['recipe_front'],
            recipe_back=request.form['recipe_back'],
            glass_type=request.form['glass_type'],
            length=int(request.form['length']),
            thickness=int(request.form['thickness']),
            height=int(request.form['height']),
            cleaning=cleaning,
            coating=coating,
            annealing=annealing,
            done=done,
            sample_image=sample_image,
            image_description=request.form.get('image_description')
        )
        db.session.add(new_sample)

        # Create experiment if any experiment data is provided
        if any(request.form.get(field) for field in ['transmittance', 'reflectance', 'absorbance', 
                                                    'plqy', 'sem', 'edx', 'xrd']):
            experiment = Experiment(
                id=full_sample_id,
                transmittance=request.form.get('transmittance'),
                reflectance=request.form.get('reflectance'),
                absorbance=request.form.get('absorbance'),
                plqy=request.form.get('plqy'),
                sem=request.form.get('sem'),
                edx=request.form.get('edx'),
                xrd=request.form.get('xrd')
            )
            db.session.add(experiment)

        db.session.commit()
        flash('Sample added successfully!', 'success')
        return redirect(url_for('index'))
    return render_template('add.html', prefixes=prefixes)

@app.route('/edit/<string:id>', methods=['GET', 'POST'])
@login_required
def edit_sample(id):
    sample = Sample.query.get_or_404(id)
    prefixes = Prefix.query.order_by(Prefix.full_form).all()

    if request.method == 'POST':
        # Handle image upload
        if 'sample_image' in request.files:
            file = request.files['sample_image']
            if file and file.filename:
                # Check if the file extension is allowed
                allowed_extensions = {'jpg', 'jpeg', 'png'}
                if '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions:
                    # Delete old image if it exists
                    if sample.sample_image:
                        old_image_path = os.path.join('static', sample.sample_image)
                        if os.path.exists(old_image_path):
                            os.remove(old_image_path)
                    
                    # Create a unique filename
                    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
                    # Save the file
                    file.save(os.path.join('static', 'sample_images', filename))
                    sample.sample_image = f"sample_images/{filename}"

        # Update sample fields
        sample.company_name = request.form['company_prefix']
        sample.ERB = request.form['ERB']
        sample.ERB_description = request.form.get('ERB_description')
        sample.date = request.form['date']
        sample.time = request.form['time']
        sample.am_pm = request.form['am_pm']
        sample.recipe_front = request.form['recipe_front']
        sample.recipe_back = request.form['recipe_back']
        sample.glass_type = request.form['glass_type']
        sample.length = int(request.form['length'])
        sample.thickness = int(request.form['thickness'])
        sample.height = int(request.form['height'])
        
        # Update process status values
        sample.cleaning = 'Y' if request.form.get('cleaning') == 'on' else 'N'
        sample.coating = 'Y' if request.form.get('coating') == 'on' else 'N'
        sample.annealing = 'Y' if request.form.get('annealing') == 'on' else 'N'
        # Set done to 'Y' only if all processes are 'Y'
        sample.done = 'Y' if all([sample.cleaning == 'Y', sample.coating == 'Y', sample.annealing == 'Y']) else 'N'
        
        sample.image_description = request.form.get('image_description')

        db.session.commit()
        return redirect(url_for('index'))
    return render_template('edit.html', sample=sample, prefixes=prefixes)

@app.route('/delete/<string:id>')
@login_required
def delete_sample(id):
    try:
        # Get the sample and its experiment
        sample = Sample.query.get_or_404(id)
        experiment = Experiment.query.get(id)
        
        # Get associated plot entries
        plot_entries = Plots.query.filter_by(sample_id=id).all()
        
        # Create trash records
        sample_trash = SampleTrash(
            id=sample.id,
            company_name=sample.company_name,
            ERB=sample.ERB,
            ERB_description=sample.ERB_description,
            date=sample.date,
            time=sample.time,
            am_pm=sample.am_pm,
            recipe_front=sample.recipe_front,
            recipe_back=sample.recipe_back,
            glass_type=sample.glass_type,
            length=sample.length,
            thickness=sample.thickness,
            height=sample.height,
            cleaning=sample.cleaning,
            coating=sample.coating,
            annealing=sample.annealing,
            done=sample.done,
            deleted_by=session.get('username')
        )
        
        # First add the sample trash record
        db.session.add(sample_trash)
        
        if experiment:
            experiment_trash = ExperimentTrash(
                id=experiment.id,
                transmittance=experiment.transmittance,
                reflectance=experiment.reflectance,
                absorbance=experiment.absorbance,
                plqy=experiment.plqy,
                sem=experiment.sem,
                edx=experiment.edx,
                xrd=experiment.xrd,
                deleted_by=session.get('username')
            )
            db.session.add(experiment_trash)
            db.session.delete(experiment)
        
        # Delete associated plot entries
        for plot_entry in plot_entries:
            db.session.delete(plot_entry)
            
        # Delete the original sample
        db.session.delete(sample)
        db.session.commit()
        flash('Record moved to trash successfully!', 'success')
        
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting record: {str(e)}', 'error')
        print(f"Error: {str(e)}")
        
    return redirect(url_for('index'))

@app.route('/experiments')
@login_required
def experiments():
    experiments = Experiment.query.all()
    return render_template('experiments.html', experiments=experiments)

@app.route('/add_experiment/<string:sample_id>', methods=['GET', 'POST'])
@login_required
def add_experiment(sample_id):
    sample = Sample.query.get_or_404(sample_id)
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

        # Process each measurement type
        experiment = Experiment(
            id=sample_id,
            transmittance=process_data(request.files.get('transmittance_file')),
            reflectance=process_data(request.files.get('reflectance_file')),
            absorbance=process_data(request.files.get('absorbance_file')),
            plqy=process_data(request.files.get('plqy_file')),
            sem=request.form.get('sem'),
            edx=request.form.get('edx'),
            xrd=request.form.get('xrd')
        )
        
        db.session.add(experiment)
        db.session.commit()
        return redirect(url_for('experiments'))
        
    return render_template('add_experiment.html', sample=sample)

@app.route('/edit_experiment/<string:id>', methods=['GET', 'POST'])
@login_required
def edit_experiment(id):
    experiment = Experiment.query.get_or_404(id)
    if request.method == 'POST':
        experiment.transmittance = request.form['transmittance']
        experiment.reflectance = request.form['reflectance']
        experiment.absorbance = request.form['absorbance']
        experiment.plqy = request.form['plqy']
        experiment.sem = request.form['sem']
        experiment.edx = request.form['edx']
        experiment.xrd = request.form['xrd']
        db.session.commit()
        return redirect(url_for('experiments'))
    return render_template('edit_experiment.html', experiment=experiment)

@app.route('/combined_view')
@login_required
def combined_view():
    # Join Sample and Experiment tables with proper ordering
    results = db.session.query(Sample, Experiment)\
        .outerjoin(Experiment, Sample.id == Experiment.id)\
        .order_by(db.func.lower(Sample.company_name).asc())\
        .all()
    # Additional Python-level sorting for sequence numbers
    results.sort(key=lambda x: (
        x[0].company_name.lower(),
        int(x[0].id.split('-')[-1]) if x[0].id.split('-')[-1].isdigit() else float('inf')
    ))
    return render_template('combined_view.html', results=results)

@app.route('/chatbot', methods=['GET', 'POST'])
@login_required
def chatbot():
    results = None
    query = None
    error = None
    response = None
    selected_columns = None
    
    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        if not query:
            response = get_help_message()
        else:
            try:
                # Get all data first
                all_data = db.session.query(Sample, Experiment)\
                    .outerjoin(Experiment, Sample.id == Experiment.id)\
                    .all()
                
                # Process the query dynamically
                query_type, conditions = analyze_query(query)
                
                if query_type == 'help':
                    response = get_help_message()
                elif query_type == 'error':
                    error = conditions  # In this case, conditions contains the error message
                else:
                    results = process_dynamic_query(query_type, conditions, all_data)
                    if isinstance(results, tuple):
                        results, selected_columns = results
                    
                    if not results or (isinstance(results, list) and len(results) == 0):
                        error = "No matching records found."
                        results = None
                    
            except Exception as e:
                error = f"An error occurred while processing your query: {str(e)}"
                print(f"Error in chatbot: {str(e)}")
    else:
        response = get_help_message()
    
    return render_template('chatbot.html', 
                         results=results, 
                         query=query, 
                         error=error, 
                         response=response,
                         selected_columns=selected_columns)

@app.route('/chatbot_new', methods=['GET', 'POST'])
@login_required
def chatbot_new():
    results = None
    query = None
    error = None
    response = None
    selected_columns = None
    
    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        if not query:
            response = get_help_message()
        else:
            try:
                print(f"Processing query: {query}")  # Debug print
                
                # Get all data first
                all_data = db.session.query(Sample, Experiment)\
                    .outerjoin(Experiment, Sample.id == Experiment.id)\
                    .all()
                
                print(f"Found {len(all_data)} total records")  # Debug print
                
                # Process the query dynamically
                query_type, conditions = analyze_query(query)
                print(f"Query type: {query_type}, Conditions: {conditions}")  # Debug print
                
                if query_type == 'help':
                    response = get_help_message()
                elif query_type == 'error':
                    error = conditions  # In this case, conditions contains the error message
                else:
                    results = process_dynamic_query(query_type, conditions, all_data)
                    if isinstance(results, tuple):
                        results, selected_columns = results
                        print(f"Selected columns: {selected_columns}")  # Debug print
                    
                    print(f"Found {len(results) if results else 0} matching records")  # Debug print
                    
                    if not results or (isinstance(results, list) and len(results) == 0):
                        error = "No matching records found."
                        results = None
                    
            except Exception as e:
                import traceback
                print(f"Error in chatbot: {str(e)}")
                print(traceback.format_exc())  # Print full traceback
                error = f"An error occurred while processing your query: {str(e)}"
    else:
        response = get_help_message()
    
    return render_template('chatbot_new.html', 
                         results=results, 
                         query=query, 
                         error=error, 
                         response=response,
                         selected_columns=selected_columns)

@app.route('/chatbot_llm', methods=['GET', 'POST'])
@login_required
def chatbot_llm():
    if client is None:
        flash('LLM Chatbot is not available. OpenAI API key is not configured.', 'error')
        return render_template('chatbot_llm.html')
        
    if request.method == 'POST':
        user_input = request.form.get('user_input')
        if not user_input:
            flash('Please enter a query', 'error')
            return render_template('chatbot_llm.html')

        try:
            # Call OpenAI to convert natural language to SQL
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": """You are a SQL expert. Convert the following natural language query to a PostgreSQL query. 
                    The database has these tables:
                    - sample (id, company_name, ERB, ERB_description, date, time, am_pm, recipe_front, recipe_back, glass_type, length, thickness, height, cleaning, coating, annealing, done)
                    - experiment (id [foreign key to sample.id], transmittance, reflectance, absorbance, plqy, sem, edx, xrd)
                    
                    Important rules for PostgreSQL queries:
                    1. Use exact column names as defined in the schema (case-sensitive)
                    2. When comparing text values, use single quotes: 'value'
                    3. For numeric comparisons, use numbers without quotes: 1, 2, 3
                    4. For ERB comparisons, use: \"ERB\" = 'value' (note the double quotes for column name)
                    5. The 'date' column in the sample table is stored as text in MM/DD/YYYY format. To filter by date, use TO_DATE(sample.date, 'MM/DD/YYYY') in WHERE clauses.
                    6. For date phrases like 'last month', 'last year', 'this year', 'between X and Y', 'after X', 'before Y', convert the phrase to a date range and use BETWEEN, >=, <= as appropriate with TO_DATE.
                    7. If you use any column from the sample table (such as sample.date), always include a JOIN sample ON experiment.id = sample.id in the FROM clause.
                    
                    Only output the SQL query, nothing else."""},
                    {"role": "user", "content": user_input}
                ]
            )
            
            generated_sql = completion.choices[0].message.content.strip()
            # Post-process to ensure JOIN sample if needed
            generated_sql = ensure_sample_join(generated_sql)
            
            # Execute the generated SQL query
            result = db.session.execute(generated_sql)
            results = [dict(row) for row in result]
            
            return render_template('chatbot_llm.html', 
                                generated_sql=generated_sql,
                                results=results)
                                
        except Exception as e:
            flash(f'Error: {str(e)}', 'error')
            return render_template('chatbot_llm.html')
            
    return render_template('chatbot_llm.html')

def parse_date_range(query):
    now = datetime.now()
    # Last month
    if "last month" in query:
        first_day_this_month = datetime(now.year, now.month, 1)
        last_month_end = first_day_this_month - timedelta(days=1)
        last_month_start = datetime(last_month_end.year, last_month_end.month, 1)
        return last_month_start, last_month_end
    # Last week
    if "last week" in query:
        start = now - timedelta(days=now.weekday() + 7)
        end = start + timedelta(days=6)
        return start, end
    # Today
    if "today" in query:
        start = datetime(now.year, now.month, now.day)
        end = start + timedelta(days=1)
        return start, end
    # This month
    if "this month" in query:
        start = datetime(now.year, now.month, 1)
        if now.month == 12:
            end = datetime(now.year + 1, 1, 1) - timedelta(days=1)
        else:
            end = datetime(now.year, now.month + 1, 1) - timedelta(days=1)
        return start, end
    # Last year
    if "last year" in query:
        start = datetime(now.year - 1, 1, 1)
        end = datetime(now.year - 1, 12, 31)
        return start, end
    # This year
    if "this year" in query:
        start = datetime(now.year, 1, 1)
        end = datetime(now.year, 12, 31)
        return start, end
    # Between X and Y
    match = re.search(r'between (\d{1,2}/\d{1,2}/\d{4}) and (\d{1,2}/\d{1,2}/\d{4})', query)
    if match:
        try:
            start = datetime.strptime(match.group(1), "%m/%d/%Y")
            end = datetime.strptime(match.group(2), "%m/%d/%Y")
            return start, end
        except Exception:
            pass
    # After X
    match = re.search(r'after (\d{1,2}/\d{1,2}/\d{4})', query)
    if match:
        try:
            start = datetime.strptime(match.group(1), "%m/%d/%Y")
            return start, now
        except Exception:
            pass
    # Before Y
    match = re.search(r'before (\d{1,2}/\d{1,2}/\d{4})', query)
    if match:
        try:
            end = datetime.strptime(match.group(1), "%m/%d/%Y")
            return datetime(1900, 1, 1), end
        except Exception:
            pass
    return None, None

def analyze_query(query):
    query = query.lower().strip()
    
    # Check for help-related queries
    help_patterns = ['help', 'what can you do', 'show commands', 'available commands']
    if any(pattern in query for pattern in help_patterns):
        return 'help', None
    
    # Handle natural language patterns for company name queries
    company_patterns = [
        r'(?:show|find|display|get).*(?:from|by|for|where|with)\s+(?:company\s+name\s+(?:is|=)\s*["\']?([^"\']+?)["\']?|company\s+name\s*["\']?([^"\']+?)["\']?|(?:company\s+)?([^"\']+?)(?:\s+(?:where|with|and|or)|$))',
        r'company\s*name\s*(?:is|=)\s*["\']?([^"\']+?)["\']?(?:\s|$)',
    ]
    
    for pattern in company_patterns:
        match = re.search(pattern, query)
        if match:
            # Take the first non-None group
            company_name = next((g for g in match.groups() if g is not None), '').strip()
            print(f"Extracted company name: {company_name}")  # Debug print
            conditions = {'_conditions': {'company_name': company_name}, '_operator': 'AND'}
            
            # Check for additional status conditions
            status_fields = ['cleaning', 'coating', 'annealing', 'done']
            for field in status_fields:
                status_match = re.search(rf'{field}\s*(?:is|=)\s*["\']?([YNyn])["\']?', query)
                if status_match:
                    conditions['_conditions'][field] = status_match.group(1).upper()
                    if ' or ' in query:
                        conditions['_operator'] = 'OR'
            
            # Check if specific columns are requested
            if 'show' in query and any(word in query for word in ['column', 'columns']):
                columns = extract_columns(query)
                return 'combined', {'columns': columns or [], 'conditions': conditions}
            return 'status', conditions
    
    # Handle natural language patterns for ERB queries
    erb_patterns = [
        r'(?:show|find|display|get).*erb\s*(?:number|#)?\s*["\']?(\d+)["\']?',
        r'erb\s*(?:is|=)\s*["\']?(\d+)["\']?',
    ]
    
    for pattern in erb_patterns:
        match = re.search(pattern, query)
        if match:
            erb_number = match.group(1)
            conditions = {'_conditions': {'erb': erb_number}, '_operator': 'AND'}
            if 'show' in query and any(word in query for word in ['column', 'columns']):
                columns = extract_columns(query)
                return 'combined', {'columns': columns or [], 'conditions': conditions}
            return 'status', conditions
    
    # Handle natural language patterns for status queries
    status_fields = ['cleaning', 'coating', 'annealing', 'done']
    status_patterns = [
        r'(?:show|find|display|get).*(?:with|where)?\s+(\w+)\s*(?:is|=)\s*["\']?([YNyn])["\']?',
        r'(\w+)\s*(?:is|=)\s*["\']?([YNyn])["\']?',
    ]
    
    status_conditions = {}
    for pattern in status_patterns:
        matches = re.finditer(pattern, query)
        for match in matches:
            field = match.group(1).lower()
            if field in status_fields:
                value = match.group(2).upper()
                status_conditions[field] = value
    
    if status_conditions:
        operator = 'OR' if ' or ' in query else 'AND'
        conditions = {'_conditions': status_conditions, '_operator': operator}
        if 'show' in query and any(word in query for word in ['column', 'columns']):
            columns = extract_columns(query)
            return 'combined', {'columns': columns or [], 'conditions': conditions}
        return 'status', conditions
    
    # Handle column selection queries
    if 'show' in query and any(word in query for word in ['column', 'columns']):
        columns = extract_columns(query)
        if columns:
            return 'columns', columns
    
    # Handle ID queries
    id_patterns = [
        r'(?:show|find|display|get).*id\s*[="\']([^"\']+)["\']',
        r'id\s*[="\']([^"\']+)["\']',
        r'(?:show|find|display|get).*(?:with|where)?\s+id\s+(?:is|=)\s*["\']?([^"\']+?)["\']?(?:\s|$)',
    ]
    
    for pattern in id_patterns:
        match = re.search(pattern, query)
        if match:
            return 'id', {'id': match.group(1).strip()}
    
    # Check for combined queries as a last resort
    combined = analyze_combined_query(query)
    if combined:
        return 'combined', combined
    
    # If no pattern matches, check if it's a simple column request
    words = query.split()
    if len(words) <= 3 and not any(word in ['where', 'with', 'and', 'or'] for word in words):
        potential_columns = [word.strip(',') for word in words if word not in ['show', 'me', 'the', 'columns']]
        if potential_columns:
            return 'columns', potential_columns
    
    # Date range support
    date_start, date_end = parse_date_range(query)
    if date_start and date_end:
        return 'date_range', {'start': date_start, 'end': date_end}
    
    return 'error', "I couldn't understand your query. Please try using one of the example formats or ask for help."

def extract_columns(query):
    # More flexible pattern matching for column names
    patterns = [
        r'show\s+([\w\s,]+?)(?:\s+(?:for|where|$))',  # matches "show col1, col2 for..."
        r'display\s+([\w\s,]+?)(?:\s+(?:for|where|$))', # matches "display col1, col2 for..."
        r'get\s+([\w\s,]+?)(?:\s+(?:for|where|$))',   # matches "get col1, col2 for..."
        r'([\w\s,]+?)\s+columns?(?:\s+(?:for|where|$))', # matches "col1, col2 columns for..."
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query.lower())
        if match:
            # Split by comma and clean up each column name
            columns = [col.strip().replace(' ', '_') for col in match.group(1).split(',')]
            # Remove empty strings and common words
            columns = [col for col in columns if col and col not in ['columns', 'column', 'show', 'me', 'the']]
            return columns
    
    # If no pattern matches but query contains column names, try to extract them
    words = query.lower().split()
    if 'show' in words or 'display' in words or 'get' in words:
        potential_columns = []
        for word in words:
            word = word.strip(',').replace(' ', '_')
            if word and word not in ['show', 'display', 'get', 'me', 'the', 'columns', 'column']:
                potential_columns.append(word)
        if potential_columns:
            return potential_columns
    
    return None

def analyze_combined_query(query):
    combined = {
        'columns': [],
        'conditions': {}
    }
    
    # Extract columns if specified
    if 'show' in query:
        cols = re.findall(r'show\s+([\w\s,]+?)(?:\s+for|where|$)', query)
        if cols:
            combined['columns'] = [col.strip() for col in cols[0].split(',')]
    
    # Extract conditions
    id_match = extract_id(query)
    if id_match:
        combined['conditions']['id'] = id_match
    
    status_match = extract_status(query)
    if status_match:
        combined['conditions'].update(status_match)
    
    return combined if (combined['columns'] or combined['conditions']) else None

def process_dynamic_query(query_type, conditions, all_data):
    if query_type == 'columns':
        return process_column_selection(conditions, all_data)
    elif query_type == 'id':
        return process_id_query(conditions['id'], all_data)
    elif query_type == 'status':
        return process_status_query(conditions, all_data)
    elif query_type == 'combined':
        return process_combined_query(conditions, all_data)
    elif query_type == 'date_range':
        return process_date_range_query(conditions, all_data)
    return None

def process_column_selection(columns, all_data):
    valid_columns = {
        'id': lambda s, e: s.id,
        'company_name': lambda s, e: s.company_name,
        'erb': lambda s, e: s.ERB,
        'erb_description': lambda s, e: s.ERB_description,
        'date': lambda s, e: s.date,
        'time': lambda s, e: s.time,
        'recipe_front': lambda s, e: s.recipe_front,
        'recipe_back': lambda s, e: s.recipe_back,
        'glass_type': lambda s, e: s.glass_type,
        'dimensions': lambda s, e: f"{s.length}x{s.thickness}x{s.height}",
        'cleaning': lambda s, e: s.cleaning,
        'coating': lambda s, e: s.coating,
        'annealing': lambda s, e: s.annealing,
        'done': lambda s, e: s.done,
        'transmittance': lambda s, e: e.transmittance if e else None,
        'reflectance': lambda s, e: e.reflectance if e else None,
        'absorbance': lambda s, e: e.absorbance if e else None,
        'plqy': lambda s, e: e.plqy if e else None,
        'sem': lambda s, e: e.sem if e else None,
        'edx': lambda s, e: e.edx if e else None,
        'xrd': lambda s, e: e.xrd if e else None
    }
    
    # Convert all column names to lowercase for case-insensitive comparison
    # and replace spaces with underscores
    columns = [col.lower().replace(' ', '_') for col in columns]
    
    # Validate and collect the requested columns
    valid_cols = []
    for col in columns:
        if col in valid_columns:
            valid_cols.append((col, valid_columns[col]))
        else:
            print(f"Warning: Invalid column name '{col}'")
    
    # If no valid columns were found, return all data
    if not valid_cols:
        return all_data, None
    
    # Always include 'id' column for reference if not already included
    if 'id' not in [col[0] for col in valid_cols]:
        valid_cols.insert(0, ('id', valid_columns['id']))
    
    # Transform the data to include only requested columns
    transformed_data = []
    for sample, exp in all_data:
        row = {}
        for col_name, col_func in valid_cols:
            row[col_name] = col_func(sample, exp)
        transformed_data.append(row)
    
    return transformed_data, [col[0] for col in valid_cols]

def process_id_query(id_value, all_data):
    return [(sample, exp) for sample, exp in all_data if sample.id.lower() == id_value.lower()]

def process_status_query(conditions, all_data):
    print(f"Processing status query with conditions: {conditions}")  # Debug print
    filtered_data = all_data
    
    if not conditions.get('_conditions'):
        return filtered_data
    
    operator = conditions.get('_operator', 'AND')
    status_conditions = conditions.get('_conditions', {})
    
    print(f"Operator: {operator}, Status conditions: {status_conditions}")  # Debug print
    
    # Handle combined company name and status conditions
    matching_data = []
    company_name = None
    
    # First extract company name if present
    if 'company_name' in status_conditions:
        company_name = status_conditions['company_name']
        del status_conditions['company_name']  # Remove it so we can process other conditions separately
    
    # Filter by company name first if present
    if company_name:
        filtered_data = [
            (sample, exp) for sample, exp in filtered_data
            if sample.company_name.lower() == company_name.lower()
        ]
        print(f"Filtering by company name '{company_name}', found {len(filtered_data)} matches")
    
    # Then process status conditions
    if operator == 'AND':
        # All conditions must match
        for field, value in status_conditions.items():
            if field == 'erb':
                filtered_data = [
                    (sample, exp) for sample, exp in filtered_data
                    if sample.ERB == str(value)
                ]
            else:
                filtered_data = [
                    (sample, exp) for sample, exp in filtered_data
                    if str(getattr(sample, field, '')).upper() == str(value).upper()
                ]
    else:  # OR
        # Any condition can match
        for field, value in status_conditions.items():
            if field == 'erb':
                matches = [
                    (sample, exp) for sample, exp in filtered_data
                    if sample.ERB == str(value)
                ]
            else:
                matches = [
                    (sample, exp) for sample, exp in filtered_data
                    if str(getattr(sample, field, '')).upper() == str(value).upper()
                ]
            matching_data.extend(matches)
        
        # Remove duplicates while preserving order
        seen = set()
        filtered_data = [x for x in matching_data if not (x in seen or seen.add(x))]
    
    print(f"Returning {len(filtered_data)} filtered records")  # Debug print
    return filtered_data

def process_combined_query(conditions, all_data):
    filtered_data = all_data
    
    # Apply conditions first with AND logic
    if conditions['conditions']:
        for field, value in conditions['conditions'].items():
            if field == 'id':
                filtered_data = [
                    (sample, exp) for sample, exp in filtered_data
                    if sample.id.lower() == value.lower()
                ]
            else:
                filtered_data = [
                    (sample, exp) for sample, exp in filtered_data
                    if getattr(sample, field, '').upper() == value.upper()
                ]
    
    # Return with selected columns if specified
    if conditions['columns']:
        return process_column_selection(conditions['columns'], filtered_data)
    
    return filtered_data

def process_date_range_query(conditions, all_data):
    start = conditions['start']
    end = conditions['end']
    results = []
    for sample, exp in all_data:
        try:
            sample_date = datetime.strptime(sample.date, "%m/%d/%Y")
            if start <= sample_date <= end:
                results.append((sample, exp))
        except Exception:
            continue
    return results

def get_help_message():
    return (
        "I can help you find information about:\n\n"
        "1. View specific columns:\n"
        "   - Example: 'Show id, company name, glass type columns'\n"
        "   - Example: 'Show glass type column'\n"
        "   - Example: 'Show cleaning, coating, annealing columns'\n\n"
        "2. Search by ID:\n"
        "   - Example: 'Show me the records of ID=\"AWI001\"'\n"
        "   - Example: 'Show records with ID=\"AWI001\"'\n\n"
        "3. Search by Company Name:\n"
        "   - Example: 'Show records where company name=\"Sun Density\"'\n"
        "   - Example: 'Show transmittance data where company name=\"Sun Density\"'\n\n"
        "4. Search by ERB:\n"
        "   - Example: 'Show records where ERB=\"1\"'\n"
        "   - Example: 'Show id, erb columns where ERB=\"1\"'\n\n"
        "5. Search by process status:\n"
        "   Using AND:\n"
        "   - Example: 'Show records with cleaning=\"Y\" and coating=\"Y\"'\n"
        "   - Example: 'Show records where cleaning=\"Y\" and coating=\"Y\"'\n\n"
        "   Using OR:\n"
        "   - Example: 'Show records with cleaning=\"Y\" or coating=\"Y\"'\n"
        "   - Example: 'Show records where cleaning=\"Y\" or coating=\"Y\"'\n\n"
        "   Combined with columns:\n"
        "   - Example: 'Show id, date columns where cleaning=\"Y\" and coating=\"Y\"'\n"
        "   - Example: 'Show id, date columns where cleaning=\"Y\" or coating=\"Y\"'\n\n"
        "6. Search by date range (advanced):\n"
        "   - Example: 'Show all experiments for last month'\n"
        "   - Example: 'Show all experiments for last year'\n"
        "   - Example: 'Show all experiments for this year'\n"
        "   - Example: 'Show all experiments between 01/01/2023 and 03/31/2023'\n"
        "   - Example: 'Show all experiments after 01/01/2023'\n"
        "   - Example: 'Show all experiments before 12/31/2023'\n\n"
        "Available columns:\n"
        "- Basic Info: id, company name, erb\n"
        "- Document Links: erb description (URL)\n"
        "- Date/Time: date, time\n"
        "- Recipe Info: recipe front, recipe back\n"
        "- Material Info: glass type, dimensions\n"
        "- Process Info: cleaning, coating, annealing, done\n"
        "- Experimental Data: transmittance, reflectance, absorbance, plqy, sem, edx, xrd\n\n"
        "You can combine these in any way, for example:\n"
        "- 'Show id, company name, cleaning, coating columns'\n"
        "- 'Show recipe front, glass type columns'\n"
        "- 'Show transmittance, reflectance, absorbance columns'\n"
        "- 'Show transmittance data where company name=\"Sun Density\"'\n"
        "- 'Show id, date columns where cleaning=\"Y\" and coating=\"Y\"'\n"
        "- 'Show id, date columns where cleaning=\"Y\" or coating=\"Y\"'\n"
        "- 'Show id, erb columns where ERB=\"1\"'"
    )

@app.route('/prefix_table', methods=['GET', 'POST'])
@login_required
def prefix_table():
    try:
        if request.method == 'POST':
            # Handle adding new prefix
            prefix = request.form.get('prefix')
            full_form = request.form.get('full_form')
            
            if prefix and full_form:
                # Check if prefix already exists
                existing_prefix = Prefix.query.get(prefix)
                if existing_prefix:
                    flash('Prefix already exists!', 'error')
                else:
                    new_prefix = Prefix(prefix=prefix, full_form=full_form)
                    db.session.add(new_prefix)
                    db.session.commit()
                    flash('Prefix added successfully!', 'success')
                
        # Get all prefixes
        prefixes = Prefix.query.order_by(Prefix.prefix).all()
        return render_template('prefix_table.html', prefixes=prefixes)
    
    except Exception as e:
        # Log the error (you might want to use proper logging)
        print(f"Error in prefix_table: {str(e)}")
        flash('An error occurred while loading the prefix table.', 'error')
        return render_template('prefix_table.html', prefixes=[])

@app.route('/delete_prefix/<string:prefix>')
@login_required
def delete_prefix(prefix):
    try:
        prefix_entry = Prefix.query.get_or_404(prefix)
        db.session.delete(prefix_entry)
        db.session.commit()
        flash('Prefix deleted successfully!', 'success')
    except Exception as e:
        flash('Error deleting prefix!', 'error')
    return redirect(url_for('prefix_table'))

# Public registration route
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

    # Check if username or email already exists
    existing_user = User.query.filter((User.username == username) | (User.email == email)).first()
    if existing_user:
        if existing_user.username == username:
            flash('Username already exists.', 'error')
        else:
            flash('Email already exists.', 'error')
        return redirect(url_for('login'))

    # Hash the password
    hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
    
    # Create new user with default preferences
    new_user = User(
        username=username,
        email=email,
        password=hashed_password,
        notification_preferences={
            'email_notifications': True,
            'system_notifications': True
        },
        dashboard_preferences={
            'recent_activity': True,
            'saved_queries': []
        }
    )
    
    db.session.add(new_user)
    db.session.commit()
    flash('Registration successful! You can now log in.', 'success')
    return redirect(url_for('login'))

# Add route to view trash
@app.route('/trash')
@login_required
def view_trash():
    # Get all trash records with their deletion info
    trash_records = db.session.query(SampleTrash, ExperimentTrash)\
        .outerjoin(ExperimentTrash, SampleTrash.id == ExperimentTrash.id)\
        .order_by(SampleTrash.deleted_at.desc())\
        .all()
    return render_template('trash.html', trash_records=trash_records)

# Add route to restore from trash
@app.route('/restore/<string:id>')
@login_required
def restore_from_trash(id):
    try:
        # Get trash records
        sample_trash = SampleTrash.query.get_or_404(id)
        experiment_trash = ExperimentTrash.query.get(id)
        
        # Check if a sample with this ID already exists
        if Sample.query.get(sample_trash.id):
            flash(f'A sample with ID {sample_trash.id} already exists!', 'error')
            return redirect(url_for('view_trash'))
        
        # Restore sample
        sample = Sample(
            id=sample_trash.id,
            company_name=sample_trash.company_name,
            ERB=sample_trash.ERB,
            ERB_description=sample_trash.ERB_description,
            date=sample_trash.date,
            time=sample_trash.time,
            am_pm=sample_trash.am_pm,
            recipe_front=sample_trash.recipe_front,
            recipe_back=sample_trash.recipe_back,
            glass_type=sample_trash.glass_type,
            length=sample_trash.length,
            thickness=sample_trash.thickness,
            height=sample_trash.height,
            cleaning=sample_trash.cleaning,
            coating=sample_trash.coating,
            annealing=sample_trash.annealing,
            done=sample_trash.done
        )
        db.session.add(sample)
        
        # Restore experiment if it exists
        if experiment_trash:
            experiment = Experiment(
                id=experiment_trash.id,
                transmittance=experiment_trash.transmittance,
                reflectance=experiment_trash.reflectance,
                absorbance=experiment_trash.absorbance,
                plqy=experiment_trash.plqy,
                sem=experiment_trash.sem,
                edx=experiment_trash.edx,
                xrd=experiment_trash.xrd
            )
            db.session.add(experiment)
            db.session.delete(experiment_trash)
            
        # Delete trash records
        db.session.delete(sample_trash)
        db.session.commit()
        flash('Record restored successfully!', 'success')
        
    except Exception as e:
        db.session.rollback()
        flash(f'Error restoring record: {str(e)}', 'error')
        print(f"Error: {str(e)}")
        
    return redirect(url_for('view_trash'))

# Add these new routes for password reset
@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')
        user = User.query.filter_by(email=email).first()  # Changed from username to email
        
        if user:
            # Generate a secure token
            token = secrets.token_urlsafe(32)
            user.reset_token = token
            user.reset_token_expiry = datetime.utcnow() + timedelta(hours=1)
            db.session.commit()
            
            # Create reset link
            reset_link = url_for('reset_password', token=token, _external=True)
            
            try:
                # Initialize SendGrid client
                sg = SendGridAPIClient(app.config['SENDGRID_API_KEY'])
                
                # Create email message
                message = Mail(
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
                                    <p>We received a request to reset your password for the Project Tracker application. 
                                    If you didn't make this request, you can safely ignore this email.</p>
                                    <p>To reset your password, click the button below:</p>
                                    <div style="text-align: center; margin: 30px 0;">
                                        <a href="{reset_link}" 
                                           style="background-color: #ff1825; color: white; padding: 12px 24px; 
                                                  text-decoration: none; border-radius: 4px; font-weight: bold;">
                                            Reset Password
                                        </a>
                                    </div>
                                    <p>This link will expire in 1 hour.</p>
                                    <p>If you're having trouble clicking the button, copy and paste this URL into your browser:</p>
                                    <p style="word-break: break-all; color: #666;">{reset_link}</p>
                                    <hr style="border: 1px solid #eee; margin: 20px 0;">
                                    <p style="color: #666; font-size: 12px;">
                                        This is an automated message, please do not reply to this email.
                                    </p>
                                </div>
                            </body>
                        </html>
                        '''
                    )
                )
                
                # Send email
                response = sg.send(message)
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
    user = User.query.filter_by(reset_token=token).first()
    
    if not user or user.reset_token_expiry < datetime.utcnow():
        flash('Invalid or expired password reset link.', 'error')
        return redirect(url_for('forgot_password'))
    
    if request.method == 'POST':
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return render_template('reset_password.html')
        
        user.password = generate_password_hash(password)
        user.reset_token = None
        user.reset_token_expiry = None
        db.session.commit()
        
        flash('Your password has been reset successfully.', 'success')
        return redirect(url_for('login'))
    
    return render_template('reset_password.html')

@app.route('/plots', methods=['GET', 'POST'])
@login_required
def plots():
    if request.method == 'POST':
        # Handle adding new plot entry
        sample_id = request.form.get('sample_id')
        sharepoint_link = request.form.get('sharepoint_link')
        
        if sample_id and sharepoint_link:
            # Check if sample exists
            sample = Sample.query.get(sample_id)
            if not sample:
                flash('Sample ID not found! Please enter a valid sample ID.', 'error')
            else:
                # Check if plot entry already exists for this sample
                existing_plot = Plots.query.filter_by(sample_id=sample_id).first()
                if existing_plot:
                    flash('A plot entry already exists for this sample ID!', 'error')
                else:
                    # Create new plot entry
                    new_plot = Plots(
                        sample_id=sample_id,
                        sharepoint_link=sharepoint_link,
                        created_by=session.get('username')
                    )
                    db.session.add(new_plot)
                    db.session.commit()
                    flash('Plot entry added successfully!', 'success')
        else:
            flash('Both Sample ID and SharePoint Link are required!', 'error')
    
    # Query all experiments
    experiments = db.session.query(Sample, Experiment).join(Experiment).all()
    
    # Initialize empty lists for each measurement type
    plot_data = {
        'transmittance': [],
        'reflectance': [],
        'absorbance': [],
        'plqy': [],
        'sem': [],
        'edx': [],
        'xrd': []
    }
    
    # Process each experiment
    for sample, experiment in experiments:
        for measurement_type in plot_data.keys():
            data = getattr(experiment, measurement_type)
            if data:  # If data exists for this measurement
                try:
                    # Try to parse the data (assuming it's stored as a string representation of data)
                    data_dict = {
                        'id': sample.id,
                        'data': data,
                        'recipe_front': sample.recipe_front,
                        'recipe_back': sample.recipe_back,
                        'glass_type': sample.glass_type
                    }
                    plot_data[measurement_type].append(data_dict)
                except:
                    continue
    
    # Get all plots entries
    plots_entries = db.session.query(Plots, Sample).outerjoin(Sample, Plots.sample_id == Sample.id).order_by(Plots.created_at.desc()).all()
    
    # Convert plot data to JSON for JavaScript
    plot_json = json.dumps(plot_data)
    
    return render_template('plots.html', plot_data=plot_json, plots_entries=plots_entries)

@app.route('/delete_plot/<int:plot_id>')
@login_required
def delete_plot(plot_id):
    try:
        plot_entry = Plots.query.get_or_404(plot_id)
        db.session.delete(plot_entry)
        db.session.commit()
        flash('Plot entry deleted successfully!', 'success')
    except Exception as e:
        flash('Error deleting plot entry!', 'error')
    return redirect(url_for('plots'))

def update_password_hashes():
    users = User.query.all()
    for user in users:
        # Only update if the hash doesn't start with 'pbkdf2:sha256'
        if not user.password.startswith('pbkdf2:sha256'):
            # Assuming you have a default password or way to reset
            new_hash = generate_password_hash('temporary_password', method='pbkdf2:sha256')
            user.password = new_hash
    db.session.commit()

# Add a route to reset admin password
@app.route('/reset_admin', methods=['GET'])
def reset_admin():
    try:
        with app.app_context():
            # Find admin user
            admin = User.query.filter_by(username='admin').first()
            if admin:
                # Update admin password
                admin.password = generate_password_hash('admin123', method='pbkdf2:sha256')
                db.session.commit()
                return 'Admin password reset successfully to "admin123"'
            else:
                # Create new admin user
                admin_user = User(
                    username='admin',
                    email='admin@example.com',
                    password=generate_password_hash('admin123', method='pbkdf2:sha256'),
                    is_admin=True,
                    is_active=True,
                    notification_preferences={
                        'email_notifications': True,
                        'system_notifications': True
                    },
                    dashboard_preferences={
                        'recent_activity': True,
                        'saved_queries': []
                    }
                )
                db.session.add(admin_user)
                db.session.commit()
                return 'New admin user created with password "admin123"'
    except Exception as e:
        return f'Error: {str(e)}'

# MongoDB connection (optional - compare feature will be disabled without it)
MONGODB_URI = os.getenv('MONGODB_URI')
mongo_client = None
mongo_db = None

if MONGODB_URI:
    try:
        print(f"Attempting to connect to MongoDB...")
        mongo_client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000, tlsAllowInvalidCertificates=True)
        # Force a connection attempt
        mongo_client.server_info()
        print("Successfully connected to MongoDB")
        # Specify database name separately
        mongo_db = mongo_client.AWI_users
    except Exception as e:
        print(f"Warning: Error connecting to MongoDB: {str(e)}")
        print("Compare feature will be disabled.")
        mongo_client = None
        mongo_db = None
else:
    print("Warning: MONGODB_URI not found in environment variables. Compare feature will be disabled.")

@app.route('/compare', methods=['GET', 'POST'])
@login_required
def compare():
    try:
        if mongo_client is None or mongo_db is None:
            raise Exception("MongoDB connection is not available")
            
        # Fetch all available files from MongoDB
        try:
            pre_data_files = list(mongo_db.pre_data.find({}, {'design_name': 1}))
            post_data_files = list(mongo_db.post_data.find({}, {'design_name': 1}))
            
            # Convert ObjectId to string for template rendering
            for doc in pre_data_files + post_data_files:
                doc['_id'] = str(doc['_id'])
                
        except Exception as e:
            print(f"Error fetching file list from MongoDB: {str(e)}")
            flash(f"Error fetching file list from MongoDB: {str(e)}", 'error')
            return render_template('compare.html', error=True)

        # If this is just the initial page load or no files selected yet
        if request.method == 'GET' or not (request.form.get('pre_file_id') and request.form.get('post_file_id')):
            return render_template('compare.html',
                                pre_data_files=pre_data_files,
                                post_data_files=post_data_files,
                                show_selection=True)

        # Get selected file IDs and convert to ObjectId
        try:
            pre_file_id = ObjectId(request.form.get('pre_file_id'))
            post_file_id = ObjectId(request.form.get('post_file_id'))
            
            # Fetch the selected documents
            selected_pre = mongo_db.pre_data.find_one({'_id': pre_file_id})
            selected_post = mongo_db.post_data.find_one({'_id': post_file_id})
            
            print("\nPre-data document:")
            print(selected_pre)
            print("\nPost-data document:")
            print(selected_post)
            
            if not selected_pre or not selected_post:
                raise Exception("Selected files not found")
            
        except Exception as e:
            print(f"Error fetching selected files: {str(e)}")
            flash("Error fetching selected files", 'error')
            return render_template('compare.html',
                                pre_data_files=pre_data_files,
                                post_data_files=post_data_files,
                                show_selection=True,
                                error=True)

        # Initialize data arrays
        wavelengths = []
        pre_transmittance = []
        pre_reflectance = []
        pre_absorbance = []
        post_transmittance = []
        post_reflectance = []
        post_absorbance = []

        # Get all wavelengths from both documents
        all_wavelengths = set()
        stats_wavelengths = set()  # For 400-1200nm statistics
        for doc in [selected_pre, selected_post]:
            print("\nDocument structure:")
            print(f"Keys at root level: {doc.keys()}")
            data = doc.get('data', {})
            print(f"Data field type: {type(data)}")
            print(f"Data field keys: {data.keys() if isinstance(data, dict) else 'Not a dictionary'}")
            
            if isinstance(data, dict):
                for key in data.keys():
                    if key not in ['_id', 'design_name']:  # Exclude non-wavelength keys
                        try:
                            wavelength = float(key)
                            all_wavelengths.add(wavelength)  # Add all wavelengths for plotting
                            if 400 <= wavelength <= 1200:  # Filter wavelengths for statistics
                                stats_wavelengths.add(wavelength)
                        except ValueError:
                            continue

        # Sort wavelengths
        wavelengths = sorted(list(all_wavelengths))
        stats_wavelengths = sorted(list(stats_wavelengths))
        print(f"\nProcessing {len(wavelengths)} total wavelengths")
        print(f"Processing {len(stats_wavelengths)} wavelengths between 400-1200nm for statistics")

        # Process pre-data
        print("\nProcessing pre-data:")
        for wavelength in wavelengths:
            str_wavelength = str(wavelength)
            values = selected_pre.get('data', {}).get(str_wavelength)
            print(f"Pre-data - Wavelength {wavelength}:")
            print(f"  Raw values: {values}")
            
            if isinstance(values, list) and len(values) >= 3:
                try:
                    t_val = float(values[0])  # Transmittance is at index 0
                    r_val = float(values[1])  # Reflectance is at index 1
                    a_val = float(values[2])  # Absorbance is at index 2
                    print(f"  Processed values - T: {t_val}, R: {r_val}, A: {a_val}")
                    
                    pre_transmittance.append(t_val)
                    pre_reflectance.append(r_val)
                    pre_absorbance.append(a_val)
                except (ValueError, TypeError, IndexError) as e:
                    print(f"  ⚠️ Error processing values: {e}")
                    pre_transmittance.append(None)
                    pre_reflectance.append(None)
                    pre_absorbance.append(None)
            else:
                print(f"  ⚠️ Invalid data format at wavelength {wavelength}: {values}")
                pre_transmittance.append(None)
                pre_reflectance.append(None)
                pre_absorbance.append(None)

        # Process post-data
        print("\nProcessing post-data:")
        for wavelength in wavelengths:
            str_wavelength = str(wavelength)
            values = selected_post.get('data', {}).get(str_wavelength)
            print(f"Post-data - Wavelength {wavelength}:")
            print(f"  Raw values: {values}")
            
            if isinstance(values, list) and len(values) >= 3:
                try:
                    t_val = float(values[0])  # Transmittance is at index 0
                    r_val = float(values[1])  # Reflectance is at index 1
                    a_val = float(values[2])  # Absorbance is at index 2
                    print(f"  Processed values - T: {t_val}, R: {r_val}, A: {a_val}")
                    
                    post_transmittance.append(t_val)
                    post_reflectance.append(r_val)
                    post_absorbance.append(a_val)
                except (ValueError, TypeError, IndexError) as e:
                    print(f"  ⚠️ Error processing values: {e}")
                    post_transmittance.append(None)
                    post_reflectance.append(None)
                    post_absorbance.append(None)
            else:
                print(f"  ⚠️ Invalid data format at wavelength {wavelength}: {values}")
                post_transmittance.append(None)
                post_reflectance.append(None)
                post_absorbance.append(None)

        # Calculate averages (excluding zeros, infinity, and None values)
        def safe_mean(values):
            filtered_values = [v for v in values if v is not None and v > 0 and not np.isinf(v) and not np.isnan(v)]
            print(f"\nCalculating mean from values: {filtered_values}")
            if not filtered_values:
                print("No valid values found for averaging")
                return 0.0
            mean_val = float(np.mean(filtered_values))
            print(f"Mean value: {mean_val}")
            return mean_val

        print("\nPre-data arrays:")
        print("Transmittance:", pre_transmittance)
        print("Reflectance:", pre_reflectance)
        print("Absorbance:", pre_absorbance)

        print("\nPost-data arrays:")
        print("Transmittance:", post_transmittance)
        print("Reflectance:", post_reflectance)
        print("Absorbance:", post_absorbance)

        # Calculate averages for the 400-1200nm range only
        stats_indices = [i for i, w in enumerate(wavelengths) if 400 <= w <= 1200]
        pre_stats_transmittance = [pre_transmittance[i] for i in stats_indices]
        pre_stats_reflectance = [pre_reflectance[i] for i in stats_indices]
        pre_stats_absorbance = [pre_absorbance[i] for i in stats_indices]
        post_stats_transmittance = [post_transmittance[i] for i in stats_indices]
        post_stats_reflectance = [post_reflectance[i] for i in stats_indices]
        post_stats_absorbance = [post_absorbance[i] for i in stats_indices]

        # Calculate averages using the filtered data
        pre_avg_transmittance = safe_mean(pre_stats_transmittance)
        pre_avg_reflectance = safe_mean(pre_stats_reflectance)
        pre_avg_absorbance = safe_mean(pre_stats_absorbance)

        post_avg_transmittance = safe_mean(post_stats_transmittance)
        post_avg_reflectance = safe_mean(post_stats_reflectance)
        post_avg_absorbance = safe_mean(post_stats_absorbance)

        # Calculate gains (as percentages)
        def calculate_gain(pre_val, post_val):
            if pre_val == 0:
                return 0.0
            return ((post_val - pre_val) / pre_val) * 100

        transmittance_gain = calculate_gain(pre_avg_transmittance, post_avg_transmittance)
        reflectance_gain = calculate_gain(pre_avg_reflectance, post_avg_reflectance)
        absorbance_gain = calculate_gain(pre_avg_absorbance, post_avg_absorbance)

        print("\nAverage values and gains:")
        print(f"Transmittance: Pre={pre_avg_transmittance:.2f}%, Post={post_avg_transmittance:.2f}%, Gain={transmittance_gain:+.2f}%")
        print(f"Reflectance: Pre={pre_avg_reflectance:.2f}%, Post={post_avg_reflectance:.2f}%, Gain={reflectance_gain:+.2f}%")
        print(f"Absorbance: Pre={pre_avg_absorbance:.2f}, Post={post_avg_absorbance:.2f}, Gain={absorbance_gain:+.2f}%")

        # Create plots
        def create_plot_data(x_data, y1_data, y2_data, title, y_label, name1, name2):
            return {
                'data': [
                    {
                        'type': 'scatter',
                        'x': list(x_data),
                        'y': list(y1_data),
                        'name': f'Pre-data ({name1})',
                        'mode': 'markers',
                        'marker': {
                            'size': 6,
                            'color': 'blue'
                        },
                        'hovertemplate': 
                            '<b>Pre-data</b><br>' +
                            'Wavelength: %{x:.0f} nm<br>' +
                            f'{y_label}: %{{y:.5f}}%<br>' +
                            '<extra></extra>'
                    },
                    {
                        'type': 'scatter',
                        'x': list(x_data),
                        'y': list(y2_data),
                        'name': f'Post-data ({name2})',
                        'mode': 'markers',
                        'marker': {
                            'size': 6,
                            'color': 'orange'
                        },
                        'hovertemplate': 
                            '<b>Post-data</b><br>' +
                            'Wavelength: %{x:.0f} nm<br>' +
                            f'{y_label}: %{{y:.5f}}%<br>' +
                            '<extra></extra>'
                    }
                ],
                'layout': {
                    'title': {
                        'text': title,
                        'font': {'size': 24}
                    },
                    'xaxis': {
                        'title': 'Wavelength (nm)',
                        'gridcolor': '#E5E5E5',
                        'showgrid': True,
                        'zeroline': False,
                        'tickformat': 'd',
                        'showline': True,
                        'linewidth': 1,
                        'linecolor': 'black',
                        'showticklabels': True,
                        'ticks': 'outside'
                    },
                    'yaxis': {
                        'title': y_label,
                        'gridcolor': '#E5E5E5',
                        'showgrid': True,
                        'zeroline': False,
                        'tickformat': 'd',
                        'showline': True,
                        'linewidth': 1,
                        'linecolor': 'black',
                        'showticklabels': True,
                        'ticks': 'outside'
                    },
                    'plot_bgcolor': 'white',
                    'paper_bgcolor': 'white',
                    'hovermode': 'closest',
                    'showlegend': True,
                    'legend': {
                        'x': 1,
                        'y': 1,
                        'xanchor': 'right',
                        'yanchor': 'top',
                        'bgcolor': 'rgba(255, 255, 255, 0.8)',
                        'bordercolor': '#E5E5E5'
                    }
                }
            }

        transmittance_plot = create_plot_data(
            wavelengths, pre_transmittance, post_transmittance,
            'Transmittance Comparison', 'Transmittance (%)',
            selected_pre.get('design_name', 'Unknown'),
            selected_post.get('design_name', 'Unknown')
        )

        reflectance_plot = create_plot_data(
            wavelengths, pre_reflectance, post_reflectance,
            'Reflectance Comparison', 'Reflectance (%)',
            selected_pre.get('design_name', 'Unknown'),
            selected_post.get('design_name', 'Unknown')
        )

        absorbance_plot = create_plot_data(
            wavelengths, pre_absorbance, post_absorbance,
            'Absorbance Comparison', 'Absorbance',
            selected_pre.get('design_name', 'Unknown'),
            selected_post.get('design_name', 'Unknown')
        )

        return render_template('compare.html',
            pre_data_files=pre_data_files,
            post_data_files=post_data_files,
            show_selection=True,
            transmittance_plot=json.dumps(transmittance_plot),
            reflectance_plot=json.dumps(reflectance_plot),
            absorbance_plot=json.dumps(absorbance_plot),
            pre_avg_transmittance=pre_avg_transmittance,
            pre_avg_reflectance=pre_avg_reflectance,
            pre_avg_absorbance=pre_avg_absorbance,
            post_avg_transmittance=post_avg_transmittance,
            post_avg_reflectance=post_avg_reflectance,
            post_avg_absorbance=post_avg_absorbance,
            transmittance_gain=transmittance_gain,
            reflectance_gain=reflectance_gain,
            absorbance_gain=absorbance_gain,
            selected_pre_file=str(pre_file_id),
            selected_post_file=str(post_file_id),
            error=False
        )
            
    except Exception as e:
        print(f"Unexpected error in compare route: {str(e)}")
        flash(f"An unexpected error occurred: {str(e)}", 'error')
        return render_template('compare.html', error=True)

def ensure_sample_join(sql):
    # If sample. is referenced but no join or from sample is present, add the join
    if re.search(r'sample\\.', sql, re.IGNORECASE) and not re.search(r'join\\s+sample|from\\s+sample', sql, re.IGNORECASE):
        # Find FROM experiment (or FROM "experiment")
        match = re.search(r'(from\\s+experiment\\b)', sql, re.IGNORECASE)
        if match:
            insert_pos = match.end()
            sql = sql[:insert_pos] + ' JOIN sample ON experiment.id = sample.id' + sql[insert_pos:]
    return sql

# Admin routes for user management
@app.route('/admin/users')
@login_required
@admin_required
def admin_users():
    users = User.query.all()
    return render_template('admin/users.html', users=users)

@app.route('/admin/users/<int:user_id>/toggle_admin', methods=['POST'])
@login_required
@admin_required
def toggle_admin_status(user_id):
    user = User.query.get_or_404(user_id)
    if user.id == session['user_id']:
        flash('You cannot modify your own admin status', 'error')
        return redirect(url_for('admin_users'))
    
    user.is_admin = not user.is_admin
    db.session.commit()
    flash(f'Admin status {"granted" if user.is_admin else "revoked"} for {user.username}', 'success')
    return redirect(url_for('admin_users'))

@app.route('/admin/users/<int:user_id>/toggle_active', methods=['POST'])
@login_required
@admin_required
def toggle_user_active(user_id):
    user = User.query.get_or_404(user_id)
    if user.id == session['user_id']:
        flash('You cannot deactivate your own account', 'error')
        return redirect(url_for('admin_users'))
    
    user.is_active = not user.is_active
    db.session.commit()
    flash(f'User {user.username} has been {"activated" if user.is_active else "deactivated"}', 'success')
    return redirect(url_for('admin_users'))

@app.route('/admin/users/<int:user_id>/delete', methods=['POST'])
@login_required
@admin_required
def delete_user(user_id):
    user = User.query.get_or_404(user_id)
    if user.id == session['user_id']:
        flash('You cannot delete your own account', 'error')
        return redirect(url_for('admin_users'))
    
    db.session.delete(user)
    db.session.commit()
    flash(f'User {user.username} has been deleted', 'success')
    return redirect(url_for('admin_users'))

# Add Plots model to store sample ID and SharePoint image links
class Plots(db.Model):
    __tablename__ = 'plots'
    id = db.Column(db.Integer, primary_key=True)
    sample_id = db.Column(db.String(100), db.ForeignKey('sample.id', ondelete='SET NULL'), nullable=True)
    sharepoint_link = db.Column(db.String(500), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    created_by = db.Column(db.String(80))
    
    # Add relationship to Sample
    sample = db.relationship('Sample', backref='plots')
    
    def __repr__(self):
        return f'<Plot {self.sample_id}>'

if __name__ == '__main__':
    # Get port from Replit environment if available
    port = int(os.environ.get('PORT', 5111))  # Using port 5000 as default
    
    with app.app_context():
        # Create tables
        db.create_all()
        print("Database tables created successfully!")
        
        # Check if any users exist, if not create admin user
        try:
            existing_users = User.query.first()
            if not existing_users:
                print("No users found, creating admin user...")
                
                # Create fresh admin user
                admin_user = User(
                    username='admin',
                    email='admin@example.com',  # Add default admin email
                    password=generate_password_hash('admin123', method='pbkdf2:sha256'),
                    is_admin=True,
                    is_active=True,
                    notification_preferences={
                        'email_notifications': True,
                        'system_notifications': True
                    },
                    dashboard_preferences={
                        'recent_activity': True,
                        'saved_queries': []
                    }
                )
                db.session.add(admin_user)
                db.session.commit()
                print("Created admin user with username: admin, password: admin123")
            else:
                print("Users already exist in database, skipping admin creation")
                
        except Exception as e:
            print(f"Error during user setup: {str(e)}")
            db.session.rollback()
    
    # Use Replit's host and port
    app.run(
        host='0.0.0.0',
        port=port,
        debug='development' in os.environ.get('PYTHON_ENV', '').lower()
    )