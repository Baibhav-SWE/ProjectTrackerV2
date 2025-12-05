import os

class Config:
    # Base configuration
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = os.environ.get('SECRET_KEY', 'your-secret-key-here')

class DevelopmentConfig(Config):
    DB_HOST = 'localhost'
    DB_PORT = '5432'
    DB_NAME = 'Project_Tracker'
    DB_USER = 'postgres'
    DB_PASS = 'AWI123'
    
    SQLALCHEMY_DATABASE_URI = f'postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}'

class ReplitConfig(Config):
    # Use Replit's database URL if available
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')

class ProductionConfig(Config):
    DB_HOST = os.environ.get('DB_HOST', '192.168.1.x')
    DB_PORT = os.environ.get('DB_PORT', '5432')
    DB_NAME = os.environ.get('DB_NAME', 'Project_Tracker')
    DB_USER = os.environ.get('DB_USER', 'project_user')
    DB_PASS = os.environ.get('DB_PASS', 'secure_password')
    
    SQLALCHEMY_DATABASE_URI = f'postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}' 