
import os
from supabase import create_client
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Supabase credentials
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

# Initialize Supabase client
supabase = None
if supabase_url and supabase_key:
    supabase = create_client(supabase_url, supabase_key)
else:
    print("Warning: Supabase credentials not found in environment variables.")
