import os
import psycopg2
from dotenv import load_dotenv

def test_postgres_db():
    """Test Replit PostgreSQL connection"""
    load_dotenv()

    db_url = os.getenv("DATABASE_URL")

    if not db_url:
        print("Error: DATABASE_URL not set in .env file")
        return False

    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        cur.execute("SELECT version();")
        version = cur.fetchone()
        print(f"PostgreSQL connection successful! Version: {version[0]}")

        # Try to query the database for tables
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)
        tables = cur.fetchall()
        print("Available tables:")
        for table in tables:
            print(f"- {table[0]}")

        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"PostgreSQL connection error: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing PostgreSQL connection...")
    db_success = test_postgres_db()

    if db_success:
        print("\nPostgreSQL connection is working correctly!")
    else:
        print("\nPostgreSQL connection failed. Please check your configuration.")