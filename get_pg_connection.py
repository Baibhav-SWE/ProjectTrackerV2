
import os
from dotenv import load_dotenv

def get_pg_connection_details():
    """Get PostgreSQL connection details for pgAdmin"""
    load_dotenv()
    
    db_url = os.getenv("DATABASE_URL")
    
    if not db_url:
        print("Error: DATABASE_URL not set in .env file")
        return
    
    # Parse the connection string
    # Format typically: postgresql://username:password@hostname:port/database
    try:
        # Split by :// to get protocol and rest
        parts = db_url.split("://", 1)
        if len(parts) != 2:
            raise ValueError("Invalid connection string format")
        
        protocol, rest = parts
        
        # Split credentials and host info
        credentials_host = rest.split("@", 1)
        if len(credentials_host) != 2:
            raise ValueError("Invalid connection string format")
        
        credentials, host_info = credentials_host
        
        # Get username and password
        username_password = credentials.split(":", 1)
        if len(username_password) != 2:
            username = username_password[0]
            password = ""
        else:
            username, password = username_password
        
        # Get host, port, and database
        host_port_db = host_info.split("/", 1)
        if len(host_port_db) != 2:
            raise ValueError("Invalid connection string format")
        
        host_port, database = host_port_db
        
        # Split host and port
        if ":" in host_port:
            host, port = host_port.split(":", 1)
        else:
            host = host_port
            port = "5432"  # Default PostgreSQL port
        
        print("\n=== PostgreSQL Connection Details for pgAdmin 4 ===")
        print(f"Host: {host}")
        print(f"Port: {port}")
        print(f"Database: {database}")
        print(f"Username: {username}")
        print(f"Password: {'*' * len(password)} (hidden)")
        print("\nIn pgAdmin 4:")
        print("1. Right-click on 'Servers' and select 'Create' > 'Server...'")
        print("2. Give it a name (e.g., 'Replit PostgreSQL')")
        print("3. In the 'Connection' tab, enter the details above")
        print("4. Click 'Save' to connect")
        print("\nNote: You may need to ensure your local network can reach the Replit database.")
        print("      If you have connection issues, consider using Replit's Database tool instead.")
        
    except Exception as e:
        print(f"Error parsing connection string: {str(e)}")

if __name__ == "__main__":
    get_pg_connection_details()
