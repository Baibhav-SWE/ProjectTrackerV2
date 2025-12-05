# AWI Project Tracker

A comprehensive project tracking and management system designed for managing samples, experiments, and research data. This application provides a robust platform for tracking, analyzing, and visualizing experimental data with advanced features for data management and analysis.

## Features

### 1. User Authentication and Authorization
- Secure login and registration system
- Role-based access control (Admin and Regular users)
- Password reset functionality via email
- Session management and security features

### 2. Sample Management
- Create, read, update, and delete (CRUD) operations for samples
- Detailed sample information tracking including:
  - Company name
  - ERB (Experimental Reference Number)
  - Date and time tracking
  - Recipe details (front and back)
  - Glass type specifications
  - Physical dimensions (length, thickness, height)
  - Process status (cleaning, coating, annealing)
  - Sample images and descriptions

### 3. Experiment Management
- Comprehensive experiment data tracking
- Support for multiple types of experimental data:
  - Transmittance
  - Reflectance
  - Absorbance
  - PLQY (Photoluminescence Quantum Yield)
  - SEM (Scanning Electron Microscopy)
  - EDX (Energy Dispersive X-ray Spectroscopy)
  - XRD (X-ray Diffraction)

### 4. Data Analysis and Visualization
- Interactive plots and graphs using Plotly
- Comparative analysis tools
- Data filtering and sorting capabilities
- Custom query support for data analysis
- Statistical analysis features

### 5. Advanced Search and Filtering
- Multi-criteria search functionality
- Date range filtering
- Status-based filtering
- Company-specific filtering
- Advanced query parsing and processing

### 6. Data Recovery and Management
- Trash management system for deleted items
- Restore functionality for accidentally deleted data
- Audit trail for data modifications
- Data backup and recovery options

### 7. Prefix Management
- Custom prefix system for sample identification
- Prefix table management
- Full form tracking for prefixes

### 8. AI-Powered Features
- Intelligent chatbot for data queries
- Natural language processing for data analysis
- Automated data interpretation
- Smart search capabilities

### 9. Administrative Features
- User management dashboard
- Admin privilege control
- User activity monitoring
- System configuration management

### 10. Data Export and Integration
- CSV export functionality
- Google Drive integration
- Data migration tools
- External system integration capabilities

## Technical Stack

- **Backend**: Flask (Python)
- **Database**: 
  - Primary: PostgreSQL
  - Backup: SQLite
  - Additional: MongoDB for specific features
- **Frontend**: HTML, CSS, JavaScript
- **Data Visualization**: Plotly
- **AI Integration**: OpenAI API
- **Email Service**: SendGrid
- **Authentication**: Custom implementation with secure password hashing

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   - Create a `.env` file with necessary configurations
   - Configure database connections
   - Set up API keys for external services

4. Initialize the database:
   ```bash
   flask db upgrade
   ```

5. Run the application:
   ```bash
   python app.py
   ```

## Security Features

- Password hashing using Werkzeug's security functions
- Session-based authentication
- CSRF protection
- Secure password reset mechanism
- Role-based access control
- Input validation and sanitization

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is proprietary and confidential. All rights reserved.

## Support

For support and queries, please contact the development team or raise an issue in the repository. 