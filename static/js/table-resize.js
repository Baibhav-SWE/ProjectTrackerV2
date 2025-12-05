// Create a new file for the table resize functionality
document.addEventListener('DOMContentLoaded', function() {
    const tables = document.querySelectorAll('table');
    
    tables.forEach(table => {
        const headers = table.querySelectorAll('thead tr:first-child th');
        
        headers.forEach((header, index) => {
            // Skip process header columns
            if (!header.classList.contains('process-header')) {
                const resizer = document.createElement('div');
                resizer.className = 'resizer';
                header.appendChild(resizer);
                
                let startX, startWidth;
                
                resizer.addEventListener('mousedown', function(e) {
                    startX = e.pageX;
                    startWidth = header.offsetWidth;
                    
                    const mouseMoveHandler = function(e) {
                        // Calculate the new width
                        const width = Math.max(100, startWidth + (e.pageX - startX));
                        
                        // Update header width
                        header.style.width = `${width}px`;
                        
                        // Update all cells in this column
                        const cells = table.querySelectorAll(`tbody tr td:nth-child(${index + 1})`);
                        cells.forEach(cell => {
                            cell.style.width = `${width}px`;
                        });
                    };
                    
                    const mouseUpHandler = function() {
                        document.removeEventListener('mousemove', mouseMoveHandler);
                        document.removeEventListener('mouseup', mouseUpHandler);
                    };
                    
                    document.addEventListener('mousemove', mouseMoveHandler);
                    document.addEventListener('mouseup', mouseUpHandler);
                });
            }
        });
    });
}); 