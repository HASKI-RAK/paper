from IPython.display import HTML
import pandas as pd

def customized_html_export(df, filename, display_simple_table_here=True):
    """
    Exports a pandas DataFrame as a simple HTML table and an advanced HTML table with search and filter functionality.
    
    Parameters:
    - df: pd.DataFrame, the DataFrame to export.
    - filename: str, the name of the HTML file for the advanced table with filters.
    - display_simple_table_here: bool, whether to display a simple table in the Jupyter Notebook output.
    
    Returns:
    - HTML: Displays the simple table in the notebook if display_simple_table_here is True.
    """
    # Generate simple HTML table
    simple_table = df.style.set_table_styles(
        [{'selector': 'thead th', 'props': [('background-color', '#f7f7f7')]}]
    ).to_html()

    # Generate HTML with DataTables and column filters
    advanced_html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
    table {{
        border-collapse: collapse;
        width: 100%;
    }}
    th, td {{
        text-align: left;
        padding: 8px;
        border: 1px solid #ddd;
    }}
    thead th {{
        background-color: #f7f7f7;
    }}
    tfoot input {{
        width: 100%;
        padding: 3px;
        box-sizing: border-box;
    }}
    </style>
    <!-- DataTables CSS -->
    <link rel="stylesheet" href="https://cdn.datatables.net/1.13.5/css/jquery.dataTables.min.css">
    </head>
    <body>
    {simple_table}
    <!-- DataTables JS -->
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.5/js/jquery.dataTables.min.js"></script>
    <script>
        // DataTables initialization with column filters
        $(document).ready(function() {{
            // Add filter inputs below headers
            $('table thead').append('<tr></tr>');
            $('table thead tr').eq(1).html(
                $('table thead th').map(function() {{
                    return '<th><input type="text" placeholder="Filter ' + $(this).text() + '" /></th>';
                }}).get().join('')
            );

            // Initialize DataTable
            var table = $('table').DataTable();

            // Event listener for column filtering
            table.columns().every(function() {{
                var column = this;
                $('input', column.header()).on('keyup change', function() {{
                    if (column.search() !== this.value) {{
                        column.search(this.value).draw();
                    }}
                }});
            }});
        }});
    </script>
    </body>
    </html>
    """

    # Save the advanced table to an external HTML file
    with open(filename, "w") as f:
        f.write(advanced_html_content)

    # Optionally display the simple table in the notebook
    if display_simple_table_here:
        return HTML(simple_table)