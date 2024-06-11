from flask import Flask, render_template_string

app = Flask(__name__)

@app.route('/')
def check_imports():
    results = {}
    packages = ['flask', 'requests', 'numpy']
    for package in packages:
        try:
            exec(f"import {package}")
            results[package] = 'Successfully imported'
        except ImportError as e:
            results[package] = f'Failed to import: {e}'
    
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Package Import Check</title>
    </head>
    <body>
        <h1>Package Import Check</h1>
        <ul>
            {% for package, status in results.items() %}
                <li><strong>{{ package }}:</strong> {{ status }}</li>
            {% endfor %}
        </ul>
    </body>
    </html>
    """
    return render_template_string(html, results=results)

if __name__ == "__main__":
    app.run()
