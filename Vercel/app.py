from flask import Flask, render_template

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
    
    return render_template('update_vercel.html', results=results)

if __name__ == "__main__":
    app.run()
