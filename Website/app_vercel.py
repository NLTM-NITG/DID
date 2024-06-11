from flask import Flask, render_template_string

app = Flask(__name__)

@app.route('/')
def hello():
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Hello, World!</title>
    </head>
    <body>
        <h1>Hello, World!</h1>
        <p>This is a simple HTML page rendered inline with Flask.</p>
    </body>
    </html>
    """
    return render_template_string(html)

if __name__ == "__main__":
    app.run()
