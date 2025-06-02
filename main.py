from flask import Flask
from flask_talisman import Talisman

app = Flask(__name__)
Talisman(app)  # This enforces HTTPS and adds security headers

@app.route('/')
def home():
    return "Hello, Flask with HTTPS!"

if __name__ == '__main__':
    app.run(ssl_context='adhoc')  # For dev only, generates self-signed cert
