from flask import Flask, render_template, jsonify
app = Flask(__name__, template_folder='templates', static_folder='static')
@app.route('/')
def index(): return render_template('index.html')
@app.route('/api/languages')
def languages(): return jsonify({'ro': 'Ro'})
@app.route('/api/models')
def models(): return jsonify({'device': 'cpu'})
if __name__ == '__main__': app.run(port=5000)
