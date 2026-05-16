from flask import Flask, jsonify, render_template
from model import segment_customers

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/segment")
def segment():
    return jsonify(segment_customers())

if __name__ == "__main__":
    app.run(debug=True)
