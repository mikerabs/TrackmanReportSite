from flask import Flask, request, render_template, send_file
import pandas as pd
from report_generator import generate_report  # <- refactored function
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['csv_file']
        pitcher_name = request.form['pitcher_name']
        innings_pitched = request.form['innings_pitched']

        if file and pitcher_name:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            df = pd.read_csv(filepath)
            pdf_path = generate_report(df, pitcher_name, innings_pitched)
            return send_file(pdf_path, as_attachment=True)
    
    return render_template("index.html")

if __name__ == '__main__':
    #app.run(debug=True)
    app.run(host="0.0.0.0", port=5000)
