from flask import Flask, render_template, request, jsonify
import requests
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['API_URL'] = 'http://localhost:8000/predict'  # FastAPI endpoint

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    
    if request.method == 'POST':
        # Check if file is in the request
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')
            
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return render_template('index.html', error='No file selected')
            
        if file:
            # Send the file to the FastAPI endpoint
            files = {'file': (file.filename, file.read(), file.content_type)}
            try:
                response = requests.post(app.config['API_URL'], files=files)
                
                if response.status_code == 200:
                    prediction = response.json()
                else:
                    return render_template('index.html', error=f'API Error: {response.text}')
            except requests.exceptions.RequestException as e:
                return render_template('index.html', error=f'Connection error: {str(e)}')
    
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True, port=5000) 