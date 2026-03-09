from fastapi import FastAPI,UploadFile,File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import numpy as np
from pathlib import Path
import os
import pandas as pd

app = FastAPI(title='House price prediction api')

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH  = BASE_DIR / "model"/ "pipeline_stacking.pkl"
print(MODEL_PATH)
model = joblib.load(MODEL_PATH)

class PredictRequest(BaseModel):
    file_path: str

@app.get("/", response_class=HTMLResponse)
async def home():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Predict your house value</title>
        <meta charset="UTF-8">
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 50px auto;

                background-color: #f5f5f5;
            }
            .container {
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
                text-align: center;
            }
            .upload-form {
                margin: 20px 0;
            }
            input[type="file"] {
                margin: 10px 0;
                padding: 10px;
                width: 100%;
            }
            button {
                background-color: #4CAF50;
                color: white;
                padding: 12px 30px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                width: 100%;
            }
            button:hover {
                background-color: #45a049;
            }
            #result {
                margin-top: 20px;
                padding: 15px;
                background-color: #f9f9f9;
                border-radius: 5px;
                display: none;
            }
            .loading {
                text-align: center;
                padding: 20px;
                display: none;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 10px;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            th {
                background-color: #4CAF50;
                color: white;
            }
            tr:nth-child(even) {
                background-color: #f2f2f2;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏠 Predict your house </h1>
            <p style="text-align: center; color: #666;">upload .csv file and predict</p>

            <div class="upload-form">
                <input type="file" id="fileInput" accept=".csv" />
                <button onclick="uploadFile()">upload and predict</button>
            </div>

            <div class="loading" id="loading">
                <p>processing...</p>
            </div>

            <div id="result"></div>
        </div>

        <script>
            async function uploadFile() {
                const fileInput = document.getElementById('fileInput');
                const file = fileInput.files[0];

                if (!file) {
                    alert('choose a file to upload');
                    return;
                }

                const formData = new FormData();
                formData.append('file', file);

                document.getElementById('loading').style.display = 'block';
                document.getElementById('result').style.display = 'none';

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });

                    if (!response.ok) {
                        throw new Error('predict failed');
                    }

                    const data = await response.json();
                    displayResults(data);
                } catch (error) {
                    alert('error: ' + error.message);
                } finally {
                    document.getElementById('loading').style.display = 'none';
                }
            }

            function displayResults(data) {
                const resultDiv = document.getElementById('result');

                let html = '<h2>results:</h2>';
                html += `<p>total ${data.length} records</p>`;
                html += '<table>';
                html += '<tr><th>ID</th><th>predoct price ($)</th></tr>';

                data.slice(0, 20).forEach(item => {
                    html += `<tr><td>${item.Id}</td><td>$${item.SalePrice.toLocaleString()}</td></tr>`;
                });

                if (data.length > 20) {
                    html += `<tr><td colspan="2" style="text-align: center; font-style: italic;">... 只显示前20条结果 ...</td></tr>`;
                }

                html += '</table>';

                resultDiv.innerHTML = html;
                resultDiv.style.display = 'block';
            }
        </script>
    </body>
    </html>
    """
    return html_content

@app.get('/health')
def health():
    return {'status':'ok'}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    df = pd.read_csv(file.file)

    X_test_ids = df["Id"]
    X_test = df.drop(columns=["Id"])

    months = 12
    X_test["sin_MoSold"] = np.sin(2*np.pi*(X_test.MoSold-1)/months)
    X_test["cos_MoSold"] = np.cos(2*np.pi*(X_test.MoSold-1)/months)

    preds_log = model.predict(X_test)
    preds = np.exp(preds_log)

    results = pd.DataFrame({
        "Id": X_test_ids,
        "SalePrice": np.round(preds,2)
    })

    return results.to_dict(orient="records")
