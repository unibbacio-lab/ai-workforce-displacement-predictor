from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# =========================
# LOAD MODEL
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(
    BASE_DIR,
    '..',
    'models',
    'ai_workforce_model.pkl'
)

model = joblib.load(model_path)

# =========================
# ROUTE
# =========================

@app.route('/', methods=['GET', 'POST'])
def index():

    prediction = None

    if request.method == 'POST':

        ai_adoption_index = float(request.form['ai_adoption_index'])
        automation_risk = float(request.form['automation_risk'])
        gdp = float(request.form['gdp'])
        ai_tool = float(request.form['ai_tool'])
        policy = float(request.form['policy'])

        data = np.array([[
            ai_adoption_index,
            automation_risk,
            gdp,
            ai_tool,
            policy
        ]])

        prediction = model.predict(data)[0]

    return render_template(
        'index.html',
        prediction=prediction
    )

# =========================
# RUN APP
# =========================

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
