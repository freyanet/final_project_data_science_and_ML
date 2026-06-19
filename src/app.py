import os
import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template


base_dir = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(base_dir, "templates"))

model_salary = pickle.load(open(os.path.join(base_dir, "..", "models", "random_forest_regressor_salary.sav"), "rb"))
model_job = pickle.load(open(os.path.join(base_dir, "..", "models", "random_forest_classifier_job.sav"), "rb"))
scaler = pickle.load(open(os.path.join(base_dir, "..", "models", "scaler_salary.sav"), "rb"))
df_model = pd.read_csv(os.path.join(base_dir, "..", "data", "processed", "df_model.csv"))

salary_cols = [c for c in df_model.columns if c != "salary_usd"]
job_cols = [c for c in df_model.columns if c.startswith("job_role_")]
jobs_cols = [c for c in df_model.columns if c not in job_cols and c != "salary_usd"]


AI_SPECS = [
    "Computer Vision", 
    "Forecasting", 
    "Generative AI", 
    "LLM", 
    "MLOps", 
    "NLP", 
    "Reinforcement Learning"
]


EDU_MAP = {"Diploma": 0, "Bootcamp": 1, "Bachelor": 2, "Master": 3, "PhD": 4}

def build_user_profile(perfil, expected_columns):
    X_user = pd.DataFrame(0, index=[0], columns=expected_columns)

    # Rellenar con medianas del dataset
    for col in expected_columns:
        if col in df_model.columns:
            X_user[col] = df_model[col].median()

    # Sobrescribir con valores del usuario
    for col, val in perfil.items():
        if col in X_user.columns:
            X_user[col] = val

    return X_user

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/recomendar", methods=["POST"])
def api_recomendar():
    perfil_usuario = request.get_json()
    if 'experience_years' not in perfil_usuario or 'education_required' not in perfil_usuario:
        return jsonify({"error": "Missing required fields: experience_years and education_level"}), 400
    if 'ai_specialization' not in perfil_usuario:
        return jsonify({"error": "Missing required field: ai_specialization"}), 400
    
    exp_years = int(perfil_usuario['experience_years'])
    edu_level = perfil_usuario['education_required']

    if isinstance(edu_level, str):
        edu_level = EDU_MAP.get(edu_level, 2)

    ai_selected = perfil_usuario.get("ai_specialization", "LLM")
    perfil = {
        "experience_years": exp_years,
        "education_required": edu_level,
    }

    for spec in AI_SPECS:
        perfil[f"ai_specialization_{spec}"] = 1 if spec == ai_selected else 0

    X_salary = build_user_profile(perfil, salary_cols)
    salary_prediction = model_salary.predict(scaler.transform(X_salary))[0]

    X_job = build_user_profile(perfil, salary_cols)
    job_prediction = model_job.predict(X_job)[0].replace("job_role_", "")

    aspiraciones = df_model[
    (df_model["experience_years"] >= exp_years - 2) & 
    (df_model["experience_years"] <= exp_years + 2)
    ]


    edu_map_inv = {v: k for k, v in EDU_MAP.items()}
    education_prediction = edu_map_inv.get(int(aspiraciones["education_required"].mode()[0]), "Master")

    recom_horas = aspiraciones['weekly_hours'].median()

    ai_cols = [c for c in salary_cols if c.startswith("ai_specialization_")]
    faltantes = [c.replace("ai_specialization_", "") for c in ai_cols if perfil.get(c, 0) == 0]
    
    return jsonify({
        "salario estimado en usd": round(float(salary_prediction), 2),
        "puestos": job_prediction,
        "recomendaciones": {
            "especializaciones a aprender": faltantes,
            "educacion recomendada": education_prediction,
            "horas semanales recomendadas": round(float(recom_horas), 1),
            }
        }
    )



if __name__ == "__main__":
    app.run(debug=True)