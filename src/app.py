import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import os
import warnings
warnings.filterwarnings("ignore")

# ─── Configuración de página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Tech Career Recommender",
    page_icon="🤖",
    layout="wide"
)

# ─── Estilos ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .rec-card {
        background: #1e1e2e;
        border: 1px solid #313244;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    .match-badge {
        background: #a6e3a1;
        color: #1e1e2e;
        font-weight: 700;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.85rem;
    }
    .gap-ok  { color: #a6e3a1; }
    .gap-bad { color: #f38ba8; }
    .metric-label { color: #cdd6f4; font-size: 0.8rem; }
    .metric-value { font-size: 1.1rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ─── Carga y preparación del modelo ──────────────────────────────────────────
@st.cache_data(show_spinner="Cargando dataset…")
def load_and_prepare():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "..", "data", "processed", "df_model.csv")
    df_model = pd.read_csv(csv_path)


    df = df_model[df_model["ai_adoption_score"] >= 20].copy()

    FEATURES_KNN = ["experience_years", "education_required", "ai_specialization",
                    "work_mode", "country", "industry", "weekly_hours"]
    FEATURES_OUTPUT = ["job_role", "salary_usd", "salary_percentile", "skill_demand_score",
                       "hiring_difficulty_score", "experience_level", "company_size",
                       "ai_adoption_score", "automation_risk", "career_growth_score",
                       "work_life_balance_score", "employee_satisfaction"]
    df_model = df[FEATURES_KNN + FEATURES_OUTPUT].copy()

    cat_cols = ["country", "ai_specialization", "experience_level",
                "education_required", "industry", "company_size", "work_mode"]
    num_cols = ["experience_years", "salary_usd", "weekly_hours",
                "hiring_difficulty_score", "ai_adoption_score", "skill_demand_score",
                "automation_risk", "career_growth_score", "work_life_balance_score",
                "salary_percentile", "employee_satisfaction"]

    encoders = {}
    df_encoded = df_model.copy()
    for col in cat_cols:
        le = LabelEncoder()
        df_encoded[col + "_enc"] = le.fit_transform(df_model[col].astype(str))
        encoders[col] = le

    scaler = StandardScaler()
    num_scaled = scaler.fit_transform(df_model[num_cols])

    enc_cols = [c + "_enc" for c in cat_cols]
    feature_matrix = np.hstack([num_scaled, df_encoded[enc_cols].values])

    return df_model, encoders, scaler, num_cols, cat_cols, enc_cols, feature_matrix

df_model, encoders, scaler, num_cols, cat_cols, enc_cols, feature_matrix = load_and_prepare()

# ─── Funciones del modelo ─────────────────────────────────────────────────────
def build_user_vector(user_profile):
    user_num_raw = []
    for col in num_cols:
        # Si el dato viene del formulario (experiencia o horas), lo usamos.
        if col in user_profile:
            user_num_raw.append(float(user_profile[col]))
        else:
            # Para variables de salida del puesto (salario, riesgo, etc.),
            # usamos 0.0 para no sesgar drásticamente la distancia hacia el promedio
            user_num_raw.append(0.0)
            
    user_num_scaled = scaler.transform([user_num_raw])[0]

    user_cat = []
    for col in cat_cols:
        le = encoders[col]
        val = user_profile.get(col, None)
        
        # Si la columna categórica no está en el perfil (ej. company_size, experience_level), 
        # tomamos la moda del dataset de manera segura
        if val is None:
            val = df_model[col].mode()[0]
            
        if val in le.classes_:
            user_cat.append(float(le.transform([val])[0]))
        else:
            user_cat.append(0.0)
            
    return np.hstack([user_num_scaled, user_cat])


def normalize(arr):
    min_v, max_v = arr.min(), arr.max()
    if max_v > min_v:
        return (arr - min_v) / (max_v - min_v)
    return arr


def calcular_skill_gap(user_profile, row):
    gaps = {}
    gaps["experience_gap_years"] = max(0, row["experience_years"] - float(user_profile.get("experience_years", 0)))
    spec_rol = row["ai_specialization"]
    spec_usuario = user_profile.get("ai_specialization", "")
    gaps["specialization_match"] = spec_rol == spec_usuario
    if not gaps["specialization_match"]:
        gaps["specialization_needed"] = spec_rol

    edu_orden = {"Diploma": 0, "Bachelor": 1, "Bootcamp": 2, "Master": 3, "PhD": 4}
    edu_usuario = edu_orden.get(user_profile.get("education_required", ""), 0)
    edu_rol = edu_orden.get(row["education_required"], 0)
    gaps["education_gap"] = max(0, edu_rol - edu_usuario)
    if gaps["education_gap"] > 0:
        gaps["education_needed"] = row["education_required"]

    gaps["work_mode_match"] = row["work_mode"] == user_profile.get("work_mode", "")
    gaps["salary_bench_usd"] = row["salary_usd"]
    gaps["salary_percentile"] = row["salary_percentile"]
    return gaps


def content_based_scores(user_profile):
    """
    Calcula la similitud basándose estrictamente en las características que el usuario 
    puede seleccionar en el formulario, evitando que las variables de salida distorsionen el score.
    """
    # 1. Similitud Numérica (Años de experiencia y Horas semanales)
    # Escalamos de forma manual y controlada en base a los rangos del formulario
    user_exp = float(user_profile.get("experience_years", 3))
    user_hours = float(user_profile.get("weekly_hours", 40))
    
    dist_exp = np.abs(df_model["experience_years"] - user_exp) / 20.0
    dist_hours = np.abs(df_model["weekly_hours"] - user_hours) / 30.0
    sim_num = 1.0 - ((dist_exp + dist_hours) / 2.0)
    
    # 2. Similitud Categórica (Comparación exacta binaria elemento a elemento)
    # Esto evita el comportamiento errático de LabelEncoder en distancias
    sim_cat = (
        (df_model["ai_specialization"] == user_profile.get("ai_specialization", "")) * 2.0 + # Más peso a la especialización
        (df_model["country"] == user_profile.get("country", "")) * 1.5 +                    # Más peso al país
        (df_model["education_required"] == user_profile.get("education_required", "")) * 1.0 +
        (df_model["work_mode"] == user_profile.get("work_mode", "")) * 1.0 +
        (df_model["industry"] == user_profile.get("industry", "")) * 1.0
    ) / 6.5 # Normalizamos por la suma de los pesos (2 + 1.5 + 1 + 1 + 1)

    # El score de contenido final es la combinación de ambos aspectos
    return (sim_num * 0.3) + (sim_cat * 0.7)


def collaborative_scores(user_profile):
    """
    Busca profesionales en tu mismo ecosistema (Especialización, Modalidad o Industria)
    y prioriza los roles que ofrecen los mejores salarios promedio para ese grupo.
    """
    mask = (
        (df_model["work_mode"] == user_profile.get("work_mode", "")) |
        (df_model["industry"] == user_profile.get("industry", "")) |
        (df_model["ai_specialization"] == user_profile.get("ai_specialization", ""))
    )
    similar_group = df_model[mask] if len(df_model[mask]) >= 5 else df_model
    
    # 🌟 CAMBIO CLAVE: Agrupamos por SALARIO promedio en vez de satisfacción
    role_salary_mean = similar_group.groupby("job_role")["salary_usd"].mean()
    
    # Mapeamos los salarios. Si un rol no está en el grupo similar, recibe el salario mínimo del dataset
    collab_salary = df_model["job_role"].map(role_salary_mean).fillna(df_model["salary_usd"].min()).values.astype(float)
    
    return collab_salary

def recomendar(
    user_profile: dict,
    df_model: pd.DataFrame,
    top_n: int = 3,
    peso_contenido: float = 0.6,
    peso_colaborativo: float = 0.4
) -> list[dict]:

    # 1. Calcular scores base directo sin pasar por matrices corruptas
    scores_cb = normalize(content_based_scores(user_profile))
    scores_cf = normalize(collaborative_scores(user_profile))

    # 2. Calcular Score Híbrido
    total_peso = peso_contenido + peso_colaborativo
    w_cb = peso_contenido / total_peso
    w_cf = peso_colaborativo / total_peso
    scores_hibrido = w_cb * scores_cb + w_cf * scores_cf

    # 3. Penalización estricta por Experiencia Faltante (Filtro duro)
    user_exp = float(user_profile.get("experience_years", 0))
    for idx in range(len(scores_hibrido)):
        row_exp = float(df_model.iloc[idx]["experience_years"])
        if row_exp > user_exp:
            gap = row_exp - user_exp
            # Penalización severa automática para evitar que aparezcan puestos inalcanzables
            scores_hibrido[idx] -= (gap * 0.15) 
            scores_hibrido[idx] = max(0.0, scores_hibrido[idx])

    # 4. Crear dataframe de evaluación temporal
    df_temp = df_model.copy()
    df_temp["_score_hibrido"] = scores_hibrido
    df_temp["_score_cb"] = scores_cb
    df_temp["_score_cf"] = scores_cf
    df_temp["_original_idx"] = df_model.index

    # 5. Agrupar por rol para forzar la diversidad real
    df_mejores_por_rol = (
        df_temp.sort_values(by="_score_hibrido", ascending=False)
        .groupby("job_role")
        .first() 
        .reset_index()
    )

    # 6. Extraer el top definitivo de roles ordenados de manera dinámica
    df_top_roles = df_mejores_por_rol.sort_values(by="_score_hibrido", ascending=False).head(top_n)

    # 7. Construcción del output
    resultados = []
    for _, row in df_top_roles.iterrows():
        resultado = row.to_dict()
        resultado['match_score_pct'] = round(float(resultado['_score_hibrido']) * 100, 1)
        resultado['content_score'] = round(float(resultado['_score_cb']) * 100, 1)
        resultado['collab_score'] = round(float(resultado['_score_cf']) * 100, 1)
        
        idx_original = resultado["_original_idx"]
        for k in ["_score_hibrido", "_score_cb", "_score_cf", "_original_idx"]:
            resultado.pop(k, None)
            
        row_original = df_model.loc[idx_original]
        resultado['skill_gap'] = calcular_skill_gap(user_profile, row_original)
        resultados.append(resultado)

    return resultados
# ─── UI ───────────────────────────────────────────────────────────────────────
st.title("🤖 Tech Career Recommender")
st.caption("Descubre qué puesto tech encaja mejor con tu perfil y cómo mejorar tu salario.")

st.divider()

# Sidebar con el formulario
with st.sidebar:
    st.header("📋 Your professional profile")

    exp_years = st.slider("Experience years", 0, 20, 3)

    education = st.selectbox(
        "Education level",
        ["Diploma", "Bootcamp", "Bachelor", "Master", "PhD"]
    )

    specialization = st.selectbox(
        "AI specialization",
        sorted(df_model["ai_specialization"].unique())
    )

    work_mode = st.selectbox(
        "Work mode",
        sorted(df_model["work_mode"].unique())
    )

    industry = st.selectbox(
        "Industry",
        sorted(df_model["industry"].unique())
    )

    country = st.selectbox(
        "Country",
        sorted(df_model["country"].unique()),
        index=sorted(df_model["country"].unique()).index("Spain")
        if "Spain" in df_model["country"].unique() else 0
    )

    weekly_hours = st.slider("Weekly hours", 30, 60, 40)

    st.divider()

    top_n = st.slider("Number of recomendations", 1, 5, 3)

    # Definimos los pesos fijos internamente
    peso_cb = 0.6
    peso_cf = 0.4

    with st.popover("ℹ️ ¿Cómo calculamos tu afinidad?"):
        st.markdown(f"""
        Utilizamos un **sistema híbrido de recomendación** balanceado automáticamente:
        * **Afinidad por Perfil ({int(peso_cb*100)}%):** Analiza matemáticamente la coincidencia exacta de tus habilidades, tecnologías, modalidad y país con las vacantes.
        * **Satisfacción del Mercado ({int(peso_cf*100)}%):** Cruza datos de profesionales similares en tu área para priorizar puestos con mejor balance vida-trabajo y felicidad laboral.
        """)
    
    st.write("") # Espaciado visual
    buscar = st.button("🔍 Recomend", type="primary", use_container_width=True)

# ─── Resultados ───────────────────────────────────────────────────────────────
if buscar:
    user = {
        "experience_years": exp_years,
        "education_required": education,
        "ai_specialization": specialization,
        "work_mode": work_mode,
        "industry": industry,
        "country": country,
        "weekly_hours": weekly_hours,
    }

    with st.spinner("Buscando los mejores puestos para ti…"):
        # CORRECCIÓN: Pasar explícitamente todos los parámetros por nombre
        recomendaciones = recomendar(
            user_profile=user, 
            df_model=df_model,
            top_n=top_n,
            peso_contenido=peso_cb,
            peso_colaborativo=peso_cf
        )

    st.subheader(f"🎯 Top {top_n} recomendations for your profile")

    for i, rec in enumerate(recomendaciones, 1):
        gap = rec["skill_gap"]
        with st.expander(
            f"#{i}  **{rec['job_role']}** — 💰 ${rec['salary_usd']:,.0f}/año",
            expanded=(i == 1)
        ):
            col1, col2, col3 = st.columns(3,vertical_alignment="center")


            with col1:
                st.metric("💵 Estimated Salary", f"${rec['salary_usd']:,.0f}")
                st.metric("📊 Salary percentile", f"{int(rec['salary_percentile'])}°")
                st.metric("🌍 Country", rec["country"])

            with col2:
                st.metric("🏢 Company size", rec["company_size"])
                st.metric("⚡ Work mode", rec["work_mode"])
                st.metric("💻 Automation risk", f"{int(rec['automation_risk'])}%")

            with col3:
                st.metric("📈 Career growth", f"{int(rec['career_growth_score'])}%")
                st.metric("⚖️ Work-Life Balance", f"{int(rec['work_life_balance_score'])}%")
                st.markdown("""
                                <style>
                                    div[data-baseweb="tooltip"] {
                                        width: 280px !important;
                                        white-space: normal !important;
                                        word-wrap: break-word !important;
                                        text-align: justify !important;
                                    }
                                </style>
                            """, unsafe_allow_html=True)
                st.metric("😊 Employee satisfaction", f"{int(rec['employee_satisfaction'])}%", help="This percentage indicates the satisfaction that employees usually have in this type of role, taking into account all the metrics mentioned.")

            st.divider()
            st.markdown("**🔍 Skill Gap — what separates you from this position:**")
            

            g1, g2, g3, g4= st.columns(4)

            with g1:
                exp_gap = gap["experience_gap_years"]
                color = "gap-ok" if exp_gap == 0 else "gap-bad"
                label = "✅ Enough" if exp_gap == 0 else f"⚠️ You need {exp_gap:.0f} years"
                st.markdown(f"<p class='metric-label'>Experience</p><p class='{color} metric-value'>{label}</p>",
                            unsafe_allow_html=True)

            with g2:
                spec_ok = gap["specialization_match"]
                color = "gap-ok" if spec_ok else "gap-bad"
                label = "✅ Match" if spec_ok else f"⚠️ You need: {gap.get('specialization_needed', '')}"
                st.markdown(f"<p class='metric-label'>Specialization</p><p class='{color} metric-value'>{label}</p>",
                            unsafe_allow_html=True)

            with g3:
                edu_gap = gap["education_gap"]
                color = "gap-ok" if edu_gap == 0 else "gap-bad"
                label = "✅ OK" if edu_gap == 0 else f"⚠️ You need: {gap.get('education_needed', '')}"
                st.markdown(f"<p class='metric-label'>Education</p><p class='{color} metric-value'>{label}</p>",
                            unsafe_allow_html=True)

            with g4:
                wm_ok = gap["work_mode_match"]
                color = "gap-ok" if wm_ok else "gap-bad"
                label = "✅ Match" if wm_ok else "⚠️ Different work mode"
                st.markdown(f"<p class='metric-label'>Work Mode</p><p class='{color} metric-value'>{label}</p>",
                            unsafe_allow_html=True)
                
    
            st.divider()
            st.markdown(
                f"AI adoption in the company: **{int(rec['ai_adoption_score'])}%** &nbsp; ")
            

else:
    st.info("👈 Fill out your profile in the left panel and click **Recommend** to see your results.")

    # Mostrar estadísticas generales mientras se espera
    st.subheader("📊 Labor market summary (2022–2026)")
    st.warning("The recommendations are based on surveys conducted between 2022 and 2026 and do not guarantee hiring for the recommended roles or the salary shown.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Records analyzed", f"{len(df_model):,}")
    c2.metric("Unique roles", df_model["job_role"].nunique())
    c3.metric("Median salary", f"${df_model['salary_usd'].mean():,.0f}")

    col_a, col_b, colc = st.columns(3)
    with col_a:
        st.markdown("**Top 10 highest paying jobs (median)**")
        top_roles = (
            df_model.groupby("job_role")["salary_usd"]
            .median()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        top_roles.columns = ["Rol", "Median salary (USD)"]
        top_roles["Median salary (USD)"] = top_roles["Median salary (USD)"].apply(lambda x: f"${x:,.0f}")
        st.dataframe(top_roles, hide_index=True, use_container_width=True)

    with col_b:
        st.markdown("**Country distribution**")
        wm_counts = df_model["country"].value_counts().reset_index()
        wm_counts.columns = ["Country", "Records"]
        st.dataframe(wm_counts, hide_index=True, use_container_width=True)

    with colc:
        st.markdown("**Industry distribution**")
        wm_counts = df_model["industry"].value_counts().reset_index()
        wm_counts.columns = ["industry", "Records"]
        st.dataframe(wm_counts, hide_index=True, use_container_width=True)