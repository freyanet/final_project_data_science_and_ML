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
    /* 1. Fondo principal y texto general */
    .stApp {
        background-color: #FAFAFA !important;
        color: #2B2124 !important;
    }

    /* 2. Barra lateral (Sidebar) con tono rosa pastel muy claro */
    [data-testid="stSidebar"] {
        background-color: #F4E8EA !important;
        border-right: 1px solid #E5C3C8 !important;
    }

    /* 3. Botones (Recommend) en Rosa Ash Vivo con texto claro */
    div.stButton > button:first-child {
        background-color: #C07A8B !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease;
    }
    
    div.stButton > button:first-child:hover {
        background-color: #A96273 !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 12px rgba(192, 122, 139, 0.3);
    }

    /* 4. Tarjetas / Expanders de recomendaciones en Fondo Claro */
    .streamlit-expanderHeader {
        background-color: #FFFFFF !important;
        border-radius: 8px !important;
        color: #8C4253 !important;
        border: 1px solid #E5C3C8 !important;
        font-weight: 600 !important;
    }
    
    .streamlit-expanderContent {
        background-color: #FAF4F5 !important;
        border: 1px solid #E5C3C8 !important;
        border-top: none !important;
        border-radius: 0 0 8px 8px !important;
    }

    /* 5. Títulos y métricas destacadas en Rosa Ceniza Oscuro */
    h1, h2, h3, [data-testid="stMetricValue"] {
        color: #8C4253 !important;
    }

    /* 6. Sliders en tono rosa */
    div[data-baseweb="slider"] div {
        color: #C07A8B !important;
    }

    /* 7. Tablas / Dataframes estilo claro */
    .stDataFrame {
        border: 1px solid #E5C3C8 !important;
        border-radius: 8px !important;
        background-color: #FFFFFF !important;
    }
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

def calcular_afinidad_perfil(user_profile, df):
    """
    Mide qué tan bien encaja cada fila del dataset con el perfil actual del usuario,
    permitiendo un margen de experiencia realista de máximo 2-3 años.
    """
    user_exp = float(user_profile.get("experience_years", 3))
    user_hours = float(user_profile.get("weekly_hours", 40))
    
    # 1. Distancias numéricas base
    dist_exp = np.abs(df["experience_years"] - user_exp) / 20.0
    dist_hours = np.abs(df["weekly_hours"] - user_hours) / 30.0
    sim_num = 1.0 - ((dist_exp + dist_hours) / 2.0)
    
    # 2. Coincidencia de etiquetas duras (país, modalidad, especialización, educación)
    sim_cat = (
        (df["country"] == user_profile.get("country", "")) * 2.0 +
        (df["work_mode"] == user_profile.get("work_mode", "")) * 1.5 +
        (df["industry"] == user_profile.get("industry", "")) * 1.0 +
        (df["ai_specialization"] == user_profile.get("ai_specialization", "")) * 2.0 +
        (df["education_required"] == user_profile.get("education_required", "")) * 1.5
    ) / 8.0

    score_afinidad = (sim_num * 0.3) + (sim_cat * 0.7)
    
    # Permitimos un margen de hasta 2-3 años adicionales. Si pide más de eso, 
    # la penalización se vuelve destructiva para sacarlo del Top 3.
    for idx in score_afinidad.index:
        row_exp = float(df.loc[idx, "experience_years"])
        if row_exp > user_exp:
            gap = row_exp - user_exp
            if gap <= 3.0:
                # Penalización leve por estar ligeramente arriba (es un salto alcanzable)
                score_afinidad.loc[idx] -= (gap * 0.05)
            else:
                # Penalización masiva si el puesto exige demasiados años más
                score_afinidad.loc[idx] -= (gap * 0.25)
            
    return score_afinidad.clip(lower=0.0)


def recomendar(user_profile: dict, df_model: pd.DataFrame, top_n: int = 3, **kwargs) -> list[dict]:
    """
    Genera el top de recomendaciones:
    - Puestos 1 al 3: Encaje con perfil actual (permitiendo máximo +3 años de exp), ordenados por mejor salario.
    - Puestos 4 y 5: Puestos con UPGRADE de educación/especialización que pagan más, manteniendo el límite de experiencia.
    """
    df_temp = df_model.copy()
    user_exp = float(user_profile.get("experience_years", 3))
    
    # Añadimos una columna índice original antes de cualquier filtrado para no perder el rastro
    df_temp["_original_idx"] = df_temp.index
    
    # Calcular afinidad para todo el dataset
    df_temp["_score_afinidad"] = calcular_afinidad_perfil(user_profile, df_temp)
    
    # ─── FASE 1: TRABAJOS DE ENCAJE DIRECTO (Puestos 1 al 3) ───
    # Filtramos de forma estricta: afinidad aceptable y máximo 3 años por encima de la experiencia del usuario
    df_filtrado_directo = df_temp[
        (df_temp["_score_afinidad"] >= 0.4) & 
        (df_temp["experience_years"] <= user_exp + 3.0)
    ]
    
    df_encaje_directo = (
        df_filtrado_directo.sort_values(by=["_score_afinidad", "salary_usd"], ascending=[False, False])
        .groupby("job_role")
        .first()
        .reset_index()
        .sort_values(by="salary_usd", ascending=False)
    )
    
    top_directo = df_encaje_directo.head(3)
    roles_seleccionados = set(top_directo["job_role"].unique()) if not top_directo.empty else set()
    
    # ─── FASE 2: UPGRADES ASPIRACIONALES (Puestos 4 y 5) ───
    top_aspiracional = pd.DataFrame()
    if top_n > 3 and not top_directo.empty:
        max_salario_directo = top_directo["salary_usd"].max()
        
        # Buscamos puestos que paguen MÁS, que NO estén repetidos en el top 3,
        # pero respetando estrictamente que NO superen en más de 3 años la experiencia del usuario
        df_potenciales_upgrades = df_temp[
            (~df_temp["job_role"].isin(roles_seleccionados)) & 
            (df_temp["salary_usd"] > max_salario_directo) &
            (df_temp["experience_years"] <= user_exp + 3.0)
        ].copy()
        
        if not df_potenciales_upgrades.empty:
            df_aspiracionales_unicos = (
                df_potenciales_upgrades.sort_values(by="salary_usd", ascending=False)
                .groupby("job_role")
                .first()
                .reset_index()
            )
            top_aspiracional = df_aspiracionales_unicos.head(top_n - 3)
            
    # Combinar ambas listas respetando el orden
    df_final_recomendaciones = pd.concat([top_directo, top_aspiracional], ignore_index=True)
    
    # ─── FASE 3: CONSTRUCCIÓN DE LA RESPUESTA PARA STREAMLIT ───
    resultados = []
    for i, row in df_final_recomendaciones.iterrows():
        resultado = row.to_dict()
        
        resultado['match_score_pct'] = round(float(resultado['_score_afinidad']) * 100, 1)
        resultado['content_score'] = round(float(resultado['_score_afinidad']) * 100, 1)
        resultado['collab_score'] = 100.0 
        
        resultado['es_aspiracional'] = (i >= 3)
        
        idx_original = resultado["_original_idx"]
        for k in ["_score_afinidad", "_score_cf", "_original_idx"]:
            resultado.pop(k, None)
            
        row_original = df_model.loc[idx_original]
        resultado['skill_gap'] = calcular_skill_gap(user_profile, row_original)
        resultados.append(resultado)
        
    return resultados
# ─── UI ───────────────────────────────────────────────────────────────────────
st.title("🤖 Tech Career Recommender")
st.caption("Discover which tech job best suits your profile and maximize your career growth in the AI era." \
)

st.divider()

# Sidebar con el formulario
with st.sidebar:
    st.header("📋 Your professional profile")
    st.caption("For a better experience and recommendations, please fill out your profile")

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

    top_n = st.slider("Number of recomendations", 5,7)

    # Definimos los pesos fijos internamente
    peso_cb = 0.8
    peso_cf = 0.2

    with st.popover("ℹ️ How do we calculate your affinity?"):
        st.markdown(f"""
        We use a **hybrid recommendation system** automatically balanced:
        * **Profile Affinity ({int(peso_cb*100)}%):** Mathematically analyzes the exact match of your skills, technologies, work mode, and country with job postings.
        * **Market Salary ({int(peso_cf*100)}%):** Cross-references data from similar professionals in your field to prioritize positions with the best salary.""")
    
    st.write ("") # Espaciado visual
    buscar = st.button("🔍 Recommend me", type="primary", use_container_width=True)

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

        recomendaciones = recomendar(
            user_profile=user, 
            df_model=df_model,
            top_n=top_n,
            peso_contenido=peso_cb,
            peso_colaborativo=peso_cf
        )

    st.subheader(f"🎯 Top {top_n} recomendations for your profile")
    st.caption("The first three recommendations are the best fit for your current profile, while the others are aspirational positions that could pay more if you upgrade your skills or education.")

    for i, rec in enumerate(recomendaciones, 1):
        gap = rec["skill_gap"]
        
    
        tipo_puesto = "✨ RECOMENDATION" if i <= 3 else "🚀 UPGRADE YOUR PROFILE FOR A BETTER SALARY"
        
        with st.expander(
            f"#{i} {tipo_puesto} **{rec['job_role']}** — 💰 ${rec['salary_usd']:,.0f}/año",
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
    st.info("👈 Fill out your profile in the left panel and click **Recommend me** to see your results.")

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