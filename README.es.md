# Tech Career Recommender — Sistema de recomendación salarial para perfiles de IA y Data Science

Proyecto Final de Bootcamp — Data Science & Machine Learning (4Geeks Academy)
Por: Joseline Proaño y Valeria Urbina

## Descripción del proyecto

Este proyecto desarrolla un sistema de recomendación para profesionales con perfiles tecnológicos (IA, Machine Learning, Data Science). A partir del perfil de un usuario — experiencia, formación, especialización en IA, modalidad de trabajo, industria, país y horas semanales — el sistema recomienda:

- **Puestos de encaje directo** (top 3): roles que se ajustan a su perfil actual, permitiendo un margen realista de hasta 3 años de experiencia por encima de la suya.
- **Puestos aspiracionales** (posiciones 4 y 5): roles que pagan más que los de encaje directo, pensados como una meta de mejora salarial alcanzable si el usuario amplía su formación o especialización.

El recomendador combina dos señales en un **score híbrido**:

1. **Score de contenido** (`calcular_afinidad_perfil`): mide el encaje directo entre el perfil del usuario y cada puesto, combinando similitud numérica (experiencia, horas semanales) y coincidencia de variables categóricas (país, modalidad, especialización, industria, educación).
2. **Score colaborativo**: calidad objetiva del puesto, calculada a partir de la satisfacción del empleado, el crecimiento de carrera, el equilibrio vida-trabajo y el percentil salarial, todos normalizados.

Ambos se combinan con un reparto **80% contenido / 20% colaborativo**, validado mediante un grid search documentado en el notebook (ver sección de Modelado y Evaluación).

## Dataset

Se utiliza el dataset [Global AI & Data Jobs Salary Dataset](https://www.kaggle.com/datasets/mohankrishnathalla/global-ai-and-data-jobs-salary-dataset) de Kaggle, con información salarial y profesional de puestos relacionados con IA y Data Science a nivel global (2020-2026). Para el modelado se filtró a los registros de 2022 en adelante, con el objetivo de reflejar mejor las condiciones actuales del mercado laboral tecnológico.

## Estructura del repositorio

```
├── data/
│   └── processed/          # Datasets procesados (df_model.csv, df_encoded.csv)
├── models/                 # Artefactos serializados (scaler.pkl, enc_cols.pkl)
├── src/
│   ├── explore.ipynb        # Notebook completo: EDA, preprocesamiento, modelado y evaluación
│   └── app.py                # Aplicación Streamlit del recomendador
├── requirements.txt
└── README.md
```

## Cómo ejecutar el proyecto

### 1. Clonar el repositorio e instalar dependencias

```bash
git clone <url-del-repo>
cd <nombre-del-repo>
pip install -r requirements.txt
```

### 2. Notebook de análisis y modelado

Abrir `src/explore.ipynb` para revisar el proceso completo: obtención y exploración del dataset, justificación de las decisiones de preprocesamiento, construcción y evolución del modelo de recomendación, y evaluación con métricas (Precision@K, catalog coverage, profile match rate, grid search de pesos del híbrido).

### 3. Ejecutar la aplicación

```bash
streamlit run src/app.py
```

La app también está desplegada en **Streamlit Community Cloud**: *[añadir aquí el enlace de la app desplegada]*.

## Metodología

El proyecto siguió el flujo de trabajo habitual de un proyecto de Data Science: comprensión y exploración de los datos, preprocesamiento, iteración sobre distintos enfoques de modelado, y evaluación cuantitativa antes de llegar a la versión final desplegada.

Un aspecto relevante del proceso fue la **evolución iterativa del modelo de recomendación**:

1. Un primer enfoque basado en similitud del coseno sobre una matriz de features (incluyendo categóricas codificadas con `LabelEncoder`) se descartó al detectar que introducía relaciones artificiales entre categorías sin orden real.
2. Se sustituyó por un modelo híbrido con score de contenido y score colaborativo (basado en salario medio de perfiles similares), pero este último sesgaba las recomendaciones hacia los roles mejor pagados independientemente del ajuste real al perfil.
3. La versión final combina un score de contenido explícito e interpretable (`calcular_afinidad_perfil`) con un score colaborativo rediseñado en torno a la calidad del puesto (satisfacción, crecimiento, equilibrio vida-trabajo, percentil salarial), evitando el sesgo hacia el salario absoluto.

El detalle completo de esta evolución, con la justificación de cada decisión, está documentado en `src/explore.ipynb`.

## Métricas de evaluación

El sistema se evalúa con tres métricas complementarias:

- **Precision@K (leave-one-out):** ¿recupera el sistema el rol real de un perfil cuando se le presenta como usuario nuevo?
- **Catalog coverage:** ¿el sistema recomienda de forma diversa entre todos los roles disponibles, o repite siempre los mismos?
- **Profile match rate:** de las recomendaciones generadas, ¿cuánto respetan las preferencias declaradas del usuario (modalidad, experiencia, especialización, educación)?

Los resultados detallados, junto con el grid search que valida el reparto 80/20 del score híbrido, están en `src/explore.ipynb`.

## Limitaciones y trabajo futuro

- **Integración de ofertas de empleo en tiempo real**: sustituir o complementar el dataset estático actual con conexión a APIs de portales de empleo (LinkedIn, Indeed, InfoJobs, etc.), de forma que las recomendaciones reflejen vacantes activas.
- **Ampliación de la personalización**: incorporar nuevas preferencias del usuario (tamaño de empresa, tolerancia al riesgo de automatización, prioridad salario vs. equilibrio vida-trabajo, sector específico) para un perfil de entrada más rico.
- **Cobertura de perfiles junior y entry-level**: ajustar la lógica de afinidad y los filtros del recomendador para usuarios con muy poca o ninguna experiencia, un segmento que el diseño actual puede no cubrir bien.

## Autoras

- Joseline Proaño
- Valeria Urbina

Proyecto Final — Bootcamp de Data Science y Machine Learning, 4Geeks Academy España.
