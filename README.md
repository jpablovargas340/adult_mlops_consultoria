# 🧠 Adult Income Prediction --- MLOps Project

Proyecto académico de **MLOps Nivel 1** utilizando el dataset **Adult
(UCI ML Repository)** para predecir si una persona gana más de 50K USD
anuales, basado en variables demográficas y laborales.

------------------------------------------------------------------------

## 📊 1. Objetivo del Proyecto

Construir un pipeline reproducible de Machine Learning que:

-   Descargue automáticamente el dataset desde UCI.
-   Valide la calidad de los datos.
-   Realice feature engineering.
-   Entrene un modelo de clasificación.
-   Evalúe su desempeño.
-   Genere artefactos versionados.
-   Permita reproducibilidad total con `dvc repro`.

Este proyecto implementa prácticas de **MLOps Nivel 1**: -
Modularización del código - Versionamiento de datos con DVC - Separación
clara entre datos, artefactos y código - Pipeline automatizado

------------------------------------------------------------------------

## 📁 2. Dataset

Dataset: **Adult Income (UCI ML Repository)**\
Registros: 48,842\
Features: 14 variables\
Clases: `<=50K`, `>50K`

Descarga automática mediante:

``` python
from ucimlrepo import fetch_ucirepo
adult = fetch_ucirepo(id=2)
```

Los datos se almacenan en formato **Parquet** para eficiencia y
reproducibilidad.

------------------------------------------------------------------------

## ⚙️ 3. Pipeline MLOps

Definido en `dvc.yaml`:

    ingest → validate → features → train → evaluate

### 🔹 Ingest

-   Descarga dataset
-   Guarda datos en `data/raw/` en formato Parquet
-   Genera `artifacts/ingest_summary.json`

### 🔹 Validate

-   Verificación de estructura, nulos y rangos
-   Genera `artifacts/validation_report.json`

### 🔹 Features

-   Limpieza de target y valores faltantes
-   Train/Test split estratificado
-   Preprocesamiento:
    -   StandardScaler (numéricas)
    -   OneHotEncoder (categóricas)
-   Serializa `preprocessor.joblib`
-   Guarda datos procesados en Parquet

### 🔹 Train

Modelo utilizado:

**Logistic Regression** - solver: saga - max_iter: 3000

Genera: - `models/model.pkl` - `artifacts/metrics.json` -
`artifacts/classification_report.json`

### 🔹 Evaluate

-   Evaluación en conjunto de test
-   Genera:
    -   `artifacts/eval_metrics.json`
    -   `artifacts/confusion_matrix.png`

------------------------------------------------------------------------

## 📈 4. Resultados del Modelo

  Métrica    Valor
  ---------- --------
  Accuracy   \~85%
  F1 Macro   \~0.78

El modelo logra un buen balance considerando el desbalance natural del
dataset.

------------------------------------------------------------------------

## 🖼️ Matriz de Confusión

La matriz de confusión se genera automáticamente en:

    artifacts/confusion_matrix.png

GitHub renderiza la imagen directamente.

------------------------------------------------------------------------

## 🗂️ 5. Estructura del Proyecto

    adult_mlops_consultoria/
    │
    ├── data/
    │   ├── raw/
    │   └── processed/
    │
    ├── src/
    │   ├── ingest.py
    │   ├── validate.py
    │   ├── features.py
    │   ├── train.py
    │   └── evaluate.py
    │
    ├── artifacts/
    ├── models/
    ├── dvc.yaml
    ├── dvc.lock
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

## 🔁 6. Reproducibilidad

### Crear entorno virtual

``` bash
python -m venv venv
```

Windows:

``` bash
venv\Scripts\activate
```

Instalar dependencias:

``` bash
pip install -r requirements.txt
```

Ejecutar pipeline:

``` bash
dvc repro
```

------------------------------------------------------------------------

## 🧪 7. Versionamiento con DVC

-   Git versiona código y métricas.
-   DVC versiona datos y modelo.
-   `dvc.lock` contiene hashes reproducibles.

Esto permite reproducibilidad total del pipeline.

------------------------------------------------------------------------

## 🛠️ 8. Stack Tecnológico

-   Python
-   pandas
-   scikit-learn
-   matplotlib
-   joblib
-   DVC
-   ucimlrepo

------------------------------------------------------------------------

## 👨‍💻 Autores

Juan Pablo Vargas - Mildreth Diaz\
Universidad Santo Tomás --- Facultad de Estadística\
Proyecto académico MLOps
