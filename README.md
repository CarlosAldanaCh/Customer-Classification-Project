# Customer Classification Project – (Machine Learning)

## Project Description

The goal of this project is to build and compare different machine learning models to predict whether a customer belongs to the **Ultra tariff plan** based on their behavior (calls, minutes, messages, and internet data usage).

We tested three classification models:

- Decision Tree
- Random Forest
- Logistic Regression

The best model was selected according to accuracy on the test set.

---

## Workflow

1. Import libraries (`pandas`, `scikit-learn`, etc.).
2. Load and explore the dataset (basic EDA).
3. Define features (X) and target (y).
4. Split the data into training and test sets.
5. Train and tune models using **GridSearchCV** with cross-validation.
6. Evaluate the models on the test set.
7. Compare against a baseline (Dummy Classifier).
8. Select the best model and save it with **joblib**.

---

## Results

- Baseline (Dummy Classifier): **69.5% accuracy**
- Decision Tree: lower performance, sensitive to depth.
- Logistic Regression: fast and interpretable, but less accurate.
- **Random Forest: 79.5% accuracy (best model)** ✅

The Random Forest clearly outperforms the baseline, proving it captures useful patterns in the data.

---

## Repository Structure

```markdown
data/ # dataset
notebook/ # Jupyter notebook with analysis
models/ # saved best model
README.md # project documentation
requirements.txt # required libraries
```

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/CarlosAldanaCh/Customer-Classification-Project.git
cd Customer-Classification-Project
pip install -r requirements.txt
```

---

## How to Run

1. Open the notebook:

   ```bash
   jupyter notebook notebook/telecom-customer-classification.ipynb
   ```

2. Run all cells.
3. The best model (`RandomForestClassifier`) will be saved in `models/`.

---

## Technologies Used

- Python 3.13+
- Pandas
- Scikit-learn
- Joblib
- Jupyter Notebook

---

## License

This project is for educational purposes (TripleTen Bootcamp).

---

---

# Proyecto de Clasificación de Clientes – (Machine Learning)

## Descripción del Proyecto

El objetivo de este proyecto es construir y comparar distintos modelos de **machine learning** para predecir si un cliente pertenece a la **tarifa Ultra** en función de su comportamiento (llamadas, minutos, mensajes y uso de datos de internet).

Se probaron tres modelos de clasificación:

- Árbol de decisión
- Bosque aleatorio
- Regresión logística

El mejor modelo fue seleccionado según su exactitud en el conjunto de prueba.

---

## Flujo de Trabajo

1. Importación de librerías (`pandas`, `scikit-learn`, etc.).
2. Carga y exploración del dataset (EDA básico).
3. Definición de variables: features (X) y target (y).
4. División de los datos en entrenamiento y prueba.
5. Entrenamiento y ajuste de modelos con **GridSearchCV** y validación cruzada.
6. Evaluación en el conjunto de prueba.
7. Comparación con un baseline (Dummy Classifier).
8. Selección del mejor modelo y guardado con **joblib**.

---

## Resultados

- Baseline (Dummy Classifier): **69.5% accuracy**
- Árbol de decisión: menor rendimiento, sensible a la profundidad.
- Regresión logística: rápida e interpretable, pero menos precisa.
- **Bosque aleatorio: 79.5% accuracy (mejor modelo)** ✅

El Bosque Aleatorio supera claramente al baseline, demostrando que captura patrones útiles en los datos.

---

## Estructura del Repositorio

```markdown
data/ # dataset
notebook/ # notebook de Jupyter con el análisis
models/ # modelo guardado
README.md # documentación del proyecto
requirements.txt # librerías necesarias
```

---

## Instalación

Clonar el repositorio e instalar dependencias:

```bash
git clone https://github.com/CarlosAldanaCh/Customer-Classification-Project.git
cd Customer-Classification-Project
pip install -r requirements.txt
```

---

## Cómo Ejecutar

1. Abrir el notebook:

   ```bash
   jupyter notebook notebook/telecom-customer-classification.ipynb
   ```

2. Ejecutar todas las celdas.
3. El mejor modelo (`RandomForestClassifier`) se guarda en `models/`.

---

## Tecnologías Usadas

- Python 3.13+
- Pandas
- Scikit-learn
- Joblib
- Jupyter Notebook

---

## Licencia

Proyecto con fines educativos (Bootcamp TripleTen).

---
