# ♻️ RecycleNet

<p align="center">
  <img src="https://images.unsplash.com/photo-1532996122724-e3c354a0b15b?q=80&w=1170&auto=format&fit=crop" width="700" alt="Recycling Bins">
</p>

<h3 align="center">
Sistema Inteligente de Clasificación y Segmentación de Residuos mediante Visión por Computadora y Deep Learning
</h3>

<p align="center">
Proyecto desarrollado para la materia <strong>Redes Neuronales Profundas</strong><br>
Ingeniería en Sistemas de Información - Universidad Tecnológica Nacional, Facultad Regional Mendoza (UTN-FRM)
</p>

---

## 📖 Descripción

RecycleNet es una aplicación de visión por computadora orientada a la clasificación y segmentación automática de residuos urbanos.

El proyecto busca simular el funcionamiento de una línea de clasificación inteligente utilizada en plantas modernas de reciclaje, permitiendo identificar distintos tipos de residuos a partir de imágenes mediante modelos de Deep Learning entrenados con PyTorch.

La solución combina:

- Clasificación multiclase de residuos.
- Segmentación de objetos.
- Fine-Tuning de modelos preentrenados.
- Aplicación web interactiva desarrollada con Streamlit.
- Despliegue en servicios de hosting gratuitos.

<p align="center">
  <img src="https://solutionops.cl/wp-content/uploads/2024/12/red-neuronal2-solutionops.png" width="600" alt="Neural Network">
</p>

## 🛠️ Tecnologías Utilizadas

<p>
   <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
   <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
   <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"/>
   <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
   <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
   <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
   <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white"/>
   <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white"/>
</p>

---

## 👥 Integrantes

### Antonella Aldao

[![GitHub](https://img.shields.io/badge/GitHub-AntoAldao-181717?style=for-the-badge&logo=github)](https://github.com/AntoAldao)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Antonella_Aldao-0A66C2?style=for-the-badge&logo=linkedin)](https://ar.linkedin.com/in/antonellaaldao)

### Ignacio Berridy

[![GitHub](https://img.shields.io/badge/GitHub-NachoBerridy-181717?style=for-the-badge&logo=github)](https://github.com/NachoBerridy)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Ignacio_Berridy-0A66C2?style=for-the-badge&logo=linkedin)](https://ar.linkedin.com/in/ignacioberridy)

### Carlos Gitto

[![GitHub](https://img.shields.io/badge/GitHub-CarlosGitto-181717?style=for-the-badge&logo=github)](https://github.com/CarlosGitto)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Carlos_Gitto-0A66C2?style=for-the-badge&logo=linkedin)](https://ar.linkedin.com/in/carlosgitto)

### Diandra Malca

[![GitHub](https://img.shields.io/badge/GitHub-DiandraMalca-181717?style=for-the-badge&logo=github)](https://github.com/DiandraMalca)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Diandra_Malca-0A66C2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/diandra-malca-8099791b3/)

### Ignacio Ramos

[![GitHub](https://img.shields.io/badge/GitHub-IgnacioGRamos-181717?style=for-the-badge&logo=github)](https://github.com/IgnacioGRamos)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Ignacio_Ramos-0A66C2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/ignacio-ramos-developer/)

---

## ⚡ Inicio Rápido

```bash
git clone https://github.com/<organization>/recyclenet.git

cd recyclenet

python scripts/setup_environment.py

# Activar entorno virtual
source .venv/bin/activate
# o .venv\Scripts\activate en Windows

python scripts/download_dataset.py

jupyter notebook dev/01_dataset_preparation.ipynb
```

---

# 📂 Dataset

## Descripción

RecycleNet utiliza múltiples datasets públicos de imágenes de residuos para las tareas de clasificación y segmentación mediante técnicas de Deep Learning y Visión por Computadora.

Las imágenes **no se almacenan dentro del repositorio** debido a las limitaciones de tamaño de GitHub y para mantener el repositorio liviano y reproducible.

---

## Fuentes de Datos

### Dataset Base de Clasificación

[![Kaggle](https://img.shields.io/badge/Kaggle-Garbage_Dataset_Classification-20BEFF?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/datasets/zlatan599/garbage-dataset-classification)

Dataset utilizado como base para la clasificación multiclase de residuos.

---

### Dataset Complementario

[![Kaggle](https://img.shields.io/badge/Kaggle-Garbage_Classification-20BEFF?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)

Dataset utilizado para ampliar la cantidad de muestras disponibles y mejorar la capacidad de generalización del modelo.

---

## Descarga Automática

Para descargar todos los datasets necesarios ejecutar:

```bash
python scripts/download_dataset.py
```

El script descargará automáticamente los datasets desde Kaggle utilizando la librería `kagglehub`.

---

## Estructura Esperada

Luego de la descarga, la estructura de carpetas debería quedar de la siguiente manera:

```text
data/
│
├── raw/
│   ├── garbage-dataset-classification/
│   └── garbage-classification/
│
├── train.csv
├── val.csv
└── test.csv
```

---

## Particiones del Dataset

Los archivos:

- train.csv
- val.csv
- test.csv

se encuentran versionados dentro del repositorio y contienen:

- Ruta relativa de cada imagen.
- Etiqueta correspondiente.
- Partición asignada.

Esto garantiza que todos los integrantes del equipo trabajen con exactamente la misma división de datos.

---

## Reproducibilidad

Para reconstruir completamente el entorno de trabajo:

1. Clonar el repositorio.
2. Instalar las dependencias definidas en `requirements.txt`.
3. Ejecutar:

```bash
python scripts/download_dataset.py
```

1. Ejecutar el notebook:## Reproducibilidad

Para reconstruir completamente el entorno de trabajo:

1. Clonar el repositorio.
2. Crear el entorno virtual e instalar las dependencias:

```bash
python scripts/setup_environment.py
```

1. Activar el entorno virtual.

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

1. Descargar los datasets:

```bash
python scripts/download_dataset.py
```

1. Ejecutar el notebook de preparación:

```bash
jupyter notebook dev/01_dataset_preparation.ipynb
```

No deberían ser necesarios pasos manuales adicionales.

```bash
jupyter notebook dev/01_dataset_preparation.ipynb
```

No deberían ser necesarios pasos manuales adicionales.

---

## Nota

Este proyecto fue desarrollado para la asignatura **Redes Neuronales Profundas** de la carrera **Ingeniería en Sistemas de Información** de la **Universidad Tecnológica Nacional - Facultad Regional Mendoza (UTN-FRM)**.

---

## 📁 Estructura del Proyecto

```text
recyclenet/
│
├── data/
│   ├── README.md
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
│
├── dev/
│   └── 01_dataset_preparation.ipynb
│
├── prod/
│
├── scripts/
│   ├── setup_environment.py
│   └── download_dataset.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/<organization>/recyclenet.git

cd recyclenet
```

### 2. Crear entorno virtual e instalar dependencias

```bash
python scripts/setup_environment.py
```

### 3. Activar entorno virtual

#### Windows

```bash
.venv\Scripts\activate
```

#### Linux / macOS

```bash
source .venv/bin/activate
```

### 4. Descargar datasets

```bash
python scripts/download_dataset.py
```

### 5. Ejecutar notebook de preparación

```bash
jupyter notebook dev/01_dataset_preparation.ipynb
```

---

## 🎯 Objetivos del Proyecto

- Implementar una solución basada en Deep Learning para clasificación de residuos.
- Incorporar técnicas de segmentación de imágenes.
- Aplicar Fine-Tuning sobre modelos preentrenados.
- Construir una aplicación web funcional para inferencia.
- Desplegar la solución en un entorno accesible públicamente.

---

## 📜 Licencia

Proyecto académico desarrollado con fines educativos para la asignatura Redes Neuronales Profundas de UTN-FRM.
