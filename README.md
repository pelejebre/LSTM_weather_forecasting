# **📈 Predicción temporal con LSTM 🧠**

## 📌 Descripción del Proyecto

Este proyecto se centra en la construcción y entrenamiento de un modelo LSTM para la predicción meteorológica.  

Las redes LSTM (Long Short-Term Memory) son un tipo de red neuronal recurrente (RNN) diseñadas para capturar dependencias a largo plazo en secuencias de datos. Gracias a su arquitectura basada en celdas de memoria y puertas de control, pueden retener información relevante durante periodos prolongados, evitando problemas como el desvanecimiento del gradiente. Esto las hace especialmente útiles para tareas de forecasting, ya que permiten modelar patrones temporales complejos en datos históricos y mejorar la precisión de las predicciones.

El conjunto de datos proviene del Instituto Planck para Biogeoquímica e incluye variables climáticas (temperatura, presión, humedad, etc.).  

## 🔬 Motivación y Contexto

El pronóstico meteorológico es una tarea crítica en muchos campos, desde la agricultura hasta la gestión de desastres.

El uso de redes LSTM (Long Short-Term Memory) permite capturar las dependencias temporales de las series históricas, facilitando predicciones precisas con datos de series temporales.

Este proyecto ayuda a:
- Comprender el funcionamiento de un modelo LSTM aplicado a series temporales.
- Experimentar con la partición de datos, generación de secuencias y escalamiento adecuado para datos meteorológicos.
- Monitorear y evaluar el rendimiento del modelo utilizando herramientas como TensorBoard.

## 📁 Estructura del Proyecto

El proyecto está organizado en la siguiente estructura de carpetas y archivos:

├── requirements.txt

├── helpers/

│   ├── __init__.py

│   └── helpers_module.py

├── 01_Procesamiento_datos.ipynb

└── 02_LSTM_univariado_unistep.ipynb

### Descripción de Archivos y Carpetas

1. **`helpers/`**

   • Contiene funciones auxiliares utilizadas en el proyecto.   

2. **`requirements.txt`**

   • Archivo que contiene todas las librerías necesarias para ejecutar el proyecto.

   • Para instalar todas las dependencias indicadas en el archivo `requirements.txt`, abre una terminal en el directorio raíz del proyecto y ejecuta el siguiente comando:
```bash
pip install -r requirements.txt
```

3. **`01.Procesamiento_datos.ipynb`**

   • Se encarga de preparar los datos brutos para la predicción meteorológica utilizando un modelo LSTM. Generando un archivo de salida de datos preporcesados.  

4. **`02.LSTM_univariado_unistep.ipynb`**
   
   • Implementa un modelo LSTM para una predicción univariable (Temperatura) con predicción unipaso (1 hora después)


### 📊 Datos Utilizados

- **Fuente:** [Instituto Planck para Biogeoquímica](https://www.kaggle.com/datasets/arashnic/max-planck-weather-dataset)


### 🚀 Requisitos e Instalación
Clonar el Repositorio
```
bash
git clone https://github.com/tu-usuario/forecasting-lstm.git
cd forecasting-lstm
```

Instalar Dependencias
```
pip install -r requirements.txt
```
Ejecutar el Notebook oportuno
```
jupyter notebook ______.ipynb
```

### 🤝 Contribuciones
Las contribuciones son bienvenidas. Si deseas colaborar:

1️⃣ Haz un fork del repositorio.

2️⃣ Crea una rama con tus mejoras o correcciones.

3️⃣ Envía un pull request para revisión.

### 📄 Referencias
📚 Hochreiter, S., & Schmidhuber, J. (1997). [Long Short-Term Memory](https://www.researchgate.net/publication/13853244_Long_Short-Term_Memory).

📊 Instituto Planck para Biogeoquímica. [Max Planck Weather Dataset](https://www.kaggle.com/datasets/arashnic/max-planck-weather-dataset)