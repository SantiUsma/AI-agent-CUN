# Instrucciones

## 0. Requirements
Se requiere Python 3.13 y Ollama descargado.

## 1. Crear el entorno virtual
En la carpeta de tu proyecto, ejecuta:

```bash
python -m venv .venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. Descargar el modelo de lenguaje
Abre la terminal una vez tengas instalado Ollama y corre el siguiente comando:

```bash
ollama pull llama3.1
```

## 3. Cargar documentos
Para cargar documentos debes correr el codigo load_docs.py usando el siguiente comando:

```bash
python load_docs.py -name name
```

Donde "name" es el path al documento que deseas almacenar en la base de datos Chroma.

## 4. Ejecutar el agente
Para ejecutar la aplicación, se debe ejecutar el siguiente comando:

```bash
streamlit run app.py
```

Esto abrirá una pestaña del buscador con un chat con el cual podrás interactuar con el agente IA.
