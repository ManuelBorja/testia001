# Instlación
## Creación de entorno virtual
El módulo trabaja con la versión de python 3.12.4
Instalar el entorno virtual en esa versión
    python -m venv .venv

Pero si se desea cambiar a otra versión se puede cambiar el parametro :
    python = "^3.12"

en el archivo pyproject.toml

## Activar el entorno virtual
    source .venv/bin/activate

## Instalar Poetry en
pip install poetry

## crear entorno poetry
poetry install

## ejecutar script de consulta
poetry run python consulta.py

## ejecutar web de consulta
poetry run streamlit run page.py

GPT RESEARCHER
