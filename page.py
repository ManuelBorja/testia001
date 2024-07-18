import streamlit as st
from applocal import descarga_documentos, generar_respuesta

# Título de la aplicación
st.title("Consultas Tribunal Constitucional")

# Botón para enviar la pregunta
if st.button("Cargar"):
    descarga_documentos()

# Campo de entrada para la pregunta del usuario
pregunta = st.text_input("Haz tu pregunta:")

# Botón para enviar la pregunta
if st.button("Enviar"):
    # Obtener la respuesta generada por la función
    respuesta = generar_respuesta(pregunta)
    # Mostrar la respuesta en la aplicación
    st.write("Respuesta:", respuesta)