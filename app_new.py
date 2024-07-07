import streamlit as st

# Función de Python para generar respuestas
def generar_respuesta(pregunta):
    # Lógica para generar la respuesta basada en la pregunta
    respuestas = {
        "¿Cuál es tu nombre?": "Mi nombre es ChatGPT.",
        "¿Qué es Streamlit?": "Streamlit es una biblioteca de Python para crear aplicaciones web interactivas y de fácil uso.",
        "¿Cómo funciona Python?": "Python es un lenguaje de programación interpretado y de alto nivel, conocido por su legibilidad y versatilidad."
    }
    return respuestas.get(pregunta, "Lo siento, no tengo una respuesta para esa pregunta.")

# Título de la aplicación
st.title("Aplicación de Preguntas y Respuestas")

# Campo de entrada para la pregunta del usuario
pregunta = st.text_input("Haz tu pregunta:")

# Botón para enviar la pregunta
if st.button("Enviar"):
    # Obtener la respuesta generada por la función
    respuesta = generar_respuesta(pregunta)
    
    # Mostrar la respuesta en la aplicación
    st.write("Respuesta:", respuesta)
