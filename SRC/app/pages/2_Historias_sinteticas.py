import streamlit as st

st.title("Consulta del paciente")

edad = st.number_input("Edad", min_value=0, max_value=125, step=1)
sexo = st.selectbox("Sexo", ["Femenino", "Masculino", "Otro", "Prefiero no decir"])
motivo = st.text_area("Motivo de consulta", height=120, placeholder="Describa brevemente...")

if st.button("Guardar"):
    st.session_state["consulta"] = {
        "edad": int(edad),
        "sexo": sexo,
        "motivo": motivo.strip()
    }
    st.success("Datos guardados")
