import streamlit as st
from pathlib import Path
import sys
import requests

from services.synthetic_data import load_casos
from services.synthetic_generator import generar_historia_sintetica
from utils.pdf_export import text_to_pdf_bytes

SRC_PATH = Path(__file__).resolve().parents[1] 
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))




OPENROUTER_KEY = "API KEY"
GEMINI_API_KEY = "API KEY"



CASOS = load_casos()


st.title("Generador de Historia Clinica Sintetica – Infectologia")

modo = st.radio(
    "Modo de generacion:",
    [
        "Manual (edad, sexo, patologia, motivo)",
        "Patologia y motivo libres (infecc.)",
        "Totalmente libre",
    ]
)

patologia_usuario = None
motivo = None

if modo.startswith("Manual"):
    edad = st.number_input("Edad", min_value=0, max_value=120)
    sexo = st.selectbox("Sexo", ["Femenino", "Masculino", "Otro"])
    patologia_usuario = st.text_input("Patologia")
    motivo = st.text_area(
        "Datos iniciales (comentarios libres: procedencia, motivo, convivencia, nacionalidad, etc.)",
        placeholder="Ej: Venezolano, radicado en Uruguay. Vive con pareja. Consulta por tos productiva."
    )

elif modo.startswith("Patologia"):
    edad = None
    sexo = None
    patologia_usuario = None
    motivo = st.text_area(
        "Datos iniciales (comentarios libres: procedencia, motivo, convivencia, nacionalidad, etc.)",
        placeholder="Ej: Venezolano, radicado en Uruguay. Vive con pareja. Consulta por tos productiva."
    )

else:
    edad = None
    sexo = None
    patologia_usuario = None
    motivo = None
    st.write("(Modo totalmente libre: no se ingresa edad, sexo ni motivo.)")


if st.button("Generar historia clinica"):

    # Validacion simple antes de llamar APIs
    if OPENROUTER_KEY.strip() in {"", "API KEY"} or GEMINI_API_KEY.strip() in {"", "API KEY"}:
        st.info(
            "Faltan API keys. Agrega tus claves de OpenRouter y Gemini en el codigo "
            "para poder generar historias."
        )
        st.stop()

    with st.spinner("Generando historia clinica..."):
        try:
            historia, few_shot, patologia_final = generar_historia_sintetica(
                modo=modo,
                edad=edad,
                sexo=sexo,
                patologia_usuario=patologia_usuario,
                motivo=motivo,
                casos=CASOS,
                openrouter_key=OPENROUTER_KEY,
                gemini_key=GEMINI_API_KEY,
            )

            
            st.session_state["historia_generada"] = historia
            st.session_state["few_shot_generado"] = few_shot
            st.session_state["patologia_final"] = patologia_final

        except requests.exceptions.HTTPError as e:
            status = getattr(e.response, "status_code", None)

            if status == 401:
                st.warning("No se pudo acceder a OpenRouter.")
                st.info(
                    "Tu API key es invalida, no esta habilitada o no tiene permisos para el modelo. "
                    "Revisa tu clave de OpenRouter."
                )
                st.stop()

            st.warning("Hubo un problema al comunicarse con el servicio.")
            st.info("Verifica tus credenciales, el modelo configurado y tu conexion.")
            st.stop()

        except requests.exceptions.RequestException:
            st.warning("Problema de red o timeout.")
            st.info("No se pudo completar la llamada al modelo. Intenta nuevamente.")
            st.stop()

        except ValueError as e:
      
            st.info(str(e))
            st.stop()

        except Exception:
            st.warning("Ocurrio un error inesperado.")
            st.info(
                "Revisa las claves, el archivo casos.json y la configuracion del proyecto."
            )
            st.stop()



historia_ss = st.session_state.get("historia_generada", "")
fewshot_ss = st.session_state.get("few_shot_generado", "")
pat_ss = st.session_state.get("patologia_final", "infectologia")

if historia_ss or fewshot_ss:
    st.text_area("Historia clinica generada:", historia_ss, height=420)
    st.text_area("Few shots tenidos en cuenta:", fewshot_ss, height=420)

    st.subheader("Descargas")

    safe_pat = str(pat_ss).replace(" ", "_").replace("/", "_")

    pdf_historia = text_to_pdf_bytes(historia_ss if historia_ss else "(vacio)")
    st.download_button(
        "Descargar HISTORIA final en PDF",
        data=pdf_historia,
        file_name=f"historia_sintetica_{safe_pat}.pdf",
        mime="application/pdf",
    )

    pdf_fewshot = text_to_pdf_bytes(fewshot_ss if fewshot_ss else "(vacio)")
    st.download_button(
        "Descargar FEW-SHOT en PDF",
        data=pdf_fewshot,
        file_name=f"fewshot_{safe_pat}.pdf",
        mime="application/pdf",
    )
else:
    st.caption("Aun no hay resultados para mostrar.")

