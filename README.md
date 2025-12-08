# Herramienta de anonimización de historias clínicas en PDF

Aplicación en **Streamlit** que anonimiza historias clínicas en formato PDF utilizando **modelos Qwen3** servidos a través de **Ollama**.  

El sistema:

- Extrae datos identificatorios del encabezado del documento (nombre, documento y dirección).
- Censura esos datos en el resto del texto.
- Anonimiza nombres/apellidos de profesionales (por ejemplo, en “Responsables del registro”).
- Detecta menciones de carga viral y redondea los valores numéricos según reglas predefinidas.
- Genera un **PDF anonimizado** a partir de las secciones del documento en un rango de fechas seleccionado.

---

## 1. Requisitos previos

- **Python** 3.10+ (recomendado 3.11).
- **Ollama** instalado y funcionando en `http://localhost:11434`.
- Sistema operativo con capacidad para ejecutar los modelos Qwen3 (idealmente con GPU).

### Modelos de lenguaje

La app está pensada para trabajar con estos modelos de Ollama:

- `qwen3:8b`
- `qwen3:4b`

Debés tenerlos descargados en tu máquina.

---

## 2. Instalación

### 2.1. Crear y activar entorno virtual

```bash
# Crear entorno virtual (Windows)
python -m venv .venv

# Activar entorno
.venv\Scripts\activate
