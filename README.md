# Sistema de Anonimización y Generación Sintética de Historias Clínicas

Proyecto de tesis que combina dos herramientas complementarias para el procesamiento de historias clínicas de infectología utilizando modelos de lenguaje grandes (LLMs y SLMs):

1. ** Anonimizador**: Censura información sensible de historias clínicas en PDF
2. ** Generador Sintético**: Crea historias clínicas sintéticas realistas para investigación y entrenamiento

> [!IMPORTANT]
> Ambas herramientas están diseñadas específicamente para **historias clínicas de infectología** del sistema de salud uruguayo, imitando el estilo telegráfico y las abreviaciones locales del corpus hospitalario real.


---



###  Soporte Multi-Modelo

- **Ollama** (local): Qwen 2.5 (3B, 7B) para máxima privacidad
- **Google Gemini** (API): para generación sintética de alta calidad
- **OpenRouter** (API): para selección semántica de casos (One-Shot Selection)

---

##  Arquitectura del Proyecto

```
Tesis/
├── src/
│   ├── app/                                    # Aplicaciones Streamlit
│   │   ├── Anonimizador_de_historias_clínicas.py    # App principal de anonimización
│   │   └── pages/
│   │       └── Historias_sintéticas.py              # Generador de datos sintéticos
│   │
│   ├── llm/                                    # Clientes de modelos LLM
│   │   ├── ollama_client.py                        # Cliente Ollama (local)
│   │   ├── gemini_client.py                        # Cliente Google Gemini
│   │   ├── openrouter_client.py                    # Cliente OpenRouter
│   │   ├── model_config.py                         # Configuración de modelos
│   │   └── prompts.py                              # Plantillas de prompts
│   │
│   ├── services/                               # Lógica de negocio
│   │   ├── pipeline.py                             # Pipeline de anonimización
│   │   ├── patient_extraction.py                   # Extracción de datos del paciente
│   │   ├── fewshot_service.py                      # Selección semántica de casos (OSS)
│   │   ├── synthetic_data.py                       # Carga de casos desde JSON
│   │   └── synthetic_generator.py                  # Generación de historias sintéticas
│   │
│   └── utils/                                  # Utilidades
│       ├── pdf_utils.py                            # Lectura y procesamiento de PDFs
│       ├── pdf_export.py                           # Generación y merge de PDFs
│       ├── section_utils.py                        # Manejo de secciones y fechas
│       └── text_utils.py                           # Utilidades de texto
│
├── data_partesintetica/                        # Datos sintéticos
│   └── casos.json                                  # 20 casos reales para few-shot
│
├── .venv/                                      # Entorno virtual Python
├── requirements.txt                            # Dependencias
└── README.md
```

###  Pipeline de Anonimización

```mermaid
graph TD
    A[PDF Original] --> B[Extracción de Texto]
    B --> C[Segmentación por Secciones]
    C --> D[Mapeo de Fechas]
    D --> E[Filtrado por Rango]
    E --> F{Por cada bloque}
    F --> G[Prompt 1: Extracción de Datos Paciente]
    G --> H[Prompt 2: Censura de Datos]
    H --> I[Prompt 3: Detección Carga Viral]
    I --> J[Prompt 5: Anonimización Profesionales]
    J --> K[Generación PDF Bloque]
    K --> L{¿Más bloques?}
    L -->|Sí| F
    L -->|No| M[Merge de PDFs]
    M --> N[PDF Anonimizado Final]
```

###  Pipeline de Generación Sintética

```mermaid
graph TD
    A[Parámetros Usuario] --> B{Decidir Patología}
    B --> C[20 Casos del Corpus]
    C --> D[OSS: Seleccionar 8 Casos Similares]
    D --> E[Elegir 3 Random]
    E --> F[Generar Few-Shot]
    F --> G[Construir Prompt Sistema]
    G --> H[Generar con Gemini]
    H --> I[Historia Sintética]
    I --> J[Validación Clínica]
    J --> K[Exportar PDF]
```

---

##  Instalación

### Requisitos Previos

- **Python 3.9** (recomendado)
- **Ollama** (opcional): para usar modelos locales - [ollama.ai](https://ollama.ai)
- **API Keys**:
  - **Google Gemini**: para generación sintética - [AI Studio](https://makersuite.google.com/app/apikey)
  - **OpenRouter**: para selección semántica de casos - [openrouter.ai](https://openrouter.ai)

### Instalación Paso a Paso

#### 1. Clonar el repositorio

```bash
git clone https://github.com/tuusuario/Tesis.git
cd Tesis
```

#### 2. Crear y activar entorno virtual

**Windows:**
```bash
# Crear entorno virtual con Python 3.9
py -3.9 -m venv .venv

# Activar entorno
.venv\Scripts\activate
```

**Linux/macOS:**
```bash
# Crear entorno virtual
python3.9 -m venv .venv

# Activar entorno
source .venv/bin/activate
```

#### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

#### 4. Configurar modelos de lenguaje

##### Ollama (local, recomendado para anonimización)

1. Instalar Ollama desde [ollama.ai](https://ollama.ai)

2. Descargar los modelos Qwen:

```bash
ollama pull qwen2.5:7b
ollama pull qwen2.5:3b
```

3. Verificar instalación:

```bash
ollama list
```

4. Asegurar que Ollama esté ejecutándose:

```bash
ollama serve
```

##  Uso

### 1. Anonimizador de Historias Clínicas

#### Ejecutar la aplicación

```bash
streamlit run src/app/Anonimizador_de_historias_clínicas.py
```

La aplicación se abrirá en `http://localhost:8501`.

#### Flujo de trabajo

1. **Seleccionar modelo LLM**: 
   - Qwen 8B (Ollama) - recomendado
   - Qwen 4B (Ollama) - más rápido, menor precisión

2. **Cargar PDF**: 
   - Subir la historia clínica a anonimizar
   - El sistema extraerá automáticamente el texto y detectará secciones

3. **Vista previa**:
   - Revisar las primeras líneas del documento
   - Verificar el rango de fechas detectado
   - Ver cantidad de secciones identificadas

4. **Configurar rango de fechas**:
   - Fecha inicial: primera evolución a incluir
   - Fecha final: última evolución a incluir
   - Solo se procesarán secciones dentro de este rango

5. **Procesar**:
   - Click en "Procesar PDF"
   - El sistema ejecutará:
     - **Prompt 1**: Extracción de datos del paciente (nombre, documento, dirección)
     - **Prompt 2**: Censura de datos identificatorios
     - **Prompt 3**: Detección y redondeo de carga viral
     - **Prompt 5**: Anonimización de profesionales
   - Barra de progreso mostrará avance por bloques

6. **Descargar PDF anonimizado**:
   - Click en "Descargar PDF anonimizado"
   - El archivo estará listo



### 2. Generador de Historias Sintéticas

#### Ejecutar la aplicación

```bash
streamlit run src/app/pages/Historias_sintéticas.py
```

La aplicación se abrirá en `http://localhost:8501`.

> [!NOTE]
> Asegurate de haber configurado las API keys de **Gemini** y **OpenRouter** antes de usar esta herramienta.

#### Modos de generación

##### Modo 1: Manual (Control Total)

Ideal para generar casos específicos con parámetros exactos.

**Parámetros**:
- Edad: 0-120 años
- Sexo: Femenino / Masculino / Otro
- Patología: elegir entre las 13 patologías disponibles
- Motivo: descripción libre del contexto clínico

**Patologías disponibles**:
- VIH
- Tuberculosis pulmonar
- Tuberculosis ganglionar
- Meningitis bacteriana
- Endocarditis
- Infección de prótesis osteoarticular
- Celulitis
- Fiebre prolongada
- Neumonía adquirida en la comunidad
- Infección urinaria
- Candidemia
- Infección de sitio quirúrgico
- Sepsis de origen desconocido



#####  Modo 2: Patología y Motivo Libres

El modelo decide edad y sexo, pero respeta el contexto que proveas.

**Parámetros**:
- Motivo: descripción de la situación clínica



#####  Modo 3: Totalmente Libre

El modelo genera un caso completo de infectología de forma autónoma.

**Sin parámetros**: solo hacer click en "Generar historia clínica".

#### Pipeline de generación

1. **Selección Semántica de Casos (OSS)**:
   - OpenRouter analiza los 20 casos del corpus (`casos.json`)
   - Selecciona los 8 casos más similares clínicamente a la patología objetivo
   - No busca coincidencias de palabras, sino similitud fisiopatológica real

2. **Construcción del Few-Shot**:
   - De los 8 casos, elige 3 aleatoriamente
   - Genera un prompt con estos casos como ejemplos

3. **Generación con Gemini**:
   - Usa el prompt del sistema (estilo telegráfico uruguayo)
   - Incorpora los few-shots seleccionados
   - Genera la historia clínica sintética

4. **Validación**:
   - Verifica que el caso sea de infectología
   - Rechaza patologías fuera del ámbito (fracturas, migraña, etc.)

#### Resultado

La aplicación mostrará:

1. **Historia clínica generada**: en formato texto, lista para revisar
2. **Few-shots utilizados**: los 3 casos del corpus que sirvieron como referencia
3. **Botones de descarga**:
   - Descargar HISTORIA final en PDF
   - Descargar FEW-SHOT en PDF


## 🔧 Configuración

### Modelos disponibles

Editar `src/llm/model_config.py` para:

- **Cambiar modelos por defecto**:
```python
MODEL_CONFIGS = {
    "Qwen 8B (Ollama)": {
        "model_name": "qwen2.5:7b",  # cambiar aquí
        "temperature": 0.0,
        # ...
    }
}
```

- **Ajustar parámetros**:
  - `temperature`: creatividad del modelo (0.0 = determinista)
  - `num_ctx`: tamaño del contexto
  - `num_predict`: tokens máximos a generar
  - `top_p`, `top_k`: muestreo
  - `repeat_penalty`: penalización de repeticiones

- **Configurar procesamiento**:
```python
SECTIONS_PER_BLOCK = 1  # secciones por bloque
USE_CHUNKING = True     # dividir en chunks
MAX_CHARS_PER_CHUNK = 15000
```

### Personalización de prompts

Editar `src/llm/prompts.py` para modificar:

- **Prompt 2**: Censura de datos del paciente
- **Prompt 3**: Detección de carga viral y reglas de redondeo
- **Prompt 5**: Anonimización de profesionales

Editar `src/services/synthetic_generator.py` para ajustar:

- **System prompt**: estilo de escritura, abreviaciones, estructura de secciones
- **Patologías válidas**: lista de patologías de infectología permitidas

### Configuración de casos sintéticos

Editar `data_partesintetica/casos.json` para:

- Agregar nuevos casos de referencia
- Modificar casos existentes
- El formato es un array de strings, cada string es un caso completo

---


### Recomendaciones de seguridad

#### Para Anonimización

1. **Usar modelos locales (Ollama)**:
   - Los datos nunca salen de tu máquina
   - No hay riesgo de filtración a terceros
   - Cumplimiento total con normativas de privacidad

2. **Revisar PDFs anonimizados**:
   - Siempre verificar el resultado antes de compartir
   - El sistema puede no detectar el 100% de los datos sensibles
   - Buscar manualmente menciones del nombre en contextos inusuales

3. **No usar APIs públicas con datos reales**:
   - Gemini y OpenRouter envían datos a servidores externos
   - Solo usar para pruebas con datos ficticios
   - Para producción en hospitales: **solo Ollama local**

#### Para Generación Sintética

1. **Datos sintéticos no son datos reales**:
   - Útiles para investigación, entrenamiento y pruebas
   - No reemplazan historias clínicas reales para diagnóstico
   - Pueden contener inconsistencias clínicas menores

2. **API Keys**:
   - No compartir tus claves de Gemini/OpenRouter
   - Usar variables de entorno




---

## 📦 Dependencias Principales

```txt
streamlit>=1.37.0        # Framework de interfaz web
requests>=2.32.0         # Cliente HTTP para APIs
pymupdf>=1.23.0          # Procesamiento y generación de PDFs
google-generativeai      # Cliente de Google Gemini
```


---






