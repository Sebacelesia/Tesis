import re

regex_numeros = r"""
    (?<!\d)
    (?: 
        # ---- CÉDULAS URUGUAY ----
        \d{1,2}\.?\d{3}\.?\d{3}[-/]\d       # 1.234.567-8 / 1234567-8 / 1.234.567/8
        |
        \d{7,8}                             # 7 u 8 dígitos seguidos (cédula sin guion)
        |

        # ---- TELÉFONOS ----
        \+?\d[\d ]{6,}                      # internacionales o secuencias largas con +

        # En fijos y celulares no separamos más
    )
    (?!\d)
"""

# ------------------------------------
# REGEX para DIRECCIÓN (solo si está declarada)
# ------------------------------------

regex_direccion_sin_tilde = r"""
    (?i)                       # case insensitive
    direccion\s*:\s*           # Dirección: (con o sin espacio)
   (.+)                        # capturar lo que sigue
"""
regex_direccion_con_tilde = r"""
    (?i)                       # case insensitive
    dirección\s*:\s*           # Dirección: (con o sin espacio)
   (.+)                        # capturar lo que sigue
"""

def extraer_datos(texto):
    """
    Extrae el nombre desde una línea que comienza con 'Nombre:'.
    Retorna una lista con cada palabra del nombre por separado.
    Maneja formatos:
      - 'Nombre: Juan Perez Silva'
      - 'Nombre: Perez Silva, Juan'
    """
    # 1) Números (teléfonos + cédulas + internacionales)
    numeros = re.findall(regex_numeros, texto, flags=re.VERBOSE)
    numeros = [n.strip() for n in numeros]
    numeros = list(dict.fromkeys(numeros))   # quitar duplicados

    # 2) Dirección SOLO si aparece como "Direccion:"
    
    match_dir = re.search(regex_direccion_con_tilde, texto, flags=re.VERBOSE)
    direccion = match_dir.group(1).strip() if match_dir else None

    if direccion == None:
        match_dir = re.search(regex_direccion_sin_tilde, texto, flags=re.VERBOSE)
        direccion = match_dir.group(1).strip() if match_dir else None
    

    # Buscar la línea que contiene "Nombre:"
    match = re.search(r'Nombre:\s*(.+)', texto, flags=re.IGNORECASE)
    if not match:
        return []  # No encontró el campo

    nombre_raw = match.group(1).strip()

    # Normalizar espacios y quitar signos que estorban
    nombre_raw = re.sub(r'[\.,]', ' ', nombre_raw)  # reemplazar comas y puntos por espacios
    nombre_raw = re.sub(r'\s+', ' ', nombre_raw)    # colapsar espacios múltiples

    # Caso especial: "Apellido(s) ..., Nombre"
    # Si había una coma originalmente, la orden era AP → NOMBRE
    if ',' in match.group(1):
        partes = [p.strip() for p in match.group(1).replace('.', '').split(',')]
        # partes[0] = apellidos
        # partes[1] = nombres
        nombre_raw = partes[1] + " " + partes[0]

    # Separar por espacios
    palabras = [p for p in nombre_raw.split() if p]
    nombre_formateado = ", ".join(palabras)
            #    -> "Juan, Perez, Herrera"

    elementos = [
        nombre_formateado,  # "Juan, Perez, Herrera"
        *numeros,           # "12345678", "+598 99010203"
        direccion           # "Av. Italia 3333"
        ]
    resultado = '"' + ", ".join(elementos) + '"'

    return resultado


# ===== Ejemplo =====
texto = """
Nombre: Perez Herrera, Juan
C.I.: 12345678
Telefono: +598 99010203
Edad: 56 años
Sexo: Masculino
Estado civil: Casado
Ocupación: Administrativo
Direccion: Av. Italia 3333

Motivo de consulta: “Dolor opresivo en el pecho desde hace 2 horas”.

Enfermedad actual: Juan Perez de 56 años que refiere dolor retroesternal de inicio súbito mientras caminaba al trabajo. Describe el dolor como opresivo, de intensidad 8/10, con irradiación al brazo izquierdo y mandíbula. Se asocia a disnea, sudoración fría y náuseas. No cedió con reposo. Niega fiebre, tos o expectoración.

Antecedentes personales:

•⁠  ⁠Hipertensión arterial diagnosticada hace 10 años, en tratamiento irregular con enalapril.
•⁠  ⁠Dislipemia conocida, no en tratamiento.
•⁠  ⁠Tabaquismo: 20 cigarrillos/día desde los 20 años.
•⁠  ⁠No diabetes conocida.
•⁠  ⁠No alergias medicamentosas conocidas.

Antecedentes familiares:

•⁠  ⁠Padre (Arturo Perez) fallecido a los 62 años por infarto agudo de miocardio.
•⁠  ⁠Madre (Raquel Cardona) con hipertensión arterial.

Examen físico:

•⁠  ⁠Estado general: sudoroso, ansioso.
•⁠  ⁠TA: 150/95 mmHg
•⁠  ⁠FC: 105 lpm
•⁠  ⁠FR: 22 rpm
•⁠  ⁠Temp: 36,8 °C
•⁠  ⁠Saturación oxigeno: 92% ambiente
•⁠  ⁠Cardiovascular: ruidos rítmicos, taquicardia, sin soplos audibles.
•⁠  ⁠Pulmones: murmullo vesicular conservado, sin ruidos agregados.
•⁠  ⁠Abdomen: blando, depresible, sin dolor.
•⁠  ⁠Extremidades: sin edemas.
"""

print(extraer_datos(texto))




