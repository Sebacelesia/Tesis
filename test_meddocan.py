from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# 1. Nombre del modelo en Hugging Face
MODEL_NAME = "IIC/roberta-large-bne-meddocan"

# 2. Texto de prueba (tu historia clínica)
clinical_text = """
Nombre: 
Juan Pérez
C.I.: 1234567-8
Telefono: 099010203
Edad: 56 años
Sexo: Masculino
Estado civil: Casado
Ocupación: Administrativo

Motivo de consulta: “Dolor opresivo en el pecho desde hace 2 horas”.

Enfermedad actual: Juan Perez de 56 años que refiere dolor retroesternal de inicio súbito mientras caminaba al trabajo. Describe el dolor como opresivo, de intensidad 8/10, con irradiación al brazo izquierdo y mandíbula. Se asocia a disnea, sudoración fría y náuseas. No cedió con reposo. Niega fiebre, tos o expectoración.

Antecedentes familiares:

- Padre (Arturo Perez) fallecido a los 62 años por infarto agudo de miocardio.
- Madre (Raquel Cardona) con hipertensión arterial.

"""

def main():
    print("Cargando modelo y tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)

    # 3. Creamos un pipeline de NER (token classification)
    ner_pipeline = pipeline(
        task="token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",  # une sub-tokens en entidades completas
    )

    print("\nEjecutando NER sobre la historia clínica...\n")
    entities = ner_pipeline(clinical_text)

    # 4. Imprimir resultados de forma prolija
    print("Entidades detectadas:\n")
    for ent in entities:
        word = ent["word"]
        label = ent["entity_group"]
        score = ent["score"]
        start = ent["start"]
        end = ent["end"]
        print(f"- {word!r:25}  -> {label:10}  (score={score:.3f}, span={start}-{end})")

    # Si querés ver solo el texto con resaltado muy simple:
    print("\nTexto con etiquetas inline:\n")
    marked = list(clinical_text)
    # marcamos de atrás hacia adelante para no mover índices
    for ent in sorted(entities, key=lambda e: e["start"], reverse=True):
        start, end, label = ent["start"], ent["end"], ent["entity_group"]
        span = clinical_text[start:end]
        marked[start:end] = list(f"[{label}:{span}]")
    print("".join(marked))


if __name__ == "__main__":
    main()
