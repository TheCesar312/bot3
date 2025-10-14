import random

def generar_texto(texto):
    respuestas = [
        "No estoy seguro, pero puedo aprender sobre eso.",
        "Podrías contarme más de eso.",
        "Interesante, no lo había pensado así.",
        "Hmm, no lo sé, pero suena intrigante."
    ]
    return random.choice(respuestas)
