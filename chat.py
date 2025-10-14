import time
import random
import json
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model("model/chatbot_model2.h5")
tokenizer = pickle.load(open("model/tokenizer2.pkl", "rb"))
label_encoder = pickle.load(open("model/label_encoder2.pkl", "rb"))

with open("data/intents.json", "r", encoding="utf-8") as f:
    intents = json.load(f)

contexto_actual = None

def responder(texto):
    global contexto_actual
    texto = texto.lower().strip()
    seq = tokenizer.texts_to_sequences([texto])
    padded = pad_sequences(seq, maxlen=model.input_shape[1], padding="post")
    pred = model.predict(padded)[0]
    tag = label_encoder.inverse_transform([np.argmax(pred)])[0]

    if np.max(pred) < 0.3:
        from generator import generar_texto
        return [generar_texto(texto)]

    for intent in intents["intents"]:

        if "context_filter" in intent:
            if intent["context_filter"] != contexto_actual:
                continue

        if intent["tag"] == tag:
            if "context_set" in intent:
                contexto_actual = intent["context_set"]
            else:
                contexto_actual = None
            respuesta = random.choice(intent["responses"])
            # Si es string, conviértelo en lista
            if isinstance(respuesta, str):
                respuesta = [respuesta]
            return respuesta

    from generator import generar_texto
    return [generar_texto(texto)]

while True:
    msg = input("Tú: ")
    if msg.lower() in ["salir", "exit"]:
        break

    respuestas = responder(msg)
    for r in respuestas:
        print("Bot:", r)
        time.sleep(random.uniform(1.2, 2.5))  # pausa natural
