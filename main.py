import json
import random
import time
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.metrics import dp
from kivy.graphics import Color, RoundedRectangle

# --- Config visual global ---
Window.clearcolor = (0.03, 0.03, 0.05, 1)

# --- Cargar modelo y datos ---
model = load_model("chatbot_model2.h5")
tokenizer = pickle.load(open("tokenizer2.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder2.pkl", "rb"))
with open("intents.json", "r", encoding="utf-8") as f:
    intents = json.load(f)

contexto_actual = None


# --- Función responder ---
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
        if "context_filter" in intent and intent["context_filter"] != contexto_actual:
            continue
        if intent["tag"] == tag:
            contexto_actual = intent.get("context_set")
            respuesta = random.choice(intent["responses"])
            if isinstance(respuesta, str):
                respuesta = [respuesta]
            return respuesta

    from generator import generar_texto
    return [generar_texto(texto)]


# --- Chat UI ---
class ChatLayout(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation="vertical", spacing=5, padding=8, **kwargs)

        # Área scrollable
        self.scroll = ScrollView(size_hint=(1, 0.9))
        self.chat = GridLayout(cols=1, spacing=10, size_hint_y=None, padding=5)
        self.chat.bind(minimum_height=self.chat.setter("height"))
        self.scroll.add_widget(self.chat)
        self.add_widget(self.scroll)

        # Input y botón
        self.input_box = BoxLayout(size_hint=(1, 0.1), spacing=5)
        self.text_input = TextInput(
            hint_text="Escribe un mensaje...",
            background_color=(0.1, 0.1, 0.12, 1),
            foreground_color=(1, 1, 1, 1),
            cursor_color=(1, 0.5, 0, 1),
            multiline=False,
            padding=[10, 10, 10, 10],
        )
        self.text_input.bind(on_text_validate=lambda _: self.send_message())

        self.send_button = Button(
            text="✉️",
            background_color=(1, 0.4, 0, 1),
            color=(1, 1, 1, 1),
            font_size=20,
            size_hint_x=None,
            width=dp(60),
            on_press=lambda _: self.send_message(),
        )

        self.input_box.add_widget(self.text_input)
        self.input_box.add_widget(self.send_button)
        self.add_widget(self.input_box)

        # Bienvenida
        self.add_bot_message("¡Hola! Soy tu asistente 🤖")

    # --- Mensaje del usuario ---
    def add_user_message(self, text):
        msg = Label(
            text=text,
            size_hint_y=None,
            height=self.get_text_height(text),
            halign="right",
            valign="middle",
            text_size=(Window.width * 0.8, None),
            color=(1, 1, 1, 1),
            padding=(10, 10),
        )
        msg.bind(size=msg.setter("text_size"))

        # Fondo gris oscuro redondeado
        with msg.canvas.before:
            Color(0.15, 0.15, 0.18, 1)
            msg.bg = RoundedRectangle(radius=[15, 15, 15, 0])
        msg.bind(pos=self.update_bg, size=self.update_bg)

        self.chat.add_widget(msg)
        Clock.schedule_once(lambda _: self.scroll.scroll_to(msg), 0.1)

    # --- Mensaje del bot ---
    def add_bot_message(self, text):
        msg = Label(
            text=f"[color=ff8800]{text}[/color]",
            markup=True,
            size_hint_y=None,
            height=self.get_text_height(text),
            halign="left",
            valign="middle",
            text_size=(Window.width * 0.8, None),
            padding=(10, 10),
        )
        msg.bind(size=msg.setter("text_size"))

        # Fondo naranja translúcido
        with msg.canvas.before:
            Color(1, 0.4, 0, 0.25)
            msg.bg = RoundedRectangle(radius=[15, 15, 0, 15])
        msg.bind(pos=self.update_bg, size=self.update_bg)

        self.chat.add_widget(msg)
        Clock.schedule_once(lambda _: self.scroll.scroll_to(msg), 0.1)

    # --- Actualiza fondo redondeado ---
    def update_bg(self, instance, *args):
        instance.bg.pos = instance.pos
        instance.bg.size = instance.size

    # --- Calcula altura del texto ---
    def get_text_height(self, text):
        lines = len(text) // 40 + 1
        return dp(25 * lines + 20)

    # --- Enviar mensaje ---
    def send_message(self):
        user_text = self.text_input.text.strip()
        if not user_text:
            return
        self.add_user_message(user_text)
        self.text_input.text = ""
        Clock.schedule_once(lambda _: self.bot_reply(user_text), 0.7)

    # --- Responder ---
    def bot_reply(self, user_text):
        respuestas = responder(user_text)
        for r in respuestas:
            self.add_bot_message(r)
            time.sleep(random.uniform(0.2, 0.5))


# --- App principal ---
class ChatApp(App):
    def build(self):
        self.title = "ChatBot 🧠"
        return ChatLayout()


if __name__ == "__main__":
    ChatApp().run()

