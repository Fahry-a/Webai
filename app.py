import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, AIMessage, SystemMessage
import json

load_dotenv()

app = Flask(__name__)

class ChatMemory:
    def __init__(self, max_size=25, filename='chat.txt'):
        self.conversation_history = []
        self.max_size = max_size
        self.filename = filename
        self.load_from_file()

    def load_from_file(self):
        if os.path.exists(self.filename) and os.path.getsize(self.filename) > 0:
            try:
                with open(self.filename, 'r') as f:
                    data = json.load(f)
                    self.conversation_history = [HumanMessage(content=msg['content']) if msg['type'] == 'HumanMessage' else AIMessage(content=msg['content']) for msg in data['conversation_history']]
            except json.JSONDecodeError:
                self.conversation_history = []
        else:
            self.conversation_history = []

    def save_to_file(self):
        data = {'conversation_history': [{'content': msg.content, 'type': type(msg).__name__} for msg in self.conversation_history]}
        with open(self.filename, 'w') as f:
            json.dump(data, f)

    def add_message(self, message):
        self.conversation_history.append(message)
        if len(self.conversation_history) > self.max_size:
            self.conversation_history.pop(0)
        self.save_to_file()

    def get_context(self):
        return self.conversation_history

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['message']
    if user_input.lower() == "exit":
        return jsonify({'response': "Keluar dari chat."})

    # Inisialisasi GroqChatModel dari langchain-groq
    groq_api_key = os.getenv('GROQ_API_KEY')
    model = 'llama3-8b-8192'
    
    # Initialize Groq Langchain chat object and conversation
    groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name=model)
    chat_memory = ChatMemory()

    # Tambahkan pesan ke dalam chat memory
    chat_memory.add_message(HumanMessage(content=user_input))

    # Kirim pesan ke Groq API melalui langchain-groq
    messages = [
        SystemMessage(content="Kamu adalah seorang programmer ahli yang dikenal dengan nama Hanz Ai. Dengan keahlian mendalam dalam berbagai bahasa pemrograman, kamu selalu up-to-date dengan tren dan inovasi terbaru dalam dunia pengembangan perangkat lunak. Bahasa pemrograman utama yang kamu kuasai adalah Java dan Python, namun kamu juga terampil dalam banyak bahasa pemrograman lainnya, seperti JavaScript, C++, dan SQL. Selain itu, kamu memiliki pengetahuan luas di bidang teknologi, algoritma, dan pemecahan masalah, serta kemampuan untuk menerapkan solusi teknis yang efisien dan tepat guna."),
    ] + chat_memory.get_context()

    # Mendapatkan respons model
    response = groq_chat.invoke(messages)

    # Tambahkan respons ke dalam chat memory
    chat_memory.add_message(AIMessage(content=response.content))

    return jsonify({'response': response.content})

if __name__ == "__main__":
    app.run(debug=True)