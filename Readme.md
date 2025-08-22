# 🌈✨ Language Translator App ✨🌈

![Translator Banner](https://img.icons8.com/color/96/translate.png)

## 🚀 What is Language Translator?

Language Translator is a vibrant, modern web application that lets you instantly translate text between multiple languages, including English, Hindi, Urdu, Hinglish, Spanish, French, German, and Chinese! 🌍🗣️

---

## 🎨 Features

- 🌟 Colorful, aesthetic UI
- 🔥 Fast and accurate translations
- 🏆 Supports many languages
- 🖥️ Built with Python, Flask, and Groq LLM
- 🧩 Easy to use and extend
- 🎉 Emoji, stickers, and fun design

---

## 🛠️ How It Works

1. **Frontend:**
   - `index.html` (Home page)
   - `translate.html` (Translation page)
   - Modern, responsive, and full of colors!

2. **Backend:**
   - `app.py` (Flask server)
   - Uses Groq LLM for translation
   - Handles form submissions and returns results

3. **Schema:**
   - `/` : Home page with info and link to translator
   - `/translate` : Translation form and results

---

## 📝 Usage

1. Clone the repo:
   ```bash
   git clone https://github.com/adnan-qazii/Language-translator.git
   ```
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Set your Groq API key in `.env`:
   ```env
   GROQ_API_KEY=your_key_here
   ```
4. Run the app:
   ```bash
   python app.py
   ```
5. Open your browser and go to `http://localhost:5000`

---

## 🌐 Supported Languages

| 🌍 Language | Code      |
|------------|-----------|
| 🇬🇧 English | en        |
| 🇮🇳 Hindi   | hi        |
| 🇵🇰 Urdu    | ur        |
| 🗣️ Hinglish | hinglish  |
| 🇪🇸 Spanish | es        |
| 🇫🇷 French  | fr        |
| 🇩🇪 German  | de        |
| 🇨🇳 Chinese | zh-cn     |

---

## 💡 Example

> "Hello, how are you?" → "नमस्ते, आप कैसे हैं?" (Hindi)

---

## 🧑‍💻 Contributing

Pull requests, issues, and suggestions are welcome! Add your favorite language, improve the UI, or share your ideas. 🤗

---

## 📦 Requirements

- Python 3.8+
- Flask
- langchain
- langchain_groq
- dotenv

---

## 🖼️ Stickers & Emojis

![Emoji](https://img.icons8.com/color/48/000000/happy.png) ![Emoji](https://img.icons8.com/color/48/000000/party.png) ![Emoji](https://img.icons8.com/color/48/000000/translate.png)

---

## 📚 License

MIT License

---

## 🏅 Author

Made with ❤️ by **adnan-qazii**

![Made with Love](https://img.icons8.com/color/48/000000/like--v3.png)
