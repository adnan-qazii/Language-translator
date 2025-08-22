from flask import Flask , render_template, request 
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
import os
load_dotenv()

app = Flask(__name__)


groq_api_key = os.getenv("GROQ_API_KEY")
model = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)


generic_template="Translate the following into {language} but only provide one suitable answer:"

prompt=ChatPromptTemplate.from_messages(
    [("system",generic_template),("user","{text}")]
)


parser = StrOutputParser()
chain = prompt | model | parser

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')




@app.route('/translate', methods=['POST', 'GET'])
def translate():
    translated_text = None
    if request.method == 'POST':
        text = request.form.get('text')
        source_lang = request.form.get('source_lang')
        target_lang = request.form.get('target_lang')
        # Use target_lang for translation
        if text and target_lang:
            translated_text = chain.invoke({"text": text, "language": target_lang})
    return render_template('translate.html', translated_text=translated_text)



if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)