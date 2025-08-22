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




@app.route('/translate', methods=['POST'])
def translate():
    text = request.form['text']
    language = request.form['language']
    result = chain.invoke({"text": text, "language": language})
    return render_template('translate.html', result=result)



if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)