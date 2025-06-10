from flask import Flask, request, render_template
import os
import PyPDF2
import docx2txt 
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnableMap, RunnableLambda, RunnableSequence
from langchain.schema.output_parser import StrOutputParser

#add an option for user to chose a book from which they want a paragraph
#fine tune the prompt and make sure the response from book if its a novel can be more about similarities between protagoists life and your life
#add memory and store it in a folder all the conversations
#add problamatics to the prompts, help user ask better questions
#multilingual speech to text

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/' #all the resume files will be dynamically stored here

os.environ["GROQ_API_KEY"] = "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"# Setup base LLM

llm = ChatOpenAI(
    temperature=0.7, 
    model="llama3-70b-8192",  # or mixtral-8x7b-32768, gemma-7b-it, etc.
    openai_api_key=os.environ["GROQ_API_KEY"],
    openai_api_base="https://api.groq.com/openai/v1",  # Groq’s base URL
)

# Output parser
parser = StrOutputParser()


# Prompt for generating a daily journal prompt
extract_theme_prompt = ChatPromptTemplate.from_template(
    "What is the main emotional or reflective theme of this journal entry?\n\n{entry}\n\nRespond with a thoughtful prompt that user can reflect on"
)

# Prompt for inspirational quote
quote_prompt = ChatPromptTemplate.from_template(
    "Give a paragraph from Thus Spoke Zarathustra by Friedrich Nietzche based on the theme of the {entry}, write the paragraph as it as instead of changing it"
)


# Chain: Generate prompt
prompt_chain = extract_theme_prompt | llm | parser

# Chain: Generate quote
quote_chain = quote_prompt | llm | parser

# Lambda to detect if an entry was provided
def with_optional_reflection(inputs):
    result = {
        "daily_prompt": prompt_chain.invoke(inputs),
        "quote": quote_chain.invoke(inputs),
    }
    return result
# Wrap the workflow
journal_chain = RunnableLambda(with_optional_reflection)

@app.route("/") #to connect ui
def matcher(): 
    return render_template('Untitled-2.html')


@app.route("/matcher", methods=["POST"])
def handle_journal():
    entry = request.form.get("entry")
    if not entry:
        return render_template('Untitled-2.html', message="Please write something.")

    result = journal_chain.invoke({"entry": entry})

    return render_template(
        "Untitled-2.html",
        message="Here’s your reflection and quote!",
        daily_prompt=result["daily_prompt"],
        quote=result["quote"]
    )


        


if __name__ == "__main__":
    
    app.run(debug=True)