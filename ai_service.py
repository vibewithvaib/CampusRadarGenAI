from flask import Flask, request, jsonify
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

app = Flask(__name__)

llm = Ollama(model="gemma:2b", temperature=0)
embeddings = OllamaEmbeddings(model="nomic-embed-text")


vector_store = FAISS.from_texts(["initialization text"], embeddings)
print("In-memory vector store initialized.")


RECRUITER_PROMPT_TEMPLATE = """
You are an expert AI assistant for tech recruiters. Your answer MUST be based exclusively on the context provided below.

**Context of available student profiles:**
{context}

**Recruiter's request:** "{question}"

Based ONLY on the context provided, suggest the top 2-3 candidates that match the recruiter's request.
If the context is empty or no candidates in the context match the request, you MUST state that no suitable candidates were found in the provided information.
Under no circumstances should you invent or mention candidates not present in the context.
"""

STUDENT_PROMPT_TEMPLATE = """
You are a helpful career advisor AI for college students. Your answer MUST be based exclusively on the context provided below.

**Context of available internships:**
{context}

**Student's request:** "{question}"

Based ONLY on the context provided, suggest 1-2 internships that best fit the student's request.
If the context is empty or no internships in the context match the request, you MUST state that no suitable internships were found in the provided information.
Under no circumstances should you invent or mention internships not present in the context.
"""


recruiter_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    chain_type_kwargs={"prompt": PromptTemplate.from_template(RECRUITER_PROMPT_TEMPLATE)}
)
student_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    chain_type_kwargs={"prompt": PromptTemplate.from_template(STUDENT_PROMPT_TEMPLATE)}
)


@app.route('/ingest', methods=['POST'])
def ingest_document():
    data = request.get_json()
    text = data.get('text')
    if not text:
        return jsonify({"error": "No text provided"}), 400


    vector_store.add_texts([text])
    print(f"Successfully ingested document.")
    return jsonify({"status": "success"}), 200

@app.route('/recommend/candidates', methods=['POST'])
def recommend_candidates():
    data = request.get_json()
    prompt = data.get('prompt')
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    

    result = recruiter_chain.invoke({"query": prompt})
    return jsonify({"recommendation": result['result']})

@app.route('/recommend/internships', methods=['POST'])
def recommend_internships():
    data = request.get_json()
    prompt = data.get('prompt')
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    

    result = student_chain.invoke({"query": prompt})
    return jsonify({"recommendation": result['result']})

if __name__ == '__main__':
    app.run(port=5001, debug=True)

