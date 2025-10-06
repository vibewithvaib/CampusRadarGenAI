from flask import Flask, request, jsonify
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import re 

app = Flask(__name__)

llm = Ollama(model="gemma:2b", temperature=0)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_store = FAISS.from_texts(["initialization text"], embeddings)
print("In-memory vector store initialized.")



def extract_skills_from_text(text):
   
    match = re.search(r"Skills: (.*)", text, re.IGNORECASE)
    if match:
        return {skill.strip().lower() for skill in match.group(1).split(',')}
    return set()


@app.route('/ingest', methods=['POST'])
def ingest_document():
    data = request.get_json()
    text = data.get('text')
    metadata = data.get('metadata')
    if not text or not metadata:
        return jsonify({"error": "Request must include 'text' and 'metadata'"}), 400

    doc = Document(page_content=text, metadata=metadata)
    vector_store.add_documents([doc])
    print(f"Successfully ingested document with metadata: {metadata}")
    return jsonify({"status": "success"}), 200

@app.route('/recommend/candidates', methods=['POST'])
def recommend_candidates():
    data = request.get_json()
    description = data.get('internship_description')
    required_skills = data.get('required_skills', [])
    
    if not description or not required_skills:
        return jsonify({"error": "Request must include 'internship_description' and 'required_skills'"}), 400

    results = vector_store.similarity_search(description, k=10, filter={'type': 'student'})

    recommendations = []
    required_skills_set = {skill.lower() for skill in required_skills}

    for doc in results:
        student_skills = extract_skills_from_text(doc.page_content)
        
        if student_skills.intersection(required_skills_set):
            recommendations.append({
                "studentId": doc.metadata.get("id"),
                "profileText": doc.page_content
            })
    
    return jsonify(recommendations)

@app.route('/recommend/internships', methods=['POST'])
def recommend_internships():
    data = request.get_json()
    profile = data.get('student_profile')
    
    student_skills = data.get('student_skills', [])

    if not profile or not student_skills:
        return jsonify({"error": "Request must include 'student_profile' and 'student_skills'"}), 400

   
    results = vector_store.similarity_search(profile, k=10, filter={'type': 'internship'})

    recommendations = []
    student_skills_set = {skill.lower() for skill in student_skills}

    for doc in results:
        internship_skills = extract_skills_from_text(doc.page_content)
        if internship_skills.intersection(student_skills_set):
            recommendations.append({
                "internshipId": doc.metadata.get("id"),
                "postingText": doc.page_content
            })

    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(port=5001, debug=True)

