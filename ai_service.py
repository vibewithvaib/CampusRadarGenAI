from flask import Flask, request, jsonify
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import re
import json

app = Flask(__name__)


try:
    llm = Ollama(model="gemma:2b", temperature=0)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = FAISS.from_texts(["initialization text"], embeddings)
    print("In-memory vector store initialized successfully.")
except Exception as e:
    print(f"CRITICAL ERROR: Could not connect to Ollama. Please ensure it's running. Error: {e}")
    vector_store = None



FILTERING_PROMPT_TEMPLATE = """
You are an expert AI assistant for tech recruiters. Your task is to analyze a list of candidates who have already applied for a specific job and identify the best fits.

**Internship Description:**
{description}

**List of Candidates who have applied:**
{candidates}

Based on the internship description, analyze the provided list of candidates and identify the top 3 best fits.
Your response MUST be a valid JSON array containing ONLY the integer IDs of the recommended students. For example: [101, 105, 203]
If no candidates are a good fit or if the list of candidates is empty, you MUST return an empty array [].
Do not provide any explanation or introductory text, only the JSON array.
"""


def extract_skills_from_text(text):
    """
    Finds a line starting with "Skills:" in a profile or posting text,
    and returns a set of the skills for easy comparison.
    """
    match = re.search(r"Skills: (.*)", text, re.IGNORECASE)
    if match:
        return {skill.strip().lower() for skill in match.group(1).split(',')}
    return set()




@app.route('/ingest', methods=['POST'])
def ingest_document():
    if vector_store is None: return jsonify({"error": "Vector store is not initialized."}), 500
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
def filter_applicants():
    if vector_store is None: return jsonify({"error": "Vector store is not initialized."}), 500
    data = request.get_json()
    description = data.get('internship_description')
    applicants = data.get('applicant_profiles')
    print(data)
    if not description or not applicants:
        return jsonify({"error": "Request must include 'internship_description' and 'applicant_profiles'"}), 400

    candidate_context = "\n---\n".join(applicants)
    prompt = FILTERING_PROMPT_TEMPLATE.format(description=description, candidates=candidate_context)
    
    response_text = llm.invoke(prompt)
    print(f"LLM Raw Response for filtering: '{response_text}'")

    try:
        match = re.search(r'\[(.*?)\]', response_text)
        if not match: raise ValueError("No JSON array found in LLM response.")
        json_string = f"[{match.group(1)}]"
        recommended_ids = json.loads(json_string)
        if not isinstance(recommended_ids, list): raise ValueError("Parsed JSON is not a list.")
        return jsonify({"recommended_student_ids": [int(id) for id in recommended_ids]})
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing LLM response: {e}")
        return jsonify({"recommended_student_ids": []})

@app.route('/recommend/internships', methods=['POST'])
def recommend_internships():
    """
    Implements the "Find then Verify" hybrid recommendation for students.
    """
    if vector_store is None: return jsonify({"error": "Vector store is not initialized."}), 500
    data = request.get_json()
    profile = data.get('student_profile')
    student_skills = data.get('student_skills', [])
    print(data)
    print(profile)
    
    if not profile or not student_skills:
        return jsonify({"error": "Request must include 'student_profile' and 'student_skills'"}), 400


    results = vector_store.similarity_search(profile, k=10, filter={'type': 'internship'})

    recommendations = []
    student_skills_set = {skill.lower() for skill in student_skills}

    # 2. VERIFY: Apply the strict skill-matching filter.
    for doc in results:
        internship_skills = extract_skills_from_text(doc.page_content)

        # Only include the internship if its required skills match at least one of the student's skills.
        if internship_skills.intersection(student_skills_set):
            recommendations.append({
                "internshipId": doc.metadata.get("id"),
                "postingText": doc.page_content
            })

    return jsonify(recommendations)


if __name__ == '__main__':
    app.run(port=5001, debug=True)

