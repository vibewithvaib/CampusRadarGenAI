from flask import Flask, request, jsonify
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import json # Import the JSON library for parsing

app = Flask(__name__)

# --- INITIALIZATION ---
# Using a model with good reasoning capabilities is important here.
llm = Ollama(model="gemma:2b", temperature=0)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# The vector store is still used for ingestion, but not for direct recommendations in this version.
vector_store = FAISS.from_texts(["initialization text"], embeddings)
print("In-memory vector store initialized.")


# --- NEW PROMPT TEMPLATE FOR AI-POWERED FILTERING ---
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


# --- API ENDPOINTS ---

@app.route('/ingest', methods=['POST'])
def ingest_document():
    """
    Ingests a new document with its text and metadata into the vector store.
    This is still useful for potential future features.
    """
    data = request.get_json()
    text = data.get('text')
    metadata = data.get('metadata')
    if not text or not metadata:
        return jsonify({"error": "Request must include 'text' and 'metadata'"}), 400

    doc = Document(page_content=text, metadata=metadata)
    vector_store.add_documents([doc])
    print(f"Successfully ingested document with metadata: {metadata}")
    return jsonify({"status": "success"}), 200

# --- NEW ENDPOINT FOR AI-ASSISTED SHORTLISTING ---
@app.route('/filter/applicants', methods=['POST'])
def filter_applicants():
    """
    Receives an internship description and a list of applicant profiles,
    and uses the LLM to determine which ones to shortlist.
    """
    data = request.get_json()
    description = data.get('internship_description')
    applicants = data.get('applicant_profiles') # Expects a list of formatted strings

    if not description or not applicants:
        return jsonify({"error": "Request must include 'internship_description' and 'applicant_profiles'"}), 400

    # Combine all applicant profiles into a single block of text for the context
    candidate_context = "\n---\n".join(applicants)

    # Create the detailed prompt for the LLM using our template
    prompt = FILTERING_PROMPT_TEMPLATE.format(description=description, candidates=candidate_context)
    
    # Get the raw text response from the LLM
    print("\n--- Sending Prompt to LLM ---")
    print(prompt)
    print("-----------------------------\n")
    response_text = llm.invoke(prompt)
    print(f"LLM Response Text: {response_text}")

    try:
        # Attempt to parse the LLM's response as a JSON array of integers
        recommended_ids = json.loads(response_text)
        if not isinstance(recommended_ids, list):
            raise ValueError("LLM response was not a JSON list.")
        
        # Ensure all items in the list are integers
        recommended_ids = [int(id) for id in recommended_ids]
        print(f"Successfully parsed recommended IDs: {recommended_ids}")
        return jsonify({"recommended_student_ids": recommended_ids})

    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing LLM response: {e}\nRaw Response was: '{response_text}'")
        # If parsing fails, return an empty list as a safe fallback
        return jsonify({"recommended_student_ids": []})


if __name__ == '__main__':
    app.run(port=5001, debug=True)

