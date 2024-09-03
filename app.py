import os
from flask import Flask, request, jsonify
from functools import wraps
from ibm_watsonx_ai.client import APIClient
from ibm_watsonx_ai.foundation_models.embeddings.sentence_transformer_embeddings import SentenceTransformerEmbeddings
from ibm_watsonx_ai.foundation_models import Model
import chromadb
from huggingface_hub import login
import gzip
import json
import random
import string
import time
import datetime

app = Flask(__name__)

# Authentication
def check_auth(username, password):
    """Check if a username/password combination is valid."""
    valid_username = os.environ.get('AUTH_USERNAME')
    valid_password = os.environ.get('AUTH_PASSWORD')
    return username == valid_username and password == valid_password

def authenticate():
    """Sends a 401 response that enables basic auth"""
    return jsonify({"message": "Authentication required."}), 401, {'WWW-Authenticate': 'Basic realm="Login Required"'}

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated

# Helper function to get configuration values
def get_config_value(key):
    value = os.getenv(key)
    if value is None:
        raise KeyError(f"Environment variable '{key}' not found.")
    return value

# Get credentials from environment variables
def get_credentials():
    return {
        "url": get_config_value('URL'),
        "apikey": get_config_value('API_KEY'),
    }

# Configuration values
model_id = "ibm/granite-13b-chat-v2"
parameters = {
    "decoding_method": "sample",
    "max_new_tokens": 1024,
    "temperature": 0.7,
    "top_p": 1,
    "top_k": 50,
    "repetition_penalty": 1.1
}
project_id = get_config_value('PROJECT_ID')
vector_index_id = get_config_value('VECTOR_INDEX_ID')

# Hugging Face login
login(get_config_value('HF_LOGIN'))

# Initialize model and API client
model = Model(
    model_id=model_id,
    params=parameters,
    credentials=get_credentials(),
    project_id=project_id
)

wml_credentials = get_credentials()
client = APIClient(credentials=wml_credentials, project_id=project_id)

vector_index_details = client.data_assets.get_details(vector_index_id)
vector_index_properties = vector_index_details["entity"]["vector_index"]

emb = SentenceTransformerEmbeddings('sentence-transformers/all-MiniLM-L6-v2')

def hydrate_chromadb():
    data = client.data_assets.get_content(vector_index_id)
    content = gzip.decompress(data)
    stringified_vectors = str(content, "utf-8")
    vectors = json.loads(stringified_vectors)

    chroma_client = chromadb.Client()

    # make sure collection is empty if it already existed
    collection_name = "my_collection"
    try:
        collection = chroma_client.delete_collection(name=collection_name)
    except:
        print("Collection didn't exist - nothing to do.")
    collection = chroma_client.create_collection(name=collection_name)

    vector_embeddings = []
    vector_documents = []
    vector_metadatas = []
    vector_ids = []

    for vector in vectors:
        vector_embeddings.append(vector["embedding"])
        vector_documents.append(vector["content"])
        metadata = vector["metadata"]
        lines = metadata["loc"]["lines"]
        clean_metadata = {}
        clean_metadata["asset_id"] = metadata["asset_id"]
        clean_metadata["asset_name"] = metadata["asset_name"]
        clean_metadata["url"] = metadata["url"]
        clean_metadata["from"] = lines["from"]
        clean_metadata["to"] = lines["to"]
        vector_metadatas.append(clean_metadata)
        asset_id = vector["metadata"]["asset_id"]
        random_string = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
        id = "{}:{}-{}-{}".format(asset_id, lines["from"], lines["to"], random_string)
        vector_ids.append(id)

    collection.add(
        embeddings=vector_embeddings,
        documents=vector_documents,
        metadatas=vector_metadatas,
        ids=vector_ids
    )
    return collection

chroma_collection = hydrate_chromadb()

def proximity_search(question):
    query_vectors = emb.embed_query(question)
    query_result = chroma_collection.query(
        query_embeddings=query_vectors,
        n_results=vector_index_properties["settings"]["top_k"],
        include=["documents", "metadatas", "distances"]
    )

    documents = list(reversed(query_result["documents"][0]))

    return "\n".join(documents)

@app.route('/chat', methods=['POST'])
@requires_auth
def chat():
    data = request.json
    prompt = data.get('prompt')
    context = data.get('context', "")

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    grounding = proximity_search(prompt)
    
    # Update conversation context
    context += f"\n<|user|>\n[Document]\n{grounding}\n[End]\n{prompt}\n<|assistant|>"
    
    try:
        response = model.generate_text(prompt=context, guardrails=False)
        
        # Your original prompt
        system_prompt = """You are an AI assistant for Volta Inc., analyzing SOWs and answering queries based ONLY on the information provided in the context. Always refer to the specific SOW documents mentioned in the context when answering.

Instructions:
1. Use ONLY information from the SOW excerpts provided in each query.
2. State clearly if the required information is not in the provided excerpts.
3. Do not use external knowledge or speculate beyond the given information.
4. Maintain a professional tone.
5. Provide source references (e.g., "According to the SOW for Customer X...").
6. When listing information, use bullet points for clarity.

Remember: Your knowledge is limited to the SOW excerpts provided with each query. If information is not in these excerpts, say so clearly.

Here are some examples of good responses:

Q: What are we charging rubicon for the services we provide?
A: We are charging Rubicon for the following services:
- Meraki Support Services Hourly: $175/hr
- VMware Managed Services 3: $330/month
- Managed Backup Services 3: $330/month
- IBM Storage Managed Services 2: $330/month
- Cisco ISE Managed Service 2: $220/month
- Cisco ASA Managed Service 2: $220/month
- Cisco Firepower Managed Service 1: $220/month
- Cisco Core Switch Managed Service 1: $220/month
Total: $3,960

Q: When did man first walk on the moon?
A: Information not provided in the document.

Q: What is the response time for a critical issue at LWC?
A: The response time for a critical issue (Severity 1) at LWC is 15-30 minutes. A Volta support engineer will respond within this timeframe and work to resolve the issue within 1 hour for remote resolution and 2 hours for onsite resolution.

Always strive for accuracy and completeness in your responses, based solely on the information provided in the SOW excerpts."""

        # Combine system prompt, context, and response
        full_response = f"{system_prompt}\n\n{context}\n{response}"
        
        return jsonify({
            "response": full_response,
            "context": context
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
