import os
import uuid
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from vector_store import VectorStore
from llm import LLM
from ingest import ingest_file
from jinja2 import Environment, FileSystemLoader

# Your Google Cloud API Key for Gemini
API_KEY = "AIzaSyD0IAGLrie7AgaumBuFrkUdS_ua2xX7J_8"

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)

app = Flask(__name__)
CORS(app)  # allow frontend JS calls

store = VectorStore(index_path="faiss.index")
llm = LLM(api_key=API_KEY)  # Pass the API key to LLM
jinja_env = Environment(loader=FileSystemLoader("templates"))

# 🏠 FRONTEND ROUTE
@app.route("/")
def home():
    return render_template("index.html")

# 📂 UPLOAD DOCUMENTS
@app.route("/upload", methods=["POST"])
def upload():
    files = request.files.getlist("files")
    saved = []
    for f in files:
        fname = f"{uuid.uuid4().hex}_{f.filename}"
        path = os.path.join(UPLOAD_DIR, fname)
        f.save(path)
        ingest_file(path, store, source_name=f.filename)
        saved.append(f.filename)
    return jsonify({"status": "ok", "uploaded": saved})

# 💾 CREATE/UPDATE TEMPLATE
@app.route("/templates", methods=["POST"])
def create_template():
    data = request.json
    name = data.get("name", "proposal_template.j2")
    content = data.get("content")
    if not content:
        return jsonify({"error": "no content"}), 400
    with open(os.path.join("templates", name), "w", encoding="utf-8") as f:
        f.write(content)
    return jsonify({"status": "ok", "name": name})

# ⚙️ GENERATE PROPOSAL
# ⚙️ GENERATE PROPOSAL
@app.route("/generate", methods=["POST"])
def generate_proposal():
    data = request.json
    user_prompt = data.get("prompt", "")
    template_name = data.get("template", "proposal_template.j2")
    top_k = int(data.get("top_k", 5))

    # 1️⃣ Retrieve top documents
    results = store.query(user_prompt, top_k=top_k)
    contexts = []
    for meta, score, text in results:
        contexts.append({"source": meta.get("source"), "score": score, "text": text})

    if not contexts:
        return jsonify({"error": "No relevant documents found for the given prompt."}), 400

    # 2️⃣ Prepare RAG prompt
    template = jinja_env.get_template(template_name)
    rag_context = "\n\n".join([f"Source: {c['source']}\n\n{c['text']}" for c in contexts])
    prompt_for_llm = template.render(user_prompt=user_prompt, rag_context=rag_context)

    # Debug: Log the generated prompt
    print("Generated prompt for LLM:")
    print(prompt_for_llm)

    # 3️⃣ Generate text
    generated = llm.generate(prompt_for_llm)

    return jsonify({
        "generated": generated,
        "used_contexts": contexts,
        "llm_prompt": prompt_for_llm
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
