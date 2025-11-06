# app.py
import os
import uuid
from flask import Flask, request, jsonify
from vector_store import VectorStore
from llm import LLM
from ingest import ingest_file
from jinja2 import Environment, FileSystemLoader

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
store = VectorStore(index_path="faiss.index")
llm = LLM()
jinja_env = Environment(loader=FileSystemLoader("templates"))

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

@app.route("/templates", methods=["POST"])
def create_template():
    # store jinja2 template content in templates dir
    data = request.json
    name = data.get("name", "proposal_template.j2")
    content = data.get("content")
    if not content:
        return jsonify({"error":"no content"}), 400
    with open(os.path.join("templates", name), "w", encoding="utf-8") as f:
        f.write(content)
    return jsonify({"status":"ok", "name": name})

@app.route("/generate", methods=["POST"])
def generate_proposal():
    data = request.json
    user_prompt = data.get("prompt", "")
    template_name = data.get("template", "proposal_template.j2")
    top_k = int(data.get("top_k", 5))

    # 1. retrieve docs
    results = store.query(user_prompt, top_k=top_k)
    contexts = []
    for meta, score, text in results:
        contexts.append({"source": meta.get("source"), "score": score, "text": text})

    # 2. craft RAG prompt using template
    template = jinja_env.get_template(template_name)
    rag_context = "\n\n".join([f"Source: {c['source']}\n\n{c['text']}" for c in contexts])
    # Template expects keys: user_prompt, rag_context
    prompt_for_llm = template.render(user_prompt=user_prompt, rag_context=rag_context)

    # 3. generate
    generated = llm.generate(prompt_for_llm)

    return jsonify({
        "generated": generated,
        "used_contexts": contexts,
        "llm_prompt": prompt_for_llm
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
