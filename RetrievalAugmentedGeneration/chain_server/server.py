"""The definition of the Llama Index chain server."""
import base64
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from chain_server import utils
from chain_server import chains

# create the FastAPI server
app = FastAPI()
# prestage the embedding model
_ = utils.get_embedding_model()
# set the global service context for Llama Index
utils.set_service_context()


class Prompt(BaseModel):
    """Definition of the Prompt API data type."""

    question: str
    context: str
    use_knowledge_base: bool = True
    num_tokens: int = 50


class DocumentSearch(BaseModel):
    """Definition of the DocumentSearch API data type."""

    content: str
    num_docs: int = 4


@app.post("/uploadDocument")
async def upload_document(file: UploadFile = File(...)) -> JSONResponse:
    """Upload a document to the vector store."""
    if not file.filename:
        return JSONResponse(content={"message": "No files provided"}, status_code=200)

    upload_folder = "uploaded_files"
    upload_file = os.path.basename(file.filename)
    if not upload_file:
        raise RuntimeError("Error parsing uploaded filename.")
    file_path = os.path.join(upload_folder, upload_file)
    uploads_dir = Path(upload_folder)
    uploads_dir.mkdir(parents=True, exist_ok=True)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    chains.ingest_docs(file_path, upload_file)

    return JSONResponse(
        content={"message": "File uploaded successfully"}, status_code=200
    )


@app.post("/generate")
async def generate_answer(prompt: Prompt) -> StreamingResponse:
    """Generate and stream the response to the provided prompt."""
    if prompt.use_knowledge_base:
        generator = chains.rag_chain(prompt.question, prompt.num_tokens)
        return StreamingResponse(generator, media_type="text/event-stream")

    generator = chains.llm_chain(prompt.context, prompt.question, prompt.num_tokens)
    return StreamingResponse(generator, media_type="text/event-stream")


@app.post("/documentSearch")
def document_search(data: DocumentSearch) -> List[Dict[str, Any]]:
    """Search for the most relevant documents for the given search parameters."""
    retriever = utils.get_doc_retriever(num_nodes=data.num_docs)
    nodes = retriever.retrieve(data.content)
    output = []
    for node in nodes:
        file_name = nodes[0].metadata["filename"]
        decoded_filename = base64.b64decode(file_name.encode("utf-8")).decode("utf-8")
        entry = {"score": node.score, "source": decoded_filename, "content": node.text}
        output.append(entry)

    return output
