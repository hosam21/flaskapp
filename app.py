# -*- coding: utf-8 -*-
import pysqlite3
import sys
sys.modules["sqlite3"] = pysqlite3
import tempfile
import os
import uuid
import base64
from io import BytesIO
from PIL import Image
from chromadb.config import Settings
from unstructured.partition.pdf import partition_pdf
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from transformers.models.blip import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.stores import BaseStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.chains import RetrievalQA
from typing import List, Optional
from collections.abc import Iterator
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev_key_123')
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp(prefix='rag_uploads_')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Check required API keys
if not all([os.getenv("GROQ_API_KEY"), os.getenv("LANGCHAIN_API_KEY")]):
    raise EnvironmentError("Missing required API keys in environment variables.")

# ----------------------------
# Custom Components (From Original Script)
# ----------------------------
class CustomInMemoryDocStore:
    def __init__(self):
        self._store = {}

    def mset(self, key_value_pairs: List[tuple[str, Document]]) -> None:
        for key, value in key_value_pairs:
            self._store[key] = value

    def mget(self, keys: List[str]) -> List[Optional[Document]]:
        return [self._store.get(key) for key in keys]

class OpenSourceEmbeddings:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, docs: List[str]) -> List[List[float]]:
        processed_docs = [doc if (isinstance(doc, str) and doc.strip()) else "unknown" for doc in docs]
        return self.model.encode(processed_docs).tolist()

    def embed_query(self, query: str) -> List[float]:
        query = query if query.strip() else "unknown"
        return self.model.encode([query]).tolist()[0]

# ----------------------------
# Helper Functions
# ----------------------------
def initialize_models():
    """Initialize ML models once at startup"""
    return {
        'blip_processor': BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base"),
        'blip_model': BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base"),
        'summarize_model': ChatGroq(groq_api_key=os.environ["GROQ_API_KEY"],temperature=0.5, model="llama-3.1-8b-instant")
    }

models = initialize_models()

def process_pdf(file_path: str):
    """Process PDF and create retriever"""
    # Extract elements
    chunks = partition_pdf(
        filename=file_path,
        infer_table_structure=True,
        strategy="hi_res",
        extract_image_block_types=["Image"],
        chunking_strategy="by_title",
        max_characters=10000,
    )

    # Separate elements
    texts = [chunk for chunk in chunks if "CompositeElement" in str(type(chunk))]
    tables = [chunk for chunk in chunks if "Table" in str(type(chunk))]
    
    # Generate summaries
    text_summaries = generate_text_summaries(texts)
    table_summaries = generate_table_summaries(tables)
    image_summaries = generate_image_summaries(chunks)
    
    # Create vectorstore and retriever
    embedding_function = OpenSourceEmbeddings()
    vectorstore = Chroma(
        collection_name=f"rag_collection_{uuid.uuid4()}",
        embedding_function=embedding_function,
        persist_directory=session.get('persist_dir')
    )
    
    docstore = CustomInMemoryDocStore()
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key="doc_id",
        search_kwargs={"k": 5}
    )
    
    # Index documents
    index_documents(text_summaries, [str(t) for t in texts], "text", retriever)
    index_documents(table_summaries, [str(t) for t in tables], "table", retriever)
    index_documents(image_summaries, image_summaries, "image", retriever)
    
    return retriever

def index_documents(summaries, originals, doc_type, retriever):
    """Index documents into vectorstore and docstore"""
    doc_ids = [str(uuid.uuid4()) for _ in summaries]
    retriever.vectorstore.add_documents(
        [Document(page_content=s, metadata={"doc_id": doc_id, "type": doc_type}) 
        for doc_id, s in zip(doc_ids, summaries)])
    retriever.docstore.mset(zip(doc_ids, [Document(page_content=o) for o in originals]))

def generate_text_summaries(texts):
    """Generate text summaries using LLM"""
    prompt_text = "Summarize this text chunk concisely:\n\n{element}"
    return [models['summarize_model'].invoke(prompt_text.format(element=text)).content for text in texts]

def generate_table_summaries(tables):
    """Generate table summaries using LLM"""
    prompt_text = "Summarize this table concisely:\n\n{element}"
    return [models['summarize_model'].invoke(prompt_text.format(element=table.metadata.text_as_html)).content for table in tables]

def generate_image_summaries(chunks):
    """Generate image captions using BLIP model"""
    images = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            for el in chunk.metadata.orig_elements:
                if "Image" in str(type(el)):
                    images.append(el.metadata.image_base64)
    
    return [generate_image_caption(img) for img in images]

def generate_image_caption(image_base64):
    """Generate caption for a single image"""
    try:
        image_data = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_data)).convert('RGB')
        inputs = models['blip_processor'](images=image, return_tensors="pt")
        outputs = models['blip_model'].generate(**inputs)
        return models['blip_processor'].decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return f"Image processing failed: {str(e)}"

# ----------------------------
# Flask Routes
# ----------------------------
@app.route('/', methods=['GET'])
def upload_form():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def handle_upload():
    if 'file' not in request.files:
        return redirect(url_for('upload_form'))
    
    file = request.files['file']
    if file.filename == '' or not file.filename.lower().endswith('.pdf'):
        return redirect(url_for('upload_form'))
    
    try:
        # Save uploaded file
        temp_dir = tempfile.mkdtemp(dir=app.config['UPLOAD_FOLDER'])
        file_path = os.path.join(temp_dir, secure_filename(file.filename))
        file.save(file_path)
        
        # Process PDF and store retriever in session
        retriever = process_pdf(file_path)
        session['retriever'] = {
            'collection_name': retriever.vectorstore._collection.name,
            'persist_dir': temp_dir
        }
        return redirect(url_for('query_interface'))
    
    except Exception as e:
        return f"Error processing PDF: {str(e)}", 500

@app.route('/query', methods=['GET'])
def query_interface():
    return render_template('query.html')

@app.route('/query', methods=['POST'])
def handle_query():
    if 'retriever' not in session:
        return redirect(url_for('upload_form'))
    
    try:
        # Reinitialize retriever
        retriever_info = session['retriever']
        embedding = OpenSourceEmbeddings()
        vectorstore = Chroma(
            collection_name=retriever_info['collection_name'],
            embedding_function=embedding,
            persist_directory=retriever_info['persist_dir']
        )
        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            docstore=CustomInMemoryDocStore(),
            search_kwargs={"k": 5}
        )
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=models['summarize_model'],
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        # Process query
        question = request.form.get('question', '')
        result = qa_chain.invoke({"query": question})
        return render_template('result.html',
                             question=question,
                             answer=result['result'],
                             sources=result['source_documents'])
    
    except Exception as e:
        return f"Error processing query: {str(e)}", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
