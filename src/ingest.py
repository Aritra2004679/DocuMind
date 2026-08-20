import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
def load_documents_from_uploads(uploaded_files):
    """
    uploaded_files: list of Streamlit UploadedFile objects
    Streamlit gives file-like objects in memory, but PyPDFLoader needs a path,
    so we write each to a temp file first.
    """
    documents = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        # tag each chunk with the ORIGINAL filename, not the temp path
        for doc in docs:
            doc.metadata["source"] = uploaded_file.name

        documents.extend(docs)
        os.unlink(tmp_path)  # clean up temp file

    print(f"Loaded {len(documents)} pages from {len(uploaded_files)} file(s)")
    return documents

def chunk_documents(documents, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    return chunks