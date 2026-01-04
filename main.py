import os
import time
import logging
import lancedb
import json

from pathlib import Path
from typing import List, Any, Dict
from dotenv import load_dotenv
from tempfile import mkdtemp
import numpy as np

from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.chunking import HybridChunker
load_dotenv()

class DocumentProcesser:
    def __init__(self):
        self.api_key=os.getenv("OPENAI_API_KEY")
        self.setup_document_converter()
        self.setup_ml_components()
    
    def setup_document_converter(self):
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr=True
        pipeline_options.ocr_options.lang=["en"]
        pipeline_options.do_table_structure=True
        pipeline_options.table_structure_options

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                    backend=PyPdfiumDocumentBackend
                )
            }
        )

    def setup_ml_components(self):
        # Can try out hugging face pipeline in langchain
        self.embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        # self.embed_model = OpenAIEmbeddings(
        #     model="text-embedding-3-small",
        #     api_key=self.api_key
        # )
        self.llm = ChatOpenAI(
            model="gpt-4o",
            api_key=self.api_key
        )


    def extract_chunk_metadata(self, chunk):
        metadata = {
            "text": chunk.text,
            "headings": [],
            "page_info": None,
            "content_type":None
        }
        if hasattr(chunk, "meta"):
            # Extract headings
            if hasattr(chunk.meta, "headings") and chunk.meta.headings:
                metadata["headings"] = chunk.meta.headings

        # Extract page information and content_type
        if hasattr(chunk.meta, "doc_items"):
            for item in chunk.meta.doc_items:
                if hasattr(item, "label"):
                    metadata["content_type"]=str(item.label)
                
                if hasattr(item, "prov"):
                    for prov in item.prov:
                        if hasattr(prov, "page_no"):
                            metadata["page_info"]=prov.page_no

        return metadata
    
    def process_document(self, pdf_path:str):
        print(f"Processing document: {pdf_path}")
        start_time = time.time()

        result = self.converter.convert(source=pdf_path)
        doc = result.document

        chunker = HybridChunker()
        chunks = list(chunker.chunk(dl_doc=doc))

        processed_chunks=[]
        for i, chunk in enumerate(chunks):
            metadata = self.extract_chunk_metadata(chunk)
            processed_chunks.append(metadata)

            # Print chunk information for inspection
            print(f"\nChunk {i}:")
            if metadata['headings']:
                print(f"Section: {' > '.join(metadata['headings'])}")
            print(f"Page: {metadata['page_info']}")
            print(f"Type: {metadata['content_type']}")
            print("-" * 40)

        print("Creating vectorbase")
        db_uri = str(Path(mkdtemp())/"docling.db")
        self.db = lancedb.connect(db_uri)

        data=[]
        for chunk in processed_chunks:
            embeddings = self.embed_model.encode(chunk["text"])
            # embeddings = np.array(embeddings, dtype=np.float32)
            data_item = {
                "vector": embeddings,
                "text": chunk['text'],
                "headings": json.dumps(chunk['headings']),
                "page_info": chunk['page_info'],
                "content_type":chunk['content_type']
            }
            data.append(data_item)
        
        self.index = self.db.create_table("document_chunks", data=data, exist_ok=True)

        processing_time = time.time()-start_time
        print(f"\nDocument processing completed in {processing_time:.2f} seconds")
        return self.index
    
    def format_context(self, chunks: List[Dict]):
        context_parts=[]
        for chunk in chunks:
            try:
                headings = json.loads(chunk['headings'])
                if headings:
                    context_parts.append(f"Section: {' > '.join(headings)}")
            except:
                pass

            if chunk['page_info']:
                context_parts.append(f"Page {chunk['page_info']}:")
            
            context_parts.append(chunk['text'])
            context_parts.append("-" * 40)

        return "\n".join(context_parts)
    
    def query(self, query:str, k:int=5):
        query_embedding = self.embed_model.encode(query)
        results = self.index.search(query_embedding, vector_column_name="vector").limit(5)
        chunks = results.to_pandas()

        context = self.format_context(chunks.to_dict('records'))

        prompt = f"""Based on the following excerpts from a document:
        {context}

Please answer this quedtion: {query}

Make sure of the section information and page numbers in your answer when relevant.

"""
        return self.llm.invoke(prompt)


def main():
    logging.basicConfig(level=logging.INFO)
    document_processer = DocumentProcesser()

    pdf_path = "financial_reports\Meta Q3 2025 Earning Release.pdf"
    document_processer.process_document(pdf_path)
    
    question = "Can you summarise earning of the company in this quarter, and its future prospect."
    answer = document_processer.query(question)
    print(f"Answer: {answer.content}")


if __name__=="__main__":
    main()
