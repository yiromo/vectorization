import asyncio
import os
import sys
import time
import traceback
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import dependable_faiss_import
from langchain_huggingface import HuggingFaceEmbeddings
from mongo_utils import db
import logging
import numpy as np
from typing import List, Dict, Any, Tuple
import psutil
import gc
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("target_vectorization.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

faiss_logger = logging.getLogger('faiss')
faiss_logger.setLevel(logging.ERROR)
dependable_faiss_import()

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # in MB

DOCUMENT_BATCH_SIZE = 50  
CHUNK_BATCH_SIZE = 50
EMBEDDING_BATCH_SIZE = 16 

RUS_ARTICLE_PATTERN = r'^(\*\*Статья \d+\.\s*.*?\*\*)'
KAZ_ARTICLE_PATTERN = r'^(\*\*\d+-бап\.\s*.*?\*\*)'
CHAPTER_PATTERN = r'^(Глава \d+\.\s*.*|[\d-]+\s*тарау\.\s*.*)'

def get_article_chunks(text: str, language: str) -> List[str]:
    if language == 'rus':
        article_pattern = RUS_ARTICLE_PATTERN
        chapter_pattern = r'^(Глава \d+\.\s*.*)'
    elif language == 'kaz':
        article_pattern = KAZ_ARTICLE_PATTERN
        chapter_pattern = r'^([\d-]+\s*тарау\.\s*.*)'
    else:
        return None
    
    chapters = re.findall(chapter_pattern, text, re.MULTILINE)
    articles = re.findall(article_pattern, text, re.MULTILINE)
    
    if not articles:
        return None
    
    chunks = []
    current_chunk = ""
    
    if chapters:
        current_chunk += chapters[0] + "\n\n"
    
    for line in text.split('\n'):
        article_match = re.match(article_pattern, line)
        if article_match:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            current_chunk = line + "\n"
        else:
            current_chunk += line + "\n"
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

logger.info(f"Initial memory usage: {get_memory_usage():.2f} MB")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True, 'batch_size': EMBEDDING_BATCH_SIZE}
)

default_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=50,
    length_function=len,
)

async def create_mongodb_indexes():
    try:
        indexes = await db.db.target_documents.list_indexes().to_list(length=None)
        existing_indexes = [idx['name'] for idx in indexes]
        
        if "text_index" not in existing_indexes:
            logger.info("Creating text index on target_documents collection")
            await db.db.target_documents.create_index(
                [
                    ("decompressedText", "text"),
                    ("zg", "text"),
                    ("voa", "text")
                ],
                name="text_index"
            )
            logger.info("Text index created successfully")
        else:
            logger.info("Text index already exists")
            
        if "_id_" not in existing_indexes:
            await db.db.target_documents.create_index("_id")
            logger.info("_id index created successfully")
        else:
            logger.info("_id index already exists")
            
    except Exception as e:
        logger.error(f"Error creating MongoDB indexes: {str(e)}")
        logger.error(traceback.format_exc())

async def fetch_target_documents_in_batches(batch_size=DOCUMENT_BATCH_SIZE):
    try:
        if "target_documents" not in await db.db.list_collection_names():
            logger.error("target_documents collection does not exist")
            return
        
        cursor = db.db.target_documents.find({})
        total_docs = await db.db.target_documents.count_documents({})
        logger.info(f"Total target documents to process: {total_docs}")
        
        if total_docs == 0:
            logger.warning("No documents found in target_documents collection")
            return
        
        batch = []
        processed = 0
        
        async for doc in cursor:
            if "decompressedText" in doc and doc["decompressedText"]:
                batch.append({
                    "_id": str(doc["_id"]),
                    "text": doc["decompressedText"],
                    "metadata": {
                        "voa": doc.get("voa", ""),
                        "zg": doc.get("zg", ""),
                        "doc_id": str(doc["_id"]),
                        "lg": doc.get("lg", ""),
                        "ngr": doc.get("ngr", ""),
                        "st": doc.get("st", "")
                    }
                })
            
            if len(batch) >= batch_size:
                yield batch
                processed += len(batch)
                logger.info(f"Processed {processed}/{total_docs} documents ({processed/total_docs*100:.2f}%)")
                batch = []
        
        if batch:
            yield batch
            processed += len(batch)
            logger.info(f"Processed {processed}/{total_docs} documents ({processed/total_docs*100:.2f}%)")
    
    except Exception as e:
        logger.error(f"Error fetching target documents: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def custom_chunk_documents(documents):
    processed_documents = []
    
    for doc in documents:
        text = doc['text']
        metadata = doc['metadata']
        doc_id = doc['_id']
        language = metadata.get('lg', '')
        
        structured_chunks = get_article_chunks(text, language)
        
        if structured_chunks:
            for chunk in structured_chunks:
                processed_documents.append({
                    'text': chunk,
                    'metadata': {**metadata, 'source_type': 'structured'}
                })
        else:
            default_chunks = default_text_splitter.split_text(text)
            for chunk in default_chunks:
                processed_documents.append({
                    'text': chunk,
                    'metadata': {**metadata, 'source_type': 'unstructured'}
                })
        
        if metadata.get('voa'):
            voa_chunks = default_text_splitter.split_text(metadata['voa'])
            for chunk in voa_chunks:
                processed_documents.append({
                    'text': chunk,
                    'metadata': {**metadata, 'source_type': 'voa'}
                })
        
        if metadata.get('zg'):
            zg_chunks = default_text_splitter.split_text(metadata['zg'])
            for chunk in zg_chunks:
                processed_documents.append({
                    'text': chunk,
                    'metadata': {**metadata, 'source_type': 'zg'}
                })
    
    return processed_documents

async def embed_texts(texts, metadatas, batch_size=EMBEDDING_BATCH_SIZE):
    all_texts = []
    all_vectors = []
    all_metadatas = []
    
    for i in range(0, len(texts), batch_size):
        end_idx = min(i + batch_size, len(texts))
        batch_texts = texts[i:end_idx]
        batch_metadatas = metadatas[i:end_idx]
        
        try:
            logger.info(f"Embedding batch {i//batch_size + 1}/{len(texts)//batch_size + 1} ({len(batch_texts)} texts)")
            vectors = await asyncio.to_thread(embeddings.embed_documents, batch_texts)
            
            all_texts.extend(batch_texts)
            all_vectors.extend(vectors)
            all_metadatas.extend(batch_metadatas)
            
            await asyncio.sleep(0.01)
            
        except Exception as e:
            logger.error(f"Error embedding batch starting at index {i}: {str(e)}")
            logger.error(traceback.format_exc())
    
    return all_texts, all_vectors, all_metadatas

async def process_chunks(chunks, metadatas, max_chunk_size=CHUNK_BATCH_SIZE):
    vector_store = None
    
    try:
        all_texts, all_vectors, all_metadatas = await embed_texts(chunks, metadatas, batch_size=max_chunk_size)
        
        if not all_texts or not all_vectors:
            logger.warning("No embedding results were generated")
            return None
        
        logger.info(f"Creating vector store with {len(all_texts)} embeddings")
        logger.info(f"Memory before vector store creation: {get_memory_usage():.2f} MB")
        
        vector_store = FAISS.from_embeddings(
            text_embeddings=list(zip(all_texts, all_vectors)),
            embedding=embeddings,
            metadatas=all_metadatas
        )
        
        logger.info(f"Memory after vector store creation: {get_memory_usage():.2f} MB")
        return vector_store
                
    except Exception as e:
        logger.error(f"Error processing chunk batch: {str(e)}")
        logger.error(traceback.format_exc())
        raise

async def process_target_documents():
    try:
        await create_mongodb_indexes()
        
        vector_store = None
        batch_num = 0
        
        total_docs = await db.db.target_documents.count_documents({})
        
        if total_docs == 0:
            logger.warning("No documents found in target_documents collection")
            return None
            
        progress_bar = tqdm(total=total_docs, desc="Processing target documents")
        
        save_frequency = 2  
        
        async for batch in fetch_target_documents_in_batches(batch_size=DOCUMENT_BATCH_SIZE): 
            batch_num += 1
            batch_start_time = time.time()
            logger.info(f"Processing batch #{batch_num} with {len(batch)} target documents")
            
            chunked_documents = custom_chunk_documents(batch)
            
            texts = [doc['text'] for doc in chunked_documents]
            metadatas = [doc['metadata'] for doc in chunked_documents]
            
            logger.info(f"Chunked batch #{batch_num}: Created {len(texts)} chunks")
            logger.info(f"Memory after chunking: {get_memory_usage():.2f} MB")
            
            try:
                if not texts:
                    logger.warning(f"Batch #{batch_num} produced no chunks, skipping")
                    progress_bar.update(len(batch))
                    continue
                
                batch_vector_store = await process_chunks(texts, metadatas, max_chunk_size=CHUNK_BATCH_SIZE)
                
                if batch_vector_store:
                    if vector_store is None:
                        vector_store = batch_vector_store
                    else:
                        before_merge = get_memory_usage()
                        logger.info(f"Memory before merge: {before_merge:.2f} MB")
                        vector_store.merge_from(batch_vector_store)
                        after_merge = get_memory_usage()
                        logger.info(f"Memory after merge: {after_merge:.2f} MB")
                        logger.info(f"Merge memory delta: {after_merge - before_merge:.2f} MB")
                    
                    del batch_vector_store
                    gc.collect()
                    
                    if batch_num % save_frequency == 0:
                        logger.info(f"Saving checkpoint at batch #{batch_num}")
                        logger.info(f"Memory before saving: {get_memory_usage():.2f} MB")
                        vector_store.save_local(f"target_docs_faiss_index_checkpoint_{batch_num}")
                        logger.info(f"Memory after saving: {get_memory_usage():.2f} MB")
                        logger.info(f"Saved checkpoint at batch #{batch_num}")
                
                batch_duration = time.time() - batch_start_time
                logger.info(f"Batch #{batch_num} completed in {batch_duration:.2f} seconds")
                
                progress_bar.update(len(batch))
                
            except Exception as e:
                logger.error(f"Error creating vectors for batch #{batch_num}: {str(e)}")
                logger.error(traceback.format_exc())
                progress_bar.update(len(batch))
                continue
            
            gc.collect()
        
        progress_bar.close()
        
        if vector_store:
            logger.info("Saving final vector store")
            vector_store.save_local("faiss_index_final")
            logger.info("Final vector store saved to 'faiss_index_final'")
            return vector_store
        else:
            logger.error("No vector store was created!")
            return None
            
    except Exception as e:
        logger.error(f"Error in process_target_documents: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def load_vector_store(path="faiss_index_final"):
    return FAISS.load_local(path, embeddings)

def create_target_retriever(k=3):
    """Create a retriever for the target documents vector store"""
    vector_store = load_vector_store()
    return vector_store.as_retriever(search_kwargs={"k": k})

async def test_search():
    """Test the vector store with a sample query"""
    try:
        vector_store = load_vector_store()
        if not vector_store:
            logger.error("Failed to load vector store for testing")
            return
            
        query = "Қазақстан Республикасы"
        logger.info(f"Testing search with query: '{query}'")
        
        results = await asyncio.to_thread(vector_store.similarity_search, query, k=3)
        
        logger.info(f"Found {len(results)} results for test query")
        for i, doc in enumerate(results):
            logger.info(f"Result {i+1}:")
            logger.info(f"  Content: {doc.page_content[:100]}...")
            logger.info(f"  Metadata: {doc.metadata}")
            logger.info("-" * 40)
            
    except Exception as e:
        logger.error(f"Error testing search: {str(e)}")

async def main():
    try:
        logger.info("Starting target documents vectorization process...")
        logger.info(f"Initial memory usage: {get_memory_usage():.2f} MB")
        start_time = time.time()
        
        vector_store = await process_target_documents()
        
        if vector_store:
            await test_search()
        
        duration = time.time() - start_time
        logger.info(f"Vectorization complete in {duration:.2f} seconds.")
        logger.info(f"Final memory usage: {get_memory_usage():.2f} MB")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unhandled error: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)