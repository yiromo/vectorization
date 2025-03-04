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
from utils.mongo_utils import db
import logging
import numpy as np
from typing import List, Dict, Any, Tuple
import psutil
import gc
import re
import concurrent.futures
import torch

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
    return process.memory_info().rss / 1024 / 1024 

def calculate_optimal_batch_size():
    total_memory = psutil.virtual_memory().total
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024) if device == 'cuda' else 0
    
    BASE_DOCUMENT_BATCH_SIZE = 50
    BASE_CHUNK_BATCH_SIZE = 50
    BASE_EMBEDDING_BATCH_SIZE = 16
    
    memory_factor = total_memory / (16 * 1024 * 1024 * 1024)  # Base memory 16 GB
    gpu_factor = gpu_memory / (8 * 1024)  # Base GPU memory 8 GB
    
    document_batch_size = int(BASE_DOCUMENT_BATCH_SIZE * memory_factor)
    chunk_batch_size = int(BASE_CHUNK_BATCH_SIZE * memory_factor)
    embedding_batch_size = int(BASE_EMBEDDING_BATCH_SIZE * memory_factor * (gpu_factor if device == 'cuda' else 1))
    
    document_batch_size = max(min(document_batch_size, 200), 10)
    chunk_batch_size = max(min(chunk_batch_size, 100), 10)
    embedding_batch_size = max(min(embedding_batch_size, 64), 8)
    
    logger.info(f"Device: {device}")
    logger.info(f"Optimal batch sizes: "
                f"Documents: {document_batch_size}, "
                f"Chunks: {chunk_batch_size}, "
                f"Embeddings: {embedding_batch_size}")
    
    return document_batch_size, chunk_batch_size, embedding_batch_size, device

DOCUMENT_BATCH_SIZE, CHUNK_BATCH_SIZE, EMBEDDING_BATCH_SIZE, DEVICE = calculate_optimal_batch_size()

logger.info(f"Initial memory usage: {get_memory_usage():.2f} MB")

def create_embeddings(device):
    """
    Create embeddings model with GPU support
    """
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            model_kwargs={
                'device': device
            },
            encode_kwargs={
                'normalize_embeddings': True, 
                'batch_size': EMBEDDING_BATCH_SIZE
            }
        )
        return embeddings
    except Exception as e:
        logger.error(f"Error creating embeddings: {str(e)}")
        raise

embeddings = create_embeddings(DEVICE)

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

default_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=50,
    length_function=len,
)

class DocumentStructureParser:
    STRUCTURE_PATTERNS = {
        'rus': {
            'main_section': r'^(ОСОБЕННАЯ ЧАСТЬ|ОБЩАЯ ЧАСТЬ)',
            'section': r'^РАЗДЕЛ \d+\.\s*(.+)',
            'chapter': r'^Глава \d+\.\s*(.+)',
            'article': r'^Статья \d+\.\s*(.+)',
        },
        'kaz': {
            'main_section': r'^(ЕРЕКШЕ БӨЛІК|ЖАЛПЫ БӨЛІК)',
            'section': r'^(\d+-БӨЛІМ)\.\s*(.+)',
            'chapter': r'^(\d+-тарау)\.\s*(.+)',
            'article': r'^(\*\*\d+-бап)\.\s*(.+)',
        }
    }

    @classmethod
    def parse_document_structure(cls, text: str, language: str) -> Dict[str, Any]:
        if language not in cls.STRUCTURE_PATTERNS:
            logger.warning(f"Unsupported language: {language}")
            return {}

        patterns = cls.STRUCTURE_PATTERNS[language]
        structure = {
            'main_section': None,
            'section': None,
            'chapter': None,
            'articles': []
        }

        main_section_match = re.search(patterns['main_section'], text, re.MULTILINE)
        if main_section_match:
            structure['main_section'] = main_section_match.group(1)

        section_match = re.search(patterns['section'], text, re.MULTILINE)
        if section_match:
            structure['section'] = section_match.group(1)
        chapter_match = re.search(patterns['chapter'], text, re.MULTILINE)
        if chapter_match:
            structure['chapter'] = chapter_match.group(1)

        article_matches = re.finditer(patterns['article'], text, re.MULTILINE)
        for match in article_matches:
            structure['articles'].append({
                'number': match.group(1),
                'title': match.group(2).strip()
            })

        return structure

def custom_chunk_documents(documents):
    processed_documents = []
    
    for doc in documents:
        text = doc['text']
        metadata = doc['metadata']
        language = metadata.get('lg', '')
        
        structure = DocumentStructureParser.parse_document_structure(text, language)
        
        base_metadata = {
            **metadata,
            'main_section': structure.get('main_section'),
            'section': structure.get('section'),
            'chapter': structure.get('chapter')
        }
        
        if structure.get('articles'):
            for article in structure['articles']:
                article_pattern = re.escape(article['number']) + r'\.\s*' + re.escape(article['title'])
                article_match = re.search(article_pattern, text, re.MULTILINE)
                
                if article_match:
                    start_idx = article_match.start()
                    
                    next_article_match = re.search(r'\*\*\d+-бап\.\s*', text[start_idx+1:], re.MULTILINE)
                    if next_article_match:
                        end_idx = start_idx + 1 + next_article_match.start()
                    else:
                        end_idx = len(text)
                    
                    article_text = text[start_idx:end_idx].strip()
                    
                    processed_documents.append({
                        'text': article_text,
                        'metadata': {
                            **base_metadata,
                            'article_number': article['number'],
                            'article_title': article['title'],
                            'source_type': 'structured_article'
                        }
                    })
        
        if not processed_documents:
            default_chunks = default_text_splitter.split_text(text)
            
            for chunk in default_chunks:
                processed_documents.append({
                    'text': chunk,
                    'metadata': {
                        **base_metadata,
                        'source_type': 'unstructured'
                    }
                })
    
    return processed_documents

def manage_gpu_memory():
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()


async def embed_texts_parallel(texts, metadatas, batch_size=EMBEDDING_BATCH_SIZE, max_workers=None):
    if max_workers is None:
        max_workers = min(len(texts), os.cpu_count() * 2)
    
    all_texts = []
    all_vectors = []
    all_metadatas = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(0, len(texts), batch_size):
            end_idx = min(i + batch_size, len(texts))
            batch_texts = texts[i:end_idx]
            batch_metadatas = metadatas[i:end_idx]
            
            future = executor.submit(embeddings.embed_documents, batch_texts)
            futures.append((future, batch_texts, batch_metadatas))
        
        for future, batch_texts, batch_metadatas in futures:
            try:
                vectors = future.result()
                all_texts.extend(batch_texts)
                all_vectors.extend(vectors)
                all_metadatas.extend(batch_metadatas)
            except Exception as e:
                logger.error(f"Error embedding batch: {str(e)}")
                
                if DEVICE == 'cuda':
                    torch.cuda.empty_cache()
    
    return all_texts, all_vectors, all_metadatas


async def process_chunks(chunks, metadatas, max_chunk_size=CHUNK_BATCH_SIZE):
    vector_store = None
    
    try:
        all_texts, all_vectors, all_metadatas = await embed_texts_parallel(chunks, metadatas, batch_size=max_chunk_size)
        
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

async def process_target_documents():
    try:
        await create_mongodb_indexes()
        
        vector_store = None
        total_docs = await db.db.target_documents.count_documents({})
        
        if total_docs == 0:
            logger.warning("No documents found in target_documents collection")
            return None
            
        progress_bar = tqdm(total=total_docs, desc="Processing target documents")
        
        async def process_batch(batch):
            try:
                batch_start_time = time.time()
                
                chunked_documents = custom_chunk_documents(batch)
                texts = [doc['text'] for doc in chunked_documents]
                metadatas = [doc['metadata'] for doc in chunked_documents]
                
                if not texts:
                    logger.warning(f"Batch produced no chunks, skipping")
                    return None
                
                batch_vector_store = await process_chunks(texts, metadatas, max_chunk_size=CHUNK_BATCH_SIZE)
                
                progress_bar.update(len(batch))
                
                manage_gpu_memory()
                
                return {
                    'vector_store': batch_vector_store, 
                    'batch_duration': time.time() - batch_start_time
                }
            except Exception as e:
                logger.error(f"Error processing batch: {str(e)}")
                progress_bar.update(len(batch))
                
                manage_gpu_memory()
                
                return None
        
        
    except Exception as e:
        logger.error(f"Error in process_target_documents: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def load_vector_store(path="faiss_index_final"):
    return FAISS.load_local(path, embeddings)

def create_target_retriever(k=3):
    vector_store = load_vector_store()
    return vector_store.as_retriever(search_kwargs={"k": k})

async def test_search():
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
        logger.info(f"Starting target documents vectorization process on {DEVICE}...")
        logger.info(f"Initial memory usage: {get_memory_usage():.2f} MB")
        start_time = time.time()
        
        vector_store = await process_target_documents()
        
        if vector_store:
            await test_search()
        
        duration = time.time() - start_time
        logger.info(f"Vectorization complete in {duration:.2f} seconds.")
        logger.info(f"Final memory usage: {get_memory_usage():.2f} MB")
        
        manage_gpu_memory()
        
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