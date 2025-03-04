import asyncio
from bson.objectid import ObjectId
import zlib
import gzip
import io
from bs4 import BeautifulSoup
import re
from concurrent.futures import ThreadPoolExecutor
from utils.mongo_utils import db
import time

async def get_target_ids():
    raw_db = db.client[db.db.name]
    
    query = {
        "actual": True, 
        "in": "0230", 
        "st": {"$in": ["upd", "new"]}, 
        "lg": {"$in": ["rus", "kaz"]}
    }
    
    cursor = raw_db.doc_meta.find(query).limit(100)
    target_ids = []
    
    async for doc in cursor:
        target_ids.append(doc['_id'])
    
    print(f"Found {len(target_ids)} target documents matching criteria")
    return target_ids

def decompress_text_sync(compressed_data):
    if not compressed_data:
        return None
    
    decompression_methods = [
        lambda data: gzip.GzipFile(fileobj=io.BytesIO(data)).read().decode('utf-8'),
        lambda data: zlib.decompress(data, 15 + 32).decode('utf-8'),
        lambda data: zlib.decompress(data).decode('utf-8'),
        lambda data: zlib.decompress(data[2:]).decode('utf-8')
    ]
    
    for method in decompression_methods:
        try:
            return method(compressed_data)
        except Exception:
            continue
    
    return None

def extract_plain_text_sync(html_content):
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        plain_text = soup.get_text(' ', strip=True)
    except:
        plain_text = re.sub(r'<[^>]+>', ' ', html_content)
        plain_text = plain_text.replace('&nbsp;', ' ')
        plain_text = re.sub(r'&[a-zA-Z]+;', ' ', plain_text)
        plain_text = re.sub(r'\s+', ' ', plain_text).strip()
    
    return plain_text

async def process_target_document(doc_id, raw_db, executor):
    try:
        print(f"Processing target document: {doc_id}")
        
        meta_doc = await raw_db.doc_meta.find_one({"_id": doc_id})
        if not meta_doc:
            return False, f"TARGET NOT FOUND: No metadata for document {doc_id}"
        
        data_doc = await raw_db.doc_data.find_one({"_id": doc_id})
        if not data_doc or "compressedText" not in data_doc:
            return False, f"TARGET INCOMPLETE: No compressed text for document {doc_id}"
        
        compressed_data = data_doc.get("compressedText")
        
        decompressed_text = await asyncio.get_event_loop().run_in_executor(
            executor, decompress_text_sync, compressed_data)
        
        if decompressed_text:
            plain_text = await asyncio.get_event_loop().run_in_executor(
                executor, extract_plain_text_sync, decompressed_text)
            
            new_doc = {
                "_id": doc_id,
                "decompressedText": plain_text,
                "in": meta_doc.get("in"),
                "st": meta_doc.get("st"),
                "voa": meta_doc.get("voa"),
                "zg": meta_doc.get("zg"),
                "lg": meta_doc.get("lg"),
                "ngr": meta_doc.get("ngr"),
                "processed_timestamp": time.time()
            }
            
            await raw_db.target_documents.insert_one(new_doc)
            return True, f"TARGET FOUND AND PROCESSED: {doc_id}"
        else:
            return False, f"TARGET DECOMPRESSION FAILED: Could not decompress data for document {doc_id}"
            
    except Exception as e:
        return False, f"ERROR processing target document {doc_id}: {str(e)}"

async def find_target_documents():
    raw_db = db.client[db.db.name]
    
    TARGET_IDS = await get_target_ids()
    
    if "target_documents" not in await raw_db.list_collection_names():
        await raw_db.create_collection("target_documents")
    
    existing_targets = set()
    cursor = raw_db.target_documents.find({}, {"_id": 1})
    async for doc in cursor:
        existing_targets.add(doc["_id"])
    
    targets_to_process = [doc_id for doc_id in TARGET_IDS if doc_id not in existing_targets]
    
    print(f"Total target documents: {len(TARGET_IDS)}")
    print(f"Already processed: {len(existing_targets)}")
    print(f"Remaining to process: {len(targets_to_process)}")
    
    if not targets_to_process:
        print("All target documents have already been processed.")
        return
    
    max_workers = min(8, len(targets_to_process))
    print(f"Using {max_workers} worker threads")
    
    start_time = time.time()
    found_count = 0
    missing_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        tasks = [process_target_document(doc_id, raw_db, executor) for doc_id in targets_to_process]
        results = await asyncio.gather(*tasks)
        
        for success, message in results:
            print(message)
            if success:
                found_count += 1
            else:
                missing_count += 1
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\nSearch completed!")
    print(f"Found and processed: {found_count} documents")
    print(f"Unable to find or process: {missing_count} documents")
    print(f"Total time: {total_time:.2f} seconds")
    
    print("\nStatus of all target documents:")
    all_processed = set()
    cursor = raw_db.target_documents.find({"_id": {"$in": TARGET_IDS}})
    async for doc in cursor:
        all_processed.add(doc["_id"])
    
    print(f"Total successfully processed: {len(all_processed)}/{len(TARGET_IDS)}")
    
    if len(all_processed) < len(TARGET_IDS):
        print("\nMissing target documents:")
        for doc_id in TARGET_IDS:
            if doc_id not in all_processed:
                print(f"  - {doc_id}")

async def main():
    await find_target_documents()

if __name__ == "__main__":
    asyncio.run(main())