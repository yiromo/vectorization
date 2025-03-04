import asyncio
from bson.objectid import ObjectId
import zlib
import gzip
import io
from bs4 import BeautifulSoup
import re
from concurrent.futures import ThreadPoolExecutor
from mongo_utils import db
import time

TARGET_IDS = [
    ObjectId('67923a2ed5de6593e98dc92b'),
    ObjectId('678e4b50d5de6593e9032e1c'),
    ObjectId('6787ae52d5de6593e8622612'),
    ObjectId('6785193dd5de6593e80e76e2'),
    ObjectId('678a5307d5de6593e8acd030'),
    ObjectId('678a52f5d5de6593e8accb77'),
    ObjectId('669de330d5de7840c55da19a'),
    ObjectId('66991286d5de7840c4dcec4d'),
    ObjectId('67486895d5def82c2c04b601'),
    ObjectId('6748688bd5def82c2c04b2b4'),
    ObjectId('677bd9efd5de6593e6fe868f'),
    ObjectId('677bd9e0d5de6593e6fe8277'),
    ObjectId('6784e6dad5de6593e8071b51'),
    ObjectId('6785193ed5de6593e80e76fc'),
    ObjectId('677dffc9d5de6593e7539a5d'),
    ObjectId('677a7f74d5de6593e6d63b1a'),
    ObjectId('677bd9f0d5de6593e6fe869e'),
    ObjectId('677a7f76d5de6593e6d63b56'),
    ObjectId('6798d285d5de6f061a757c71'),
    ObjectId('678fa358d5de6593e92c07eb'),
    ObjectId('66cdd46cd5de07b24eb016a8'),
    ObjectId('678a530ad5de6593e8acd074'),
    ObjectId('678a52f2d5de6593e8accacf'),
    ObjectId('66cdd45ed5de07b24eb012de'),
    ObjectId('678a5308d5de6593e8acd035'),
    ObjectId('678a52f5d5de6593e8accb85'),
    ObjectId('67923a2ed5de6593e98dc945'),
    ObjectId('678e4b50d5de6593e9032e24'),
    ObjectId('678119d6d5de6593e7ad925c'),
    ObjectId('678119e3d5de6593e7ad95c7'),
    ObjectId('67923a2fd5de6593e98dc958'),
    ObjectId('678e4b50d5de6593e9032e31'),
    ObjectId('67486898d5def82c2c04b68b'),
    ObjectId('67486889d5def82c2c04b22e'),
    ObjectId('66a2560fd5de7840c5fdb9ca'),
    ObjectId('66a25601d5de7840c5fdb657'),
    ObjectId('6661b757d5dee306ff38e8bc'),
    ObjectId('6661b74cd5dee306ff38e47f')
]

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
        
        # Find document metadata
        meta_doc = await raw_db.doc_meta.find_one({"_id": doc_id})
        if not meta_doc:
            return False, f"TARGET NOT FOUND: No metadata for document {doc_id}"
        
        # Find document data
        data_doc = await raw_db.doc_data.find_one({"_id": doc_id})
        if not data_doc or "compressedText" not in data_doc:
            return False, f"TARGET INCOMPLETE: No compressed text for document {doc_id}"
        
        compressed_data = data_doc.get("compressedText")
        
        # Decompress text in a separate thread
        decompressed_text = await asyncio.get_event_loop().run_in_executor(
            executor, decompress_text_sync, compressed_data)
        
        if decompressed_text:
            # Extract plain text in a separate thread
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