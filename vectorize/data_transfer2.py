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
import sys
import logging

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s: %(message)s')

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
        except Exception as e:
            logging.debug(f"Decompression method failed: {e}")
            continue

    return None

def extract_plain_text_sync(html_content):
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        plain_text = soup.get_text(' ', strip=True)
        
        if not plain_text:
            raise ValueError("Empty text from BeautifulSoup")
        
        return plain_text
    except Exception:
        plain_text = re.sub(r'<[^>]+>', ' ', html_content)
        plain_text = plain_text.replace('&nbsp;', ' ')
        plain_text = re.sub(r'&[a-zA-Z]+;', ' ', plain_text)
        plain_text = re.sub(r'\s+', ' ', plain_text).strip()

    return plain_text

async def get_target_ids():
    """
    Retrieve target document IDs based on specific criteria
    """
    raw_db = db.client[db.db.name]

    query = {
        "actual": True,
        "in": {"$in": ["0100", "0230", "0300", "0400"]},
        "st": {"$in": ["upd", "new"]},
        "lg": {"$in": ["rus", "kaz"]}
    }

    start_time = time.time()
    
    cursor = raw_db.doc_meta.find(query)
    target_ids = []

    doc_count = 0
    async for doc in cursor:
        target_ids.append(doc['_id'])
        doc_count += 1
        
        if doc_count % 500 == 0:
            logging.info(f"Retrieved {doc_count} document IDs...")

    retrieval_time = time.time() - start_time
    logging.info(f"ID Retrieval Time: {retrieval_time:.2f} seconds")
    logging.info(f"Total Documents Retrieved: {len(target_ids)}")

    return target_ids

async def find_missing_document_ids(raw_db, all_target_ids):
    existing_targets = set()
    cursor = raw_db.target_documents.find({}, {"_id": 1})
    async for doc in cursor:
        existing_targets.add(doc["_id"])

    missing_ids = [doc_id for doc_id in all_target_ids if doc_id not in existing_targets]

    logging.info(f"Total target documents: {len(all_target_ids)}")
    logging.info(f"Already processed: {len(existing_targets)}")
    logging.info(f"Remaining to process: {len(missing_ids)}")

    logging.info("\nMissing Document IDs:")
    for i in range(0, len(missing_ids), 50):
        # Convert ObjectId to string explicitly
        chunk = [str(doc_id) for doc_id in missing_ids[i:i+50]]
        logging.info(", ".join(chunk))

    with open('missing_document_ids.txt', 'w') as f:
        for doc_id in missing_ids:
            f.write(f"{doc_id}\n")
    logging.info(f"\nFull list of missing IDs written to 'missing_document_ids.txt'")

    return missing_ids

async def process_target_document(doc_id, raw_db, executor, total_docs, progress_queue):
    start_doc_time = time.time()
    try:
        meta_doc = await raw_db.doc_meta.find_one({"_id": doc_id})
        data_doc = await raw_db.doc_data.find_one({"_id": doc_id})

        if not meta_doc or not data_doc or "compressedText" not in data_doc:
            await progress_queue.put(1)
            return False, f"INCOMPLETE: Document {doc_id}"

        decompressed_text = await asyncio.get_event_loop().run_in_executor(
            executor, decompress_text_sync, data_doc.get("compressedText"))

        if not decompressed_text:
            await progress_queue.put(1)
            return False, f"DECOMPRESSION FAILED: {doc_id}"

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

        await progress_queue.put(1)
        return True, f"PROCESSED: {doc_id}"

    except Exception as e:
        await progress_queue.put(1)
        logging.error(f"ERROR processing {doc_id}: {e}")
        return False, f"ERROR: {doc_id}"

async def progress_tracker(total_docs, progress_queue):
    processed = 0
    start_time = time.time()
    while processed < total_docs:
        count = await progress_queue.get()
        processed += count
        percentage = (processed / total_docs) * 100
        elapsed = time.time() - start_time
        est_total_time = (elapsed / processed) * total_docs if processed > 0 else 0
        est_remaining = max(0, est_total_time - elapsed)

        sys.stdout.write(
            f"\rProgress: {processed}/{total_docs} ({percentage:.2f}%) | "
            f"Elapsed: {elapsed:.2f}s | Est. Remaining: {est_remaining:.2f}s"
        )
        sys.stdout.flush()

        if processed >= total_docs:
            break

async def process_missing_documents(raw_db, missing_ids):
    if not missing_ids:
        logging.info("No documents to process.")
        return

    progress_queue = asyncio.Queue()

    max_workers = min(8, len(missing_ids))
    logging.info(f"Using {max_workers} worker threads")

    progress_task = asyncio.create_task(progress_tracker(len(missing_ids), progress_queue))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        tasks = [process_target_document(doc_id, raw_db, executor, 
                                         len(missing_ids), progress_queue) 
                 for doc_id in missing_ids]
        results = await asyncio.gather(*tasks)

    await progress_task

    found_count = sum(1 for success, _ in results if success)
    missing_count = sum(1 for success, _ in results if not success)

    logging.info("\nSearch completed!")
    logging.info(f"Found and processed: {found_count} documents")
    logging.info(f"Unable to find or process: {missing_count} documents")

async def main():
    raw_db = db.client[db.db.name]

    if "target_documents" not in await raw_db.list_collection_names():
        await raw_db.create_collection("target_documents")

    all_target_ids = await get_target_ids()

    missing_ids = await find_missing_document_ids(raw_db, all_target_ids)

    if missing_ids:
        await process_missing_documents(raw_db, missing_ids)
    else:
        logging.info("All documents have already been processed.")

if __name__ == "__main__":
    asyncio.run(main())