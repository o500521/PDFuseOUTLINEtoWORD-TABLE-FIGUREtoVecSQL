import asyncio, traceback
from utility.llm_google import chapter_to_json

MAX_RETRY = 5
RATE_DELAY = 1.0

async def process_chunk(text, ic_model, page, section, doc_title):
    retry = 0
    while retry < MAX_RETRY:
        try:
            await chapter_to_json(text, ic_model, page, section, doc_title)
            return True
        except Exception as e:
            retry += 1
            wait = 2 ** retry
            print(f"⚠️ AI錯誤. {wait} 秒後重試... ({retry}/{MAX_RETRY})")
            print("Error:", e)
            print(traceback.format_exc())
            await asyncio.sleep(wait)
    
    print("❌ 最終失敗, 已寫入 error.log")
    with open("error.log", "a", encoding="utf-8") as f:
        f.write(f"\n-----\nChunk:\n{text[:200]}...\nError:\n{traceback.format_exc()}\n")
    
    return False

async def ingest_chunks(chunks):
    for idx, chunk_dict in enumerate(chunks):
        text = chunk_dict["text"]
        ic_model = chunk_dict.get("ic_model", "Unknown")
        page = chunk_dict.get("page", "Unknown")
        section = chunk_dict.get("section", "Unknown")
        doc_title = chunk_dict.get("title", "Unknown")
        
        print(f"🧠 AI處理塊 {idx+1}/{len(chunks)} : {section}")
        
        await process_chunk(text, ic_model, page, section, doc_title)
        
        await asyncio.sleep(RATE_DELAY)