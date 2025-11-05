import tkinter as tk
from tkinter import filedialog
import asyncio
import aiofiles

from utility.llm_google import extract_ic_model, extract_page, extract_section
from utility.chapter_parser import detect_chapter_title
from utility.ingest_queue import ingest_chunks

async def read_txt_chunk(filepath):
    with open(filepath, "r", encoding="utf-8-sig") as f:
        buffer = []
        async with aiofiles.open(filepath, "r", encoding="utf-8-sig") as f:
            async for line in f:
                if line.strip() == "" and buffer:
                    yield "\n".join(buffer).strip()
                    buffer = []
                else:
                    buffer.append(line)
        
            if buffer:
                yield "\n".join(buffer).strip()

async def proceess_file(file_path):
    print(f"✅ 已選擇檔案: {file_path}")
    print("🚀 開始分析 TXT 並建立 chunk list...\n")
    
    chunks = []
    ic_file_title = file_path.split("/")[-1]
    
    async for chunk in read_txt_chunk(file_path):
        if not chunk.strip():
            continue
        
        ic_model = extract_ic_model(chunk) or "Unknown"
        page = extract_page(chunk) or "Unknown"
        section = extract_section(chunk) or "Unknown"
        chapter = detect_chapter_title(chunk) or section
        
        chunks.append({
            "text": chunk,
            "ic_model": ic_model,
            "page": page,
            "section": section,
            "chapter": chapter,
            "title": ic_file_title
        })
        
        print(f"📦 解析完成, 共 {len(chunks)} 個 chunks, 開始送入 AI & 資料庫 ...\n")
        await ingest_chunks(chunks)

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    
    print("📂 請選擇要匯入的 TXT 檔案 ...")
    file_path = filedialog.askopenfilename(title="選擇 TXT 檔案", filetypes=[("Text Files", "*.txt")])
    
    if not file_path:
        print("❌ 未選擇檔案, 程式已退出")
        exit()
    
    asyncio.run(proceess_file(file_path))
    