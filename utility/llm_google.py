import google.generativeai as genai
import os, json, re, asyncio
import utility.db
from utility.config import config
from utility.chapter_parser import detect_chapter_title

genai.configure(api_key=config.google_ai_studio_apikey)

def load_prompt():
    with open("prompt_ic.txt", "r", encoding="utf-8") as f:
        return f.read()

def extract_page(text: str):
    match = re.search(r'(\d{1,4})/\d{1,4}', text)
    
    if match:
        return str(match.group(1))
    
    match = re.search(r'Page\s+(\d+)', text, re.IGNORECASE)
    if match:
        return str(match.group(1))
    
    return None

def extract_ic_model(text: str):
    matches = re.findall(r'\b[A-Z][A-Z0-9_-]{3,20}\b', text)

    if matches:
        blacklist = {"PAGE", "TABLE", "FIGURE", "END", "REV", "MCU", "CPU"}
        filtered = [m for m in matches if m not in blacklist]
        if filtered:
            return max(set(filtered), key=filtered.count)

    match = re.search(r'(RM\d{3,5})', text)
    if match:
        return match.group(1)

    match = re.search(r'(STM32[A-Za-z0-9]+)', text)
    if match:
        return match.group(1)

    return None

def extract_section(text: str):
    lines = text.splitlines()

    for line in lines[:3]:
        if "[TABLE" in line:
            return line.strip()

    for line in lines:
        l = line.strip()
        if (
            l 
            and not l[0].isdigit() 
            and not re.search(r'RM\d{3,5}', l)
            and len(l) > 3
        ):
            return l

    return "Unknown Section"


def chapter_to_json(text, ic_model, page, section, document_title):
    chapter_guess = detect_chapter_title(text)
    prompt = load_prompt()
    prompt = (
        prompt.replace("[TEXT_CHUNK]", text)
                .replace("[IC_MODEL]", ic_model)
                .replace("[SOURCE_PAGE]", str(page))
                .replace('[SOURCE_SECTION]', section)
                .replace('[SOURCE_CHAPTER]', chapter_guess or section)
    )
    
    async def run_ai():
        model = genai.GenerativeModel("gemini-2.5-flash")
        resp = await model.generate_content(prompt)
        return resp.text
    
    #生成AI回覆
    try:
        try:
            loop = asyncio.get_running_loop()
            raw = loop.run_until_complete(run_ai())
        except RuntimeError:
            raw = asyncio.run(run_ai())
            
        raw = raw.strip().replace("```json", "").replace("```", "").strip()
        metadata = json.loads(raw)
        
    except Exception as e:
        print(f"⚠️ AI JSON parse fallback: {e}")
        metadata = {
            "ic_model": ic_model,
            "source_page": page,
            "source_section": section,
            "block_type": "paragraph",
            "content_payload": {
                "header": section,
                "content": text
            },
            "tags": []
        }
    
    metadata.setdefault("ic_model", ic_model)
    metadata.setdefault("document_title", document_title)
    metadata.setdefault("chapter_title", chapter_guess or section)
    metadata.setdefault("section_title", section)
    
    utility.db.store_chunk_to_vector_db(text, metadata)
    return metadata
