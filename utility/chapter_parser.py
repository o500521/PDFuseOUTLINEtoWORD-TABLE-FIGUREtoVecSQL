import re

chapter_pattern = re.compile(r'^\s*(\d+(\.\d+)*)\s+[A-Z][A-Za-z0-9\-/,() ]+')

def detect_chapter_title(text: str):
    """自動偵測 datasheet 章節標題"""
    lines = text.splitlines()
    for line in lines[:5]:
        if chapter_pattern.match(line.strip()):
            return line.strip()
        
    return None