# ==============================================================================
#      PDF 章節結構化擷取與輸出
# ==============================================================================

# === 匯入必要套件 ===
import os
import re
import json
import cv2
import fitz # PyMuPDF
import time
import camelot
import asyncio
import numpy as np
import pandas as pd
from PIL import Image
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from utility.llm_google import chapter_to_json, extract_ic_model, extract_section
from utility.ingest_queue import ingest_chunks

# GPU 功能已完全移除

# === 多執行緒設定 ===
def get_optimal_workers():
    """
    自動計算最佳執行緒數量
    """
    cpu_count = os.cpu_count() or 2

    # CPU 模式：使用 75% 的執行緒數，保留一些給系統
    optimal = max(4, int(cpu_count * 0.75))
    print(f"🔧 偵測到 {cpu_count} 個邏輯處理器（執行緒）")
    print(f"💻 CPU 模式：使用 {optimal} 個執行緒（約 {optimal/cpu_count*100:.0f}% 使用率）")

    return optimal

# 全域變數：用於追蹤已儲存圖片的內容（Hash值 -> 原始檔名/路徑）
IMAGE_HASH_CACHE = {}

# ==============================================================================
# === 預編譯正則表達式 (效能優化) ===
# ==============================================================================

# 目錄解析相關
CATALOG_LINE_PATTERN_DOTS = re.compile(
    r"^(.+?)\s*\.{2,}\s*([0-9]+|[IVXLCDMivxlcdm]+|[零一二三四五六七八九十百千萬]+)\s*$"
)
CATALOG_LINE_PATTERN_SPACE = re.compile(
    r"^(.+?)\s+([0-9]+|[IVXLCDMivxlcdm]+|[零一二三四五六七八九十百千萬]+)\s*$"
)
CATALOG_CHAPTER_PATTERN = re.compile(
    r"(第[一二三四五六七八九十百]+章)|(\.{3,}\s*\d+)"
)

# 圖表標題檢測 (最高頻，影響最大)
FIGURE_TABLE_TITLE_PATTERN = re.compile(
    r"^(圖|图|Figure|Fig\.|圖例|图例|表|Tab\.|Table|図|表|그림)\s*[\.0-9一二三四五六七八九十百]+",
    re.IGNORECASE
)

# 文字清理相關
TRAILING_PUNCTUATION_PATTERN = re.compile(
    r'[\s\.]*([\.，。；、:：!！?？]+|\.{3,})\s*$'
)
UNSAFE_FILENAME_CHARS_PATTERN = re.compile(r'[\\/:*?"<>|]+')
WHITESPACE_PATTERN = re.compile(r'\s+')

# 章節偵測相關
CHAPTER_NUMBER_PREFIX_PATTERN = re.compile(
    r"^\s*[0-9]{1,2}(\.[0-9]{1,2}){0,2}\s*"
)
CHAPTER_PREFIX_PATTERN = re.compile(
    r"^(圖|Figure|圖例|表|Table|章|節)\s*"
)
CHAPTER_REGEX_STRONG = re.compile(
    r"(第\s*[0-9一二三四五六七八九十百]+\s*章)"
)


# 修正方案：將 MockFiles 定義在 try/except 之外
class MockFiles:
    """用於模擬非 Colab 環境的 files.upload() 行為"""
    def upload(self):
        print("請在本地環境手動將 PDF 檔案放在與腳本相同的目錄，並修改 pdf_path 變數。")
        return {"dummy.pdf": None}


# 嘗試在 Colab 環境匯入 files，若失敗則使用 MockFiles
try:
    from google.colab import files # type: ignore
    IS_COLAB = True
except ImportError:
    files = MockFiles() # 在非 Colab 環境中使用 Mock 實例
    IS_COLAB = False

# 預設的鄰近距離（像素），用於判斷文字和圖形是否「緊鄰」
NEIGHBOR_GAP_PX = 30
OUT_CHAPTERS_DIR = "structured_chapters_final"
INDEX_FILENAME = "chapters_index.json"
RAW_OUTLINE_FILENAME = "chapters_raw_outline.json"

# ==============================================================================
# === 輔助函式：數字與邊界處理 (保留) ===
# ==============================================================================

def chinese_to_arabic(chinese_num_str):
    """將中文數字字串轉換為阿拉伯數字。"""
    conversion_map = {
        '零': 0, '一': 1, '二': 2, '三': 3, '四': 4,
        '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
        '十': 10
    }
    total = 0
    if not chinese_num_str:
        return None

    if '十' in chinese_num_str:
        parts = chinese_num_str.split('十')
        if not parts[0]:
            total += 10
        else:
            total += conversion_map.get(parts[0], 0) * 10
        if len(parts) > 1 and parts[1]:
            total += conversion_map.get(parts[1], 0)
    else:
        total = conversion_map.get(chinese_num_str, None)

    return total

def get_iou(rect1, rect2):
    """計算 IoU (Intersection over Union)"""
    x1, y1, w1, h1 = rect1; x2, y2, w2, h2 = rect2
    xA = max(x1, x2); yA = max(y1, y2); xB = min(x1 + w1, x2 + w2); yB = min(y1 + h1, y2 + h2)
    inter_width = max(0, xB - xA); inter_height = max(0, yB - yA)
    intersection_area = inter_width * inter_height
    if intersection_area == 0: return 0.0
    area1 = w1 * h1; area2 = w2 * h2
    return intersection_area / (area1 + area2 - intersection_area)

def has_overlap(rect1, rect2):
    """
    檢查兩個矩形是否有重疊（效能優化版本）
    比 get_iou() > 0.0 更快，因為不計算面積和比例，只檢查是否有交集
    """
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    xA = max(x1, x2)
    yA = max(y1, y2)
    xB = min(x1 + w1, x2 + w2)
    yB = min(y1 + h1, y2 + h2)
    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    return inter_width > 0 and inter_height > 0

def filter_contained_figures(bboxes, overlap_threshold=0.9):
    """圖形包含過濾"""
    if not bboxes: return []
    kept_indices = list(range(len(bboxes)))
    for i in range(len(bboxes)):
        if i not in kept_indices: continue
        rect1 = bboxes[i]; x1, y1, w1, h1 = rect1; area1 = w1 * h1
        for j in range(len(bboxes)):
            if i == j or j not in kept_indices: continue
            rect2 = bboxes[j]; x2, y2, w2, h2 = rect2; area2 = w2 * h2
            if area1 == 0 or area2 == 0: continue
            xA = max(x1, x2); yA = max(y1, y2); xB = min(x1 + w1, x2 + w2); yB = min(y1 + h1, y2 + h2)
            intersection_area = max(0, xB - xA) * max(0, yB - yA)
            if intersection_area == 0: continue
            overlap_ratio_i_in_j = intersection_area / area1
            overlap_ratio_j_in_i = intersection_area / area2
            if overlap_ratio_i_in_j >= overlap_threshold and area1 < area2:
                if i in kept_indices: kept_indices.remove(i); break
            elif overlap_ratio_j_in_i >= overlap_threshold and area2 < area1:
                if j in kept_indices: kept_indices.remove(j)
    return [bboxes[i] for i in sorted(list(set(kept_indices)))]

def merge_overlapping_bboxes(bboxes):
    """邊界框擴展合併"""
    if not bboxes: return []
    boxes = [[x, y, x + w, y + h] for x, y, w, h in bboxes]
    merged_boxes = []
    while boxes:
        current_box = boxes.pop(0)
        should_restart = True
        while should_restart:
            should_restart = False
            indices_to_merge = []
            for i in range(len(boxes)):
                other_box = boxes[i]
                xA = max(current_box[0], other_box[0]); yA = max(current_box[1], other_box[1])
                xB = min(current_box[2], other_box[2]); yB = min(current_box[3], other_box[3])
                if max(0, xB - xA) * max(0, yB - yA) > 0:
                    current_box[0] = min(current_box[0], other_box[0]); current_box[1] = min(current_box[1], other_box[1])
                    current_box[2] = max(current_box[2], other_box[2]); current_box[3] = max(current_box[3], other_box[3])
                    indices_to_merge.append(i); should_restart = True
            for i in sorted(indices_to_merge, reverse=True): del boxes[i]
        merged_boxes.append(current_box)
    return [(x0, y0, x1 - x0, y1 - y0) for x0, y0, x1, y1 in merged_boxes]


# ==============================================================================
# === 輔助函式：創建安全檔名 ===
# ==============================================================================
def create_safe_filename(text, max_len=50, extension=""):
    """
    將標題文字轉換為安全、簡短且符合檔案系統規則的部分名稱。
    保留完整標題（包含 Figure/Table 編號）。
    """
    if not text:
        return "untitled"

    # 1. 保留完整標題，不移除 Figure/Table 前綴
    safe_text = text.strip()

    # 2. 移除結尾的省略號、冒號、句號等標點
    safe_text = TRAILING_PUNCTUATION_PATTERN.sub('', safe_text).strip()

    # 3. 移除所有不安全的字符 (保留，用於檔案名清理)
    safe_text = UNSAFE_FILENAME_CHARS_PATTERN.sub('', safe_text)

    # 4. 將空格替換為底線
    safe_text = WHITESPACE_PATTERN.sub('_', safe_text)

    # 5. 如果過長則截斷
    safe_text = safe_text[:max_len]

    if not safe_text:
        return "untitled"

    # 6. 確保結尾沒有多餘的底線
    safe_text = safe_text.rstrip('_')

    return safe_text


# ==============================================================================
# *** dHash 圖片雜湊函式 ***
# ==============================================================================

def dhash(image, hash_size=8):
    """
    計算圖片的 Difference Hash (dHash)。
    """
    # 轉換為灰度圖 (如果還不是)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # 縮放圖片到 (hash_size + 1, hash_size)
    resized = cv2.resize(gray, (hash_size + 1, hash_size))

    # CPU 版本
    diff = resized[:, 1:] > resized[:, :-1]
    return ''.join(str(int(b)) for b in diff.flatten())


def dhash_batch(images, hash_size=8):
    """
    批次計算多張圖片的 dHash

    參數：
        images: list of np.ndarray，每張圖片可以是 BGR 或灰階
        hash_size: hash 大小（預設 8）

    傳回：
        list of str，每張圖片的 hash 值（順序與輸入一致）
    """
    if not images:
        return []

    # 如果只有 1 張圖，直接呼叫單張版本
    if len(images) == 1:
        return [dhash(images[0], hash_size)]

    # 預處理：轉灰階 + 縮放
    resized_batch = []
    for img in images:
        # 轉灰階
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        # 縮放
        resized = cv2.resize(gray, (hash_size + 1, hash_size))
        resized_batch.append(resized)

    # 堆疊成 (N, H, W) 的批次陣列
    resized_batch = np.array(resized_batch, dtype=np.uint8)

    # CPU 批次處理
    hashes = []
    for i in range(len(images)):
        diff = resized_batch[i, :, 1:] > resized_batch[i, :, :-1]
        hash_value = ''.join(str(int(b)) for b in diff.flatten())
        hashes.append(hash_value)

    return hashes


def rgb_to_gray_batch(images):
    """
    批次將 RGB/BGR 圖片轉換為灰階

    參數：
        images: list of np.ndarray，BGR 圖片

    傳回：
        list of np.ndarray，灰階圖片（順序與輸入一致）
    """
    if not images:
        return []

    # CPU 版本
    return [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]


# ==============================================================================
# === 結構化擷取與目錄解析函式 (傳入 doc) ===
# ==============================================================================

def extract_text_blocks_from_page(page):
    """從指定的 PDF 頁面擷取文字區塊 (0-based page_num)。"""
    try:
        return page.get_text('blocks')
    except Exception as e:
        return []

def find_catalog_pages(doc):
    """尋找 PDF 檔案中可能的目錄頁面。"""
    catalog_pages = []
    found_catalog_start = False
    max_pages_to_check = min(len(doc), 15)

    # 設置標準化的英文關鍵字（都用大寫）
    EN_CATALOG_KEYWORDS = ["CONTENTS", "TABLE OF CONTENTS", "INDEX"]

    for i in range(max_pages_to_check):
        page = doc[i]
        # 假設 extract_text_blocks_from_page 已經定義
        text_blocks = extract_text_blocks_from_page(page)

        # 獲取頁面全部文字
        text = "".join([block[4] for block in text_blocks])

        # 核心優化：將文本轉為大寫並將多個空白字元標準化為單一空格
        # 這能解決大小寫問題和"Table of \n contents"的換行問題
        text_normalized = " ".join(text.upper().split())

        is_catalog = False

        # 1. 關鍵字匹配 (中/英文)
        if "目錄" in text_normalized:
            is_catalog = True
        # 檢查英文關鍵字 (使用標準化後的大寫文本)
        elif any(keyword in text_normalized for keyword in EN_CATALOG_KEYWORDS):
            is_catalog = True

        # 2. 正則表達式匹配 (如：點點點 + 頁碼)
        # 即使沒有明確的標題，也可以透過排版特徵來判斷
        elif CATALOG_CHAPTER_PATTERN.search(text):
            is_catalog = True

        if is_catalog:
            catalog_pages.append((i, text))
            found_catalog_start = True
        else:
            # 如果已經開始找到目錄頁，但當前頁面不再是目錄，且內容長度超過一定限制，則停止搜尋
            if found_catalog_start and text and len(text) > 100:
                break

    return catalog_pages

def parse_catalog_text(text):
    """
    從目錄文字中解析章節標題和頁碼。
    """
    chapters = []
    if not text:
        return chapters

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        m = CATALOG_LINE_PATTERN_DOTS.search(line)
        if not m:
            m = CATALOG_LINE_PATTERN_SPACE.search(line)

        if m:
            title_raw = m.group(1).strip()
            page_raw = m.group(2).strip()

            title = TRAILING_PUNCTUATION_PATTERN.sub('', title_raw).strip()

            page_int = None
            try:
                page_int = int(page_raw)
            except ValueError:
                page_int = chinese_to_arabic(page_raw)
            except Exception as e:
                pass
            if title and page_int is not None and len(title) > 2 and not title.isnumeric():
                chapters.append({"title": title, "page": page_int})
    return chapters

def find_y_coord_for_title(page, title_to_find, search_blocks):
    """在單頁中尋找特定標題的 Y 座標 (y0)。找到多個匹配時，選擇最上方的（Y 座標最小）。"""
    search_title = title_to_find.strip()

    if len(search_title) > 30:
        search_title = search_title[:30]

    # 正規化：移除多餘空白和換行，方便匹配
    search_normalized = " ".join(search_title.split())

    # 收集所有匹配的 Y 座標
    matched_y_coords = []

    # 第一輪：精確匹配（含長度限制）
    for b in search_blocks:
        x0, y0, x1, y1, text = b[:5]
        text_normalized = " ".join(text.split())

        if search_normalized in text_normalized and len(text.strip()) < len(search_title) + 50:
            matched_y_coords.append(y0)

    # 如果第一輪找到匹配，返回最小的 Y 座標（最上方）
    if matched_y_coords:
        return min(matched_y_coords)

    # 第二輪：不區分大小寫的匹配
    search_title_lower = search_normalized.lower()
    for b in search_blocks:
        x0, y0, x1, y1, text = b[:5]
        text_normalized = " ".join(text.split()).lower()

        if search_title_lower in text_normalized:
            matched_y_coords.append(y0)

    # 返回最小的 Y 座標（最上方），如果沒找到則返回 None
    return min(matched_y_coords) if matched_y_coords else None

def find_outline(doc):
    """嘗試從 PDF 的 Outline (書籤) 抓取章節結構。"""
    chapters = []
    try:
        toc = doc.get_toc(simple=True)  # [level, title, page]
        for level, title, page in toc:
            if title and page > 0:
                cleaned_title = TRAILING_PUNCTUATION_PATTERN.sub('', title).strip()
                chapters.append({"title": cleaned_title, "page": page})
    except Exception as e:
        print(f"⚠️ 偵測 Outline 發生錯誤：{e}")
    return chapters

# ==============================================================================
# === 核心輔助函式：修正後的標題偵測 (New/Modified) ===
# ==============================================================================

def find_figure_titles_from_reserved_blocks(
    potential_title_blocks, figure_blocks_px, scale_x, scale_y, page_height
):
    """
    從預先標記的「潛在標題」區塊中，尋找最靠近每個圖形邊界框的文字，
    並將其標記為圖形標題。
    """
    figure_titles_map = {}

    for i, fig_rect_px in enumerate(figure_blocks_px):
        fx, fy, fw, fh = fig_rect_px

        # 將圖形邊界框轉換為 fitz 座標
        fx0_fitz = fx / scale_x; fy0_fitz = fy / scale_y
        fx1_fitz = (fx + fw) / scale_x; fy1_fitz = (fy + fh) / scale_y

        best_title = None
        min_gap = float('inf')
        best_bbox = None

        for b in potential_title_blocks:
            tx0, ty0, tx1, ty1, text = b[:5]
            text = text.strip()

            # 檢查文字區塊是否在圖形上方或下方緊鄰，或重疊
            gap_above = fy0_fitz - ty1
            gap_below = ty0 - fy1_fitz

            is_close = False

            # 判斷是否重疊 (重疊即為緊密關聯)
            if max(0, min(fx1_fitz, tx1) - max(fx0_fitz, tx0)) * max(0, min(fy1_fitz, ty1) - max(fy0_fitz, ty0)) > 0:
                 is_close = True

            # 判斷是否緊鄰且對齊
            elif (0 <= gap_above <= NEIGHBOR_GAP_PX or 0 <= gap_below <= NEIGHBOR_GAP_PX):
                horizontal_overlap = max(0, min(fx1_fitz, tx1) - max(fx0_fitz, tx0))
                if horizontal_overlap > 0 or abs((fx0_fitz + fx1_fitz) / 2 - (tx0 + tx1) / 2) < NEIGHBOR_GAP_PX:
                    is_close = True

            if is_close:
                # 判斷誰最近 (絕對值距離，重疊時距離為 0)
                current_gap = 0
                if gap_above >= 0: current_gap = min(current_gap, gap_above)
                if gap_below >= 0: current_gap = min(current_gap, gap_below)

                if current_gap < min_gap:
                    min_gap = current_gap
                    best_title = text
                    best_bbox = (tx0, ty0, tx1, ty1)

        if best_title:
            figure_titles_map[f"figure_{i+1}"] = {
                "content": best_title,
                "bbox": best_bbox
            }

    return figure_titles_map

def process_single_page_and_get_items(
    page, page_height, chapter_assets_dir, chapter_safe_name,
    start_y_coordinate=None, end_y_coordinate=None, dpi=200, debug_draw=True,
    camelot_tables=[], chapter_image_cache=None, cache_lock=None
):
    """
    處理單頁 PDF，擷取文字、表格、圖形，並回傳排序後的項目列表。
    關鍵修正：在圖形合併前，先隔離標題文字。
    """
    global IMAGE_HASH_CACHE

    # 使用傳入的章節專屬快取，如果沒有則使用全域快取
    if chapter_image_cache is None:
        image_cache = IMAGE_HASH_CACHE
        lock = threading.Lock()  # 全域快取使用自己的鎖
    else:
        image_cache = chapter_image_cache
        lock = cache_lock if cache_lock else threading.Lock()

    page_num = page.number
    page_width = page.rect.width
    items_out = []
    successful_table_bboxes_fitz = []
    figure_blocks_px = []
    text_blocks_fitz = []
    chapter_title_bbox_fitz = None
    occupied_text_bboxes_fitz = [] # 儲存所有非正文文字 (表格標題、圖形標題) 的邊界框

    pix = page.get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    if pix.n == 4: img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3: img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    img_height, img_width, _ = img.shape
    scale_x = img_width / page_width
    scale_y = img_height / page_height

    # --- [1/6] 偵測與隔離標題文字區塊 (FIT) ---
    text_blocks_raw = page.get_text("blocks")
    potential_figure_titles_blocks = []

    for b in text_blocks_raw:
        x0, y0, x1, y1, text = b[:5]
        text = text.strip()
        if not text:
            continue

        # 檢查是否為章節標題
        if start_y_coordinate is not None and abs(y0 - start_y_coordinate) < 5:
            chapter_title_bbox_fitz = (x0, y0, x1, y1)

        # 檢查是否為潛在的圖表標題（即使被畫進圖形，也先保留）
        # 支援：繁體中文、簡體中文、英文、日文、韓文等
        if FIGURE_TABLE_TITLE_PATTERN.match(text):
            # 將它標記為潛在標題
            potential_figure_titles_blocks.append(b)
            occupied_text_bboxes_fitz.append((x0, y0, x1, y1))
            continue # 標題文字不進入常規 text_blocks_fitz

        # 過濾頁眉頁腳，非標題文字才會被處理
        if start_y_coordinate is not None and y1 < start_y_coordinate:
            continue
        if end_y_coordinate is not None and y0 >= end_y_coordinate:
            continue

        text_blocks_fitz.append(b)

    # --- [2/6] 處理表格 ---
    detected_tables_info = []
    for i, t in enumerate(camelot_tables):
        x0, y0, x1, y1 = map(float, t._bbox)
        table_top_y = page_height - y1
        table_bottom_y = page_height - y0

        is_valid_table = True
        # 修正：如果表格的底部（最下方）在章節開始座標之前，則排除整個表格
        # 表格的任何部分都不應該在章節開始之前
        if start_y_coordinate is not None and table_bottom_y < start_y_coordinate:
            is_valid_table = False
        # 如果表格的頂部（最上方）在章節結束座標之後，則排除
        if end_y_coordinate is not None and table_top_y > end_y_coordinate:
            is_valid_table = False
        if not is_valid_table:
            continue

        area = (x1 - x0) * (y1 - y0)
        df = t.df
        is_valid_csv = not (df.shape[0] < 2 or df.shape[1] < 2 or (df.replace("", pd.NA).isna().sum().sum() / max(1, df.size) > 0.5))

        if is_valid_csv:
            detected_tables_info.append({
                "camelot_table": t, "bbox": (x0, y0, x1, y1), "area": area, "name": f"table_{i+1}"
            })

    filtered_tables_info = []
    for t_info in detected_tables_info:
        t = t_info["camelot_table"]
        x0_c, y0_c, x1_c, y1_c = t_info["bbox"]
        name = t_info["name"]
        y_raw_a = page_height - y0_c
        y_raw_b = page_height - y1_c

        x0_fitz = max(0, x0_c)
        x1_fitz = min(page_width, x1_c)
        y0_fitz = max(0, min(page_height, y_raw_b))
        y1_fitz = max(0, min(page_height, y_raw_a))
        table_height = y1_fitz - y0_fitz
        if table_height <= 1.0: continue

        if end_y_coordinate is not None and y0_fitz >= end_y_coordinate:
             continue

        filtered_tables_info.append(t_info)
        successful_table_bboxes_fitz.append((x0_fitz, y0_fitz, x1_fitz, y1_fitz, name))
        occupied_text_bboxes_fitz.append((x0_fitz, y0_fitz, x1_fitz, y1_fitz)) # 將表格邊界加入佔用列表

    for t_info in filtered_tables_info:
        t = t_info["camelot_table"]
        x0_c, y0_c, x1_c, y1_c = t_info["bbox"]
        y_center = (page_height - y1_c + page_height - y0_c) / 2
        items_out.append({
            "type": "table", "y_center": y_center, "content": t.df, "name": t_info['name'], "mode": "csv", "page_num": page_num + 1
        })


    # --- [3/6] 偵測與合併圖形 ---
    img_for_opencv = img.copy()
    if successful_table_bboxes_fitz:
        for tx0, ty0, tx1, ty1, _ in successful_table_bboxes_fitz:
            pt1_px = (int(tx0 * scale_x), int(ty0 * scale_y))
            pt2_px = (int(tx1 * scale_x), int(ty1 * scale_y))
            cv2.rectangle(img_for_opencv, pt1_px, pt2_px, (255, 255, 255), -1)

    # 圖像處理 (CPU)
    gray = cv2.cvtColor(img_for_opencv, cv2.COLOR_BGR2GRAY)

    _, th = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    temp_opencv_bboxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w <= 100 or h <= 80: continue

        # 過濾單行文字框：寬高比過大（太扁）的框不當作圖片                  ╎│
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio > 10:  # 寬度超過高度10倍，視為單行文字             ╎│
            continue

        temp_opencv_bboxes.append((x, y, w, h))

    valid_opencv_bboxes = filter_contained_figures(temp_opencv_bboxes, overlap_threshold=0.9)
    merged_bboxes = merge_overlapping_bboxes(valid_opencv_bboxes)

    # 效能優化：預先計算所有文字邊界框的像素座標，避免在每個圖形迴圈中重複計算
    # 節省：若有 N 個圖形和 M 個文字，從 O(N×M) 次計算降為 O(M) 次
    text_bboxes_px = [(int(b[0]*scale_x), int(b[1]*scale_y), int((b[2]-b[0])*scale_x), int((b[3]-b[1])*scale_y)) for b in text_blocks_fitz]

    # 效能優化：預先計算所有表格邊界框的像素座標，避免在圖形擴張迴圈中重複計算
    # 節省：若有 N 個圖形、M 個文字、T 個表格，從 O(N×M×T) 次計算降為 O(T) 次
    # 注意：successful_table_bboxes_fitz 是 (x0, y0, x1, y1, name) 五元組
    table_bboxes_px = [
        (int(tb[0] * scale_x), int(tb[1] * scale_y), int(tb[2] * scale_x), int(tb[3] * scale_y))
        for tb in successful_table_bboxes_fitz  # tb[0:4] 取前四個座標值
    ]

    combined_bboxes_with_text = []
    for fx, fy, fw, fh in merged_bboxes:
        fx_fitz = fx / scale_x
        fy_fitz = fy / scale_y
        fw_fitz = fw / scale_x
        fh_fitz = fh / scale_y

        # 檢查圖形起始位置是否在章節範圍之前
        if start_y_coordinate is not None and (fy_fitz + fh_fitz) <= start_y_coordinate:
            continue

        # 檢查圖形起始位置是否完全在章節範圍之後
        if end_y_coordinate is not None and fy_fitz >= end_y_coordinate:
            continue

        x_min, y_min, x_max, y_max = fx, fy, fx + fw, fy + fh
        rect_fig = (fx, fy, fw, fh)

        # 第一次擴張（使用預先計算好的 text_bboxes_px）
        for tx, ty, tw, th in text_bboxes_px:
            rect_text = (tx, ty, tw, th)
            tx_min, ty_min, tx_max, ty_max = tx, ty, tx + tw, ty + th

            # 檢查文字區塊是否在表格內，如果是則跳過（不與圖形合併）
            # 效能優化：在表格迴圈外預先計算文字中心點，避免每個表格都重複計算
            text_center_x = (tx_min + tx_max) / 2
            text_center_y = (ty_min + ty_max) / 2
            is_text_in_table = False
            for table_x0_px, table_y0_px, table_x1_px, table_y1_px in table_bboxes_px:
                # 檢查文字中心點是否在表格內
                if table_x0_px <= text_center_x <= table_x1_px and table_y0_px <= text_center_y <= table_y1_px:
                    is_text_in_table = True
                    break

            if is_text_in_table:
                continue  # 跳過表格內的文字，不與圖形合併

            # 效能優化：使用 has_overlap() 取代 get_iou() > 0.0，省略面積和比例計算
            is_inside_or_overlapping = has_overlap(rect_fig, rect_text)
            is_outside_and_close = False

            if not is_inside_or_overlapping:
                vertical_gap = min(abs(ty_max - y_min), abs(ty_min - y_max))
                horizontal_overlap_width = max(0, min(tx_max, x_max) - max(tx_min, x_min))
                is_vertically_aligned = (vertical_gap <= NEIGHBOR_GAP_PX) and (horizontal_overlap_width > 0)
                horizontal_gap = min(abs(tx_max - x_min), abs(tx_min - x_max))
                vertical_overlap_height = max(0, min(ty_max, y_max) - max(ty_min, y_min))
                is_horizontally_aligned = (horizontal_gap <= NEIGHBOR_GAP_PX) and (vertical_overlap_height > 0)
                if is_vertically_aligned or is_horizontally_aligned: is_outside_and_close = True

            if is_inside_or_overlapping or is_outside_and_close:
                x_min = min(x_min, tx_min); y_min = min(y_min, ty_min)
                x_max = max(x_max, tx_max); y_max = max(y_max, ty_max)

        # 檢查第一次擴張後的邊界是否「壓到」任何文字
        has_overlapping_text = False
        for tx, ty, tw, th in text_bboxes_px:
            tx_min, ty_min, tx_max, ty_max = tx, ty, tx + tw, ty + th
            # 檢查是否有重疊（即使部分重疊也算）
            if max(0, min(x_max, tx_max) - max(x_min, tx_min)) > 0 and max(0, min(y_max, ty_max) - max(y_min, ty_min)) > 0:
                has_overlapping_text = True
                break

        # 第二次擴張：只有在第一次擴張後「壓到」文字時才執行
        if has_overlapping_text:
            for tx, ty, tw, th in text_bboxes_px:
                tx_min, ty_min, tx_max, ty_max = tx, ty, tx + tw, ty + th

                # 檢查文字是否在表格內
                # 效能優化：在表格迴圈外預先計算文字中心點，避免每個表格都重複計算
                text_center_x = (tx_min + tx_max) / 2
                text_center_y = (ty_min + ty_max) / 2
                is_text_in_table = False
                for table_x0_px, table_y0_px, table_x1_px, table_y1_px in table_bboxes_px:
                    if table_x0_px <= text_center_x <= table_x1_px and table_y0_px <= text_center_y <= table_y1_px:
                        is_text_in_table = True
                        break

                if is_text_in_table:
                    continue

                # 使用擴張後的邊界進行檢查
                rect_current = (x_min, y_min, x_max - x_min, y_max - y_min)
                rect_text = (tx, ty, tw, th)

                # 效能優化：使用 has_overlap() 取代 get_iou() > 0.0，省略面積和比例計算
                is_inside_or_overlapping = has_overlap(rect_current, rect_text)
                is_outside_and_close = False

                if not is_inside_or_overlapping:
                    vertical_gap = min(abs(ty_max - y_min), abs(ty_min - y_max))
                    horizontal_overlap_width = max(0, min(tx_max, x_max) - max(tx_min, x_min))
                    is_vertically_aligned = (vertical_gap <= NEIGHBOR_GAP_PX) and (horizontal_overlap_width > 0)
                    horizontal_gap = min(abs(tx_max - x_min), abs(tx_min - x_max))
                    vertical_overlap_height = max(0, min(ty_max, y_max) - max(ty_min, y_min))
                    is_horizontally_aligned = (horizontal_gap <= NEIGHBOR_GAP_PX) and (vertical_overlap_height > 0)
                    if is_vertically_aligned or is_horizontally_aligned: is_outside_and_close = True

                if is_inside_or_overlapping or is_outside_and_close:
                    x_min = min(x_min, tx_min); y_min = min(y_min, ty_min)
                    x_max = max(x_max, tx_max); y_max = max(y_max, ty_max)

        final_x = max(0, x_min); final_y = max(0, y_min)
        final_w = min(img_width, x_max) - final_x; final_h = min(img_height, y_max) - final_y

        # 不進行任何裁切，保留完整圖形
        # 跨章節的圖片會在兩個章節都完整出現
        combined_bboxes_with_text.append((final_x, final_y, final_w, final_h))

    final_merged_cutouts = merge_overlapping_bboxes(combined_bboxes_with_text)
    figure_blocks_px = final_merged_cutouts


    # --- [4/6] 偵測圖形標題 (從預先保留的區塊中尋找) ---
    figure_titles_map = find_figure_titles_from_reserved_blocks(
        potential_figure_titles_blocks, # 使用預先篩選的標題區塊
        figure_blocks_px,
        scale_x, scale_y, page_height
    )

    # --- [5/6] 儲存圖形、命名並去重複 (使用批次 dHash) ---
    figure_items_to_add = []

    # 第一步：提取所有有效的 ROI 圖片和索引
    valid_rois = []
    valid_indices = []
    valid_bboxes = []

    for i, (x, y, w, h) in enumerate(final_merged_cutouts):
        if w <= 100 or h <= 80:
            continue

        roi = img[y:y+h, x:x+w]
        valid_rois.append(roi)
        valid_indices.append(i)
        valid_bboxes.append((x, y, w, h))

    # 第二步：批次計算所有圖片的 hash
    if valid_rois:
        current_hashes = dhash_batch(valid_rois)
    else:
        current_hashes = []

    # 第三步：處理每張圖片（順序與輸入一致）
    for idx, (i, roi, current_hash, (x, y, w, h)) in enumerate(zip(valid_indices, valid_rois, current_hashes, valid_bboxes)):

        # 使用鎖保護快取的讀取
        with lock:
            is_duplicate = current_hash in image_cache
            if is_duplicate:
                first_saved_path = image_cache[current_hash]

        if is_duplicate:
            original_filename = os.path.basename(first_saved_path)
            y_center_fitz = (y / scale_y + (y + h) / scale_y) / 2

            figure_items_to_add.append({
                "type": "figure",
                "y_center": y_center_fitz,
                "content": first_saved_path,
                "name": original_filename,
                "mode": "jpg",
                "page_num": page_num + 1,
                "title": f"[重複圖形，使用: {original_filename}]"
            })
            print(f"   [去重] 頁面 {page_num + 1} 的圖形 {i + 1} (Hash: {current_hash[:10]}...) 為重複圖形，使用首次出現的檔案: {original_filename}")
            continue

        title_content = figure_titles_map.get(f"figure_{i+1}", {}).get("content", "")
        safe_title_part = create_safe_filename(title_content, max_len=40)

        fig_filename_base = f"{chapter_safe_name}_page{page_num+1}_{safe_title_part}"
        fig_filename = fig_filename_base + ".jpg"

        counter = 1
        current_fig_path = os.path.join(chapter_assets_dir, fig_filename)
        base_name_no_ext = os.path.join(chapter_assets_dir, fig_filename_base)
        while os.path.exists(current_fig_path):
            current_fig_path = f"{base_name_no_ext}_{counter}.jpg"
            counter += 1

        cv2.imwrite(current_fig_path, roi)

        # 使用鎖保護快取的寫入
        with lock:
            image_cache[current_hash] = current_fig_path

        y_center_fitz = (y / scale_y + (y + h) / scale_y) / 2

        figure_items_to_add.append({
            "type": "figure",
            "y_center": y_center_fitz,
            "content": current_fig_path,
            "name": os.path.basename(current_fig_path),
            "mode": "jpg",
            "page_num": page_num + 1,
            "title": title_content
        })

    items_out.extend(figure_items_to_add)

    # --- [6/6] 處理純文字 (已排除所有標題和表格佔用區域的文字) ---

    for b in text_blocks_fitz: # 這裡的 text_blocks_fitz 已排除潛在標題
        x0, y0, x1, y1, text = b[:5]
        if not text.strip(): continue
        y_center = (y0 + y1) / 2; x_center = (x0 + x1) / 2
        text_bbox_fitz = (x0, y0, x1, y1)

        is_chapter_title = (chapter_title_bbox_fitz is not None and abs(y0 - chapter_title_bbox_fitz[1]) < 5)

        inside_element = False

        # 排除在表格或圖形邊界內的文字
        for fx0, fy0, fx1, fy1 in occupied_text_bboxes_fitz:
            # 檢查文字中心點是否在佔用區內
            if fx0 <= x_center <= fx1 and fy0 <= y_center <= fy1:
                inside_element = True; break
        if inside_element: continue

        # 排除被圖形佔用的文字 (使用最終合併後的圖形邊界)
        for x_cv, y_cv, w_cv, h_cv in figure_blocks_px:
            fx0 = x_cv / scale_x; fy0 = y_cv / scale_y
            fx1 = (x_cv + w_cv) / scale_x; fy1 = (y_cv + h_cv) / scale_y
            if fx0 <= x_center <= fx1 and fy0 <= y_center <= fy1:
                inside_element = True; break
        if inside_element: continue

        items_out.append({
            "type": "text",
            "y_center": y_center,
            "content": text.strip(),
            "page_num": page_num + 1,
            "is_title": is_chapter_title
        })

    # [7/7] Debug 繪圖 (保留，但確保使用的是正確的邊界框)
    if debug_draw:
        img_debug = img.copy()

        # 藍色：所有原始文字區塊
        for b in text_blocks_raw:
            x0, y0, x1, y1 = b[:4]
            pt1 = (int(x0 * scale_x), int(y0 * scale_y))
            pt2 = (int(x1 * scale_x), int(y1 * scale_y))
            cv2.rectangle(img_debug, pt1, pt2, (255, 0, 0), 1)

        # 紅色：表格區塊
        for (tx0, ty0, tx1, ty1, name) in successful_table_bboxes_fitz:
            pt1 = (int(tx0 * scale_x), int(ty0 * scale_y))
            pt2 = (int(tx1 * scale_x), int(ty1 * scale_y))
            cv2.rectangle(img_debug, pt1, pt2, (0, 0, 255), 2)
            cv2.putText(img_debug, name, (pt1[0], max(0, pt1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        # 綠色：最終圖形裁剪區塊
        for x, y, w, h in figure_blocks_px:
            cv2.rectangle(img_debug, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 紫色：已識別的圖形標題區塊
        for name, title_info in figure_titles_map.items():
            x0, y0, x1, y1 = title_info['bbox']
            pt1 = (int(x0 * scale_x), int(y0 * scale_y))
            pt2 = (int(x1 * scale_x), int(y1 * scale_y))
            cv2.rectangle(img_debug, pt1, pt2, (255, 0, 255), 2)
            cv2.putText(img_debug, "Fig. Title", (pt1[0], max(0, pt1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2, cv2.LINE_AA)

        # 黃色：章節標題
        if chapter_title_bbox_fitz:
            x0, y0, x1, y1 = chapter_title_bbox_fitz
            pt1 = (int(x0 * scale_x), int(y0 * scale_y))
            pt2 = (int(x1 * scale_x), int(y1 * scale_y))
            cv2.rectangle(img_debug, pt1, pt2, (0, 255, 255), 2)
            cv2.putText(img_debug, "Chapter Title", (pt1[0], max(0, pt1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        if end_y_coordinate:
            y_px = int(end_y_coordinate * scale_y)
            cv2.line(img_debug, (0, y_px), (img_width, y_px), (0, 255, 255), 2)
            cv2.putText(img_debug, "Chapter End", (10, max(0, y_px - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        debug_img_path = os.path.join(chapter_assets_dir, f"page{page_num+1}_debug.jpg")
        cv2.imwrite(debug_img_path, img_debug)
        print(f"✅ Debug 圖輸出：{debug_img_path}")

    return items_out


# ==============================================================================
# === 後續章節處理函式 (已加入頁碼位移校正的 Print) ===
# ==============================================================================

def build_chapters_index(doc, catalog_data, raw_chapters_list, out_dir="chapters_index_only", source="toc"):
    """
    根據目錄/大綱數據建立章節的頁碼索引，並針對同頁章節進行 Y 座標排序。
    (已恢復原始的兩段式校正邏輯，並加入詳細 Print 輸出)

    參數:
        catalog_data: 目錄數據 (直接傳遞，不再從檔案讀取)
        raw_chapters_list: 原始章節列表 (直接傳遞，不再從檔案讀取)
    """
    # 已停用 chapters_index_only 資料夾輸出
    # os.makedirs(out_dir, exist_ok=True)
    # idx_file = os.path.join(out_dir, INDEX_FILENAME)

    # 在函式開頭計算一次，避免重複計算
    total_pages = len(doc)

    results_index = []
    all_chapters = sorted(raw_chapters_list, key=lambda x: x['page'])


    # --- 步驟 2: 頁碼位移補正邏輯 (已結合您的要求) ---
    print("\n--- 執行頁碼位移補正邏輯 (TOC 模式) ---")

    # ⚠️ 初始化位移量，預設為 0
    offset = 0

    if source == "toc" and catalog_data and all_chapters:
        first_catalog_page = catalog_data[0]["catalog_page"] # 目錄所在頁 (1-based)
        first_chap_title = all_chapters[0]["title"] # <-- 偵測的錨點標題
        old_first = all_chapters[0]["page"] # 目錄/TOC 解析出的起始頁碼

        # 【2.1 階段：強制預位移】
        print(f"【頁碼校正-階段1】偵測到 Source=TOC，目錄頁碼: {first_catalog_page}，TOC解析的首章頁碼: {old_first}")

        if old_first <= first_catalog_page:
            offset = (first_catalog_page + 1) - old_first # ⚠️ 計算位移量
            print(f"【頁碼校正-階段1】⚠️ 偵測到首章頁碼 ({old_first}) 位於或早於目錄頁 ({first_catalog_page})。")
            print(f"【頁碼校正-階段1】執行**強制預位移**：將所有頁碼向後推遲 **+{offset}** 頁。")

            for chap in all_chapters:
                chap["page"] += offset

            new_offset_start = all_chapters[0]["page"]
            print(f"【頁碼校正-階段1】新的首章起始頁碼為: {new_offset_start}")
        else:
            new_offset_start = old_first
            print(f"【頁碼校正-階段1】首章頁碼 ({old_first}) 在目錄頁之後，跳過強制預位移。")

        # 【2.2 階段：微調校正 - 關鍵字模糊偵測】
        max_pages = len(doc)
        found_valid_title = False
        candidate_start = all_chapters[0]["page"] # 預位移後的新頁碼

        # 1. 提取核心關鍵字
        # 移除數字編號 (如: 1, 1.1) 和多餘空格
        core_keyword_raw = CHAPTER_NUMBER_PREFIX_PATTERN.sub("", first_chap_title).strip()
        # 移除常見的章/節/圖/表前綴
        core_keyword = CHAPTER_PREFIX_PATTERN.sub("", core_keyword_raw).strip()

        if len(core_keyword) < 3: # 如果關鍵字太短（例如只有兩個字），使用原始標題
            search_keyword = re.escape(first_chap_title.strip())
        else:
            # 使用核心關鍵字，且允許前後有任意文字
            search_keyword = re.escape(core_keyword)

        # 2. 定義偵測 Regex (組合：[第X章] OR [核心關鍵字])
        # 使用預編譯的 CHAPTER_REGEX_STRONG
        keyword_regex = rf"{search_keyword}"


        # 🌟 核心修正點：根據 offset (強制預位移量) 動態調整搜尋範圍
        # range_to_search 決定了搜尋範圍的絕對值大小 (例如 offset=3, range_to_search=4, 搜尋 +/-3)
        range_to_search = max(3, offset + 1)

        # 建立三個子列表：[0], [1, 2, 3...], [-1, -2, -3...]
        delta_zero = [0]
        delta_positive = list(range(1, range_to_search)) # [1, 2, 3, ...]
        # 負向列表需要反轉，以確保 -3, -2, -1 的順序
        delta_negative = list(range(-range_to_search + 1, 0)) # [-3, -2, -1]

        # 按照您的要求順序拼接：[0] + [正向] + [反轉的負向]
        # 確保負向是從最大負數到最小負數
        # 例如：offset=3, range_to_search=4， delta_negative=[-3, -2, -1]
        search_deltas = delta_zero + delta_positive + delta_negative


        print(f"\n【頁碼校正-階段2】在預測頁碼 {candidate_start} 周圍 (±{range_to_search-1}) 執行微調搜尋...")
        print(f"【頁碼校正-階段2】使用關鍵字/模式偵測: StrongRegex='第X章', Keyword='{core_keyword}'")

        print(f"【頁碼校正-階段2】搜尋順序: {candidate_start} 頁 + {search_deltas} 頁位移")

        for delta in search_deltas:
            candidate_idx = candidate_start + delta - 1 # 轉成 0-based 索引
            candidate_page_num = candidate_idx + 1 # 1-based 頁碼

            if 0 <= candidate_idx < max_pages:

                blocks = doc[candidate_idx].get_text("blocks")
                page_text_raw = "\n".join([b[4] for b in blocks])
                page_text_for_search = page_text_raw.upper().replace(' ', '') # 移除空格後轉換大寫以利關鍵字搜尋

                match = None
                is_keyword_match = False

                # 優先使用強模式（第X章）
                match = CHAPTER_REGEX_STRONG.search(page_text_raw)

                # 如果強模式未匹配，使用關鍵字模糊匹配
                if not match and len(core_keyword) >= 3:

                    # 進行模糊匹配：尋找關鍵字
                    if core_keyword.upper().replace(' ', '') in page_text_for_search:
                         is_keyword_match = True
                         # pass # 讓 is_keyword_match = True 進入 if match: 區塊 (見下方修正)


                if match or is_keyword_match:

                    # 決定位移
                    new_actual_page = candidate_page_num
                    old_current_page = all_chapters[0]["page"]
                    shift_delta = new_actual_page - old_current_page

                    if match:
                        # 從 Strong Regex 匹配中擷取整行
                        start_pos = match.start()
                        line_start = page_text_raw.rfind('\n', 0, start_pos)
                        if line_start == -1: line_start = 0
                        line_end = page_text_raw.find('\n', start_pos)
                        if line_end == -1: line_end = len(page_text_raw)
                        detected_line = page_text_raw[line_start:line_end].strip()
                        detection_mode = "Strong Regex"
                    else: # is_keyword_match = True
                        # 從整個頁面文字中，找到關鍵字並擷取其所在的行
                        keyword_index_in_raw = page_text_raw.upper().find(core_keyword.upper())

                        if keyword_index_in_raw != -1:
                            line_start = page_text_raw.rfind('\n', 0, keyword_index_in_raw)
                            if line_start == -1: line_start = 0
                            line_end = page_text_raw.find('\n', keyword_index_in_raw)
                            if line_end == -1: line_end = len(page_text_raw)
                            detected_line = page_text_raw[line_start:line_end].strip()
                            detection_mode = f"Keyword Match: {core_keyword}"
                        else:
                             # 關鍵字搜尋失敗，可能發生在空格移除等操作後
                             print(f"【頁碼校正-階段2】... 頁碼 {candidate_page_num}：關鍵字匹配後無法擷取上下文，跳過。")
                             continue

                        # 修正：如果偵測到的行仍然是空的，則跳過，繼續尋找
                    if not detected_line:
                        print(f"【頁碼校正-階段2】... 頁碼 {candidate_page_num}：偵測模式:{detection_mode}，但內容為空，跳過。")
                        continue

                    print(f"【頁碼校正-階段2】✅ 在**第 {candidate_page_num} 頁**偵測到標題。")
                    print(f"【頁碼校正-階段2】偵測模式: **{detection_mode}**")
                    print(f"【頁碼校正-階段2】偵測到的**完整標題行**: **'{detected_line}'**")

                    if shift_delta != 0:
                        print(f"【頁碼校正-階段2】執行微調校正位移: **{shift_delta}** 頁到所有章節。")
                        for chap in all_chapters:
                            chap["page"] += shift_delta
                    else:
                        print(f"【頁碼校正-階段2】✅ 微調精確。首章頁碼: {new_actual_page} (無額外位移)。")

                    found_valid_title = True
                    break # 找到後立即退出 delta 迴圈
                else:
                    print(f"【頁碼校正-階段2】... 頁碼 {candidate_page_num}：未找到章節標題。")

        if not found_valid_title:
              # range_to_search-1 是實際位移的最大絕對值 (例如 4-1 = 3)
              print(f"【頁碼校正-階段2】❌ 儘管檢查了 {candidate_start} 頁周圍 ±{range_to_search-1} 頁，仍找不到明確章節標題。不進行微調校正。")

    elif source != "toc":
        print(f"【頁碼校正】Source={source} (非 TOC 模式)，跳過頁碼位移校正。")
    else:
        print("【頁碼校正】無有效章節數據，跳過頁碼位移校正。")

    all_chapters.sort(key=lambda x: x['page'])
    print("--- 頁碼位移補正邏輯結束 ---")


    # === 步驟 3: 針對同頁章節，按 Y 座標排序 (使用傳入的 doc) ===
    chapters_by_page = {}; final_sorted_chapters = []

    original_order_map = {chap['title']: i for i, chap in enumerate(all_chapters)}

    for chap in all_chapters:
        page_idx = chap["page"] - 1
        if page_idx not in chapters_by_page: chapters_by_page[page_idx] = []
        chapters_by_page[page_idx].append(chap)

    for page_idx in sorted(chapters_by_page.keys()):
        page_chapters = chapters_by_page[page_idx]

        if len(page_chapters) > 1:

            if source == "outline":
                sorted_page_chapters = sorted(page_chapters, key=lambda x: original_order_map[x['title']])
                final_sorted_chapters.extend(sorted_page_chapters)
                continue

            page = doc[page_idx]
            blocks = page.get_text("blocks")
            y_coords_map = {}
            for chap in page_chapters:
                y_coord = find_y_coord_for_title(page, chap["title"], blocks)
                if y_coord is not None:
                    y_coords_map[chap["title"]] = (y_coord, chap)
                else:
                    # 將找不到座標的標題移到頁面底部 (例如頁尾免責聲明)
                    y_coords_map[chap["title"]] = (float('inf') if chap["title"].lower().strip() == 'disclaimer' else float('-inf'), chap)

            sorted_by_y = sorted(y_coords_map.values(), key=lambda x: x[0])
            sorted_page_chapters = [chap_info for y_coord, chap_info in sorted_by_y if y_coord != float('inf')]
            final_sorted_chapters.extend(sorted_page_chapters)
        else:
            final_sorted_chapters.extend(page_chapters)

    all_chapters = final_sorted_chapters

    # === 步驟 4: 建立最終 Index (使用傳入的 doc) ===
    for i, chap in enumerate(all_chapters):
        title = chap["title"]
        start_page = chap["page"]

        if i + 1 < len(all_chapters):
            next_chap = all_chapters[i + 1]
            end_page = next_chap["page"]
        else:
            end_page = total_pages + 1

        if end_page < start_page:
            end_page = start_page

        safe_title = UNSAFE_FILENAME_CHARS_PATTERN.sub('', title).strip()
        safe_title = create_safe_filename(safe_title, max_len=50)

        base_name = safe_title

        temp_info = {
            "title": title,
            "page_start": start_page,
            "page_end": end_page,
            "out_file": f"{base_name}.txt",
            "out_dir": f"{base_name}",
            "text_len": 0,
            "tables_count": 0,
        }
        results_index.append(temp_info)

    # 已停用輸出 chapters_index.json
    # final_output_index = [
    #     {"title": r['title'], "page_start": r['page_start'], "page_end": r['page_end']}
    #     for r in results_index
    # ]
    # index_only_dir = "chapters_index_only"
    # os.makedirs(index_only_dir, exist_ok=True)
    # idx_file = os.path.join(index_only_dir, INDEX_FILENAME)
    #
    # with open(idx_file, "w", encoding="utf-8") as f:
    #     json.dump(final_output_index, f, ensure_ascii=False, indent=2)
    # print(f"\n📄 已輸出優化後的章節索引檔 ({INDEX_FILENAME}) → {idx_file}")

    return results_index

def process_chapters_batch(doc, chapters_index_list, all_camelot_tables, base_out_dir=OUT_CHAPTERS_DIR, dpi=200, debug_plot=False, use_multithread=True):
    """
    根據章節索引列表處理每一章節的內容。
    多執行緒：使用 ThreadPoolExecutor 平行處理頁面

    參數：
        use_multithread: 是否啟用多執行緒處理（預設 True）
    """
    os.makedirs(base_out_dir, exist_ok=True)

    total_pdf_pages = len(doc)

    # 效能優化：預先建立表格的頁碼索引，避免每頁都遍歷所有表格
    # 時間複雜度從 O(總頁數 × 總表格數) 降為 O(總表格數 + 總頁數)
    tables_by_page = {}
    for t in all_camelot_tables:
        page_num = t.page
        if page_num not in tables_by_page:
            tables_by_page[page_num] = []
        tables_by_page[page_num].append(t)

    # 取得最佳執行緒數
    max_workers = get_optimal_workers() if use_multithread else 1

    print(f"\n==========================================")
    print(f"🔄 開始處理 {len(chapters_index_list)} 個章節... 📄 文件總頁數: {total_pdf_pages}")
    if use_multithread:
        print(f"🚀 多執行緒模式：{max_workers} 個執行緒")
    else:
        print(f"📌 單執行緒模式")
    print(f"==========================================")

    updated_index_list = []

    for i, chap_info in enumerate(chapters_index_list):
        # 每個章節建立獨立的圖片快取，章節內去重
        # 使用 threading.Lock 保護快取，避免多執行緒競爭
        chapter_image_cache = {}
        cache_lock = threading.Lock()  # 保護快取的鎖

        title = chap_info["title"]
        start_page = chap_info["page_start"]
        end_page = chap_info["page_end"]
        out_file_name = chap_info["out_file"]
        assets_dir_name = chap_info["out_dir"]

        actual_end_page = min(end_page, total_pdf_pages + 1) # +1 是為了處理最後一章

        chapter_safe_name = assets_dir_name

        print(f"\n--- 📚 處理章節 {i + 1}/{len(chapters_index_list)}: **{title}** (頁碼: {start_page} - {actual_end_page - 1 if actual_end_page > total_pdf_pages else actual_end_page -1}) ---")

        # 1. 章節資產資料夾路徑
        chapter_assets_dir = os.path.join(base_out_dir, assets_dir_name)
        os.makedirs(chapter_assets_dir, exist_ok=True)

        # 2. TXT 檔路徑
        out_path = os.path.join(chapter_assets_dir, out_file_name)

        all_chapter_items = []
        current_text_len = 0
        current_tables_count = 0

        start_y_coord = None
        try:
            page = doc[start_page - 1]
            blocks = page.get_text("blocks")
            start_y_coord = find_y_coord_for_title(page, title, blocks)
        except Exception as e:
            pass

        end_y_coord = None
        if i + 1 < len(chapters_index_list):
            next_chap = chapters_index_list[i + 1]
            # 下一章節的開始頁碼與本章節的結束頁碼相同
            if next_chap["page_start"] == (end_page):
                try:
                    page = doc[next_chap["page_start"] - 1]
                    blocks = page.get_text("blocks")
                    end_y_coord = find_y_coord_for_title(page, next_chap["title"], blocks)
                except Exception as e:
                    pass


        # 多執行緒處理頁面
        if use_multithread and (end_page - start_page) > 2:
            # 準備頁面處理任務
            def process_page_task(page_num):
                """單頁處理任務（用於多執行緒）"""
                try:
                    page = doc[page_num]
                    page_height = page.rect.height

                    is_start_page_in_chapter = (page_num == start_page - 1)
                    is_last_page_in_chapter = (page_num == end_page - 1)

                    y_start_to_process = start_y_coord if is_start_page_in_chapter and start_y_coord is not None else None
                    y_end_to_process = None

                    if is_last_page_in_chapter and end_y_coord is not None and i + 1 < len(chapters_index_list):
                        next_chap = chapters_index_list[i + 1]
                        if next_chap["page_start"] == (page_num + 1):
                            y_end_to_process = min(end_y_coord, page_height)
                    elif is_last_page_in_chapter and end_page == total_pdf_pages + 1:
                        y_end_to_process = None

                    # 效能優化：使用預先建立的索引直接查詢當前頁的表格（O(1) 查詢）
                    current_page_tables = tables_by_page.get(page_num + 1, [])

                    items = process_single_page_and_get_items(
                        page, page_height, chapter_assets_dir, chapter_safe_name,
                        start_y_coordinate=y_start_to_process,
                        end_y_coordinate=y_end_to_process,
                        dpi=dpi, debug_draw=debug_plot,
                        camelot_tables=current_page_tables,
                        chapter_image_cache=chapter_image_cache,
                        cache_lock=cache_lock
                    )
                    return (page_num, items)
                except Exception as e:
                    print(f"⚠️ 處理頁面 {page_num + 1} 時發生錯誤: {e}")
                    return (page_num, [])

            # 使用多執行緒處理
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有頁面任務
                future_to_page = {
                    executor.submit(process_page_task, page_num): page_num
                    for page_num in range(start_page - 1, end_page)
                    if page_num < total_pdf_pages
                }

                # 收集結果
                for future in as_completed(future_to_page):
                    page_num, items = future.result()
                    all_chapter_items.extend(items)

            # 按頁碼和 Y 座標排序（確保順序正確）
            all_chapter_items.sort(key=lambda x: (x['page_num'], x['y_center']))

        else:
            # 單執行緒處理（原始邏輯）
            for page_num in range(start_page - 1, end_page):
                if page_num >= total_pdf_pages:
                    break

                page = doc[page_num]
                page_height = page.rect.height

                is_start_page_in_chapter = (page_num == start_page - 1)
                is_last_page_in_chapter = (page_num == end_page - 1)

                y_start_to_process = start_y_coord if is_start_page_in_chapter and start_y_coord is not None else None
                y_end_to_process = None

                if is_last_page_in_chapter and end_y_coord is not None and i + 1 < len(chapters_index_list):
                    next_chap = chapters_index_list[i + 1]
                    if next_chap["page_start"] == (page_num + 1):
                        # 如果下一章的開始頁與本章結束頁碼相同，則本頁的結束 y 座標為下一章標題的 y 座標
                        y_end_to_process = min(end_y_coord, page_height)
                elif is_last_page_in_chapter and end_page == total_pdf_pages + 1:
                    # 如果是最後一章的最後一頁，處理到頁尾
                    y_end_to_process = None

                # 效能優化：使用預先建立的索引直接查詢當前頁的表格（O(1) 查詢）
                current_page_tables = tables_by_page.get(page_num + 1, [])

                items = process_single_page_and_get_items(
                    page, page_height, chapter_assets_dir, chapter_safe_name,
                    start_y_coordinate=y_start_to_process,
                    end_y_coordinate=y_end_to_process,
                    dpi=dpi, debug_draw=debug_plot,
                    camelot_tables=current_page_tables,
                    chapter_image_cache=chapter_image_cache,
                    cache_lock=cache_lock
                )
                all_chapter_items.extend(items)

        # === 寫入 TXT 並存至向量資料庫 ===
        
        chunks = []
        with open(out_path, "w", encoding="utf-8-sig") as f_out:
            for item in sorted(all_chapter_items, key=lambda x: (x['page_num'], x['y_center'])):
                chunk_text = None
                if item["type"] == "text":
                    chunk_text = item["content"]
                    prefix = f"## {title} - " if item.get("is_title") else ""
                    f_out.write(prefix + item["content"] + "\n")
                    current_text_len += len(item["content"])
                elif item["type"] == "table":
                    chunk_text = item["content"].to_csv(index=False, sep="\t")
                    current_tables_count += 1
                    f_out.write(f"\n[TABLE {item['name']} - Page {item['page_num']}]\n")
                    item["content"].to_csv(f_out, index=False, sep="\t", encoding="utf-8-sig")
                    f_out.write(f"[END TABLE]\n")
                elif item["type"] == "figure":
                    figure_title = f"標題: {item['title']}" if item.get('title') and not item['title'].startswith('[重複圖形') else "無標題"

                    relative_path_name = os.path.basename(item['content'])

                    f_out.write(f"\n[FIGURE {relative_path_name} - Page {item['page_num']}]\n")
                    f_out.write(f"圖片路徑: {relative_path_name}\n")
                    f_out.write(f"{figure_title}\n")
                    f_out.write(f"[END FIGURE]\n")
                    
                if chunk_text:
                    ic_model = extract_ic_model(chunk_text) or title
                    page = str(item.get("page_num"))
                    section = extract_section(chunk_text) or title
                    
                    chunks.append({
                        "text": chunk_text,
                        "ic_model": ic_model,
                        "page": page,
                        "section": section
                    })


        # === 更新章節索引 (暫存資訊) ===
        chap_info["text_len"] = current_text_len
        chap_info["tables_count"] = current_tables_count
        updated_index_list.append(chap_info)

        print(f"✅ 完成章節：{title}，文字長度: {current_text_len}，表格數: {current_tables_count}, 共 {len(chunks)} 個 chunks, 開始送 AI ...")
        asyncio.run(ingest_chunks(chunks, title))

    # 已停用輸出 chapters_index.json 到 structured_chapters_final
    # final_output_index = [
    #     {"title": r['title'], "page_start": r['page_start'], "page_end": r['page_end']}
    #     for r in updated_index_list
    # ]
    #
    # idx_file = os.path.join(base_out_dir, INDEX_FILENAME)
    # with open(idx_file, "w", encoding="utf-8") as f:
    #     json.dump(final_output_index, f, ensure_ascii=False, indent=2)
    # print(f"\n📄 已輸出最終章節索引檔 (title/page_start/page_end) → {idx_file}")

    return updated_index_list


# === 主程式進入點 (優化後) ===
if __name__ == "__main__":
    pdf_file = "dummy.pdf"

    try:
        # --- 步驟 1: 上傳檔案與初始化 ---
        is_mock = isinstance(files, MockFiles)

        if is_mock:
            # 本地環境：使用檔案選擇對話框
            try:
                from tkinter import Tk, filedialog
                print("請選擇 PDF 檔案...")
                root = Tk()
                root.withdraw()  # 隱藏主視窗
                root.attributes('-topmost', True)  # 讓對話框置頂
                pdf_file = filedialog.askopenfilename(
                    title="選擇 PDF 檔案",
                    filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
                )
                root.destroy()

                if not pdf_file:
                    print("❌ 未選擇檔案，程式終止。")
                    exit()
                print(f"✅ 已選擇：{pdf_file}")
            except ImportError:
                print("❌ 無法載入檔案選擇對話框 (需要 tkinter)，程式終止。")
                exit()
        else:
            # Colab 環境：使用原本的上傳方式
            print("請上傳 PDF 檔案：")
            uploaded = files.upload()

            if not uploaded:
                print("❌ 沒有檔案上傳，程式終止。")
                exit()

            pdf_file = next(iter(uploaded))
            print(f"✅ 已上傳：{pdf_file}")

        # ⏱️ 開始計時
        program_start_time = time.time()
        print(f"\n⏱️ 開始處理... (計時開始)")

        # 🚀 效能優化點 A: 僅在此處打開一次 PDF 文件
        with fitz.open(pdf_file) as doc:
            total_pages = len(doc)
            print(f"這份 PDF 總共有 {total_pages} 頁。")

            # --- 步驟 2: 建立章節索引 ---
            catalog_json_file = "catalog.json"

            # 使用已開啟的 doc 物件
            outline_chapters = find_outline(doc)
            catalog_data = []
            source = ""

            raw_chapters_list = []

            if outline_chapters:
                print(f"\n✅ 從 Outline (書籤) 偵測到 {len(outline_chapters)} 個章節。")
                raw_chapters_list = outline_chapters
                catalog_data = [{"catalog_page": 0, "content": "\n".join([f"{c['title']} ... {c['page']}" for c in outline_chapters])}]
                source = "outline"
            else:
                # 使用已開啟的 doc 物件
                catalog_pages = find_catalog_pages(doc)
                if not catalog_pages:
                    print("\n⚠️ 沒找到 Outline 或目錄頁，無法建立章節索引，程式結束。")
                    exit()
                for page_idx, content in catalog_pages:
                    catalog_data.append({"catalog_page": page_idx + 1, "content": content})
                    raw_chapters_list.extend(parse_catalog_text(content))

                if source == "toc":
                    raw_chapters_list.sort(key=lambda x: x['page'])

                print(f"\n✅ 已找到 {len(catalog_pages)} 頁目錄。")
                source = "toc"

            # 已停用輸出 chapters_raw_outline.json
            # with open(RAW_OUTLINE_FILENAME, "w", encoding="utf-8") as f:
            #     raw_output = [{"title": r['title'], "page": r['page']} for r in raw_chapters_list]
            #     json.dump(raw_output, f, ensure_ascii=False, indent=2)
            # print(f"📄 已輸出**原始**章節順序檔 (無任何校正) → **{RAW_OUTLINE_FILENAME}**")

            # 已停用輸出 catalog.json
            # with open(catalog_json_file, "w", encoding="utf-8") as f:
            #     json.dump(catalog_data, f, ensure_ascii=False, indent=2)
            # print(f"📄 已輸出原始目錄資料檔 (包含原始文字) → {catalog_json_file}")

            # 傳入 doc 物件
            # 這裡會輸出頁碼校正的 print 信息
            chapters_index_list = build_chapters_index(doc, catalog_data, raw_chapters_list, source=source)

            if not chapters_index_list:
                print("❌ 無法從目錄或大綱解析出有效的章節列表，程式結束。")
                exit()

            # 🚀 效能優化點 B: 提前執行表格偵測（僅處理章節範圍內的頁面）
            # 從章節索引中提取所有涵蓋的頁碼
            chapter_pages = set()
            for chap in chapters_index_list:
                for page_num in range(chap['page_start'], chap['page_end']):
                    if page_num <= total_pages:
                        chapter_pages.add(page_num)

            # 將頁碼列表轉換為 Camelot 接受的範圍字串格式
            sorted_pages = sorted(chapter_pages)
            if sorted_pages:
                ranges = []
                start = sorted_pages[0]
                end = sorted_pages[0]

                for p in sorted_pages[1:]:
                    if p == end + 1:
                        end = p
                    else:
                        ranges.append(f"{start}-{end}" if start != end else f"{start}")
                        start = end = p

                ranges.append(f"{start}-{end}" if start != end else f"{start}")
                pages_str = ','.join(ranges)
            else:
                pages_str = "all"

            print(f"\n🔍 正在執行表格偵測（僅處理章節頁面: {len(chapter_pages)}/{total_pages} 頁）...")
            print(f"   頁碼範圍: {pages_str}")
            all_camelot_tables = camelot.read_pdf(pdf_file, pages=pages_str, flavor="lattice")
            print(f"✅ 表格偵測完成，共偵測到 {len(all_camelot_tables)} 個表格。")


            # --- 步驟 3: 批次處理章節內容 ---
            # 傳入 doc 物件 和 all_camelot_tables
            process_chapters_batch(
                doc,
                chapters_index_list,
                all_camelot_tables,
                debug_plot=False,  # 💡 注意：如果要啟用 Debug 繪圖，請將 debug_plot=False 改為 debug_plot=True
                use_multithread=True  # 💡 啟用多執行緒加速（設為 False 可關閉）
            )

        # PDF 文件在此處關閉 (when fitz.open(pdf_file) scope ends)

        # ⏱️ 計算總執行時間
        program_end_time = time.time()
        total_program_time = program_end_time - program_start_time

        print("\n" + "=" * 60)
        print("🎉 程式執行完成！")
        print("=" * 60)
        print(f"⏱️  總執行時間: {total_program_time:.2f} 秒")
        print(f"⏱️  總執行時間: {total_program_time/60:.2f} 分鐘")
        if total_program_time >= 3600:
            print(f"⏱️  總執行時間: {total_program_time/3600:.2f} 小時")
        print("=" * 60)

    except ImportError as e:
        print(f"❌ 缺少必要的函式庫：{e}。請確認所有必要的套件 (**PyMuPDF**, **camelot-py**, **opencv-python**, **pandas**, **numpy**) 皆已安裝。")

    except Exception as e:
        print(f"❌ 發生錯誤：{e}")
        import traceback
        traceback.print_exc()
