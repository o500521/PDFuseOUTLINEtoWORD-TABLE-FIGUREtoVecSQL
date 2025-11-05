# PDF 章節結構化擷取工具

這是一個 Python 工具, 用於從 PDF 文件中自動擷取和結構化內容, 包括文字、表格和圖片. 
使用前, 請在根目錄添加一個 config.json, 並在下方找到格式進行設定.

## 功能特色

- 📄 自動偵測 PDF 目錄並分章節處理
- 📊 使用 PyMuPDF 快速擷取表格
- ✅ 在 PDF 切換至 TXT 時, 自動匯集並統一提交給AI進行資料整理
- 🖼️ 自動偵測和儲存圖片
- 🔍 圖片去重（使用 dHash 演算法）
- 🎨 生成 debug 頁面以視覺化偵測結果
- ⚡ 支援多執行緒加速處理
- 🌍 使用 Google AI Studio 進行資料整理 (TXT 轉 Json)
- 🤖 使用 BGE-M3 進行資料向量
- 📁 使用 PostgreSQL + pgvector 進行資料存儲

### 處理+偵測章節：
  1. find_outline()      → 檢查 PDF 是否有內建書籤
     ↓ (如果沒有)
  2. find_catalog_pages() → 在前 15 頁搜尋「目錄」關鍵字或排版特徵
     
  3. parse_catalog_text() → 將目錄文字解析成章節列表
     
  4. build_chapters_index() → 校正頁碼並建立章節索引
### 批次處理章節：
  5. camelot.read_pdf() → 提前偵測所有表格
     
  6. process_chapters_batch() → 對每一章處理 起始頁 到 結束頁
### 處理單頁內容：
  7. process_single_page_and_get_items()  
        [1/6] 偵測與隔離標題文字（圖表標題識別）  
        [2/6] 擷取表格（過濾章節範圍、驗證有效性）  
        [3/6] 偵測與合併圖形（OpenCV 二值化、輪廓偵測  
        [4/6] 偵測圖形標題（匹配最近的標題文字）  
        [5/6] 儲存圖形並去重（dHash 批次計算、重複檢查）  
        [6/6] 擷取純文字（排除標題、表格、圖形區域）
### 寫入 TXT 檔案：
  8. 輸出：structured_chapters_final→ 按 Y 座標排序, 輸出文字/表格/圖片


###	頁碼校正問題
在build_chapters_index(), offset是位移量

###	圖表擷取問題 
在process_single_page_and_get_items()

####  1. 表格必須在章節範圍內（Y 座標檢查）（容易跑掉的地方）

  位置： pdf.py:592-606

    for i, t in enumerate(camelot_tables):
        x0, y0, x1, y1 = map(float, t._bbox)
        table_top_y = page_height - y1
        table_bottom_y = page_height - y0

        is_valid_table = True
        # 修正：如果表格的底部（最下方）在章節開始座標之前, 則排除整個表格
        # 表格的任何部分都不應該在章節開始之前
        if start_y_coordinate is not None and table_bottom_y < start_y_coordinate:
            is_valid_table = False
        # 如果表格的頂部（最上方）在章節結束座標之後, 則排除
        if end_y_coordinate is not None and table_top_y > end_y_coordinate:
            is_valid_table = False
        if not is_valid_table:
            continue

  說明： 檢查表格的上下邊界是否在章節的 Y 座標範圍內. 
	
####  2. 符合條件被認為是有效表格 => 現在輸出成表格（至少 2 行 2 列 + 空白部分比例 < 50%）(條件可以改掉)

  位置： pdf.py:610-615

    詳細拆解：
		↓
    is_valid_csv = not (
        df.shape[0] < 2          # ← 行數必須 ≥ 2
        or
        df.shape[1] < 2          # ← 列數必須 ≥ 2
        or
        (df.replace("", pd.NA).isna().sum().sum() / max(1, df.size) > 0.5)# ← 空白部分比例比例必須 ≤ 50%)

#### 3.抓圖片兩階段擴張 (第二次擴張可能會需要做調整)

	  第一次擴張：合併圖形內部或緊鄰的文字 (他沒什麼問題)
	  位置： pdf.py:703-738 (尚未更新代碼位置)
			↓
	  檢查第一次擴張是否「壓到」文字 (他沒什麼問題)
	  位置： pdf.py:744-750
	  只要跟文字有部分重疊就設定 has_overlapping_text = True
			↓
	  第二次擴張：只有當壓到文字時才執行 (NEIGHBOR_GAP_PX = 30  # pdf.py:99)
	  位置： pdf.py:748-786 (尚未更新代碼位置)
		
	  第一次擴張跟第二次擴張都是用NEIGHBOR_GAP_PX, 如果覺得圖抓太大或太小, 可優先考慮更改這個數字, 或是使用不同的常數
		
###	啟用 Debug 模式 (在main)
    debug_plot=True  # 會輸出整頁pdf圖片 (如果不需要： debug_plot=False  # pdf.py:1624)

藍色：所有偵測到的文字                                                              
綠色：偵測到的圖形                                                          
紫色：圖形標題                                                              
黃色：章節開頭、結尾  
紅色：表格

### config.json格式
    {
        "postgre_user": "你的PostgreSQL使用者名稱",
        "postgre_password": "你的PostgreSQL使用者密碼",
        "google_ai_studio_apikey": "Google AI Studio API權杖"
    }
