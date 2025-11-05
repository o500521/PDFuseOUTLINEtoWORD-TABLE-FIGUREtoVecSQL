from utility.db import vector_search

if __name__ == "__main__":
    query = input("請輸入你要查詢的關鍵字: ").strip()
    
    if not query:
        print("❗ 未輸入關鍵字, 程式結束.")
        exit()
        
    print(f"\n🔍 正在搜尋: {query}\n")
        
    res = vector_search(query, top_k=10)
    if not res:
        print("⚠️ 找不到相關內容")
        exit()
    
    for row in res:
        section = row[0]
        content = row[1]

        print(f"\n📌 Section: {section}")
        print("────────────────────────────")
        print(content.strip())
        print("────────────────────────────\n")