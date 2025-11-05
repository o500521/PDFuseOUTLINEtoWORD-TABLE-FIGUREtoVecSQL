import psycopg2
import json
from utility.bge_embed import embed_text
from utility.config import config

def get_db_connection(db):
    return psycopg2.connect(
        host="localhost",
        port=5432,
        database=db,
        user=config.postgre_user,
        password=config.postgre_password
    )

def sql_insert_vector_chunk(ic_model, doc_title, chapter, section, text, metadata, embedding):
    conn = get_db_connection("vector_test")
    cur = conn.cursor()

    query = """
        INSERT INTO pdf_chunks (ic_model, document_title, chapter_title, section_title, chunk_text, metadata, embedding)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    
    cur.execute(query, (ic_model, doc_title, chapter, section, text, json.dumps(metadata), embedding))
    conn.commit()

    cur.close()
    conn.close()

def store_chunk_to_vector_db(chunk_text, metadata):
    ic_model = metadata.get("ic_model", "Unknown")
    doc_title = metadata.get("document_title", "Unknown")
    chapter = metadata.get("chapter_title", "Unknown")
    section = metadata.get("source_section", "Unknown")

    embedding = embed_text(chunk_text)

    sql_insert_vector_chunk(ic_model, doc_title, chapter, section, chunk_text, metadata, embedding)

def vector_search(query_text: str, ic_model=None, top_k=10):
    embedding = embed_text(query_text)
    conn = get_db_connection("vector_test")
    cur = conn.cursor()

    if ic_model:
        sql = """
        SELECT chapter_title, section_title, chunk_title, metadata,
                embedding <-> %s::vector AS distance
        FROM pdf_chunks
        WHERE ic_model = %s
        ORDER BY embedding <-> %s::vector
        LIMIT %s
        """
        cur.execute(sql, (embedding, ic_model, embedding, top_k))
    else:
        sql = """
        SELECT chapter_title, section_title, chunk_title, metadata,
                embedding <-> %s::vector AS distance
        FROM pdf_chunks
        ORDER BY embedding <-> %s::vector
        LIMIT %s
        """
        cur.execute(sql, (embedding, embedding, top_k))
        
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows
