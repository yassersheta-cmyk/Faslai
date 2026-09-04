import streamlit as st
import json
import base64
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi
from groq import Groq
import os
import streamlit.components.v1 as components

st.set_page_config(page_title="مساعد المحاضرة", layout="wide")

CHUNK_SIZE = 4
OVERLAP = 2
TOP_K = 5
SEMANTIC_WEIGHT = 0.6

ARABIC_STOPWORDS = {"ال", "في", "من", "إلى", "على", "عن", "أن", "إن", "و", "أو", "كل", "هي", "هو"}


def tokenize(text):
    return [w for w in text.split() if w not in ARABIC_STOPWORDS]


# ---------- مفتاح Groq API ----------
def get_groq_key():
    if "GROQ_API_KEY" in st.secrets:
        return st.secrets["GROQ_API_KEY"]
    return os.environ.get("GROQ_API_KEY", "")


groq_key = get_groq_key()
if not groq_key:
    groq_key = st.sidebar.text_input("حط مفتاح Groq API هنا:", type="password")

if not groq_key:
    st.warning("لازم تحط مفتاح Groq API الأول (من الشريط الجانبي).")
    st.stop()

client = Groq(api_key=groq_key)

# ---------- رفع الملفات ----------
st.sidebar.header("ملفات المحاضرة")
audio_file = st.sidebar.file_uploader("ملف الصوت (mp3/wav)", type=["mp3", "wav"])
json_file = st.sidebar.file_uploader("ملف الـ transcript.json", type=["json"])

if not audio_file or not json_file:
    st.info("ارفع ملف الصوت وملف الـ transcript.json من الشريط الجانبي عشان تبدأ.")
    st.stop()

transcript_data = json.load(json_file)
audio_bytes = audio_file.read()


# ---------- بناء الـ RAG pipeline (يتعمل مرة واحدة بس ويتخزن) ----------
@st.cache_resource
def build_pipeline(_transcript_data):
    segments = _transcript_data["segments"]

    chunks = []
    step = max(CHUNK_SIZE - OVERLAP, 1)
    for i in range(0, len(segments), step):
        group = segments[i:i + CHUNK_SIZE]
        if not group:
            continue
        chunk_text = " ".join(seg["text"].strip() for seg in group if seg["text"].strip())
        if not chunk_text:
            continue
        chunks.append({
            "text": chunk_text,
            "start": group[0]["start"],
            "end": group[-1]["end"],
        })

    embed_model = SentenceTransformer("intfloat/multilingual-e5-large")
    chunk_texts = [c["text"] for c in chunks]
    embeddings = embed_model.encode(chunk_texts, show_progress_bar=False)
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    tokenized_chunks = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized_chunks)

    return chunks, embed_model, index, bm25


chunks, embed_model, index, bm25 = build_pipeline(transcript_data)
audio_b64 = base64.b64encode(audio_bytes).decode()


def hybrid_search(query, top_k=TOP_K, semantic_weight=SEMANTIC_WEIGHT):
    query_embedding = embed_model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, len(chunks))
    semantic_scores = {int(idx): 1 / (1 + dist) for idx, dist in zip(indices[0], distances[0])}

    bm25_scores_raw = bm25.get_scores(tokenize(query))
    max_bm25 = max(bm25_scores_raw) if max(bm25_scores_raw) > 0 else 1
    bm25_scores = {i: score / max_bm25 for i, score in enumerate(bm25_scores_raw)}

    combined_scores = {}
    for idx in range(len(chunks)):
        sem = semantic_scores.get(idx, 0)
        kw = bm25_scores.get(idx, 0)
        combined_scores[idx] = (semantic_weight * sem) + ((1 - semantic_weight) * kw)

    top_indices = sorted(combined_scores, key=combined_scores.get, reverse=True)[:top_k]
    return [chunks[i] for i in top_indices]


def format_chunk_with_citation(chunk_text, start_sec, end_sec):
    start_m = int(start_sec // 60)
    start_s = int(start_sec % 60)
    end_m = int(end_sec // 60)
    end_s = int(end_sec % 60)
    return f"[Citation: {start_m:02d}:{start_s:02d} - {end_m:02d}:{end_s:02d}]\n{chunk_text}"


def build_context(retrieved_chunks):
    parts = [format_chunk_with_citation(c["text"], c["start"], c["end"]) for c in retrieved_chunks]
    return "\n\n".join(parts)


def filter_used_chunks(answer_text, retrieved_chunks):
    used = []
    for c in retrieved_chunks:
        start_s, end_s = c["start"], c["end"]
        start_label = f"{int(start_s // 60):02d}:{int(start_s % 60):02d}"
        end_label = f"{int(end_s // 60):02d}:{int(end_s % 60):02d}"
        if start_label in answer_text and end_label in answer_text:
            used.append(c)
    # لو الفلترة رجعت فاضية لأي سبب (الموديل غيّر الصيغة شوية)، ارجع لكل الـ chunks بدل ما تفضل فاضية
    return used if used else retrieved_chunks


def answer_question(question, top_k=TOP_K):
    retrieved = hybrid_search(question, top_k=top_k)
    context = build_context(retrieved)
    prompt = f"""
أنت مساعد تعليمي بتجاوب على أسئلة الطلاب بناءً على نص المحاضرة بس.

نص المحاضرة (الأجزاء الأقرب للسؤال)، كل جزء ليه Citation مكتوب قبله:
---
{context}
---

سؤال الطالب: {question}

القواعد:
- جاوب من المحتوى الموجود فوق بس، وميتضافش معلومات من عندك.
- جاوب بأسلوبك الخاص وبشكل منظم وواضح، ومتنقلش كلام الدكتورة حرفيًا زي ما هو (متقولش "هي قالت").
- لو السؤال بيدور على قيمة أو معادلة، انقلها زي ما هي بالظبط.
- لو مفيش علاقة خالص، قول "الإجابة دي مش موجودة في الجزء المتاح من المحاضرة".
- متحسبش الأرقام بنفسك، استخدم الـ Citation زي ما هو بالظبط.
- اكتب سطر Citation منفصل في آخر إجابتك بالشكل ده بالظبط:
Citation: [من الدقيقة X إلى الدقيقة Y]
- لو استخدمت معلومات من أكتر من جزء، اكتب كل الـ Citations بتوعهم مش واحد بس.
"""
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    answer_text = response.choices[0].message.content
    used_chunks = filter_used_chunks(answer_text, retrieved)
    return answer_text, used_chunks


# ---------- واجهة المستخدم ----------
st.title("🎧 مساعد المحاضرة")

question = st.text_input("اكتب سؤالك:")

if st.button("اسأل") and question.strip():
    with st.spinner("بدور على الإجابة..."):
        answer, retrieved = answer_question(question)
    st.session_state["answer"] = answer
    st.session_state["retrieved"] = retrieved

if "answer" in st.session_state:
    st.markdown("### الإجابة")
    st.write(st.session_state["answer"])

    st.markdown("### الأجزاء اللي جاوبت منها — دوس تسمع الجزء دة")

    for idx, chunk in enumerate(st.session_state["retrieved"]):
        start_s = chunk["start"]
        end_s = chunk["end"]
        start_label = f"{int(start_s // 60):02d}:{int(start_s % 60):02d}"
        end_label = f"{int(end_s // 60):02d}:{int(end_s % 60):02d}"

        col1, col2 = st.columns([5, 1])
        with col1:
            st.write(f"**[{start_label} - {end_label}]** {chunk['text']}")
        with col2:
            player_id = f"player_{idx}"
            html_code = f"""
            <audio id="{player_id}" src="data:audio/mp3;base64,{audio_b64}" preload="none"></audio>
            <button onclick="playSegment_{idx}()"
                style="background:#ff4b4b;color:white;border:none;border-radius:6px;padding:6px 14px;cursor:pointer;">
                ▶ اسمع
            </button>
            <script>
            function playSegment_{idx}() {{
                var audio = document.getElementById("{player_id}");
                audio.currentTime = {start_s};
                audio.play();
                var checkEnd = setInterval(function() {{
                    if (audio.currentTime >= {end_s} || audio.paused) {{
                        audio.pause();
                        clearInterval(checkEnd);
                    }}
                }}, 100);
            }}
            </script>
            """
            components.html(html_code, height=50)
