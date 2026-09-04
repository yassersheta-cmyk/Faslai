# 🎧 مساعد المحاضرة (Lecture Assistant)

نظام (RAG) بيجاوب على أسئلة الطلاب بناءً على نص محاضرة متفرغة (transcript)، ويديك زرار تسمع بيه الجزء الصوتي اللي جت منه الإجابة بالظبط.

## المميزات (Features)
- بحث هجين (Hybrid Search): دمج بين البحث الدلالي (semantic search بـ FAISS) والبحث بالكلمات المفتاحية (BM25).
- استرجاع مدعوم بالاستشهاد (Citation): كل إجابة مربوطة بتوقيت محدد في المحاضرة.
- تشغيل صوتي مباشر (Inline Audio Playback): زرار "اسمع" بيشغل بالظبط الجزء اللي اتجاوب منه.
- دعم اللغة العربية: تنظيف (stopwords) وتقسيم نص عربي.

## هيكل المشروع (Project Structure)
```
.
├── app.py              # كود التطبيق (Streamlit)
├── requirements.txt    # المكتبات المطلوبة
└── README.md
```

## طريقة التشغيل محليًا (Run Locally)

1. تثبيت المكتبات:
```bash
pip install -r requirements.txt
```

2. حط مفتاح Groq API (اختياري، ينفع تحطه في التطبيق نفسه من الشريط الجانبي):
```bash
export GROQ_API_KEY="your_key_here"
```
احصل على مفتاح مجاني من [console.groq.com](https://console.groq.com).

3. شغل التطبيق:
```bash
streamlit run app.py
```

4. من التطبيق:
   - ارفع ملف الصوت (mp3/wav) الخاص بالمحاضرة.
   - ارفع ملف الـ transcript.json (لازم يكون فيه مفتاح `segments` وكل segment فيه `text` و`start` و`end`).
   - اكتب سؤالك واضغط "اسأل".

## التشغيل على Streamlit Cloud (Deploy)

1. ارفع الريبو على GitHub.
2. من [share.streamlit.io](https://share.streamlit.io) اختار الريبو وملف `app.py`.
3. من إعدادات التطبيق (App settings → Secrets) ضيف:
```toml
GROQ_API_KEY = "your_key_here"
```

## طريقة عمل نموذج البيانات (transcript.json)
لازم يكون الملف بالشكل ده:
```json
{
  "segments": [
    {"start": 0.0, "end": 5.2, "text": "..."},
    {"start": 5.2, "end": 11.0, "text": "..."}
  ]
}
```
ده الشكل الناتج طبيعي من نماذج تفريغ الصوت زي Whisper.

## الموديلات المستخدمة (Models)
- Embeddings: `intfloat/multilingual-e5-large`
- LLM: `openai/gpt-oss-20b` عبر [Groq API](https://groq.com)
