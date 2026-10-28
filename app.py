"""
EnglishBuddy - AI English Learning App
Built with Streamlit + Groq (LLM) + SQLite.

Run:   streamlit run app.py
Key:   set GROQ_API_KEY as an environment variable, in a .env file,
       or in .streamlit/secrets.toml  (never write it inside this file).
"""

import difflib
import html
import json
import os
import random
import sqlite3
from datetime import date, datetime, timedelta

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass

from groq import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    Groq,
    RateLimitError,
)

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
APP_NAME = "EnglishBuddy"
DB_PATH = os.getenv("ENGLISHBUDDY_DB", "englishbuddy.db")
MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
MAX_TEXT_CHARS = 1500
MAX_CHAT_CHARS = 500

PAGES = [
    "🏠 Home",
    "📊 Dashboard",
    "✍️ Grammar Checker",
    "✨ Sentence Improver",
    "📚 Vocabulary Builder",
    "💬 Conversation Practice",
    "📝 Daily Quiz",
    "📈 My Progress",
]

st.set_page_config(page_title=f"{APP_NAME} - Learn English", page_icon="🎓", layout="wide")

# ----------------------------------------------------------------------------
# Styling (soft blue, white, light purple) - responsive
# ----------------------------------------------------------------------------
st.markdown(
    """
<style>
:root{
  --blue:#4F8EF7; --blue-soft:#E8F1FF; --purple:#8B7CF6; --purple-soft:#F1EEFF;
  --ink:#000000; --muted:#222222; --white:#FFFFFF;
}
html{ color-scheme:light; }
.stApp{ background:#FFFFFF; color:#000000; }
.block-container{ padding-top:2rem; padding-bottom:3rem; max-width:1150px; }
section[data-testid="stSidebar"]{ background:linear-gradient(180deg,#EAF2FF,#F3EFFF); }
h1,h2,h3{ color:var(--ink); }
.hero{ background:linear-gradient(135deg,#4F8EF7 0%,#8B7CF6 100%); color:#fff; padding:2.2rem 2rem;
       border-radius:22px; box-shadow:0 10px 30px rgba(79,142,247,.25); margin-bottom:1.2rem; }
.hero h1{ color:#fff; margin:0 0 .4rem 0; font-size:2.2rem; }
.hero p{ margin:0; font-size:1.08rem; opacity:.95; }
.card{ background:var(--white); border:1px solid #E6EBFA; border-radius:18px; padding:1.1rem 1.2rem;
       box-shadow:0 4px 16px rgba(79,100,200,.07); margin-bottom:.9rem; }
.card h4{ margin:.1rem 0 .4rem 0; color:var(--ink); }
.card p{ margin:0; color:var(--muted); }
.soft-blue{ background:var(--blue-soft); border-radius:16px; padding:1rem 1.2rem; border:1px solid #D3E4FF; }
.soft-purple{ background:var(--purple-soft); border-radius:16px; padding:1rem 1.2rem; border:1px solid #DDD6FF; }
.compare-box{ background:#fff; border:1px solid #E6EBFA; border-radius:16px; padding:1rem 1.2rem; min-height:90px;
              line-height:1.7; }
.del{ background:#FFE3E3; color:#B42318; border-radius:6px; padding:0 4px; text-decoration:line-through; }
.add{ background:#DDF6E4; color:#15692F; border-radius:6px; padding:0 4px; font-weight:600; }
.badge{ display:inline-block; background:var(--purple-soft); color:#5B4BD5; border-radius:999px;
        padding:.15rem .7rem; font-size:.8rem; font-weight:600; margin-right:.3rem; }
.word-title{ font-size:2rem; font-weight:800; color:#3B5BDB; margin:0; }
.stButton>button, .stFormSubmitButton>button{
  border-radius:12px; border:0; background:linear-gradient(135deg,#4F8EF7,#8B7CF6); color:#fff;
  font-weight:600; padding:.55rem 1.2rem; transition:transform .1s ease, box-shadow .1s ease; }
.stButton>button:hover, .stFormSubmitButton>button:hover{ transform:translateY(-1px);
  box-shadow:0 6px 16px rgba(79,142,247,.35); color:#fff; }
div[data-testid="stMetric"]{ background:#fff; border:1px solid #E6EBFA; border-radius:16px; padding:.8rem 1rem;
  box-shadow:0 4px 14px rgba(79,100,200,.06); }
@media (max-width: 768px){
  .block-container{ padding-left:1rem; padding-right:1rem; }
  .hero{ padding:1.4rem 1.1rem; } .hero h1{ font-size:1.6rem; } .word-title{ font-size:1.6rem; }
}

/* ===== Make text clear: white background, black text ===== */
[data-testid="stHeader"], [data-testid="stBottom"], [data-testid="stBottom"] > div{ background:#FFFFFF !important; }
.stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp h5,.stApp h6,
.stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stMarkdownContainer"] li,
.stApp [data-testid="stMarkdownContainer"] strong,
.stApp label, .stApp label p,
.stApp [data-testid="stWidgetLabel"] p,
.stApp [data-testid="stMetricLabel"] *, .stApp [data-testid="stMetricValue"] *,
.stApp [data-testid="stExpander"] summary, .stApp [data-testid="stExpander"] summary p,
.stApp button[data-baseweb="tab"] p,
.stApp [data-testid="stChatMessage"] p,
.stApp [data-testid="stRadio"] p{ color:#000000 !important; }
.stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stCaptionContainer"] p{ color:#222222 !important; }
/* keep white text on colored areas */
.stApp .hero h1, .stApp .hero p{ color:#FFFFFF !important; }
.stApp .stButton button p, .stApp .stFormSubmitButton button p, .stApp .stButton button, .stApp .stFormSubmitButton button{ color:#FFFFFF !important; }
.stApp p.word-title{ color:#2F4FC4 !important; }
/* input boxes */
.stApp textarea, .stApp input, [data-testid="stChatInput"] textarea{
  color:#000000 !important; -webkit-text-fill-color:#000000 !important; background:#FFFFFF !important; }
.stApp textarea, .stApp input{ border:1px solid #B9C6F2 !important; border-radius:10px !important; }
.stApp textarea::placeholder, .stApp input::placeholder{ color:#555555 !important; -webkit-text-fill-color:#555555 !important; }
[data-testid="stChatInput"]{ background:#FFFFFF !important; border:1px solid #B9C6F2; border-radius:14px; }
.stApp div[data-baseweb="select"] > div{ background:#FFFFFF !important; color:#000000 !important; border:1px solid #B9C6F2 !important; }
.stApp div[data-baseweb="select"] *{ color:#000000 !important; }
div[data-baseweb="popover"] *, ul[role="listbox"] *{ color:#000000 !important; background-color:#FFFFFF !important; }
/* cards, chat, expanders, alerts */
.stApp [data-testid="stChatMessage"]{ background:#F7F9FF !important; border:1px solid #E6EBFA; border-radius:14px; }
.stApp [data-testid="stExpander"]{ background:#FFFFFF !important; border:1px solid #E6EBFA; border-radius:12px; }
.stApp [data-testid="stAlert"]{ background:#F1EEFF !important; border:1px solid #DDD6FF; }
.stApp [data-testid="stAlert"] *{ color:#000000 !important; }
section[data-testid="stSidebar"] *{ color:#000000; }
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Database (SQLite)
# ----------------------------------------------------------------------------


def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS activities(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                day TEXT NOT NULL,
                kind TEXT NOT NULL,
                detail TEXT,
                score INTEGER,
                total INTEGER
            );
            CREATE TABLE IF NOT EXISTS saved_words(
                word TEXT PRIMARY KEY,
                meaning TEXT, pronunciation TEXT, example TEXT,
                synonyms TEXT, antonyms TEXT, saved_at TEXT
            );
            CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT);
            """
        )


def log_activity(kind, detail="", score=None, total=None):
    now = datetime.now()
    with db() as conn:
        conn.execute(
            "INSERT INTO activities(ts, day, kind, detail, score, total) VALUES (?,?,?,?,?,?)",
            (now.isoformat(timespec="seconds"), now.date().isoformat(), kind, detail[:200], score, total),
        )


def get_setting(key, default):
    with db() as conn:
        row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


def set_setting(key, value):
    with db() as conn:
        conn.execute("INSERT OR REPLACE INTO settings(key,value) VALUES (?,?)", (key, str(value)))


def save_word(entry):
    with db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO saved_words VALUES (?,?,?,?,?,?,?)",
            (
                entry["word"].strip().lower(),
                entry.get("meaning", ""),
                entry.get("pronunciation", ""),
                entry.get("example", ""),
                ", ".join(entry.get("synonyms", [])),
                ", ".join(entry.get("antonyms", [])),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
    log_activity("Vocabulary", f"Saved word: {entry['word']}")


def remove_word(word):
    with db() as conn:
        conn.execute("DELETE FROM saved_words WHERE word=?", (word,))


def get_saved_words():
    with db() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM saved_words ORDER BY saved_at DESC")]


def activities_df():
    with db() as conn:
        rows = conn.execute("SELECT * FROM activities ORDER BY id").fetchall()
    return pd.DataFrame([dict(r) for r in rows])


def compute_streak(df):
    if df.empty:
        return 0
    days = set(df["day"].tolist())
    d = date.today()
    if d.isoformat() not in days:
        d -= timedelta(days=1)
    streak = 0
    while d.isoformat() in days:
        streak += 1
        d -= timedelta(days=1)
    return streak


init_db()

# ----------------------------------------------------------------------------
# Groq AI helpers (real API calls only - errors are shown, never faked)
# ----------------------------------------------------------------------------


class AIError(Exception):
    """A friendly error message that can be shown to the user."""


def read_api_key():
    key = os.getenv("GROQ_API_KEY")
    if key:
        return key.strip()
    try:
        key = st.secrets.get("GROQ_API_KEY")
        return key.strip() if key else None
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def make_client(key):
    return Groq(api_key=key)


def ask_json(system, user=None, messages=None, temperature=0.3, max_tokens=1800):
    """Call Groq and return parsed JSON. Raises AIError with a friendly message on any problem."""
    key = read_api_key()
    if not key:
        raise AIError(
            "The AI key is missing. Set GROQ_API_KEY as an environment variable "
            "(or in .streamlit/secrets.toml) and restart the app."
        )
    msgs = [{"role": "system", "content": system + "\nReturn ONLY one valid JSON object. No extra text."}]
    msgs += messages if messages else [{"role": "user", "content": user}]
    try:
        resp = make_client(key).chat.completions.create(
            model=MODEL,
            messages=msgs,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
    except AuthenticationError:
        raise AIError("The AI key is not valid. Please check your GROQ_API_KEY.")
    except RateLimitError:
        raise AIError("Too many requests right now. Please wait a minute and try again.")
    except APIConnectionError:
        raise AIError("I could not reach the AI service. Please check your internet and try again.")
    except APIStatusError as e:
        raise AIError(f"The AI service had a problem (code {e.status_code}). Please try again.")
    except Exception as e:  # noqa: BLE001
        raise AIError(f"Something went wrong while contacting the AI: {type(e).__name__}.")
    try:
        data = json.loads(resp.choices[0].message.content)
        if not isinstance(data, dict):
            raise ValueError
        return data
    except Exception:  # noqa: BLE001
        raise AIError("The AI sent an answer I could not read. Please press the button again.")


def clean_text(text, limit):
    """Validate user text. Returns (text, error)."""
    text = (text or "").strip()
    if not text:
        return None, "Please type something first."
    if len(text) > limit:
        return None, f"Your text is too long. Please use fewer than {limit} characters."
    if not any(c.isalpha() for c in text):
        return None, "Please write some English words (letters), not only numbers or symbols."
    return text, None


def as_list(value):
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str) and value:
        return [value]
    return []


# ----------------------------------------------------------------------------
# Sample learning content (built in, works without AI)
# ----------------------------------------------------------------------------
SAMPLE_WORDS = [
    {"word": "curious", "pronunciation": "/ˈkjʊə.ri.əs/", "meaning": "Wanting to know or learn about things.",
     "example": "The curious child asked many questions.", "synonyms": ["interested", "inquisitive"], "antonyms": ["bored", "uninterested"]},
    {"word": "generous", "pronunciation": "/ˈdʒen.ər.əs/", "meaning": "Happy to give time, money, or help to others.",
     "example": "My generous uncle paid for our trip.", "synonyms": ["kind", "giving"], "antonyms": ["selfish", "mean"]},
    {"word": "improve", "pronunciation": "/ɪmˈpruːv/", "meaning": "To become better, or to make something better.",
     "example": "I read every day to improve my English.", "synonyms": ["get better", "develop"], "antonyms": ["worsen", "decline"]},
    {"word": "journey", "pronunciation": "/ˈdʒɜː.ni/", "meaning": "A trip from one place to another.",
     "example": "The journey to the mountains took five hours.", "synonyms": ["trip", "voyage"], "antonyms": ["stay"]},
    {"word": "confident", "pronunciation": "/ˈkɒn.fɪ.dənt/", "meaning": "Feeling sure about yourself or your ability.",
     "example": "She felt confident before her speech.", "synonyms": ["sure", "self-assured"], "antonyms": ["nervous", "doubtful"]},
    {"word": "delicious", "pronunciation": "/dɪˈlɪʃ.əs/", "meaning": "Having a very good taste or smell.",
     "example": "The soup was warm and delicious.", "synonyms": ["tasty", "yummy"], "antonyms": ["awful", "tasteless"]},
    {"word": "patient", "pronunciation": "/ˈpeɪ.ʃənt/", "meaning": "Able to wait calmly without getting angry.",
     "example": "Be patient. Learning takes time.", "synonyms": ["calm", "tolerant"], "antonyms": ["impatient", "restless"]},
    {"word": "honest", "pronunciation": "/ˈɒn.ɪst/", "meaning": "Always telling the truth.",
     "example": "He is an honest man, so I trust him.", "synonyms": ["truthful", "sincere"], "antonyms": ["dishonest", "false"]},
    {"word": "achieve", "pronunciation": "/əˈtʃiːv/", "meaning": "To succeed in doing something, often after hard work.",
     "example": "With practice, you can achieve your goals.", "synonyms": ["accomplish", "reach"], "antonyms": ["fail", "give up"]},
    {"word": "fragile", "pronunciation": "/ˈfrædʒ.aɪl/", "meaning": "Easy to break or damage.",
     "example": "Please be careful with the fragile glass.", "synonyms": ["delicate", "breakable"], "antonyms": ["strong", "sturdy"]},
]

SAMPLE_QUIZ = {
    "Beginner": [
        {"question": "She ___ to school every day.", "options": ["go", "goes", "going", "gone"], "answer_index": 1,
         "explanation": "With he/she/it in the present simple, we add -s: 'She goes'."},
        {"question": "Choose the correct sentence.", "options": ["I am have a dog.", "I has a dog.", "I have a dog.", "I having a dog."],
         "answer_index": 2, "explanation": "Use 'have' with 'I'. 'I have a dog.'"},
        {"question": "What does 'happy' mean?", "options": ["Sad", "Glad and joyful", "Very tired", "Angry"], "answer_index": 1,
         "explanation": "'Happy' means you feel glad and joyful."},
    ],
    "Intermediate": [
        {"question": "If it ___ tomorrow, we will stay at home.", "options": ["rain", "rains", "will rain", "rained"], "answer_index": 1,
         "explanation": "In a real 'if' sentence, use the present simple after 'if': 'If it rains'."},
        {"question": "I have lived here ___ 2019.", "options": ["for", "since", "from", "during"], "answer_index": 1,
         "explanation": "Use 'since' with a starting point in time (2019). Use 'for' with a length of time."},
        {"question": "Which word means 'very tired'?", "options": ["Exhausted", "Energetic", "Curious", "Generous"], "answer_index": 0,
         "explanation": "'Exhausted' means extremely tired."},
    ],
    "Advanced": [
        {"question": "Had I known about the delay, I ___ earlier.", "options": ["would leave", "will have left", "would have left", "had left"],
         "answer_index": 2, "explanation": "This is the third conditional with inversion: 'Had I known... I would have left'."},
        {"question": "Choose the word closest in meaning to 'ubiquitous'.", "options": ["Rare", "Everywhere", "Ancient", "Hidden"],
         "answer_index": 1, "explanation": "'Ubiquitous' means found everywhere."},
        {"question": "Not only ___ late, but he also forgot the documents.", "options": ["he was", "was he", "he is", "did he was"],
         "answer_index": 1, "explanation": "After 'Not only' at the start, we invert the subject and verb: 'was he'."},
    ],
}

SCENARIOS = {
    "👋 Introducing yourself": "You are meeting a new person for the first time at a friendly event.",
    "🛍️ Shopping": "You are a shop assistant helping a customer buy clothes in a store.",
    "✈️ Travelling": "You are an airport or hotel staff member helping a traveller.",
    "💼 Job interview": "You are a kind interviewer asking simple job interview questions.",
    "🍽️ At a restaurant": "You are a waiter taking an order at a restaurant.",
    "🩺 At the doctor": "You are a friendly doctor asking a patient how they feel.",
}

# ----------------------------------------------------------------------------
# Small UI helpers
# ----------------------------------------------------------------------------


def go(page):
    st.session_state.page = page


def show_header(title, subtitle):
    st.markdown(f"<div class='hero'><h1>{title}</h1><p>{subtitle}</p></div>", unsafe_allow_html=True)


def speak_button(text, label="🔊 Listen"):
    """Uses the browser's built-in text-to-speech (no extra service needed)."""
    js_text = json.dumps(text)
    components.html(
        f"""
        <button style="background:#EEF2FF;border:1px solid #D3DBFF;border-radius:10px;padding:6px 14px;
                       color:#3B5BDB;font-weight:600;cursor:pointer;font-family:sans-serif"
          onclick='(function(){{var u=new SpeechSynthesisUtterance({js_text});u.lang="en-US";u.rate=0.9;
          window.speechSynthesis.cancel();window.speechSynthesis.speak(u);}})()'>{label}</button>
        """,
        height=45,
    )


def diff_html(original, corrected):
    a, b = original.split(), corrected.split()
    sm = difflib.SequenceMatcher(None, a, b)
    left, right = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            left.append(html.escape(" ".join(a[i1:i2])))
            right.append(html.escape(" ".join(b[j1:j2])))
        else:
            if i2 > i1:
                left.append(f"<span class='del'>{html.escape(' '.join(a[i1:i2]))}</span>")
            if j2 > j1:
                right.append(f"<span class='add'>{html.escape(' '.join(b[j1:j2]))}</span>")
    return " ".join(left), " ".join(right)


def word_card(entry, key_prefix):
    st.markdown(
        f"""<div class='card'>
        <p class='word-title'>{html.escape(entry['word'])}</p>
        <p style='margin:.2rem 0 .6rem 0'><span class='badge'>{html.escape(entry.get('part_of_speech', 'word'))}</span>
        <span style='color:#6B7391'>{html.escape(entry.get('pronunciation', ''))}</span></p>
        <p style='color:#27304A;font-size:1.05rem'><b>Meaning:</b> {html.escape(entry.get('meaning', ''))}</p>
        <p style='color:#27304A;font-size:1.05rem;margin-top:.4rem'><b>Example:</b> <i>{html.escape(entry.get('example', ''))}</i></p>
        <p style='margin-top:.6rem'><b style='color:#15692F'>Synonyms:</b> {html.escape(', '.join(entry.get('synonyms', [])) or '-')}
        &nbsp;&nbsp; <b style='color:#B42318'>Antonyms:</b> {html.escape(', '.join(entry.get('antonyms', [])) or '-')}</p>
        {f"<p style='margin-top:.6rem' class='soft-purple'>💡 {html.escape(entry['tip'])}</p>" if entry.get('tip') else ''}
        </div>""",
        unsafe_allow_html=True,
    )
    c1, c2, _ = st.columns([1, 1, 3])
    with c1:
        speak_button(f"{entry['word']}. {entry.get('example', '')}")
    with c2:
        saved = entry["word"].lower() in {w["word"] for w in get_saved_words()}
        if saved:
            st.button("✅ Saved", key=f"{key_prefix}_saved", disabled=True)
        elif st.button("⭐ Save word", key=f"{key_prefix}_save"):
            save_word(entry)
            st.rerun()


# ----------------------------------------------------------------------------
# Generic quiz renderer (used by Daily Quiz and Vocabulary Quiz)
# ----------------------------------------------------------------------------


def clear_quiz(key):
    st.session_state.pop(key, None)
    for k in [k for k in st.session_state if k.startswith(f"{key}_a")]:
        del st.session_state[k]


def render_quiz(key, kind):
    quiz = st.session_state.get(key)
    if not quiz:
        return
    qs = quiz["questions"]

    if quiz.get("result") is None:
        st.markdown(f"<span class='badge'>{html.escape(quiz['label'])}</span>", unsafe_allow_html=True)
        with st.form(f"form_{key}"):
            answers = []
            for i, q in enumerate(qs):
                st.markdown(f"**Question {i + 1}.** {q['question']}")
                answers.append(st.radio("Choose one answer", q["options"], index=None,
                                        key=f"{key}_a{i}", label_visibility="collapsed"))
                st.write("")
            submitted = st.form_submit_button("✅ Submit answers")
        if submitted:
            if any(a is None for a in answers):
                st.warning("Please answer all the questions before you submit.")
            else:
                score = sum(1 for q, a in zip(qs, answers) if a == q["options"][q["answer_index"]])
                quiz["result"] = {"score": score, "answers": answers}
                log_activity(kind, quiz["label"], score=score, total=len(qs))
                st.rerun()
        return

    res = quiz["result"]
    score, total = res["score"], len(qs)
    st.markdown(f"### Your score: {score} / {total}")
    st.progress(score / total)
    if score == total:
        st.success("Perfect score! Great job! 🎉")
        st.balloons()
    elif score >= total / 2:
        st.info("Good work! Read the explanations below and try again.")
    else:
        st.info("Nice try! Every mistake helps you learn. Read the explanations below. 💪")
    for i, (q, a) in enumerate(zip(qs, res["answers"])):
        correct = q["options"][q["answer_index"]]
        ok = a == correct
        box = "soft-blue" if ok else "soft-purple"
        st.markdown(
            f"<div class='{box}' style='margin-bottom:.7rem'><b>Q{i + 1}. {html.escape(q['question'])}</b><br>"
            f"{'✅' if ok else '❌'} Your answer: {html.escape(str(a))}<br>"
            f"{'' if ok else '✔️ Correct answer: <b>' + html.escape(correct) + '</b><br>'}"
            f"💡 {html.escape(q.get('explanation', ''))}</div>",
            unsafe_allow_html=True,
        )
    st.button("🔄 Take a new quiz", key=f"new_{key}", on_click=clear_quiz, args=(key,))


def validate_questions(raw):
    good = []
    for q in raw if isinstance(raw, list) else []:
        try:
            opts = [str(o) for o in q["options"]]
            idx = int(q["answer_index"])
            if len(opts) == 4 and 0 <= idx < 4 and len(set(opts)) == 4 and str(q["question"]).strip():
                good.append({"question": str(q["question"]).strip(), "options": opts, "answer_index": idx,
                             "explanation": str(q.get("explanation", "")).strip()})
        except Exception:  # noqa: BLE001
            continue
    return good


# ----------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------


def page_home():
    st.markdown(
        """<div class='hero'><h1>Welcome to EnglishBuddy 🎓</h1>
        <p>Learn English step by step. It is okay to make mistakes here. Every mistake helps you learn.</p></div>""",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns([1, 1])
    c1.button("🚀 Go to my Dashboard", on_click=go, args=("📊 Dashboard",), use_container_width=True)
    c2.button("✍️ Check my grammar now", on_click=go, args=("✍️ Grammar Checker",), use_container_width=True)
    st.write("")
    st.subheader("What can you do here?")
    feats = [
        ("✍️ Grammar Checker", "Find mistakes and see simple explanations.", "✍️ Grammar Checker"),
        ("✨ Sentence Improver", "Make your sentences sound more natural.", "✨ Sentence Improver"),
        ("📚 Vocabulary Builder", "Learn a new word every day and save your favorites.", "📚 Vocabulary Builder"),
        ("💬 Conversation Practice", "Talk with an AI partner. No stress, no judgment.", "💬 Conversation Practice"),
        ("📝 Daily Quiz", "Test yourself at your level and learn from answers.", "📝 Daily Quiz"),
        ("📈 My Progress", "See your streak, scores, and goals.", "📈 My Progress"),
    ]
    cols = st.columns(3)
    for i, (title, desc, page) in enumerate(feats):
        with cols[i % 3]:
            st.markdown(f"<div class='card'><h4>{title}</h4><p>{desc}</p></div>", unsafe_allow_html=True)
            st.button("Open", key=f"home_open_{i}", on_click=go, args=(page,), use_container_width=True)
    st.markdown(
        "<div class='soft-blue'>💙 <b>Tip for beginners:</b> Practice a little every day. "
        "Even 10 minutes is great!</div>",
        unsafe_allow_html=True,
    )


def page_dashboard():
    show_header("Your Dashboard 📊", "See how you are doing today and choose what to learn next.")
    df = activities_df()
    today = date.today().isoformat()
    today_count = int((df["day"] == today).sum()) if not df.empty else 0
    goal = int(get_setting("daily_goal", 3))
    quizzes = df[df["kind"].isin(["Quiz", "Vocabulary Quiz"])] if not df.empty else df
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔥 Day streak", compute_streak(df))
    c2.metric("✅ Today's activities", f"{today_count}/{goal}")
    c3.metric("📝 Quizzes done", len(quizzes))
    c4.metric("⭐ Saved words", len(get_saved_words()))

    st.markdown("#### Today's goal")
    st.progress(min(today_count / goal, 1.0))
    if today_count >= goal:
        st.success("You reached your goal today! Well done! 🎉")
    else:
        st.caption(f"Do {goal - today_count} more activit{'y' if goal - today_count == 1 else 'ies'} to reach your goal.")

    st.markdown("#### Start learning")
    cols = st.columns(4)
    quick = [("✍️ Grammar", "✍️ Grammar Checker"), ("✨ Improve", "✨ Sentence Improver"),
             ("📚 Words", "📚 Vocabulary Builder"), ("💬 Talk", "💬 Conversation Practice")]
    for col, (label, page) in zip(cols, quick):
        col.button(label, key=f"dash_{page}", on_click=go, args=(page,), use_container_width=True)

    st.markdown("#### Word of the day")
    word = SAMPLE_WORDS[date.today().toordinal() % len(SAMPLE_WORDS)]
    st.markdown(
        f"<div class='card'><p class='word-title'>{word['word']}</p>"
        f"<p>{word['meaning']}</p><p><i>{word['example']}</i></p></div>",
        unsafe_allow_html=True,
    )


def clear_grammar():
    st.session_state.pop("gc_result", None)
    st.session_state.gc_input = ""


def page_grammar():
    show_header("AI Grammar Checker ✍️", "Write a sentence or paragraph. I will find mistakes and explain them simply.")
    text = st.text_area("Your English text", height=160, max_chars=MAX_TEXT_CHARS, key="gc_input",
                        placeholder="Example: She go to school yesterday and buyed a book.")
    c1, c2, _ = st.columns([1, 1, 3])
    run = c1.button("🔍 Check grammar", use_container_width=True)
    c2.button("🧹 Clear", use_container_width=True, on_click=clear_grammar)

    if run:
        clean, err = clean_text(text, MAX_TEXT_CHARS)
        if err:
            st.error(err)
        else:
            system = (
                "You are a kind English teacher for beginners. Check the user's text for grammar, spelling and "
                "punctuation mistakes. Keep the meaning and the user's style. Explain each mistake in VERY simple "
                "English (short sentences, easy words). Respond in JSON with keys: "
                '"corrected_text" (string), "has_mistakes" (boolean), '
                '"mistakes" (list of objects with "original", "correction", "type" (Grammar, Spelling or Punctuation), '
                '"explanation"), "encouragement" (one short kind sentence). '
                "If there are no mistakes, set has_mistakes to false, return the same text and an empty list."
            )
            try:
                with st.spinner("Checking your text..."):
                    data = ask_json(system, clean, temperature=0.2)
                if not isinstance(data.get("corrected_text"), str) or not data["corrected_text"].strip():
                    raise AIError("The AI did not return a corrected text. Please try again.")
                st.session_state.gc_result = {"original": clean, "data": data}
                log_activity("Grammar Check", clean[:80])
            except AIError as e:
                st.session_state.pop("gc_result", None)
                st.error(f"😕 {e}")

    res = st.session_state.get("gc_result")
    if res:
        data, original = res["data"], res["original"]
        corrected = data["corrected_text"].strip()
        mistakes = [m for m in data.get("mistakes", []) if isinstance(m, dict)]
        left, right = diff_html(original, corrected)
        st.markdown("### Original and corrected")
        a, b = st.columns(2)
        a.markdown(f"**Your text**<div class='compare-box'>{left}</div>", unsafe_allow_html=True)
        b.markdown(f"**Corrected text**<div class='compare-box'>{right}</div>", unsafe_allow_html=True)
        if data.get("has_mistakes") and mistakes:
            st.markdown(f"### What to fix ({len(mistakes)})")
            for m in mistakes:
                st.markdown(
                    f"<div class='card'><span class='badge'>{html.escape(str(m.get('type', 'Grammar')))}</span>"
                    f"<span class='del'>{html.escape(str(m.get('original', '')))}</span> → "
                    f"<span class='add'>{html.escape(str(m.get('correction', '')))}</span>"
                    f"<p style='margin-top:.5rem;color:#27304A'>💡 {html.escape(str(m.get('explanation', '')))}</p></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.success("No mistakes found. Your English looks great! 🎉")
        if data.get("encouragement"):
            st.info(f"💙 {data['encouragement']}")
        speak_button(corrected, "🔊 Listen to corrected text")


def page_improver():
    show_header("AI Sentence Improver ✨", "Write a simple sentence. I will show you better ways to say it.")
    text = st.text_area("Your sentence", height=110, max_chars=500, key="si_input",
                        placeholder="Example: I like my job because it is good.")
    if st.button("✨ Improve my sentence"):
        clean, err = clean_text(text, 500)
        if err:
            st.error(err)
        else:
            system = (
                "You are a friendly English coach. Rewrite the user's sentence so it sounds natural and fluent. "
                "Keep the same meaning. Respond in JSON with keys: "
                '"natural_version" (string), "beginner" (string, simple words), "intermediate" (string), '
                '"advanced" (string, richer vocabulary and structure), '
                '"explanations" (list of 2 to 4 short strings that explain the improvements in very easy words).'
            )
            try:
                with st.spinner("Improving your sentence..."):
                    data = ask_json(system, clean, temperature=0.6)
                needed = ["natural_version", "beginner", "intermediate", "advanced"]
                if not all(isinstance(data.get(k), str) and data[k].strip() for k in needed):
                    raise AIError("The AI answer was incomplete. Please try again.")
                st.session_state.si_result = {"original": clean, "data": data}
                log_activity("Sentence Improver", clean[:80])
            except AIError as e:
                st.session_state.pop("si_result", None)
                st.error(f"😕 {e}")

    res = st.session_state.get("si_result")
    if res:
        d = res["data"]
        st.markdown(f"**Your sentence:** {html.escape(res['original'])}")
        st.markdown(f"<div class='soft-blue'><b>🌟 Most natural version</b><br>{html.escape(d['natural_version'])}</div>",
                    unsafe_allow_html=True)
        st.write("")
        cols = st.columns(3)
        for col, (title, key, icon) in zip(cols, [("Beginner", "beginner", "🌱"), ("Intermediate", "intermediate", "🌿"),
                                                  ("Advanced", "advanced", "🌳")]):
            col.markdown(f"<div class='card'><h4>{icon} {title}</h4><p style='color:#27304A'>{html.escape(d[key])}</p></div>",
                         unsafe_allow_html=True)
        expl = as_list(d.get("explanations"))
        if expl:
            st.markdown("### Why is it better?")
            for e in expl:
                st.markdown(f"- {e}")
        speak_button(d["natural_version"], "🔊 Listen to natural version")


def lookup_word(word):
    system = (
        "You are an English vocabulary teacher for beginners. Explain the English word in very simple English. "
        "Respond in JSON with keys: \"word\", \"part_of_speech\", \"pronunciation\" (IPA between slashes), "
        "\"meaning\" (one easy sentence), \"example\" (one easy example sentence), \"synonyms\" (list of 2-4), "
        "\"antonyms\" (list of 0-3), \"tip\" (one short memory tip)."
    )
    data = ask_json(system, f"Word: {word}", temperature=0.4)
    if not isinstance(data.get("meaning"), str) or not data.get("meaning").strip():
        raise AIError("The AI could not explain that word. Please check the spelling and try again.")
    data["word"] = str(data.get("word") or word).strip().lower()
    data["synonyms"], data["antonyms"] = as_list(data.get("synonyms")), as_list(data.get("antonyms"))
    return data


def make_vocab_quiz():
    pool = {w["word"]: w for w in SAMPLE_WORDS}
    for w in get_saved_words():
        if w["meaning"]:
            pool[w["word"]] = w
    items = list(pool.values())
    chosen = random.sample(items, min(5, len(items)))
    questions = []
    for w in chosen:
        wrong = random.sample([x["meaning"] for x in items if x["word"] != w["word"]], 3)
        options = wrong + [w["meaning"]]
        random.shuffle(options)
        questions.append({
            "question": f"What does '{w['word']}' mean?",
            "options": options,
            "answer_index": options.index(w["meaning"]),
            "explanation": f"'{w['word']}' means: {w['meaning']} Example: {w.get('example', '')}",
        })
    return questions


def page_vocab():
    show_header("Vocabulary Builder 📚", "Learn new words, save your favorites, and test yourself.")
    tab1, tab2, tab3, tab4 = st.tabs(["🌅 Word of the day", "🔎 Look up a word", "⭐ My saved words", "🧩 Vocabulary quiz"])

    with tab1:
        entry = dict(SAMPLE_WORDS[date.today().toordinal() % len(SAMPLE_WORDS)])
        entry["part_of_speech"] = "word of the day"
        word_card(entry, "daily")
        st.caption("Come back tomorrow for a new word!")
        if st.button("🤖 Give me a new AI word", key="ai_word_btn"):
            level = random.choice(["beginner", "intermediate"])
            seen = ", ".join(w["word"] for w in get_saved_words()[:20])
            system = (
                "You are an English vocabulary teacher. Pick ONE useful everyday English word for a "
                f"{level} learner (not one of: {seen or 'none'}) and explain it simply. Respond in JSON with keys: "
                "\"word\", \"part_of_speech\", \"pronunciation\" (IPA), \"meaning\", \"example\", "
                "\"synonyms\" (list), \"antonyms\" (list), \"tip\"."
            )
            try:
                with st.spinner("Finding a word..."):
                    data = ask_json(system, f"Random seed: {random.randint(1, 10**6)}", temperature=1.0)
                    if not isinstance(data.get("word"), str) or not isinstance(data.get("meaning"), str):
                        raise AIError("The AI answer was incomplete. Please try again.")
                    data["word"] = data["word"].strip().lower()
                    data["synonyms"], data["antonyms"] = as_list(data.get("synonyms")), as_list(data.get("antonyms"))
                st.session_state.ai_word = data
                log_activity("Vocabulary", f"AI word: {data['word']}")
            except AIError as e:
                st.error(f"😕 {e}")
        if st.session_state.get("ai_word"):
            word_card(st.session_state.ai_word, "aiword")

    with tab2:
        w = st.text_input("Type an English word", max_chars=40, key="lookup_input", placeholder="Example: reliable")
        if st.button("🔎 Explain this word"):
            clean, err = clean_text(w, 40)
            if err:
                st.error(err)
            elif len(clean.split()) > 3:
                st.error("Please type one word (or a short phrase).")
            else:
                try:
                    with st.spinner("Looking it up..."):
                        st.session_state.lookup = lookup_word(clean)
                    log_activity("Vocabulary", f"Looked up: {clean}")
                except AIError as e:
                    st.session_state.pop("lookup", None)
                    st.error(f"😕 {e}")
        if st.session_state.get("lookup"):
            word_card(st.session_state.lookup, "lookup")

    with tab3:
        saved = get_saved_words()
        if not saved:
            st.info("You have no saved words yet. Press ⭐ Save word on any word card.")
        for s in saved:
            with st.expander(f"⭐ {s['word']} - {s['meaning'][:60]}"):
                st.markdown(f"**Pronunciation:** {s['pronunciation'] or '-'}")
                st.markdown(f"**Meaning:** {s['meaning']}")
                st.markdown(f"**Example:** *{s['example']}*")
                st.markdown(f"**Synonyms:** {s['synonyms'] or '-'}  \n**Antonyms:** {s['antonyms'] or '-'}")
                st.button("🗑️ Remove", key=f"rm_{s['word']}", on_click=remove_word, args=(s["word"],))

    with tab4:
        st.write("Answer 5 questions about word meanings. Words you saved are included!")
        if st.button("▶️ Start vocabulary quiz"):
            clear_quiz("vquiz")
            st.session_state.vquiz = {"questions": make_vocab_quiz(), "label": "Vocabulary quiz", "result": None}
            st.rerun()
        render_quiz("vquiz", "Vocabulary Quiz")


def page_conversation():
    show_header("AI Conversation Practice 💬", "Talk about everyday topics. I will help you gently and politely.")
    scenario = st.selectbox("Choose a situation", list(SCENARIOS.keys()))
    chat = st.session_state.get("chat")
    if chat and chat["scenario"] != scenario:
        st.info("You changed the situation. Press **Start new conversation** to practice it.")

    def system_prompt(sc):
        return (
            f"You are an English conversation partner for a nervous beginner. Situation: {SCENARIOS[sc]} "
            "Stay in this role. Use simple words and short replies (1 to 3 sentences) and usually end with a "
            "simple question. Be warm and encouraging. Respond in JSON with keys: "
            '"reply" (your in-character reply), '
            '"correction" (if the user\'s LAST message had mistakes, one polite short note with the corrected '
            'sentence, like "Nice try! Say: ..."; otherwise an empty string), '
            '"suggestion" (a more natural way to say the user\'s last message, or an empty string if it was already natural).'
        )

    def call_chat(sc, api_msgs):
        data = ask_json(system_prompt(sc), messages=api_msgs[-20:], temperature=0.7, max_tokens=500)
        if not isinstance(data.get("reply"), str) or not data["reply"].strip():
            raise AIError("The AI did not reply. Please try again.")
        return data

    if st.button("🆕 Start new conversation"):
        api = [{"role": "user", "content": "(Start the conversation now with a friendly greeting and a first simple question.)"}]
        try:
            with st.spinner("Starting..."):
                data = call_chat(scenario, api)
            api.append({"role": "assistant", "content": json.dumps({"reply": data["reply"]})})
            st.session_state.chat = {"scenario": scenario, "api": api,
                                     "display": [{"role": "assistant", "reply": data["reply"], "correction": "", "suggestion": ""}]}
            log_activity("Conversation", f"Started: {scenario}")
        except AIError as e:
            st.error(f"😕 {e}")
        st.rerun() if st.session_state.get("chat") else None

    chat = st.session_state.get("chat")
    if not chat:
        st.info("Press **Start new conversation** to begin. 😊")
        return

    for m in chat["display"]:
        if m["role"] == "user":
            with st.chat_message("user"):
                st.write(m["text"])
        else:
            with st.chat_message("assistant", avatar="🎓"):
                st.write(m["reply"])
                if m.get("correction"):
                    st.markdown(f"<div class='soft-purple'>✏️ {html.escape(m['correction'])}</div>", unsafe_allow_html=True)
                if m.get("suggestion"):
                    st.markdown(f"<div class='soft-blue' style='margin-top:.4rem'>💡 More natural: {html.escape(m['suggestion'])}</div>",
                                unsafe_allow_html=True)

    user_msg = st.chat_input("Type your answer in English...", max_chars=MAX_CHAT_CHARS)
    if user_msg:
        clean, err = clean_text(user_msg, MAX_CHAT_CHARS)
        if err:
            st.error(err)
            return
        chat["api"].append({"role": "user", "content": clean})
        try:
            with st.spinner("Thinking..."):
                data = call_chat(chat["scenario"], chat["api"])
            chat["api"].append({"role": "assistant", "content": json.dumps({"reply": data["reply"]})})
            chat["display"].append({"role": "user", "text": clean})
            chat["display"].append({"role": "assistant", "reply": data["reply"],
                                    "correction": str(data.get("correction") or ""),
                                    "suggestion": str(data.get("suggestion") or "")})
            log_activity("Conversation", chat["scenario"])
        except AIError as e:
            chat["api"].pop()  # remove the message that did not get an answer
            st.error(f"😕 {e} Your message was not sent. Please try again.")
            return
        st.rerun()


def page_quiz():
    show_header("Daily Quiz 📝", "Choose your level and test your grammar and vocabulary.")
    c1, c2, c3 = st.columns(3)
    level = c1.selectbox("Level", ["Beginner", "Intermediate", "Advanced"])
    topic = c2.selectbox("Topic", ["Mixed", "Grammar", "Vocabulary"])
    source = c3.selectbox("Questions from", ["AI (new questions)", "Sample questions"])
    if st.button("▶️ Start quiz"):
        clear_quiz("quiz")
        label = f"{level} - {topic}"
        if source.startswith("Sample"):
            qs = list(SAMPLE_QUIZ[level])
            random.shuffle(qs)
            st.session_state.quiz = {"questions": qs, "label": f"{label} (sample)", "result": None}
        else:
            system = (
                f"You are an English teacher. Create 5 multiple-choice questions for a {level} learner about "
                f"{'grammar and vocabulary' if topic == 'Mixed' else topic.lower()}. Each question must have exactly 4 "
                "different options and exactly one correct answer. Respond in JSON: "
                '{"questions":[{"question":"...","options":["a","b","c","d"],"answer_index":0,'
                '"explanation":"very simple explanation of why the answer is correct"}]}. '
                "Use '___' for blanks in fill-in questions. Vary the position of the correct answer."
            )
            try:
                with st.spinner("Making your quiz..."):
                    data = ask_json(system, f"Variety seed: {random.randint(1, 10**6)}", temperature=0.9, max_tokens=2200)
                    qs = validate_questions(data.get("questions"))
                if len(qs) < 3:
                    raise AIError("The AI made a quiz that was not good enough. Please try again.")
                st.session_state.quiz = {"questions": qs, "label": label, "result": None}
            except AIError as e:
                st.error(f"😕 {e}")
        st.rerun() if st.session_state.get("quiz") else None
    if not st.session_state.get("quiz"):
        st.info("Choose your level and press **Start quiz**. You can do it! 💪")
    render_quiz("quiz", "Quiz")


def page_progress():
    show_header("My Progress 📈", "See how much you have learned. Small steps every day make a big change!")
    df = activities_df()
    goal = int(get_setting("daily_goal", 3))
    new_goal = st.number_input("Daily goal (activities per day)", 1, 20, goal)
    if int(new_goal) != goal:
        set_setting("daily_goal", int(new_goal))
        st.rerun()

    if df.empty:
        st.info("No activity yet. Try a quiz or the grammar checker, then come back here!")
        return

    quizzes = df[df["kind"].isin(["Quiz", "Vocabulary Quiz"]) & df["score"].notna()].copy()
    avg = (quizzes["score"].sum() / quizzes["total"].sum() * 100) if not quizzes.empty else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔥 Streak", f"{compute_streak(df)} days")
    c2.metric("📝 Quizzes", len(quizzes))
    c3.metric("🎯 Average score", f"{avg:.0f}%")
    c4.metric("⭐ Saved words", len(get_saved_words()))

    today_count = int((df["day"] == date.today().isoformat()).sum())
    st.markdown("#### Today's goal")
    st.progress(min(today_count / int(new_goal), 1.0))
    st.caption(f"{today_count} of {int(new_goal)} activities done today")

    st.markdown("#### Activities in the last 7 days")
    days = [(date.today() - timedelta(days=i)).isoformat() for i in range(6, -1, -1)]
    counts = df.groupby("day").size().reindex(days, fill_value=0)
    st.bar_chart(pd.DataFrame({"Activities": counts.values}, index=[d[5:] for d in days]), color="#8B7CF6")

    if not quizzes.empty:
        st.markdown("#### Quiz scores (%)")
        quizzes["Score %"] = (quizzes["score"] / quizzes["total"] * 100).round()
        st.line_chart(quizzes.reset_index(drop=True)["Score %"], color="#4F8EF7")

    st.markdown("#### What you did")
    by_kind = df.groupby("kind").size().sort_values(ascending=False)
    st.bar_chart(by_kind, color="#4F8EF7")

    st.markdown("#### 🏆 Achievements")
    ach = [
        ("First step", "Do your first activity", len(df) >= 1),
        ("Quiz starter", "Finish 1 quiz", len(quizzes) >= 1),
        ("Quiz master", "Finish 5 quizzes", len(quizzes) >= 5),
        ("Word collector", "Save 5 words", len(get_saved_words()) >= 5),
        ("On fire", "Learn 3 days in a row", compute_streak(df) >= 3),
        ("Perfect score", "Get every answer right in a quiz", bool((quizzes["score"] == quizzes["total"]).any()) if not quizzes.empty else False),
    ]
    cols = st.columns(3)
    for i, (name, desc, done) in enumerate(ach):
        cols[i % 3].markdown(
            f"<div class='card' style='opacity:{1 if done else .5}'><h4>{'🏅' if done else '🔒'} {name}</h4><p>{desc}</p></div>",
            unsafe_allow_html=True,
        )

    st.markdown("#### Recent activity")
    recent = df.sort_values("id", ascending=False).head(10)[["ts", "kind", "detail", "score", "total"]]
    st.dataframe(recent.rename(columns={"ts": "Time", "kind": "Activity", "detail": "Details", "score": "Score", "total": "Out of"}),
                 use_container_width=True, hide_index=True)


# ----------------------------------------------------------------------------
# Navigation
# ----------------------------------------------------------------------------
st.session_state.setdefault("page", PAGES[0])

with st.sidebar:
    st.markdown(f"## 🎓 {APP_NAME}")
    st.caption("Learn English without stress.")
    st.radio("Menu", PAGES, key="page", label_visibility="collapsed")
    st.divider()
    if read_api_key():
        st.success("AI is ready ✅")
    else:
        st.error("AI key not found. Set GROQ_API_KEY and restart the app.")
    st.caption(f"Model: {MODEL}")

ROUTES = {
    "🏠 Home": page_home,
    "📊 Dashboard": page_dashboard,
    "✍️ Grammar Checker": page_grammar,
    "✨ Sentence Improver": page_improver,
    "📚 Vocabulary Builder": page_vocab,
    "💬 Conversation Practice": page_conversation,
    "📝 Daily Quiz": page_quiz,
    "📈 My Progress": page_progress,
}
ROUTES[st.session_state.page]()
