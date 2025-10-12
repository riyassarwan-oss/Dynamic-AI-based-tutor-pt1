# app_dataset.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import random, time, datetime, json, re, csv
from pathlib import Path

# -------------------- BASIC SETUP --------------------
st.set_page_config(page_title="🎮 Dynamic Game-Based Tutor", layout="centered")

DATA_DIR     = Path("data")
PROFILES_DIR = DATA_DIR / "profiles"
SUBMIT_DIR   = DATA_DIR / "submissions"
for d in [DATA_DIR, PROFILES_DIR, SUBMIT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# -------------------- LOAD QUESTION BANK --------------------
@st.cache_data
def load_bank():
    df = pd.read_csv(DATA_DIR / "question_bank.csv")
    # expected columns: id, subject, topic, difficulty, question, options, correct_option, explanation
    df["options_list"] = df["options"].astype(str).apply(lambda s: s.split("|"))
    valid = {"Easy", "Medium", "Hard"}
    if not set(df["difficulty"].unique()).issubset(valid):
        def _len2diff(q):
            L = len(str(q))
            return "Easy" if L < 80 else ("Medium" if L < 160 else "Hard")
        df["difficulty"] = df.apply(
            lambda r: r["difficulty"] if r["difficulty"] in valid else _len2diff(r["question"]),
            axis=1
        )
    if "explanation" not in df.columns:
        df["explanation"] = ""
    else:
        df["explanation"] = df["explanation"].astype(str).replace({"nan": ""})
    df["id"] = df["id"].astype(str)
    return df

try:
    BANK = load_bank()
except Exception as e:
    st.error("⚠️ data/question_bank.csv not found or unreadable. Run your prep script to create it.")
    st.stop()

# -------------------- UTILITIES --------------------
DIFFS = ["Easy", "Medium", "Hard"]

def slug(name: str) -> str:
    return "_".join(name.strip().lower().split())

def profile_path(name: str) -> Path:
    return PROFILES_DIR / f"{slug(name)}.json"

def load_profile(name: str) -> dict:
    p = profile_path(name)
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    now = time.time()
    return {
        "name": name,
        "created_at": now,
        "last_seen": now,
        "cumulative": {
            "total": 0,
            "correct": 0,
            "score": 0,
            "streak_best": 0,
            "sessions": 0
        },
        "last_topic": None,
        "last_difficulty": "Easy",
    }

def save_profile(name: str, prof: dict):
    prof["last_seen"] = time.time()
    with profile_path(name).open("w", encoding="utf-8") as f:
        json.dump(prof, f, indent=2)

def step_diff(diff_name: str, harder: bool, easier: bool) -> str:
    i = DIFFS.index(diff_name)
    if easier and i > 0:  i -= 1
    if harder and i < 2:  i += 1
    return DIFFS[i]

def pick_question(topic: str, difficulty: str, asked_ids: set):
    pool = BANK[
        (BANK["topic"] == topic) &
        (BANK["difficulty"] == difficulty) &
        (~BANK["id"].isin(asked_ids))
    ]
    if pool.empty:
        return None
    row = pool.sample(1).iloc[0]
    return {
        "id": str(row["id"]),
        "text": str(row["question"]),
        "options": list(row["options_list"]),
        "correct_idx": "ABCD".index(str(row["correct_option"]).strip()[0]),
        "explanation": str(row.get("explanation", "") or "")
    }

def append_attempt(rec: dict):
    out = SUBMIT_DIR / "all_sessions.csv"
    header_needed = not out.exists()
    with out.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "timestamp","student","topic",
            "difficulty_before","difficulty_after",
            "question_id","correct","score_after","total_answered_session",
            "time_taken_sec","hint_used","session_id"
        ])
        if header_needed: w.writeheader()
        w.writerow(rec)

# -------------------- HINT ENGINE (ALWAYS 50/50) --------------------
# Only change in this version: hint ALWAYS reduces to 2 options (correct + 1 wrong).
def _trim(s: str, n: int = 140) -> str:
    s = (s or "").strip()
    return (s[:n-1]+"…") if len(s) > n else s

STOPWORDS = {"which","these","those","there","their","about","because","while","after","before","between",
             "during","would","could","should","where","when","what","from","with","into","through",
             "many","some","most","other","often","usually","generally"}

def extract_keywords(text: str, k: int = 3):
    words = re.findall(r"[A-Za-z][A-Za-z\-]{3,}", text or "")
    words = [w for w in words if w.lower() not in STOPWORDS]
    words = sorted(set(words), key=len, reverse=True)
    return ", ".join(words[:k])

def build_hint(q):
    """
    ALWAYS prefer a 50/50 reduction (keep correct + one random wrong).
    Returns (hint_text, reduced_options, new_correct_idx).
    We keep the text very short; UI doesn't need to show it.
    """
    # short tip (unused visually, but kept if you want to show later)
    expl = _trim(str(q.get("explanation") or ""), 120)
    kw   = extract_keywords(q["text"])
    hint_text = expl if expl else ("Focus on: " + (kw if kw else "the key idea"))

    correct_idx = int(q["correct_idx"])
    opts = q["options"]

    if len(opts) > 2:
        wrong = [i for i in range(len(opts)) if i != correct_idx]
        keep_wrong = random.sample(wrong, 1)
        keep = sorted([correct_idx] + keep_wrong)
        reduced = [opts[i] for i in keep]
        new_correct_idx = keep.index(correct_idx)
        return hint_text, reduced, new_correct_idx

    # already 2 options
    return hint_text, None, None

# -------------------- STATE --------------------
EPISODE_LEN = 10  # questions per session

if "sess" not in st.session_state:
    topics = sorted(BANK["topic"].dropna().unique().tolist() or ["General"])
    st.session_state.sess = {
        "student": "Player",
        "topic": topics[0],
        "difficulty": "Easy",
        "total": 0,
        "correct": 0,
        "streak": 0,
        "score": 0,
        "asked_ids": set(),
        "qpack": None,
        "answered": False,
        "last_feedback": None,
        "session_id": None,
        "session_started_at": None,
        "hint_used": False,
        "hint_payload": None,   # (reduced_options, new_correct_idx)
        "question_started_at": None,
        "episode_done": False,
    }
S = st.session_state.sess

# -------------------- THEME / HEADER (centered) --------------------
st.markdown("""
<style>
.main-wrap {max-width: 980px; margin: 0 auto;}
.hero {text-align:center;}
.hero .underline{
  height:6px; width:68%; margin:8px auto 14px auto; border-radius:6px;
  background: linear-gradient(90deg,#3bd1ff,#7aff9e);
}
.sel-row .stTextInput, .sel-row .stSelectbox {width:100%;}
.hintbox{
  background: rgba(24,119,242,.14);
  border: 1px solid rgba(24,119,242,.35);
  padding: 8px 12px; border-radius: 10px; display:inline-block;
  font-size:.92rem; margin-top:6px;
}
.agent-pill{
  display:inline-block; padding:6px 10px; margin-top:6px; border-radius:14px;
  background: linear-gradient(90deg,#6ec6ff,#ffd36e);
  color:#111; font-weight:600;
}
.diff-pill{
  display:inline-block; padding:6px 10px; margin-left:6px; border-radius:14px;
  background: linear-gradient(90deg,#ffd36e,#ff8f70);
  color:#111; font-weight:600;
}
</style>
<div class="main-wrap">
  <div class="hero">
    <h1>🎮 Dynamic Game-Based Tutor</h1>
    <div class="underline"></div>
  </div>
</div>
""", unsafe_allow_html=True)

mode = st.sidebar.radio("Mode", ["Student", "Teacher"], index=0)

# ==================================================================
#                           STUDENT MODE
# ==================================================================
if mode == "Student":
    st.markdown('<div class="main-wrap">', unsafe_allow_html=True)

    # ---- Name + Topic row ----
    st.markdown('<div class="sel-row">', unsafe_allow_html=True)
    colA, colB = st.columns([1,1])
    with colA:
        prev_name = S["student"]
        S["student"] = st.text_input("Your Name", value=S["student"])
        if S["student"].strip() and S["student"] != prev_name:
            prof = load_profile(S["student"])
            S["difficulty"] = prof.get("last_difficulty", "Easy")
            if prof.get("last_topic"): S["topic"] = prof["last_topic"]
            st.toast(f"Loaded profile for {S['student']}", icon="✅")
            save_profile(S["student"], prof)
    with colB:
        topics = sorted(BANK["topic"].dropna().unique().tolist())
        if S["topic"] not in topics: S["topic"] = topics[0]
        S["topic"] = st.selectbox("Topic", topics, index=topics.index(S["topic"]))
    st.markdown('</div>', unsafe_allow_html=True)  # /sel-row

    # ---- Lifetime panel (from profile) ----
    prof = load_profile(S["student"])
    lifetime_sessions = int(prof["cumulative"]["sessions"])

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("⭐ Score", S["score"])
    sess_acc = (S["correct"]/S["total"])*100 if S["total"]>0 else None
    c2.metric("Accuracy", f"{sess_acc:.0f}%" if sess_acc is not None else "—")
    c3.metric("🔥 Streak", S["streak"])
    c4.metric("🌿 Difficulty", S["difficulty"])
    c5.metric("📊 Sessions", lifetime_sessions)

    st.progress(min(S["total"], EPISODE_LEN) / EPISODE_LEN)

    if S["total"] >= EPISODE_LEN:
        S["episode_done"] = True

    if S["episode_done"]:
        st.success(f"🏁 Session complete! Score: **{S['score']}** • Session Accuracy: **{int(sess_acc or 0)}%**")
        colX, colY = st.columns([1,1])
        if colX.button("💾 Save Session & New"):
            prof = load_profile(S["student"])
            prof["cumulative"]["sessions"] += 1
            prof["cumulative"]["score"] += int(S["score"])
            prof["last_topic"] = S["topic"]
            prof["last_difficulty"] = S["difficulty"]
            save_profile(S["student"], prof)
            S.update({
                "difficulty": "Easy",
                "total": 0, "correct": 0, "streak": 0, "score": 0,
                "asked_ids": set(), "qpack": None, "answered": False,
                "last_feedback": None, "episode_done": False,
                "session_id": None, "session_started_at": None,
                "hint_used": False, "hint_payload": None, "question_started_at": None
            })
            st.rerun()
        if colY.button("🏠 Back to Start (keep lifetime)"):
            S.update({
                "difficulty": "Easy",
                "total": 0, "correct": 0, "streak": 0, "score": 0,
                "asked_ids": set(), "qpack": None, "answered": False,
                "last_feedback": None, "episode_done": False,
                "session_id": None, "session_started_at": None,
                "hint_used": False, "hint_payload": None, "question_started_at": None
            })
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        st.stop()

    if not S["session_id"]:
        S["session_id"] = f"{slug(S['student'])}_{int(time.time())}"
        S["session_started_at"] = datetime.datetime.now().isoformat(timespec="seconds")

    if S["qpack"] is None:
        S["qpack"] = pick_question(S["topic"], S["difficulty"], S["asked_ids"])
        S["answered"] = False
        S["last_feedback"] = None
        S["hint_used"] = False
        S["hint_payload"] = None
        S["question_started_at"] = time.time()

    if S["qpack"] is None:
        st.info("No more questions in this topic/difficulty. Change topic or start a new session.")
        st.markdown('</div>', unsafe_allow_html=True)
        st.stop()

    # ---------- render question ----------
    q = S["qpack"]
    st.markdown(f"### ❓ Q{S['total']+1}. {q['text']}")

    if S.get("hint_payload"):
        current_opts, current_corr = S["hint_payload"]
    else:
        current_opts, current_corr = q["options"], q["correct_idx"]

    with st.form("quiz_form", clear_on_submit=False):
        choice = st.radio("Choose an option", current_opts, index=0, key="choice_radio")
        c1, c2, c3 = st.columns([1,1,1])
        submit_clicked = c1.form_submit_button("✅ Submit", disabled=S["answered"])
        hint_clicked   = c2.form_submit_button("🧠 Hint (50/50, −2)", disabled=S["hint_used"] or S["answered"])
        skip_clicked   = c3.form_submit_button("⏭️ Skip", disabled=S["answered"])

    # --- HINT: before grading; ALWAYS 50/50; re-render immediately ---
    if hint_clicked and not S["hint_used"] and not S["answered"]:
        hint_text, reduced, new_idx = build_hint(q)
        S["score"] = max(0, S["score"] - 2)
        S["hint_used"] = True
        if reduced is not None:
            S["hint_payload"] = (reduced, new_idx)   # exactly 2 options
        # keep UI clean; just show reduced options
        st.rerun()

    # --- SUBMIT ---
    if submit_clicked and not S["answered"]:
        if S.get("hint_payload"):
            opts, corr = S["hint_payload"]
        else:
            opts, corr = q["options"], q["correct_idx"]
        is_correct = (choice == opts[int(corr)])
        S["total"] += 1

        if is_correct:
            S["correct"] += 1
            S["streak"]  += 1
            S["score"]   += 10 + 2*S["streak"]
        else:
            S["streak"] = 0
            S["score"]  = max(0, S["score"] - 2)

        old_diff = S["difficulty"]
        action   = "harder" if is_correct else "easier"
        new_diff = step_diff(old_diff, harder=is_correct, easier=not is_correct)
        S["difficulty"] = new_diff
        S["asked_ids"].add(q["id"])
        S["answered"] = True
        S["last_feedback"] = ("✅ Correct!" if is_correct else "❌ Wrong!", old_diff, action, new_diff)

        prof = load_profile(S["student"])
        prof["cumulative"]["total"] += 1
        if is_correct: prof["cumulative"]["correct"] += 1
        prof["cumulative"]["streak_best"] = max(prof["cumulative"]["streak_best"], S["streak"])
        prof["last_topic"] = S["topic"]; prof["last_difficulty"] = S["difficulty"]
        save_profile(S["student"], prof)

        time_taken = max(0.0, time.time() - (S["question_started_at"] or time.time()))
        append_attempt({
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "student": S["student"],
            "topic": S["topic"],
            "difficulty_before": old_diff,
            "difficulty_after": new_diff,
            "question_id": q["id"],
            "correct": int(is_correct),
            "score_after": S["score"],
            "total_answered_session": S["total"],
            "time_taken_sec": round(time_taken, 1),
            "hint_used": int(S["hint_used"]),
            "session_id": S["session_id"]
        })
        if is_correct and S["streak"] >= 3:
            st.balloons()

    # --- SKIP ---
    if skip_clicked and not S["answered"]:
        S["streak"] = 0
        S["score"]  = max(0, S["score"] - 1)
        old_diff    = S["difficulty"]
        new_diff    = step_diff(old_diff, harder=False, easier=True)
        S["difficulty"] = new_diff
        S["asked_ids"].add(q["id"])
        S["answered"] = True
        S["last_feedback"] = ("⏭️ Skipped", old_diff, "easier", new_diff)

        prof = load_profile(S["student"])
        prof["cumulative"]["total"] += 1
        prof["last_topic"] = S["topic"]; prof["last_difficulty"] = S["difficulty"]
        save_profile(S["student"], prof)

        time_taken = max(0.0, time.time() - (S["question_started_at"] or time.time()))
        append_attempt({
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "student": S["student"],
            "topic": S["topic"],
            "difficulty_before": old_diff,
            "difficulty_after": new_diff,
            "question_id": q["id"],
            "correct": 0,
            "score_after": S["score"],
            "total_answered_session": S["total"],
            "time_taken_sec": round(time_taken, 1),
            "hint_used": int(S["hint_used"]),
            "session_id": S["session_id"]
        })

    # --- feedback + nav ---
    if S["answered"] and S["last_feedback"]:
        verdict, od, act, nd = S["last_feedback"]
        (st.success if "✅" in verdict else st.error)(verdict)
        st.markdown(
            f'<span class="agent-pill">Agent action: {act}</span>'
            f'<span class="diff-pill">Difficulty: {od} → {nd}</span>',
            unsafe_allow_html=True
        )

    colN, colR = st.columns([1,1])
    if colN.button("⏭️ Next", disabled=not S["answered"]):
        S["qpack"] = None
        S["answered"] = False
        S["last_feedback"] = None
        S["hint_used"] = False
        S["hint_payload"] = None
        S["question_started_at"] = None
        st.rerun()

    if colR.button("🔁 Reset Session"):
        S.update({
            "difficulty": "Easy",
            "total": 0, "correct": 0, "streak": 0, "score": 0,
            "asked_ids": set(), "qpack": None,
            "answered": False, "last_feedback": None,
            "episode_done": False, "session_id": None,
            "session_started_at": None, "hint_used": False,
            "hint_payload": None, "question_started_at": None
        })
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)  # /main-wrap

# ==================================================================
#                          TEACHER MODE
# ==================================================================
else:
    st.subheader("👩‍🏫 Teacher Dashboard")

    all_csv = SUBMIT_DIR / "all_sessions.csv"
    if all_csv.exists():
        df = pd.read_csv(all_csv)
    else:
        df = pd.DataFrame(columns=[
            "timestamp","student","topic","difficulty_before","difficulty_after",
            "question_id","correct","score_after","total_answered_session",
            "time_taken_sec","hint_used","session_id"
        ])

    students = []
    for p in PROFILES_DIR.glob("*.json"):
        with p.open("r", encoding="utf-8") as f:
            prof = json.load(f)
        name = prof["name"]
        lif = prof["cumulative"]

        if not df.empty:
            df_s = df[df["student"] == name]
            sessions_attended = int(df_s["session_id"].nunique())
            last_seen = df_s["timestamp"].max() if not df_s.empty else "-"
            total_attempts = int(len(df_s))
            lifetime_acc = (df_s["correct"].mean()*100) if total_attempts>0 else 0.0
        else:
            sessions_attended = lif.get("sessions", 0)
            last_seen = "-"
            total_attempts = lif.get("total", 0)
            lifetime_acc = (lif.get("correct",0)/max(1,lif.get("total",1)))*100 if lif.get("total",0)>0 else 0.0

        students.append({
            "Student": name,
            "Sessions": sessions_attended,
            "Lifetime Score": lif.get("score",0),
            "Lifetime Acc %": round(lifetime_acc,1),
            "Total Qs": total_attempts,
            "Best Streak": lif.get("streak_best",0),
            "Last Activity": last_seen
        })

    st.markdown("### 🗂️ Students Overview")
    if students:
        df_students = pd.DataFrame(students).sort_values(
            ["Sessions","Lifetime Score"], ascending=[False,False]
        )
        df_display = df_students.reset_index(drop=True)
        df_display.index = df_display.index + 1
        st.dataframe(df_display, height=320, use_container_width=True)
    else:
        st.info("No student profiles yet. Ask students to play a session.")

    st.markdown("---")
    st.markdown("### 📜 Recent Attempts (last 300)")
    if not df.empty:
        df_recent = df.tail(300).reset_index(drop=True)
        df_recent.index = df_recent.index + 1
        st.dataframe(df_recent, use_container_width=True, height=300)

        # ---- Attempts by Difficulty (colored bars) ----
        st.markdown("#### 📊 Attempts by Difficulty (after)")
        counts = df["difficulty_after"].value_counts().reindex(["Easy","Medium","Hard"]).fillna(0).astype(int).reset_index()
        counts.columns = ["Difficulty","Attempts"]
        palette = {"Easy":"#3ecf8e","Medium":"#ffd166","Hard":"#f75c4c"}
        chart = (alt.Chart(counts)
                 .mark_bar()
                 .encode(x=alt.X("Difficulty:N", sort=["Easy","Medium","Hard"]),
                         y=alt.Y("Attempts:Q"),
                         color=alt.Color("Difficulty:N", scale=alt.Scale(domain=list(palette.keys()),
                                                                         range=list(palette.values())),
                                         legend=None))
                 .properties(height=260))
        st.altair_chart(chart, use_container_width=True)

        # ---- Accuracy by Student ----
        st.markdown("#### 🎯 Accuracy by Student (%)")
        acc_by_student = (df.groupby("student")["correct"].mean()*100).round(1).reset_index().sort_values("correct", ascending=False)
        acc_chart = (alt.Chart(acc_by_student)
                     .mark_bar()
                     .encode(x=alt.X("student:N", title="Student"),
                             y=alt.Y("correct:Q", title="Accuracy %"),
                             color=alt.value("#7aa0ff"))
                     .properties(height=260))
        st.altair_chart(acc_chart, use_container_width=True)

        # ---- Accuracy Progress by Session (robust; no 'ended') ----
        st.markdown("#### 📈 Accuracy Progress by Session")
        st.caption("Per-student accuracy per session (each row = one session).")
        try:
            sess = (
                df.groupby(["student","session_id"], as_index=False)
                  .agg(correct_rate=("correct","mean"),
                       attempts=("correct","size"),
                       last_event=("timestamp","max"))
            )
            sess["correct_rate"] = (sess["correct_rate"]*100).round(1)
            show = sess.rename(columns={
                "student":"Student","session_id":"Session ID",
                "correct_rate":"Accuracy %","attempts":"#Qs","last_event":"Last Event"
            }).sort_values(["Student","Last Event"])
            show.index = range(1, len(show)+1)
            st.dataframe(show, use_container_width=True, height=260)
        except Exception:
            st.info("Not enough data to compute session progression yet.")
    else:
        st.info("No attempts logged yet. They appear here as students answer questions.")