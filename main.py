from fastapi import FastAPI, HTTPException, Response, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import os, hashlib, random, pickle, io, json
from datetime import date, datetime, timedelta, timezone
import numpy as np
import pandas as pd
import jwt as pyjwt
from concurrent.futures import ThreadPoolExecutor
import asyncio
from contextlib import contextmanager

# ── PostgreSQL connection pool ────────────────────────────────────────────────

import psycopg2
from psycopg2 import pool as pg_pool
from psycopg2.extras import RealDictCursor   #

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.cluster import KMeans
import shap
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors as rl_colors
from reportlab.lib.pagesizes import letter

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
SECRET_KEY         = os.environ.get("SECRET_KEY", "change-me-in-production-please")
TOKEN_EXPIRY_HOURS = int(os.environ.get("TOKEN_EXPIRY_HOURS", 24))
DATABASE_URL       = os.environ.get("DATABASE_URL")          # Set on Render / Supabase
GROQ_API_KEY       = os.environ.get("GROQ_API_KEY", "")
FRONTEND_DIR       = os.path.dirname(os.path.abspath(__file__))

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL environment variable is not set.\n"
        "Set it to: postgresql://user:password@host:5432/dbname\n"
        "Get it from Supabase → Settings → Database → Connection String → URI"
    )

STUDY_TIPS_POOL = [
    "🧠 The Feynman Technique: Explain concepts as if teaching a 12-year-old.",
    "⏰ Spaced repetition beats cramming. Review yesterday's material first.",
    "📝 Active recall is 3x more effective than re-reading. Test yourself!",
    "🎯 Set SMART goals: Specific, Measurable, Achievable, Relevant, Time-bound.",
    "🔴 Pomodoro tip: If you break early, the timer resets. Commit fully.",
    "💧 Even mild dehydration reduces focus by 13%. Keep water at your desk.",
    "🚶 A 20-min walk before studying increases BDNF — the brain's growth hormone.",
    "🌙 Memory consolidates during sleep. Never sacrifice sleep for cramming.",
    "✍️ Handwriting notes improves retention vs typing by up to 40%.",
    "📱 Phone nearby (even face-down) reduces IQ by ~10 points. Leave it outside.",
    "🔄 Interleaving topics — switching subjects — beats blocked practice for retention.",
    "🎭 Rubber duck method: Explain your problem aloud. Solutions emerge naturally.",
    "📊 Cornell Notes: split page into cue, note, and summary columns.",
    "🌅 Tackle hardest subjects when your brain is freshest (usually morning).",
    "🎯 80/20 rule: 20% of content covers 80% of exam questions. Find it.",
    "🧘 10 min meditation before study reduces mind-wandering by 22%.",
    "🍎 Omega-3 fatty acids, blueberries, and dark chocolate genuinely help focus.",
    "📚 Read summary and chapter questions BEFORE the chapter to prime your brain.",
    "💡 Study in multiple environments to improve memory flexibility.",
    "🔬 The Testing Effect: being tested is 50% more effective than re-reading.",
    "🌿 Plants in your study space reduce cortisol and increase oxygen.",
    "📅 Planning tomorrow's sessions tonight boosts follow-through by 40%.",
    "🏆 Reward yourself after sessions. Dopamine reinforces study habits.",
    "🔁 Review at 1 day, 1 week, 1 month intervals for optimal long-term retention.",
    "💪 Exercise grows new neurons. Even 20 min/day makes a measurable difference.",
    "🤝 Teaching others is the most effective form of learning (Protégé Effect).",
    "⚡ Multitasking reduces IQ by 10-15 points temporarily. Single-task only.",
    "🏗️ Build conceptual frameworks first, fill in details later.",
    "🌊 Flow state needs: clear goals, immediate feedback, and balanced challenge.",
    "⏳ Parkinson's Law: Work expands to fill time. Set tight deadlines.",
    "🧬 Neuroplasticity is real. Your brain can always improve with practice.",
    "🌐 Socratic Method: Question everything. Deep understanding > surface recall.",
    "💭 Method of Loci: Link information to spatial memories for recall.",
    "📓 Keep a learning journal. Writing about what you learned cements it.",
    "🔮 Visualize yourself successfully using knowledge in exams. It works.",
    "🌀 Confusion is a sign of learning, not failure. Sit with it — then seek clarity.",
    "🎵 Even 5 minutes of music you love resets focus better than silence.",
    "🏃 Physical warmth (tea, warm room) signals safety to the brain, reducing anxiety.",
    "📖 SQ3R Method: Survey, Question, Read, Recite, Review.",
    "🔑 Understanding the 'why' behind facts makes them 5x easier to remember.",
]

# ══════════════════════════════════════════════════════════════════════════════
# CONNECTION POOL
# ══════════════════════════════════════════════════════════════════════════════
_pool: pg_pool.SimpleConnectionPool = None

def get_pool() -> pg_pool.SimpleConnectionPool:
    global _pool
    if _pool is None:
        _pool = pg_pool.SimpleConnectionPool(
            minconn=1, maxconn=10,
            dsn=DATABASE_URL,
            # Force SSL for Supabase — harmless if already in the URL
            sslmode="require" if "supabase" in (DATABASE_URL or "") else "prefer"
        )
    return _pool

@contextmanager
def get_conn():
    """Context manager — always returns connection back to pool."""
    p = get_pool()
    conn = p.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        p.putconn(conn)

@contextmanager
def get_cursor(conn):
    """Return a RealDictCursor so rows behave like dicts (replaces row_factory)."""
    c = conn.cursor(cursor_factory=RealDictCursor)
    try:
        yield c
    finally:
        c.close()

# ══════════════════════════════════════════════════════════════════════════════
# DATABASE INIT
# ══════════════════════════════════════════════════════════════════════════════
def init_db():
    statements = [
        """CREATE TABLE IF NOT EXISTS users(
            id SERIAL PRIMARY KEY,
            username TEXT UNIQUE,
            password_hash TEXT,
            email TEXT
        )""",
        """CREATE TABLE IF NOT EXISTS study_sessions(
            id SERIAL PRIMARY KEY,
            user_id INTEGER, subject TEXT, date TEXT, minutes INTEGER
        )""",
        """CREATE TABLE IF NOT EXISTS subjects_total(
            subject TEXT, user_id INTEGER, total_minutes INTEGER DEFAULT 0,
            PRIMARY KEY (subject, user_id)
        )""",
        """CREATE TABLE IF NOT EXISTS user_stats(
            id INTEGER PRIMARY KEY,
            xp INTEGER DEFAULT 0, level INTEGER DEFAULT 1, daily_goal INTEGER DEFAULT 180
        )""",
        """CREATE TABLE IF NOT EXISTS subjects_meta(
            user_id INTEGER, subject TEXT,
            difficulty TEXT DEFAULT 'Medium', last_test_score REAL DEFAULT 0,
            PRIMARY KEY (user_id, subject)
        )""",
        """CREATE TABLE IF NOT EXISTS tasks(
            id SERIAL PRIMARY KEY,
            user_id INTEGER, task TEXT, subject TEXT, date TEXT,
            priority TEXT DEFAULT 'Medium', completed TEXT DEFAULT 'No',
            completed_date TEXT
        )""",
        """CREATE TABLE IF NOT EXISTS subject_notes(
            id SERIAL PRIMARY KEY,
            user_id INTEGER, subject TEXT, note TEXT, created_at TEXT
        )""",
        """CREATE TABLE IF NOT EXISTS reset_tokens(
            id SERIAL PRIMARY KEY,
            user_id INTEGER, token TEXT, expires_at TEXT
        )""",
        """CREATE TABLE IF NOT EXISTS user_profiles(
            user_id INTEGER PRIMARY KEY,
            bio TEXT DEFAULT '',
            avatar_emoji TEXT DEFAULT '🎓',
            role TEXT DEFAULT 'student',
            fields_of_study TEXT DEFAULT '[]',
            country TEXT DEFAULT '',
            school TEXT DEFAULT '',
            grade TEXT DEFAULT '',
            website TEXT DEFAULT '',
            is_public INTEGER DEFAULT 1,
            joined_at TEXT
        )""",
        """CREATE TABLE IF NOT EXISTS follows(
            follower_id INTEGER,
            followed_id INTEGER,
            created_at TEXT,
            PRIMARY KEY(follower_id, followed_id)
        )""",
        """CREATE TABLE IF NOT EXISTS community_posts(
            id SERIAL PRIMARY KEY,
            user_id INTEGER,
            content TEXT,
            subject_tag TEXT DEFAULT '',
            post_type TEXT DEFAULT 'general',
            created_at TEXT
        )""",
        """CREATE TABLE IF NOT EXISTS post_likes(
            post_id INTEGER,
            user_id INTEGER,
            PRIMARY KEY(post_id, user_id)
        )""",
        """CREATE TABLE IF NOT EXISTS post_comments(
            id SERIAL PRIMARY KEY,
            post_id INTEGER,
            user_id INTEGER,
            comment TEXT,
            created_at TEXT
        )""",
        """CREATE TABLE IF NOT EXISTS chat_rooms(
            id SERIAL PRIMARY KEY,
            name TEXT,
            description TEXT DEFAULT '',
            room_type TEXT DEFAULT 'subject',
            created_by INTEGER,
            created_at TEXT
        )""",
        """CREATE TABLE IF NOT EXISTS room_members(
            room_id INTEGER,
            user_id INTEGER,
            joined_at TEXT,
            PRIMARY KEY(room_id, user_id)
        )""",
        """CREATE TABLE IF NOT EXISTS room_messages(
            id SERIAL PRIMARY KEY,
            room_id INTEGER,
            user_id INTEGER,
            message TEXT,
            created_at TEXT
        )""",
        """CREATE TABLE IF NOT EXISTS monitored_users(
            monitor_id INTEGER,
            student_id INTEGER,
            relationship TEXT DEFAULT 'teacher',
            created_at TEXT,
            PRIMARY KEY(monitor_id, student_id)
        )""",
        """CREATE TABLE IF NOT EXISTS chatbot_history(
            id SERIAL PRIMARY KEY,
            user_id INTEGER,
            role TEXT,
            content TEXT,
            created_at TEXT
        )""",
    ]

    with get_conn() as conn:
        with get_cursor(conn) as c:
            for stmt in statements:
                c.execute(stmt)

            # Seed default chat rooms if empty
            c.execute("SELECT COUNT(*) as cnt FROM chat_rooms")
            row = c.fetchone()
            if row["cnt"] == 0:
                rooms = [
                    ("🌍 General Lounge",    "Talk about anything study-related",     "general"),
                    ("🔬 Science Hub",        "Physics, Chemistry, Biology discussions","subject"),
                    ("📐 Maths Corner",       "From calculus to trigonometry",          "subject"),
                    ("📖 Literature Lounge",  "Books, essays, writing tips",            "subject"),
                    ("💻 Tech & Code",        "Programming, CS, AI",                   "subject"),
                    ("🏛 History & Arts",     "History, geography, fine arts",          "subject"),
                    ("🎯 Exam Prep Zone",     "Tips, tricks, past papers",              "general"),
                    ("🤝 Study Buddy Finder", "Find accountability partners",           "general"),
                    ("👩‍🏫 Teachers' Room",  "Pedagogy, lesson plans, resources",      "general"),
                    ("👨‍👩‍👧 Parents Corner","Support your child's education",         "general"),
                ]
                for name, desc, rtype in rooms:
                    c.execute(
                        "INSERT INTO chat_rooms(name,description,room_type,created_by,created_at)"
                        " VALUES(%s,%s,%s,0,%s)",
                        (name, desc, rtype, datetime.now().strftime("%Y-%m-%d %H:%M"))
                    )

# ── APP ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="StudyNest API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # Tighten this to your Render URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=4)

# Init DB on startup
@app.on_event("startup")
def on_startup():
    init_db()

# ── WEBSOCKET MANAGER ─────────────────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.connections: Dict[int, List[WebSocket]] = {}

    async def connect(self, room_id: int, ws: WebSocket):
        await ws.accept()
        self.connections.setdefault(room_id, []).append(ws)

    def disconnect(self, room_id: int, ws: WebSocket):
        if room_id in self.connections:
            try: self.connections[room_id].remove(ws)
            except ValueError: pass

    async def broadcast(self, room_id: int, data: dict):
        dead = []
        for ws in list(self.connections.get(room_id, [])):
            try: await ws.send_json(data)
            except: dead.append(ws)
        for ws in dead: self.disconnect(room_id, ws)

    def count(self, room_id: int) -> int:
        return len(self.connections.get(room_id, []))

manager = ConnectionManager()

# ── AUTH HELPERS ──────────────────────────────────────────────────────────────
def hash_pw(p: str) -> str:
    return hashlib.sha256(p.encode()).hexdigest()

def create_token(user_id: int, minutes: int = None) -> str:
    exp = (datetime.now(timezone.utc) +
           (timedelta(minutes=minutes) if minutes else timedelta(hours=TOKEN_EXPIRY_HOURS)))
    return pyjwt.encode({"user_id": user_id, "exp": exp}, SECRET_KEY, algorithm="HS256")

def verify_token(token: str) -> Optional[int]:
    try:
        return pyjwt.decode(token, SECRET_KEY, algorithms=["HS256"])["user_id"]
    except Exception:
        return None

def get_user(request: Request) -> int:
    token = request.cookies.get("auth_token")
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    uid = verify_token(token)
    if not uid:
        raise HTTPException(status_code=401, detail="Token expired")
    return uid

# ── ML HELPERS ────────────────────────────────────────────────────────────────
def difficulty_weight(level: str) -> float:
    return {"Easy": 1.0, "Medium": 1.2, "Hard": 1.5}.get(level, 1.0)

def detect_drift(old_mean, new_mean, threshold=0.25) -> bool:
    if old_mean == 0: return False
    return abs(new_mean - old_mean) / old_mean > threshold

def load_model(uid):
    mf, xf = f"study_model_{uid}.pkl", f"study_meta_{uid}.pkl"
    if os.path.exists(mf) and os.path.exists(xf):
        with open(mf,"rb") as f: model = pickle.load(f)
        with open(xf,"rb") as f: meta  = pickle.load(f)
        return model, meta
    return None, None

def save_model_files(model, meta, uid):
    with open(f"study_model_{uid}.pkl","wb") as f: pickle.dump(model, f)
    with open(f"study_meta_{uid}.pkl","wb")  as f: pickle.dump(meta,  f)

def compute_streak(study_log: dict, days=100) -> int:
    streak = 0
    for i in range(days):
        d = (date.today() - timedelta(days=i)).strftime("%d-%m-%Y")
        if any(study_log.get(s, {}).get(d, 0) > 0 for s in study_log):
            streak += 1
        else:
            break
    return streak

def run_ml(uid: int, study_log: dict, subjects_meta: dict):
    ml_rows = []
    for s in study_log:
        for d, m in study_log[s].items():
            ml_rows.append({"date": d, "subject": s, "minutes": m})
    if len(ml_rows) < 7: return None

    df = pd.DataFrame(ml_rows)
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
    for s in df["subject"].unique():
        if s in subjects_meta:
            diff  = subjects_meta[s].get("difficulty", "Medium")
            score = subjects_meta[s].get("score") or 50
            w = difficulty_weight(diff) + (100 - score) / 100
            df.loc[df["subject"] == s, "minutes"] *= w
    df = df.groupby("date")["minutes"].sum().reset_index()
    df = df.sort_values("date").reset_index(drop=True)
    df["dow"]   = df["date"].dt.dayofweek
    df["dom"]   = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["is_weekend"] = df["dow"].isin([5,6]).astype(int)
    df["days_since_start"] = (df["date"] - df["date"].min()).dt.days
    for w in [3, 7, 14]:
        df[f"rm_{w}"]  = df["minutes"].rolling(w).mean().fillna(0)
        df[f"rs_{w}"]  = df["minutes"].rolling(w).std().fillna(0)
    df["rm_28"]  = df["minutes"].shift(1).rolling(28).mean().fillna(0)
    df["rmax_7"] = df["minutes"].rolling(7).max().fillna(0)
    df["rmin_7"] = df["minutes"].rolling(7).min().fillna(0)
    df["rmed_7"] = df["minutes"].shift(1).rolling(7).median().fillna(0)
    for lag in [1,2,3,7,14]:
        df[f"lag_{lag}"] = df["minutes"].shift(lag).fillna(0)
    df["pct_7"]   = df["minutes"].pct_change(7).fillna(0)
    df["mom_7"]   = df["minutes"] - df["minutes"].shift(7).fillna(0)
    df["consist"] = df["rs_7"] / (df["rm_7"] + 1)
    df["overload"]= (df["rm_3"] > df["rm_7"] * 1.5).astype(int)
    df["dow_sin"] = np.sin(2*np.pi*df["dow"]/7)
    df["dow_cos"] = np.cos(2*np.pi*df["dow"]/7)
    df["dom_sin"] = np.sin(2*np.pi*df["dom"]/31)
    df["dom_cos"] = np.cos(2*np.pi*df["dom"]/31)
    df["mo_sin"]  = np.sin(2*np.pi*df["month"]/12)
    df["mo_cos"]  = np.cos(2*np.pi*df["month"]/12)
    df["wx_m7"]   = df["is_weekend"] * df["rm_7"]
    gm = df["minutes"].mean()
    df["rel_avg"] = df["minutes"] / (gm + 1)
    df["motiv"]   = df["lag_1"]*0.5 + df["lag_2"]*0.3 + df["lag_7"]*0.2
    df["dow_avg"] = df.groupby(df["date"].dt.dayofweek)["minutes"].transform("mean").fillna(0)
    try:
        df["intensity"] = pd.cut(df["minutes"],bins=[0,60,180,360,10000],labels=[0,1,2,3]).astype(int)
    except Exception:
        df["intensity"] = 1

    y = df["minutes"]
    X = df.drop(columns=["minutes","date","behaviour_cluster"],errors="ignore").fillna(0)
    cluster_result = None
    try:
        cf = df[["rm_7","consist"]].fillna(0)
        nc = min(3, max(2, len(cf)))
        km = KMeans(n_clusters=nc, random_state=42, n_init=10)
        df["behaviour_cluster"] = km.fit_predict(cf)
        cluster_result = int(df["behaviour_cluster"].iloc[-1])
    except Exception: pass

    saved_model, meta = load_model(uid)
    best_model, best_mae, best_features = None, float("inf"), list(X.columns)
    if saved_model is not None:
        try:
            td = meta["train_date"]
            if isinstance(td, str): td = datetime.strptime(td, "%Y-%m-%d").date()
            if (date.today() - td).days < 7 and not detect_drift(meta["mean_minutes"], df["minutes"].mean()):
                best_model = saved_model
                best_features = meta["features"]
                best_mae = meta.get("mae", 0)
        except Exception: pass

    if best_model is None and len(df) > 10:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, shuffle=False)
        candidates = {
            "XGBoost":  XGBRegressor(n_estimators=200,max_depth=5,learning_rate=0.05,verbosity=0),
            "LightGBM": LGBMRegressor(n_estimators=200,learning_rate=0.05,verbosity=-1),
            "RF":       RandomForestRegressor(n_estimators=200,max_depth=8,random_state=42),
            "Ridge":    Ridge(alpha=1.0),
        }
        for name, mdl in candidates.items():
            try:
                mdl.fit(X_tr, y_tr)
                try:
                    sv = np.array(shap.TreeExplainer(mdl).shap_values(X_tr))
                    imp = np.abs(sv) if sv.ndim==1 else np.abs(sv).mean(axis=0)
                    feats = X_tr.columns[imp > np.percentile(imp, 40)]
                    if len(feats) == 0: feats = X_tr.columns
                    mdl.fit(X_tr[feats], y_tr); preds = mdl.predict(X_te[feats])
                except Exception:
                    feats = X_tr.columns; preds = mdl.predict(X_te)
                mae = mean_absolute_error(y_te, preds)
                if mae < best_mae:
                    best_mae, best_model, best_features = mae, mdl, list(feats)
            except Exception: pass

        if best_model:
            save_model_files(best_model, {
                "train_date": date.today(), "mean_minutes": float(df["minutes"].mean()),
                "features": best_features, "mae": best_mae
            }, uid)

    if best_model is None: return None
    vf  = [f for f in best_features if f in X.columns] or list(X.columns)
    raw = best_model.predict(X.iloc[-1:][vf])[0]
    tomorrow = max(int(raw), 30)
    lr = X.iloc[-1:][vf].copy()
    preds_7 = []
    for _ in range(7):
        p = max(best_model.predict(lr)[0], 0)
        preds_7.append(float(p))
        if "lag_1" in lr.columns:
            lr = lr.copy(); lr["lag_1"] = p
    std_e    = float(np.std(y.values - best_model.predict(X[vf])))
    wk_total = sum(preds_7)
    return {
        "tomorrow": tomorrow, "mae": round(best_mae, 2),
        "confidence": "High" if best_mae < 20 else "Medium" if best_mae < 50 else "Low",
        "forecast": preds_7, "std_e": std_e, "weekly_total": int(wk_total),
        "lower": max(int(wk_total - 1.96 * std_e * np.sqrt(7)), 0),
        "upper": int(wk_total + 1.96 * std_e * np.sqrt(7)),
        "cluster": cluster_result,
    }

# ══════════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ══════════════════════════════════════════════════════════════════════════════
class LoginReq(BaseModel):
    username: str; password: str

class RegisterReq(BaseModel):
    username: str; password: str; email: Optional[str] = ""

class SessionReq(BaseModel):
    subject: str; date: str; minutes: int

class SubjectReq(BaseModel):
    name: str

class TaskReq(BaseModel):
    task: str; subject: str; date: str; priority: str = "Medium"

class NoteReq(BaseModel):
    subject: str; note: str

class GoalReq(BaseModel):
    daily_goal: int

class PrefReq(BaseModel):
    subject: str; difficulty: str; score: float

class ProfileReq(BaseModel):
    bio: Optional[str] = ""
    avatar_emoji: Optional[str] = "🎓"
    role: Optional[str] = "student"
    fields_of_study: Optional[str] = "[]"
    country: Optional[str] = ""
    school: Optional[str] = ""
    grade: Optional[str] = ""
    website: Optional[str] = ""
    is_public: Optional[int] = 1

class PostReq(BaseModel):
    content: str
    subject_tag: Optional[str] = ""
    post_type: Optional[str] = "general"

class CommentReq(BaseModel):
    comment: str

class RoomReq(BaseModel):
    name: str
    description: Optional[str] = ""
    room_type: Optional[str] = "subject"

class MessageReq(BaseModel):
    message: str

class MonitorReq(BaseModel):
    username: str
    relationship: str = "teacher"

class ChatbotReq(BaseModel):
    message: str
    history: Optional[List[dict]] = None


def parse_fields_of_study(value):
    if not value:
        return []
    if isinstance(value, list):
        return value
    try:
        return json.loads(value)
    except Exception:
        return []

# ══════════════════════════════════════════════════════════════════════════════
# AUTH
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/api/auth/register")
def register(req: RegisterReq):
    if len(req.password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters")
    try:
        with get_conn() as conn:
            with get_cursor(conn) as c:
                c.execute(
                    "INSERT INTO users(username,email,password_hash) VALUES(%s,%s,%s) RETURNING id",
                    (req.username.strip(), req.email.strip(), hash_pw(req.password))
                )
                uid = c.fetchone()["id"]
                c.execute(
                    "INSERT INTO user_stats(id,xp,level,daily_goal) VALUES(%s,0,1,180) ON CONFLICT DO NOTHING",
                    (uid,)
                )
                c.execute(
                    "INSERT INTO user_profiles(user_id,joined_at) VALUES(%s,%s) ON CONFLICT DO NOTHING",
                    (uid, datetime.now().strftime("%Y-%m-%d"))
                )
        return {"message": "Account created! Welcome to StudyNest."}
    except Exception as e:
        if "unique" in str(e).lower() or "duplicate" in str(e).lower():
            raise HTTPException(400, "Username already taken")
        raise HTTPException(500, str(e))

@app.post("/api/auth/login")
def login(req: LoginReq, response: Response):
    with get_conn() as conn:
        with get_cursor(conn) as c:
            c.execute("SELECT id, password_hash FROM users WHERE username=%s", (req.username,))
            row = c.fetchone()
    if not row or row["password_hash"] != hash_pw(req.password):
        raise HTTPException(401, "Invalid credentials")
    token = create_token(row["id"])
    response.set_cookie(key="auth_token", value=token, httponly=True, samesite="lax",
                        max_age=TOKEN_EXPIRY_HOURS * 3600)
    return {"message": "Login successful", "user_id": row["id"]}

@app.post("/api/auth/logout")
def logout(response: Response):
    response.delete_cookie("auth_token")
    return {"message": "Logged out"}

@app.get("/api/auth/me")
def me(request: Request):
    uid = get_user(request)
    with get_conn() as conn:
        with get_cursor(conn) as c:
            c.execute("SELECT username, email FROM users WHERE id=%s", (uid,))
            row = c.fetchone()
    if not row: raise HTTPException(404, "User not found")
    return {"user_id": uid, "username": row["username"], "email": row["email"] or ""}

@app.get("/api/auth/ws-token")
def get_ws_token(request: Request):
    uid = get_user(request)
    return {"token": create_token(uid, minutes=5)}

# ══════════════════════════════════════════════════════════════════════════════
# STATS
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/api/stats")
def get_stats(request: Request):
    uid = get_user(request)
    with get_conn() as conn:
        with get_cursor(conn) as c:
            c.execute("SELECT xp, level, daily_goal FROM user_stats WHERE id=%s", (uid,))
            row = c.fetchone()
            if not row:
                c.execute(
                    "INSERT INTO user_stats(id,xp,level,daily_goal) VALUES(%s,0,1,180) ON CONFLICT DO NOTHING",
                    (uid,)
                )
                xp, level, daily_goal = 0, 1, 180
            else:
                xp, level, daily_goal = row["xp"], row["level"], row["daily_goal"]

            c.execute("SELECT subject, total_minutes FROM subjects_total WHERE user_id=%s", (uid,))
            subjects = {r["subject"]: r["total_minutes"] for r in c.fetchall()}

            c.execute("SELECT subject, date, minutes FROM study_sessions WHERE user_id=%s", (uid,))
            study_log: dict = {}
            for r in c.fetchall():
                study_log.setdefault(r["subject"], {})
                study_log[r["subject"]][r["date"]] = study_log[r["subject"]].get(r["date"], 0) + r["minutes"]

            c.execute("SELECT subject, difficulty, last_test_score FROM subjects_meta WHERE user_id=%s", (uid,))
            subjects_meta = {
                r["subject"]: {"difficulty": r["difficulty"], "score": r["last_test_score"]}
                for r in c.fetchall()
            }

    today       = date.today().strftime("%d-%m-%Y")
    total_today = sum(study_log.get(s, {}).get(today, 0) for s in study_log)
    streak      = compute_streak(study_log)
    week_total  = sum(
        study_log.get(s, {}).get((date.today() - timedelta(days=i)).strftime("%d-%m-%Y"), 0)
        for s in study_log for i in range(7)
    )
    week_vals = []
    for i in range(6,-1,-1):
        d = (date.today()-timedelta(days=i)).strftime("%d-%m-%Y")
        week_vals.append(sum(study_log.get(s,{}).get(d,0) for s in study_log))

    recent = sum(week_vals[4:])/3 if sum(week_vals[4:]) else 0
    older  = sum(week_vals[:4])/4 if sum(week_vals[:4]) else 0
    momentum = 0 if (older==0 and recent==0) else 200 if older==0 else min(int(recent/older*100),200)

    return {
        "xp": xp, "level": level, "daily_goal": daily_goal,
        "total_today": total_today, "streak": streak,
        "week_total": week_total, "momentum": momentum,
        "subjects": subjects, "study_log": study_log,
        "subjects_meta": subjects_meta, "week_vals": week_vals,
        "tip": random.choice(STUDY_TIPS_POOL),
    }

# ══════════════════════════════════════════════════════════════════════════════
# SUBJECTS & SESSIONS
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/api/subjects")
def add_subject(req: SubjectReq, request: Request):
    uid = get_user(request)
    with get_conn() as conn:
        with get_cursor(conn) as c:
            c.execute(
                "INSERT INTO subjects_total(subject,user_id,total_minutes) VALUES(%s,%s,0) ON CONFLICT DO NOTHING",
                (req.name.strip(), uid)
            )
    return {"message": f"Subject '{req.name}' added"}

@app.delete("/api/subjects/{subject}")
def delete_subject(subject: str, request: Request):
    uid = get_user(request)
    with get_conn() as conn:
        with get_cursor(conn) as c:
            c.execute("DELETE FROM subjects_total WHERE subject=%s AND user_id=%s", (subject, uid))
            c.execute("DELETE FROM study_sessions WHERE subject=%s AND user_id=%s", (subject, uid))
    return {"message": f"Subject '{subject}' deleted"}

@app.post("/api/sessions")
def log_session(req: SessionReq, request: Request):
    uid = get_user(request)
    with get_conn() as conn:
        with get_cursor(conn) as c:
            c.execute(
                "INSERT INTO study_sessions(user_id,subject,date,minutes) VALUES(%s,%s,%s,%s)",
                (uid, req.subject, req.date, req.minutes)
            )
            c.execute(
                "INSERT INTO subjects_total(subject,user_id,total_minutes) VALUES(%s,%s,0) ON CONFLICT DO NOTHING",
                (req.subject, uid)
            )
            c.execute(
                "UPDATE subjects_total SET total_minutes=total_minutes+%s WHERE subject=%s AND user_id=%s",
                (req.minutes, req.subject, uid)
            )
            c.execute("SELECT xp, level FROM user_stats WHERE id=%s", (uid,))
            row = c.fetchone()
            if row:
                xp = row["xp"] + req.minutes//5; level = row["level"]
                if xp >= level * 100: level += 1
                c.execute("UPDATE user_stats SET xp=%s, level=%s WHERE id=%s", (xp, level, uid))
    return {"message": "Session logged", "xp_gained": req.minutes//5}

@app.delete("/api/sessions/undo")
def undo_session(request: Request):
    uid = get_user(request)
    with get_conn() as conn:
        with get_cursor(conn) as c:
            c.execute(
                "SELECT id, subject, minutes, date FROM study_sessions WHERE user_id=%s ORDER BY id DESC LIMIT 1",
                (uid,)
            )
            row = c.fetchone()
            if not row: raise HTTPException(404, "Nothing to undo")
            info = dict(row)
            c.execute("DELETE FROM study_sessions WHERE id=%s", (row["id"],))
            c.execute(
                "UPDATE subjects_total SET total_minutes=GREATEST(0,total_minutes-%s) WHERE subject=%s AND user_id=%s",
                (row["minutes"], row["subject"], uid)
            )
            c.execute(
                "UPDATE user_stats SET xp=GREATEST(0,xp-%s) WHERE id=%s",
                (row["minutes"]//5, uid)
            )
    return {
        "message": f"Undone: {info['subject']} — {info['minutes']} min on {info['date']}",
        "undone": info
    }

@app.get("/api/sessions/last")
def get_last_session(request: Request):
    uid = get_user(request)
    with get_conn() as conn:
        with get_cursor(conn) as c:
            c.execute(
                "SELECT subject, minutes, date FROM study_sessions WHERE user_id=%s ORDER BY id DESC LIMIT 1",
                (uid,)
            )
            row = c.fetchone()
    return {"last": dict(row) if row else None}

# ══════════════════════════════════════════════════════════════════════════════
# ML PREDICT
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/api/predict")
async def predict(request: Request):
    uid = get_user(request)
    with get_conn() as conn:
        with get_cursor(conn) as c:
            c.execute("SELECT subject, date, minutes FROM study_sessions WHERE user_id=%s", (uid,))
            study_log: dict = {}
            for r in c.fetchall():
                study_log.setdefault(r["subject"], {})
                study_log[r["subject"]][r["date"]] = study_log[r["subject"]].get(r["date"], 0) + r["minutes"]
            c.execute("SELECT subject, difficulty, last_test_score FROM subjects_meta WHERE user_id=%s", (uid,))
            subjects_meta = {r["subject"]: {"difficulty": r["difficulty"], "score": r["last_test_score"]}
                             for r in c.fetchall()}
    loop   = asyncio.get_running_loop()
    result = await loop.run_in_executor(executor, run_ml, uid, study_log, subjects_meta)
    if result is None:
        return {"error": "Need 7+ sessions for prediction", "tomorrow": None}
    return result

# ══════════════════════════════════════════════════════════════════════════════
# TASKS
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/api/tasks")
def get_tasks(request: Request):
    uid = get_user(request)
    with get_conn() as conn:
        with get_cursor(conn) as c:
            c.execute("""
                SELECT id,task,subject,date,priority,completed,completed_date FROM tasks
                WHERE user_id=%s
                ORDER BY CASE priority WHEN 'Urgent' THEN 1 WHEN 'High' THEN 2
                         WHEN 'Medium' THEN 3 ELSE 4 END, date ASC
            """, (uid,))
            tasks = [dict(r) for r in c.fetchall()]

            # Auto-clean completed > 1 day old
            c.execute(
                "SELECT id, completed_date FROM tasks WHERE completed='Yes' AND user_id=%s", (uid,)
            )
            for r in c.fetchall():
                if r["completed_date"]:
                    try:
                        cd = datetime.strptime(r["completed_date"], "%Y-%m-%d").date()
                        if (date.today() - cd).days >= 1:
                            c.execute("DELETE FROM tasks WHERE id=%s", (r["id"],))
                    except Exception: pass
    return tasks

@app.post("/api/tasks")
def add_task(req: TaskReq, request: Request):
    uid = get_user(request)
    with get_conn() as conn:
        with get_cursor(conn) as c:
            c.execute(
                "INSERT INTO tasks(user_id,task,subject,date,priority,completed) VALUES(%s,%s,%s,%s,%s,%s)",
                (uid, req.task.strip(), req.subject, req.date, req.priority, "No")
            )
    return {"message": "Task added"}

@app.put("/api/tasks/{task_id}/complete")
def complete_task(task_id: int, request: Request):
    uid = get_user(request)
    with get_conn() as conn:
        with get_cursor(conn) as c:
            c.execute(
                "UPDATE tasks SET completed='Yes', completed_date=%s WHERE id=%s AND user_id=%s",
                (str(date.today()), task_id, uid)
            )
    return {"message": "Task completed"}

@app.delete("/api/tasks/{task_id}")
def delete_task(task_id: int, request: Request):
    uid = get_user(request)
    with get_conn() as conn:
        with get_cursor(conn) as c:
            c.execute("DELETE FROM tasks WHERE id=%s AND user_id=%s", (task_id, uid))
    return {"message": "Task deleted"}

# ══════════════════════════════════════════════════════════════════════════════
# NOTES
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/api/notes/{subject}")
def get_notes(subject: str, request: Request):
    uid = get_user(request)
    with get_conn() as conn:
        with get_cursor(conn) as c:
            c.execute(
                "SELECT id, note, created_at FROM subject_notes WHERE user_id=%s AND subject=%s ORDER BY id DESC",
                (uid, subject)
            )
            return [dict(r) for r in c.fetchall()]

@app.post("/api/notes")
def add_note(req: NoteReq, request: Request):
    uid = get_user(request)
    with get_conn() as conn:
        with get_cursor(conn) as c:
            c.execute(
                "INSERT INTO subject_notes(user_id,subject,note,created_at) VALUES(%s,%s,%s,%s)",
                (uid, req.subject, req.note.strip(), datetime.now().strftime("%Y-%m-%d %H:%M"))
            )
    return {"message": "Note saved"}

@app.delete("/api/notes/{note_id}")
def delete_note(note_id: int, request: Request):
    uid = get_user(request)
    with get_conn() as conn:
        with get_cursor(conn) as c:
            c.execute("DELETE FROM subject_notes WHERE id=%s AND user_id=%s", (note_id, uid))
    return {"message": "Note deleted"}

# ══════════════════════════════════════════════════════════════════════════════
# ACCOUNT
# ══════════════════════════════════════════════════════════════════════════════
@app.put("/api/account/goal")
def set_goal(req: GoalReq, request: Request):
    uid = get_user(request)
    with get_conn() as conn:
        with get_cursor(conn) as c:
            c.execute("UPDATE user_stats SET daily_goal=%s WHERE id=%s", (req.daily_goal, uid))
    return {"message": "Goal updated"}

@app.post("/api/account/preferences")
def save_prefs(req: PrefReq, request: Request):
    uid = get_user(request)
    with get_conn() as conn:
        with get_cursor(conn) as c:
            c.execute("""
                INSERT INTO subjects_meta(user_id,subject,difficulty,last_test_score) VALUES(%s,%s,%s,%s)
                ON CONFLICT(user_id,subject) DO UPDATE SET
                difficulty=EXCLUDED.difficulty, last_test_score=EXCLUDED.last_test_score
            """, (uid, req.subject, req.difficulty, req.score))
    return {"message": "Preferences saved"}

@app.delete("/api/account/reset")
def reset_history(request: Request):
    uid = get_user(request)
    with get_conn() as conn:
        with get_cursor(conn) as c:
            c.execute("DELETE FROM subjects_total WHERE user_id=%s", (uid,))
            c.execute("DELETE FROM study_sessions WHERE user_id=%s", (uid,))
            c.execute("UPDATE user_stats SET xp=0,level=1 WHERE id=%s", (uid,))
    return {"message": "History reset"}

# ══════════════════════════════════════════════════════════════════════════════
# PROFILE
# ══════════════════════════════════════════════════════════════════════════════
def _get_profile_full(uid: int, viewer_id: int):
    with get_conn() as conn:
        with get_cursor(conn) as c:
            c.execute("SELECT * FROM user_profiles WHERE user_id=%s", (uid,))
            p = c.fetchone()
            c.execute("SELECT username, email FROM users WHERE id=%s", (uid,))
            u = c.fetchone()
            c.execute("SELECT xp, level FROM user_stats WHERE id=%s", (uid,))
            st = c.fetchone()
            c.execute("SELECT COUNT(*) as cnt FROM follows WHERE followed_id=%s", (uid,))
            followers = c.fetchone()["cnt"]
            c.execute("SELECT COUNT(*) as cnt FROM follows WHERE follower_id=%s", (uid,))
            following = c.fetchone()["cnt"]
            c.execute("SELECT COUNT(*) as cnt FROM follows WHERE follower_id=%s AND followed_id=%s",
                      (viewer_id, uid))
            is_following = bool(c.fetchone()["cnt"])
            c.execute("SELECT subject, total_minutes FROM subjects_total WHERE user_id=%s ORDER BY total_minutes DESC LIMIT 5", (uid,))
            top_subjects = [dict(r) for r in c.fetchall()]
            c.execute("SELECT subject, date, minutes FROM study_sessions WHERE user_id=%s", (uid,))
            sl: dict = {}
            for r in c.fetchall():
                sl.setdefault(r["subject"], {})
                sl[r["subject"]][r["date"]] = sl[r["subject"]].get(r["date"], 0) + r["minutes"]
            streak = compute_streak(sl)
            total_mins = sum(v for subj in sl.values() for v in subj.values())

    return {
        "user_id": uid,
        "username": u["username"] if u else "",
        "bio": p["bio"] if p else "",
        "avatar_emoji": p["avatar_emoji"] if p else "🎓",
        "role": p["role"] if p else "student",
        "fields_of_study": parse_fields_of_study(p["fields_of_study"]) if p else [],
        "country": p["country"] if p else "",
        "school": p["school"] if p else "",
        "grade": p["grade"] if p else "",
        "website": p["website"] if p else "",
        "is_public": p["is_public"] if p else 1,
        "joined_at": p["joined_at"] if p else "",
        "xp": st["xp"] if st else 0,
        "level": st["level"] if st else 1,
        "followers": followers,
        "following": following,
        "is_following": is_following,
        "streak": streak,
        "total_minutes": total_mins,
        "top_subjects": top_subjects,
    }

@app.get("/api/profile")
def get_my_profile(request: Request):
    uid = get_user(request)
    return _get_profile_full(uid, uid)

@app.put("/api/profile")
def update_profile(req: ProfileReq, request: Request):
    uid = get_user(request)
    with get_conn() as conn:
        with get_cursor(conn) as c:
            c.execute("""
                INSERT INTO user_profiles(user_id,bio,avatar_emoji,role,fields_of_study,country,school,grade,website,is_public,joined_at)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT(user_id) DO UPDATE SET
                bio=EXCLUDED.bio, avatar_emoji=EXCLUDED.avatar_emoji,
                role=EXCLUDED.role, fields_of_study=EXCLUDED.fields_of_study,
                country=EXCLUDED.country, school=EXCLUDED.school,
                grade=EXCLUDED.grade, website=EXCLUDED.website,
                is_public=EXCLUDED.is_public
            """, (uid, req.bio, req.avatar_emoji, req.role,
                  req.fields_of_study if isinstance(req.fields_of_study, str) else json.dumps(req.fields_of_study),
                  req.country, req.school, req.grade, req.website, req.is_public,
                  date.today().strftime("%Y-%m-%d")))
    return {"message": "Profile updated"}

@app.get("/api/profile/{uid}")
def get_user_profile(uid: int, request: Request):
    viewer = get_user(request)
    return _get_profile_full(uid, viewer)

@app.get("/api/users/search")
def search_users(q: str, request: Request):
    me = get_user(request)
    with get_conn() as conn:
        with get_cursor(conn) as c:
            c.execute("""
                SELECT u.id, u.username, p.avatar_emoji, p.role, p.fields_of_study, p.country
                FROM users u LEFT JOIN user_profiles p ON u.id=p.user_id
                WHERE u.username ILIKE %s AND u.id != %s LIMIT 20
                      """, (f"%{q}%", me))
            rows = []
            for r in c.fetchall():
                rows.append({
                    "user_id": r["id"], "username": r["username"],
                    "avatar_emoji": r["avatar_emoji"] or "🎓",
                    "role": r["role"] or "student",
                    "fields": parse_fields_of_study(r["fields_of_study"]),
                    "country": r["country"] or "",
                })
    return rows

# ══════════════════════════════════════════════════════════════════════════════
# FOLLOWS
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/api/follow/{target_id}")
def follow_user(target_id: int, request: Request):
    uid = get_user(request)
    if uid == target_id: raise HTTPException(400, "Cannot follow yourself")
    with get_conn() as conn:
        with get_cursor(conn) as c:
            c.execute(
                "INSERT INTO follows(follower_id,followed_id,created_at) VALUES(%s,%s,%s) ON CONFLICT DO NOTHING",
                (uid, target_id, datetime.now().strftime("%Y-%m-%d"))
            )
    return {"message": "Followed"}

@app.delete("/api/follow/{target_id}")
def unfollow_user(target_id: int, request: Request):
    uid = get_user(request)
    with get_conn() as conn:
        with get_cursor(conn) as c:
            c.execute("DELETE FROM follows WHERE follower_id=%s AND followed_id=%s", (uid, target_id))
    return {"message": "Unfollowed"}

@app.get("/api/followers")
def get_followers(request: Request):
    uid = get_user(request)
    with get_conn() as conn:
        with get_cursor(conn) as c:
            c.execute("""
                SELECT u.id, u.username, p.avatar_emoji, p.role FROM follows f
                JOIN users u ON f.follower_id=u.id
                LEFT JOIN user_profiles p ON u.id=p.user_id
                WHERE f.followed_id=%s
            """, (uid,))
            return [{"user_id":r["id"],"username":r["username"],"avatar_emoji":r["avatar_emoji"] or "🎓","role":r["role"] or "student"}
                    for r in c.fetchall()]

@app.get("/api/following")
def get_following(request: Request):
    uid = get_user(request)
    with get_conn() as conn:
        with get_cursor(conn) as c:
            c.execute("""
                SELECT u.id, u.username, p.avatar_emoji, p.role FROM follows f
                JOIN users u ON f.followed_id=u.id
                LEFT JOIN user_profiles p ON u.id=p.user_id
                WHERE f.follower_id=%s
            """, (uid,))
            return [{"user_id":r["id"],"username":r["username"],"avatar_emoji":r["avatar_emoji"] or "🎓","role":r["role"] or "student"}
                    for r in c.fetchall()]

@app.get("/api/study-buddies")
def get_study_buddies(request: Request):
    uid = get_user(request)
    with get_conn() as conn:
        with get_cursor(conn) as c:
            c.execute("SELECT fields_of_study FROM user_profiles WHERE user_id=%s", (uid,))
            my_row = c.fetchone()
            my_fields = parse_fields_of_study(my_row["fields_of_study"]) if my_row else []
            c.execute("""
                SELECT u.id, u.username, p.avatar_emoji, p.role, p.fields_of_study, p.country, p.school,
                       st.xp, st.level
                FROM users u
                LEFT JOIN user_profiles p ON u.id=p.user_id
                LEFT JOIN user_stats st ON u.id=st.id
                WHERE u.id != %s AND p.is_public=1
            """, (uid,))
            buddies = []
            for r in c.fetchall():
                their_fields = parse_fields_of_study(r["fields_of_study"])
                overlap = [f for f in their_fields if f in my_fields]
                if overlap or not my_fields:
                    buddies.append({
                        "user_id": r["id"], "username": r["username"],
                        "avatar_emoji": r["avatar_emoji"] or "🎓",
                        "role": r["role"] or "student",
                        "fields": their_fields, "common": overlap,
                        "country": r["country"] or "", "school": r["school"] or "",
                        "level": r["level"] or 1,
                    })
    buddies.sort(key=lambda x: -len(x["common"]))
    return buddies[:20]

# ══════════════════════════════════════════════════════════════════════════════
# COMMUNITY POSTS
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/api/feed")
def get_feed(request: Request, limit: int = 30, offset: int = 0):
    uid = get_user(request)
    with get_conn() as conn:
        with get_cursor(conn) as c:
            c.execute("""
                SELECT p.id, p.user_id, p.content, p.subject_tag, p.post_type, p.created_at,
                       u.username, pr.avatar_emoji,
                       (SELECT COUNT(*) FROM post_likes pl WHERE pl.post_id=p.id) as like_count,
                       (SELECT COUNT(*) FROM post_comments pc WHERE pc.post_id=p.id) as comment_count,
                       (SELECT COUNT(*) FROM post_likes pl2 WHERE pl2.post_id=p.id AND pl2.user_id=%s) as i_liked
                FROM community_posts p
                JOIN users u ON p.user_id=u.id
                LEFT JOIN user_profiles pr ON p.user_id=pr.user_id
                ORDER BY p.id DESC LIMIT %s OFFSET %s
            """, (uid, limit, offset))
            return [dict(r) for r in c.fetchall()]

@app.post("/api/posts")
def create_post(req: PostReq, request: Request):
    uid = get_user(request)
    with get_conn() as conn:
        with get_cursor(conn) as c:
            c.execute(
                "INSERT INTO community_posts(user_id,content,subject_tag,post_type,created_at) VALUES(%s,%s,%s,%s,%s)",
                (uid, req.content.strip(), req.subject_tag, req.post_type, datetime.now().strftime("%Y-%m-%d %H:%M"))
            )
    return {"message": "Posted!"}

@app.delete("/api/posts/{post_id}")
def delete_post(post_id: int, request: Request):
    uid = get_user(request)
    with get_conn() as conn:
        with get_cursor(conn) as c:
            c.execute("DELETE FROM community_posts WHERE id=%s AND user_id=%s", (post_id, uid))
    return {"message": "Post deleted"}

@app.post("/api/posts/{post_id}/like")
def toggle_like(post_id: int, request: Request):
    uid = get_user(request)
    with get_conn() as conn:
        with get_cursor(conn) as c:
            c.execute("SELECT 1 FROM post_likes WHERE post_id=%s AND user_id=%s", (post_id, uid))
            if c.fetchone():
                c.execute("DELETE FROM post_likes WHERE post_id=%s AND user_id=%s", (post_id, uid))
                liked = False
            else:
                c.execute("INSERT INTO post_likes(post_id,user_id) VALUES(%s,%s)", (post_id, uid))
                liked = True
    return {"liked": liked}

@app.get("/api/posts/{post_id}/comments")
def get_comments(post_id: int, request: Request):
    get_user(request)
    with get_conn() as conn:
        with get_cursor(conn) as c:
            c.execute("""
                SELECT pc.id, pc.user_id, pc.comment, pc.created_at, u.username, pr.avatar_emoji
                FROM post_comments pc JOIN users u ON pc.user_id=u.id
                LEFT JOIN user_profiles pr ON pc.user_id=pr.user_id
                WHERE pc.post_id=%s ORDER BY pc.id ASC
            """, (post_id,))
            return [dict(r) for r in c.fetchall()]

@app.post("/api/posts/{post_id}/comments")
def add_comment(post_id: int, req: CommentReq, request: Request):
    uid = get_user(request)
    with get_conn() as conn:
        with get_cursor(conn) as c:
            c.execute(
                "INSERT INTO post_comments(post_id,user_id,comment,created_at) VALUES(%s,%s,%s,%s)",
                (post_id, uid, req.comment.strip(), datetime.now().strftime("%Y-%m-%d %H:%M"))
            )
    return {"message": "Comment added"}

@app.delete("/api/posts/{post_id}/comments/{cid}")
def delete_comment(post_id: int, cid: int, request: Request):
    uid = get_user(request)
    with get_conn() as conn:
        with get_cursor(conn) as c:
            c.execute("DELETE FROM post_comments WHERE id=%s AND user_id=%s", (cid, uid))
    return {"message": "Comment deleted"}

# ══════════════════════════════════════════════════════════════════════════════
# CHAT ROOMS
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/api/rooms")
def get_rooms(request: Request):
    uid = get_user(request)
    with get_conn() as conn:
        with get_cursor(conn) as c:
            c.execute("""
                SELECT r.id, r.name, r.description, r.room_type,
                       (SELECT COUNT(*) FROM room_members rm WHERE rm.room_id=r.id) as member_count,
                       (SELECT COUNT(*) FROM room_messages rm WHERE rm.room_id=r.id) as msg_count,
                       (SELECT COUNT(*) FROM room_members rm WHERE rm.room_id=r.id AND rm.user_id=%s) as is_member
                FROM chat_rooms r ORDER BY r.id ASC
            """, (uid,))
            return [dict(r) for r in c.fetchall()]

@app.post("/api/rooms")
def create_room(req: RoomReq, request: Request):
    uid = get_user(request)
    with get_conn() as conn:
        with get_cursor(conn) as c:
            c.execute(
                "INSERT INTO chat_rooms(name,description,room_type,created_by,created_at) VALUES(%s,%s,%s,%s,%s) RETURNING id",
                (req.name.strip(), req.description, req.room_type, uid, datetime.now().strftime("%Y-%m-%d %H:%M"))
            )
            rid = c.fetchone()["id"]
            c.execute(
                "INSERT INTO room_members(room_id,user_id,joined_at) VALUES(%s,%s,%s) ON CONFLICT DO NOTHING",
                (rid, uid, datetime.now().strftime("%Y-%m-%d"))
            )
    return {"message": "Room created", "room_id": rid}

@app.post("/api/rooms/{room_id}/join")
def join_room(room_id: int, request: Request):
    uid = get_user(request)
    with get_conn() as conn:
        with get_cursor(conn) as c:
            c.execute(
                "INSERT INTO room_members(room_id,user_id,joined_at) VALUES(%s,%s,%s) ON CONFLICT DO NOTHING",
                (room_id, uid, datetime.now().strftime("%Y-%m-%d"))
            )
    return {"message": "Joined room"}

@app.delete("/api/rooms/{room_id}/leave")
def leave_room(room_id: int, request: Request):
    uid = get_user(request)
    with get_conn() as conn:
        with get_cursor(conn) as c:
            c.execute("DELETE FROM room_members WHERE room_id=%s AND user_id=%s", (room_id, uid))
    return {"message": "Left room"}

@app.get("/api/rooms/{room_id}/messages")
def get_messages(room_id: int, request: Request, since_id: int = 0):
    get_user(request)
    with get_conn() as conn:
        with get_cursor(conn) as c:
            c.execute("""
                SELECT m.id, m.user_id, m.message, m.created_at, u.username, p.avatar_emoji
                FROM room_messages m JOIN users u ON m.user_id=u.id
                LEFT JOIN user_profiles p ON m.user_id=p.user_id
                WHERE m.room_id=%s AND m.id>%s
                ORDER BY m.id ASC LIMIT 100
            """, (room_id, since_id))
            return [dict(r) for r in c.fetchall()]

@app.post("/api/rooms/{room_id}/messages")
def send_message(room_id: int, req: MessageReq, request: Request):
    uid = get_user(request)
    if not req.message.strip(): raise HTTPException(400, "Empty message")
    with get_conn() as conn:
        with get_cursor(conn) as c:
            c.execute(
                "INSERT INTO room_messages(room_id,user_id,message,created_at) VALUES(%s,%s,%s,%s) RETURNING id",
                (room_id, uid, req.message.strip(), datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )
            mid = c.fetchone()["id"]
            c.execute("SELECT username FROM users WHERE id=%s", (uid,))
            uname = c.fetchone()["username"]
            c.execute("SELECT avatar_emoji FROM user_profiles WHERE user_id=%s", (uid,))
            p = c.fetchone()
            emoji = p["avatar_emoji"] if p else "🎓"
    return {"id": mid, "user_id": uid, "username": uname, "avatar_emoji": emoji,
            "message": req.message.strip(), "created_at": datetime.now().strftime("%H:%M")}

@app.websocket("/ws/room/{room_id}")
async def ws_room(websocket: WebSocket, room_id: int):
    token = websocket.query_params.get("token")
    uid = verify_token(token)
    if not uid:
        await websocket.close(code=1008); return

    with get_conn() as conn:
        with get_cursor(conn) as c:
            c.execute("SELECT username FROM users WHERE id=%s", (uid,))
            u = c.fetchone()
            c.execute("SELECT avatar_emoji FROM user_profiles WHERE user_id=%s", (uid,))
            p = c.fetchone()
    uname = u["username"] if u else "User"
    emoji = p["avatar_emoji"] if p else "🎓"

    await manager.connect(room_id, websocket)
    await manager.broadcast(room_id, {"type": "system", "message": f"{uname} joined",
                                       "online": manager.count(room_id)})
    try:
        while True:
            data = await websocket.receive_json()
            msg = data.get("message", "").strip()
            if not msg: continue
            with get_conn() as conn2:
                with get_cursor(conn2) as c2:
                    c2.execute(
                        "INSERT INTO room_messages(room_id,user_id,message,created_at) VALUES(%s,%s,%s,%s) RETURNING id",
                        (room_id, uid, msg, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    )
                    mid = c2.fetchone()["id"]
            await manager.broadcast(room_id, {
                "type": "message", "id": mid, "user_id": uid,
                "username": uname, "avatar_emoji": emoji, "message": msg,
                "created_at": datetime.now().strftime("%H:%M")
            })
    except WebSocketDisconnect:
        manager.disconnect(room_id, websocket)
        await manager.broadcast(room_id, {"type": "system", "message": f"{uname} left",
                                           "online": manager.count(room_id)})

# ══════════════════════════════════════════════════════════════════════════════
# LEADERBOARD
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/api/leaderboard")
def get_leaderboard(request: Request, period: str = "weekly"):
    get_user(request)
    with get_conn() as conn:
        with get_cursor(conn) as c:
            if period == "weekly":
                cutoff = (date.today() - timedelta(days=7)).strftime("%d-%m-%Y")
                c.execute("""
                    SELECT ss.user_id, u.username, p.avatar_emoji, p.role, p.country,
                           SUM(ss.minutes) as total_mins, st.xp, st.level
                    FROM study_sessions ss
                    JOIN users u ON ss.user_id=u.id
                    LEFT JOIN user_profiles p ON ss.user_id=p.user_id
                    LEFT JOIN user_stats st ON ss.user_id=st.id
                    WHERE ss.date >= %s
                    GROUP BY ss.user_id, u.username, p.avatar_emoji, p.role, p.country, st.xp, st.level
                    ORDER BY total_mins DESC LIMIT 50
                """, (cutoff,))
            elif period == "alltime":
                c.execute("""
                    SELECT st2.user_id, u.username, p.avatar_emoji, p.role, p.country,
                           SUM(st2.total_minutes) as total_mins, st.xp, st.level
                    FROM subjects_total st2
                    JOIN users u ON st2.user_id=u.id
                    LEFT JOIN user_profiles p ON st2.user_id=p.user_id
                    LEFT JOIN user_stats st ON st2.user_id=st.id
                    GROUP BY st2.user_id, u.username, p.avatar_emoji, p.role, p.country, st.xp, st.level
                    ORDER BY total_mins DESC LIMIT 50
                """)
            else:  # xp
                c.execute("""
                    SELECT u.id as user_id, u.username, p.avatar_emoji, p.role, p.country,
                           st.xp, st.level, st.xp as total_mins
                    FROM user_stats st JOIN users u ON st.id=u.id
                    LEFT JOIN user_profiles p ON u.id=p.user_id
                    ORDER BY st.xp DESC LIMIT 50
                """)
            rows = []
            for i, r in enumerate(c.fetchall()):
                rows.append({
                    "rank": i+1, "user_id": r["user_id"], "username": r["username"],
                    "avatar_emoji": r["avatar_emoji"] or "🎓", "role": r["role"] or "student",
                    "country": r["country"] or "", "total_mins": r["total_mins"] or 0,
                    "xp": r["xp"] or 0, "level": r["level"] or 1,
                })
    return rows

# ══════════════════════════════════════════════════════════════════════════════
# MONITORING
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/api/monitor")
def add_monitored(req: MonitorReq, request: Request):
    uid = get_user(request)
    with get_conn() as conn:
        with get_cursor(conn) as c:
            c.execute("SELECT id FROM users WHERE username=%s", (req.username,))
            row = c.fetchone()
            if not row: raise HTTPException(404, "User not found")
            if row["id"] == uid: raise HTTPException(400, "Cannot monitor yourself")
            c.execute(
                "INSERT INTO monitored_users(monitor_id,student_id,relationship,created_at) VALUES(%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                (uid, row["id"], req.relationship, datetime.now().strftime("%Y-%m-%d"))
            )
    return {"message": f"Now monitoring @{req.username}"}

@app.delete("/api/monitor/{student_id}")
def remove_monitored(student_id: int, request: Request):
    uid = get_user(request)
    with get_conn() as conn:
        with get_cursor(conn) as c:
            c.execute("DELETE FROM monitored_users WHERE monitor_id=%s AND student_id=%s", (uid, student_id))
    return {"message": "Removed"}

@app.get("/api/monitor/students")
def get_monitored_students(request: Request):
    uid = get_user(request)
    with get_conn() as conn:
        with get_cursor(conn) as c:
            c.execute("""
                SELECT mu.student_id, mu.relationship, u.username, p.avatar_emoji
                FROM monitored_users mu JOIN users u ON mu.student_id=u.id
                LEFT JOIN user_profiles p ON mu.student_id=p.user_id
                WHERE mu.monitor_id=%s
            """, (uid,))
            students = []
            for r in c.fetchall():
                sid = r["student_id"]
                c.execute("SELECT xp, level FROM user_stats WHERE id=%s", (sid,))
                st = c.fetchone()
                c.execute(
                    "SELECT subject, date, minutes FROM study_sessions WHERE user_id=%s ORDER BY id DESC LIMIT 30",
                    (sid,)
                )
                sl: dict = {}
                for s in c.fetchall():
                    sl.setdefault(s["subject"], {})
                    sl[s["subject"]][s["date"]] = sl[s["subject"]].get(s["date"], 0) + s["minutes"]
                today = date.today().strftime("%d-%m-%Y")
                today_mins = sum(sl.get(s, {}).get(today, 0) for s in sl)
                streak = compute_streak(sl)
                students.append({
                    "user_id": sid, "username": r["username"],
                    "avatar_emoji": r["avatar_emoji"] or "🎓",
                    "relationship": r["relationship"],
                    "xp": st["xp"] if st else 0, "level": st["level"] if st else 1,
                    "today_mins": today_mins, "streak": streak,
                })
    return students

# ══════════════════════════════════════════════════════════════════════════════
# AI CHATBOT (Groq)
# ══════════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPTS = {
    "student": (
        "You are StudyNest AI, a warm and encouraging study coach. Help students with: "
        "study techniques, subject questions, time management, motivation, stress, exam prep. "
        "Be concise (2-3 paragraphs max), practical, and always end with an actionable tip. "
        "Use emojis occasionally. Never give medical/legal advice."
    ),
    "teacher": (
        "You are StudyNest AI, a professional educational assistant for teachers. "
        "Help with: lesson planning, student engagement, assessment strategies, differentiated instruction, "
        "classroom management, educational research. Be evidence-based and professional."
    ),
    "parent": (
        "You are StudyNest AI, a supportive guide for parents supporting their children's education. "
        "Help with: creating study environments, motivating children, understanding curriculum, "
        "screen time management, communication with teachers, recognizing learning difficulties. "
        "Be warm, practical, and non-judgmental."
    ),
}

@app.post("/api/chatbot")
async def chatbot(req: ChatbotReq, request: Request):
    uid = get_user(request)
    with get_conn() as conn:
        with get_cursor(conn) as c:
            c.execute("SELECT role FROM user_profiles WHERE user_id=%s", (uid,))
            p = c.fetchone()
    role   = p["role"] if p else "student"
    system = SYSTEM_PROMPTS.get(role, SYSTEM_PROMPTS["student"])
    messages = [{"role": "system", "content": system}]
    for h in (req.history or [])[-8:]:
        if h.get("role") in ("user","assistant") and h.get("content"):
            messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": req.message})

    if GROQ_API_KEY and HTTPX_AVAILABLE:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}",
                             "Content-Type": "application/json"},
                    json={"model": "llama-3.1-8b-instant", "messages": messages, "max_tokens": 512}
                )
                data = r.json()
                reply = data["choices"][0]["message"]["content"]
                return {"reply": reply, "source": "llama"}
        except Exception as e:
            return {"reply": f"AI temporarily unavailable: {str(e)}", "source": "error"}

    return {
        "reply": (
            "💡 AI chatbot requires a GROQ_API_KEY environment variable.\n\n"
            "Get a free key at console.groq.com — it gives you Llama 3.1 for free.\n"
            "Then set GROQ_API_KEY in your Render environment variables."
        ),
        "source": "fallback"
    }

# ══════════════════════════════════════════════════════════════════════════════
# EXPORT
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/api/export/pdf")
def export_pdf(request: Request):
    uid = get_user(request)
    with get_conn() as conn:
        with get_cursor(conn) as c:
            c.execute(
                "SELECT subject, date, minutes FROM study_sessions WHERE user_id=%s ORDER BY date DESC", (uid,)
            )
            rows = c.fetchall()
            c.execute("SELECT xp, level FROM user_stats WHERE id=%s", (uid,))
            stats = c.fetchone()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("StudyNest — Progress Report", styles["Title"]),
        Spacer(1,10),
        Paragraph(
            f"Generated: {date.today()} | Level: {stats['level'] if stats else 1} | XP: {stats['xp'] if stats else 0}",
            styles["Normal"]
        ),
        Spacer(1,12)
    ]
    td = [["Date","Subject","Minutes"]]
    for r in rows: td.append([r["date"], r["subject"], str(r["minutes"])])
    t = Table(td, colWidths=[100,200,80])
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0), rl_colors.HexColor("#1e40af")),
        ("TEXTCOLOR",(0,0),(-1,0),  rl_colors.white),
        ("FONTNAME",(0,0),(-1,0),   "Helvetica-Bold"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[rl_colors.white, rl_colors.HexColor("#f1f5f9")]),
        ("GRID",(0,0),(-1,-1),0.4,  rl_colors.grey),
        ("FONTSIZE",(0,0),(-1,-1),  9),
    ]))
    story.append(t); doc.build(story); buf.seek(0)
    return StreamingResponse(buf, media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=studynest_report_{date.today()}.pdf"})

@app.get("/api/export/csv")
def export_csv(request: Request):
    uid = get_user(request)
    with get_conn() as conn:
        with get_cursor(conn) as c:
            c.execute(
                "SELECT subject, date, minutes FROM study_sessions WHERE user_id=%s ORDER BY date DESC", (uid,)
            )
            rows = c.fetchall()
    lines = ["Date,Subject,Minutes"] + [f"{r['date']},{r['subject']},{r['minutes']}" for r in rows]
    return StreamingResponse(io.StringIO("\n".join(lines)), media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=studynest_{date.today()}.csv"})

# ══════════════════════════════════════════════════════════════════════════════
# STATIC / FRONTEND
# ══════════════════════════════════════════════════════════════════════════════
try:
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
except Exception:
    pass

@app.get("/")
@app.get("/{path:path}")
def serve_frontend(path: str = ""):
    index = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return {"message": "StudyNest API running. Place index.html in same directory as main.py."}
