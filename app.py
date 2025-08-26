"""
Streamlit AutoML App
--------------------
Large, single-file Streamlit application that organizes a simple AutoML
workflow with:
 - Lightweight authentication (sqlite + hashed passwords)
 - History logging (sqlite)
 - Ten ML steps (data upload -> preprocessing -> model training -> export)
 - Small chatbot using OpenAI API (requires OPENAI_API_KEY env var)
 - Model storage, download and basic deployment simulation

Notes:
 - This is intended as a well-documented, educational AutoML starter app.
 - For real production systems, split code across files, secure secrets,
   and use robust authentication and deployment.

Dependencies (install with pip):
streamlit, pandas, scikit-learn, joblib, sqlalchemy, passlib, openai

Example:
    pip install streamlit pandas scikit-learn joblib sqlalchemy passlib openai
    export OPENAI_API_KEY="sk-..."
    streamlit run streamlit_automl_app.py

"""

import os
import io
import uuid
import json
import sqlite3
import base64
import tempfile
from datetime import datetime
from typing import Tuple, Dict, Any, List

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score, classification_report
import joblib

# Security: password hashing
from passlib.context import CryptContext

# Optional OpenAI integration
try:
    import openai
except Exception:
    openai = None

# ---------- Configuration ----------
APP_TITLE = "AutoML Streamlit App"
DB_PATH = "automl_app.db"
MODEL_DIR = "saved_models"
ALLOWED_EXTENSIONS = ["csv", "parquet", "xls", "xlsx"]

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ---------- Helpers: DB and Auth ----------

def init_db(path: str = DB_PATH):
    conn = sqlite3.connect(path, check_same_thread=False)
    c = conn.cursor()
    # Users table
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE,
            password_hash TEXT,
            display_name TEXT,
            created_at TEXT
        )
        """
    )
    # History table
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY,
            username TEXT,
            action TEXT,
            details TEXT,
            timestamp TEXT
        )
        """
    )
    conn.commit()
    return conn

DB_CONN = init_db()


def create_user(username: str, password: str, display_name: str = None) -> bool:
    pw_hash = pwd_context.hash(password)
    if display_name is None:
        display_name = username
    try:
        cur = DB_CONN.cursor()
        cur.execute(
            "INSERT INTO users (username, password_hash, display_name, created_at) VALUES (?, ?, ?, ?)",
            (username, pw_hash, display_name, datetime.utcnow().isoformat()),
        )
        DB_CONN.commit()
        return True
    except sqlite3.IntegrityError:
        return False


def verify_user(username: str, password: str) -> bool:
    cur = DB_CONN.cursor()
    cur.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
    r = cur.fetchone()
    if not r:
        return False
    stored_hash = r[0]
    return pwd_context.verify(password, stored_hash)


def log_history(username: str, action: str, details: str = ""):
    cur = DB_CONN.cursor()
    cur.execute(
        "INSERT INTO history (username, action, details, timestamp) VALUES (?, ?, ?, ?)",
        (username, action, details, datetime.utcnow().isoformat()),
    )
    DB_CONN.commit()


def get_history(limit: int = 100) -> List[Dict[str, Any]]:
    cur = DB_CONN.cursor()
    cur.execute("SELECT id, username, action, details, timestamp FROM history ORDER BY id DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    return [dict(id=r[0], username=r[1], action=r[2], details=r[3], timestamp=r[4]) for r in rows]


# ---------- Utilities ----------

def allowed_file(filename: str) -> bool:
    ext = filename.rsplit(".", 1)[-1].lower()
    return ext in ALLOWED_EXTENSIONS


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name
    if not allowed_file(name):
        raise ValueError("Format de fichier non supporté. Utilisez CSV/Parquet/XLSX.")
    ext = name.rsplit(".", 1)[-1].lower()
    if ext == "csv":
        return pd.read_csv(uploaded_file)
    elif ext in ("xls", "xlsx"):
        return pd.read_excel(uploaded_file)
    elif ext == "parquet":
        return pd.read_parquet(uploaded_file)
    else:
        raise ValueError("Extension inconnue")


# ---------- AutoML pipeline pieces (10 étapes) ----------

# 1. Charger les données
# 2. Exploration (EDA)
# 3. Nettoyage / Imputation
# 4. Feature engineering
# 5. Séparation train/test
# 6. Sélection du modèle
# 7. Entraînement
# 8. Validation & évaluation
# 9. Hyperparam tuning
# 10. Export / déploiement


class AutoMLSession:
    """Container pour stocker l'état et les artefacts d'une session AutoML."""

    def __init__(self):
        self.data = None
        self.target = None
        self.task = None  # 'classification' or 'regression'
        self.features = None
        self.pipeline = None
        self.model = None
        self.trained = False
        self.metrics = {}
        self.model_path = None


@st.cache_resource
def new_session() -> AutoMLSession:
    return AutoMLSession()


# Utility to infer task

def infer_task(series: pd.Series) -> str:
    # Simple heuristic: if numeric and many unique values -> regression
    if pd.api.types.is_numeric_dtype(series):
        if series.nunique() <= 20:
            return "classification"
        else:
            return "regression"
    else:
        return "classification"


# 3. build preprocessing pipeline

def build_preprocessor(df: pd.DataFrame, numeric_threshold: int = 0.6) -> Tuple[ColumnTransformer, List[str]]:
    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # Simple imputers and scalers
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ],
        remainder="drop",
    )

    features = numeric_cols + categorical_cols
    return preprocessor, features


# 6. model selection helper

def get_model_candidates(task: str):
    if task == "classification":
        return {
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
            "LogisticRegression": LogisticRegression(max_iter=1000)
        }
    else:
        return {
            "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=42),
            "LinearRegression": LinearRegression()
        }


# 7. train function

def train_model(session: AutoMLSession, X_train: pd.DataFrame, y_train: pd.Series, model_name: str, params: dict = None) -> Any:
    candidates = get_model_candidates(session.task)
    base_model = candidates.get(model_name)
    if base_model is None:
        raise ValueError("Modèle non supporté")

    # build a pipeline: preprocessor + estimator
    pipeline = Pipeline(steps=[("preprocessor", session.pipeline), ("estimator", base_model)])

    if params:
        # Apply params to estimator if gridsearch style keys
        estimator_params = {k.replace("estimator__", ""): v for k, v in params.items()}
        base_model.set_params(**estimator_params)

    pipeline.fit(X_train, y_train)
    session.model = pipeline
    session.trained = True
    return pipeline


# 8. evaluate

def evaluate(session: AutoMLSession, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    if not session.trained:
        raise ValueError("Modèle non entraîné")
    preds = session.model.predict(X_test)
    if session.task == "classification":
        acc = accuracy_score(y_test, preds)
        try:
            f1 = f1_score(y_test, preds, average="weighted")
        except Exception:
            f1 = None
        report = classification_report(y_test, preds, output_dict=True)
        metrics = {"accuracy": acc, "f1_weighted": f1, "report": report}
    else:
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        metrics = {"mse": mse, "r2": r2}
    session.metrics = metrics
    return metrics


# 9. hyperparameter tuning (basic gridsearch)

def tune_hyperparameters(session: AutoMLSession, X_train: pd.DataFrame, y_train: pd.Series, param_grid: dict, cv: int = 3):
    if not session.pipeline:
        raise ValueError("Préprocesseur non défini")
    # we will try gridsearch on estimator
    model = RandomForestClassifier() if session.task == "classification" else RandomForestRegressor()
    pipeline = Pipeline(steps=[("preprocessor", session.pipeline), ("estimator", model)])
    grid = GridSearchCV(pipeline, param_grid=param_grid, cv=cv, n_jobs=-1)
    grid.fit(X_train, y_train)
    session.model = grid.best_estimator_
    session.trained = True
    return grid


# 10. export model

def save_model(session: AutoMLSession, name: str = None) -> str:
    if not session.trained:
        raise ValueError("Aucun modèle entraîné à sauvegarder")
    if name is None:
        name = f"model_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.joblib"
    path = os.path.join(MODEL_DIR, name)
    joblib.dump(session.model, path)
    session.model_path = path
    return path


# ---------- Small Chatbot using OpenAI ----------

def openai_chat(prompt: str, system: str = "You are a helpful assistant.") -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if openai is None:
        return "OpenAI library not installed. Install openai package to use the chatbot."
    if not api_key:
        return "OPENAI_API_KEY not set in environment. Set it to use the chatbot."
    try:
        openai.api_key = api_key
        # Using ChatCompletion (example) - adjust per your OpenAI client version
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # user can change to available model
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.2,
        )
        text = response.choices[0].message.content.strip()
        return text
    except Exception as e:
        return f"Error calling OpenAI: {e}"


# ---------- Streamlit App UI ----------

st.set_page_config(APP_TITLE, layout="wide")

# Sidebar: Authentication
st.sidebar.title("Connexion")
if "auth_user" not in st.session_state:
    st.session_state.auth_user = None

with st.sidebar.expander("Se connecter / S'inscrire", expanded=True):
    auth_mode = st.radio("Mode", ["Se connecter", "S'inscrire"], index=0)
    username = st.text_input("Nom d'utilisateur")
    password = st.text_input("Mot de passe", type="password")
    display_name = None
    if auth_mode == "S'inscrire":
        display_name = st.text_input("Nom affiché (optionnel)")
    if st.button("Valider"):
        if auth_mode == "S'inscrire":
            ok = create_user(username, password, display_name)
            if ok:
                st.success("Compte créé. Connecte-toi maintenant.")
                log_history(username, "signup", "Nouvel utilisateur")
            else:
                st.error("Nom d'utilisateur déjà pris")
        else:
            if verify_user(username, password):
                st.session_state.auth_user = username
                st.success(f"Connecté en tant que {username}")
                log_history(username, "login", "Connexion réussie")
            else:
                st.error("Échec de la connexion")

if st.session_state.auth_user:
    st.sidebar.markdown(f"**Connecté :** {st.session_state.auth_user}")
    if st.sidebar.button("Se déconnecter"):
        log_history(st.session_state.auth_user, "logout", "Utilisateur déconnecté")
        st.session_state.auth_user = None

# Main layout
st.title(APP_TITLE)
st.write("Petite plateforme AutoML pour prototyper un pipeline complet, avec historique et chatbot intégré.")

# Initialize or retrieve session
if "automl_session" not in st.session_state:
    st.session_state.automl_session = new_session()

session: AutoMLSession = st.session_state.automl_session

# Tabs: Workflow, Chatbot, Historique, Modèles
tabs = st.tabs(["Workflow AutoML", "Chatbot IA", "Historique", "Modèles"])

# ---------- Tab 1: Workflow AutoML ----------
with tabs[0]:
    st.header("Workflow AutoML - 10 étapes")

    step = st.selectbox("Choisis une étape", [
        "1 - Charger les données",
        "2 - Exploration (EDA)",
        "3 - Nettoyage & Imputation",
        "4 - Feature Engineering",
        "5 - Train/Test Split",
        "6 - Sélection du modèle",
        "7 - Entraînement",
        "8 - Validation & Évaluation",
        "9 - Hyperparam Tuning",
        "10 - Export / Déploiement"
    ])

    # ---------- Step 1: Load Data ----------
    if step.startswith("1"):
        st.subheader("1 — Charger les données")
        uploaded = st.file_uploader("Upload un fichier (CSV, Parquet, XLSX)")
        if uploaded is not None:
            try:
                df = read_uploaded_file(uploaded)
                session.data = df.copy()
                st.success(f"Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes")
                st.dataframe(df.head())
                log_history(st.session_state.auth_user or "anonymous", "load_data", f"{uploaded.name} - {df.shape}")
            except Exception as e:
                st.error(f"Erreur lors du chargement : {e}")

    # ---------- Step 2: EDA ----------
    if step.startswith("2"):
        st.subheader("2 — Exploration (EDA)")
        if session.data is None:
            st.warning("Aucune donnée. Charge un fichier d'abord.")
        else:
            df = session.data
            st.write("Aperçu des données :")
            st.dataframe(df.head())
            st.write("Dtypes :")
            st.write(df.dtypes)
            st.write("Statistiques descriptives (numériques) :")
            st.write(df.describe().T)
            st.write("Colonnes avec valeurs manquantes :")
            na = df.isna().sum()
            st.write(na[na > 0])
            log_history(st.session_state.auth_user or "anonymous", "eda", f"shape={df.shape}")

    # ---------- Step 3: Cleaning ----------
    if step.startswith("3"):
        st.subheader("3 — Nettoyage & Imputation")
        if session.data is None:
            st.warning("Aucune donnée. Charge un fichier d'abord.")
        else:
            df = session.data
            st.write("Options de nettoyage rapides :")
            if st.checkbox("Supprimer doublons"):
                before = df.shape[0]
                df = df.drop_duplicates()
                after = df.shape[0]
                st.write(f"Doublons supprimés : {before - after}")
            if st.checkbox("Supprimer lignes avec valeurs manquantes massives (>50%)"):
                thresh = int(df.shape[1] * 0.5)
                before = df.shape[0]
                df = df.dropna(thresh=thresh)
                after = df.shape[0]
                st.write(f"Lignes supprimées : {before - after}")
            if st.button("Appliquer et sauvegarder nettoyage"):
                session.data = df.copy()
                st.success("Nettoyage appliqué et sauvegardé dans la session.")
                log_history(st.session_state.auth_user or "anonymous", "cleaning", f"shape={df.shape}")
            st.dataframe(df.head())

    # ---------- Step 4: Feature Engineering ----------
    if step.startswith("4"):
        st.subheader("4 — Feature Engineering")
        if session.data is None:
            st.warning("Aucune donnée. Charge un fichier d'abord.")
        else:
            df = session.data
            st.write("Créer des features simples :")
            cols = df.columns.tolist()
            st.write(cols)
            new_col_name = st.text_input("Nom de la nouvelle colonne (ex: total_price)")
            expr = st.text_area("Expression pandas (utiliser 'df' pour référencer). Exemple: df['price'] * df['quantity']")
            if st.button("Ajouter la colonne") and new_col_name and expr:
                try:
                    local_env = {"df": df}
                    exec(f"df['{new_col_name}'] = {expr}", {}, local_env)
                    session.data = local_env["df"].copy()
                    st.success(f"Colonne {new_col_name} ajoutée")
                    log_history(st.session_state.auth_user or "anonymous", "feature_engineering", new_col_name)
                except Exception as e:
                    st.error(f"Erreur lors de la création de la colonne: {e}")
            st.write("Aperçu :")
            st.dataframe(session.data.head())

    # ---------- Step 5: Train/Test Split ----------
    if step.startswith("5"):
        st.subheader("5 — Train/Test Split")
        if session.data is None:
            st.warning("Aucune donnée. Charge un fichier d'abord.")
        else:
            df = session.data
            st.write("Colonnes disponibles :")
            st.write(df.columns.tolist())
            target_col = st.selectbox("Choisis la colonne cible (target)", [None] + df.columns.tolist())
            if target_col:
                session.target = target_col
                session.task = infer_task(df[target_col])
                st.info(f"Tâche inférée : {session.task}")
                features = st.multiselect("Choisis les features (laisser vide pour toutes sauf target)", [c for c in df.columns if c != target_col], default=[c for c in df.columns if c != target_col])
                if features:
                    session.features = features
                    test_size = st.slider("Taille du test set", min_value=0.1, max_value=0.5, value=0.2)
                    random_state = st.number_input("Seed (random_state)", value=42)
                    if st.button("Créer split"):
                        X = df[features]
                        y = df[target_col]
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_state))
                        # Save split to session
                        session.X_train = X_train
                        session.X_test = X_test
                        session.y_train = y_train
                        session.y_test = y_test
                        st.success(f"Split créé. Train: {X_train.shape}, Test: {X_test.shape}")
                        log_history(st.session_state.auth_user or "anonymous", "split", f"target={target_col} test_size={test_size}")
                else:
                    st.warning("Sélectionne au moins une feature")

    # ---------- Step 6: Model selection ----------
    if step.startswith("6"):
        st.subheader("6 — Sélection du modèle")
        if not hasattr(session, "X_train"):
            st.warning("Crée un train/test split d'abord (étape 5)")
        else:
            st.write("Modèles candidats :")
            candidates = get_model_candidates(session.task)
            st.write(list(candidates.keys()))
            chosen = st.selectbox("Choisis un modèle", list(candidates.keys()))
            if st.button("Sélectionner modèle"):
                # Build preprocessor and attach to session
                preproc, feats = build_preprocessor(pd.concat([session.X_train, session.X_test]))
                session.pipeline = preproc
                session.model_choice = chosen
                st.success(f"Modèle {chosen} sélectionné et préprocesseur construit")
                log_history(st.session_state.auth_user or "anonymous", "select_model", chosen)

    # ---------- Step 7: Training ----------
    if step.startswith("7"):
        st.subheader("7 — Entraînement")
        if not hasattr(session, "model_choice"):
            st.warning("Sélectionne un modèle d'abord (étape 6)")
        else:
            st.write(f"Entraîner : {session.model_choice}")
            if st.button("Lancer l'entraînement"):
                with st.spinner("Entraînement en cours..."):
                    model = train_model(session, session.X_train, session.y_train, session.model_choice)
                    st.success("Entraînement terminé")
                    log_history(st.session_state.auth_user or "anonymous", "train", session.model_choice)
                    st.write("Pipeline trained :")
                    st.write(session.model)

    # ---------- Step 8: Evaluation ----------
    if step.startswith("8"):
        st.subheader("8 — Validation & Évaluation")
        if not session.trained:
            st.warning("Entraîne un modèle d'abord (étape 7)")
        else:
            metrics = evaluate(session, session.X_test, session.y_test)
            st.write("Metrics :")
            st.json(metrics)
            log_history(st.session_state.auth_user or "anonymous", "evaluate", json.dumps(metrics))

    # ---------- Step 9: Hyperparam tuning ----------
    if step.startswith("9"):
        st.subheader("9 — Hyperparam Tuning")
        if not hasattr(session, "X_train"):
            st.warning("Crée un split d'abord (étape 5)")
        else:
            st.write("Grille d'exemple pour RandomForest :")
            default_grid = {
                "estimator__n_estimators": [50, 100],
                "estimator__max_depth": [None, 10]
            }
            st.write(default_grid)
            use_grid = st.checkbox("Utiliser la grille par défaut")
            if use_grid:
                param_grid = default_grid
            else:
                raw = st.text_area("Param grid JSON (ex: {\"estimator__n_estimators\": [10,50]})")
                try:
                    param_grid = json.loads(raw) if raw.strip() else {}
                except Exception:
                    param_grid = {}
            if st.button("Lancer tuning"):
                with st.spinner("Grid search..."):
                    grid = tune_hyperparameters(session, session.X_train, session.y_train, param_grid or default_grid)
                    st.success("Tuning terminé")
                    st.write("Best params:")
                    st.write(grid.best_params_)
                    log_history(st.session_state.auth_user or "anonymous", "tune", json.dumps(grid.best_params_))

    # ---------- Step 10: Export / Deployment ----------
    if step.startswith("10"):
        st.subheader("10 — Export / Déploiement")
        if not session.trained:
            st.warning("Entraîne un modèle d'abord (étape 7 ou 9)")
        else:
            name = st.text_input("Nom du modèle à enregistrer (optionnel)")
            if st.button("Sauvegarder le modèle"):
                path = save_model(session, name if name.strip() else None)
                st.success(f"Modèle sauvegardé : {path}")
                log_history(st.session_state.auth_user or "anonymous", "save_model", path)
                with open(path, "rb") as f:
                    bytes_data = f.read()
                    b64 = base64.b64encode(bytes_data).decode()
                    href = f"data:application/octet-stream;base64,{b64}"
                    st.markdown(f"[Télécharger le modèle]({href})", unsafe_allow_html=True)

# ---------- Tab 2: Chatbot IA ----------
with tabs[1]:
    st.header("Chatbot IA (intégration OpenAI)")
    st.write("Pose des questions générales ou liées à tes données/ML. Le chatbot utilise l'API OpenAI si configurée.")
    user_prompt = st.text_area("Ton message pour le chatbot")
    if st.button("Envoyer au chatbot"):
        if not user_prompt.strip():
            st.warning("Écris un message d'abord")
        else:
            res = openai_chat(user_prompt, system="Tu es un assistant qui aide à l'AutoML et à Python.")
            st.write(res)
            log_history(st.session_state.auth_user or "anonymous", "chatbot", user_prompt)

# ---------- Tab 3: Historique ----------
with tabs[2]:
    st.header("Historique des actions")
    hist = get_history(200)
    df_hist = pd.DataFrame(hist)
    st.dataframe(df_hist)
    if st.button("Exporter l'historique CSV"):
        csv = df_hist.to_csv(index=False).encode("utf-8")
        b64 = base64.b64encode(csv).decode()
        href = f"data:file/csv;base64,{b64}"
        st.markdown(f"[Télécharger historique]({href})", unsafe_allow_html=True)

# ---------- Tab 4: Modèles ----------
with tabs[3]:
    st.header("Modèles sauvegardés")
    files = sorted(os.listdir(MODEL_DIR))
    st.write(files)
    selected = st.selectbox("Choisis un modèle", [None] + files)
    if selected:
        path = os.path.join(MODEL_DIR, selected)
        st.write(f"Chemin: {path}")
        if st.button("Charger le modèle sélectionné"):
            try:
                loaded = joblib.load(path)
                st.session_state.loaded_model = loaded
                st.success("Modèle chargé dans la session (loaded_model)")
                log_history(st.session_state.auth_user or "anonymous", "load_model", selected)
            except Exception as e:
                st.error(f"Erreur lors du chargement: {e}")
        if st.session_state.get("loaded_model"):
            lm = st.session_state.loaded_model
            st.write(lm)
            # Provide a simple predict UI
            if st.button("Faire une prédiction d'exemple (aléatoire)"):
                if hasattr(session, "X_test"):
                    sample = session.X_test.sample(1)
                    st.write("Sample input:")
                    st.dataframe(sample)
                    preds = lm.predict(sample)
                    st.write("Prédiction :")
                    st.write(preds)
                else:
                    st.warning("Aucun jeu de test disponible dans la session pour exemple")

# Footer / quick tips
st.markdown("---")
st.write("**Tips:**")
st.markdown("* Pour une utilisation locale avec modèle lourd, considère d'héberger le modèle sur un serveur ou d'utiliser Hugging Face / Streamlit Cloud.*")
st.markdown("* Assure-toi de définir la variable d'environnement `OPENAI_API_KEY` pour utiliser le chatbot.*")

# ----------------------------
# End of file
# ----------------------------
