import ast
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.model_selection import train_test_split


# -----------------------------
# Helpers and cached functions
# -----------------------------

def _read_csv_with_fallback(path: Path):
    last_err = None
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc), enc
        except Exception as e:
            last_err = e
    try:
        return pd.read_csv(path, encoding="latin-1", engine="python", on_bad_lines="skip"), "latin-1 (python, skip bad lines)"
    except Exception as e:
        raise e if last_err is None else last_err

def _find_csv(filename: str) -> Path | None:
    """Search sensible locations so bundled CSVs load even if Streamlit is launched from elsewhere."""
    candidates = []
    try:
        script_dir = Path(__file__).parent.resolve()
        candidates += [script_dir / filename, script_dir / "data" / filename, script_dir / "nigel" / filename]
    except NameError:
        pass

    cwd = Path.cwd().resolve()
    candidates += [cwd / filename, cwd / "data" / filename, cwd / "nigel" / filename,
                   cwd.parent / filename, cwd.parent / "data" / filename, cwd.parent / "nigel" / filename]

    for p in candidates:
        if p.exists() and p.is_file():
            return p.resolve()
    return None

def _parse_tokens(value, mode: str) -> List[str]:
	"""Parse a cell into list of tokens based on user-selected mode.

	mode options:
	- 'python-list': cell contains a Python list literal (e.g., "['a', 'b']")
	- 'space-separated': tokens are space-separated string
	- 'comma-separated': tokens are comma-separated string
	- 'auto': attempt list -> literal_eval -> space split
	"""
	if isinstance(value, list):
		return [str(t) for t in value]

	s = "" if pd.isna(value) else str(value)

	try_modes = []
	if mode == "python-list":
		try_modes = ["python-list"]
	elif mode == "space-separated":
		try_modes = ["space-separated"]
	elif mode == "comma-separated":
		try_modes = ["comma-separated"]
	else:  # auto
		try_modes = ["python-list", "space-separated"]

	for m in try_modes:
		if m == "python-list":
			try:
				lit = ast.literal_eval(s)
				if isinstance(lit, list):
					return [str(t) for t in lit]
			except Exception:
				pass
		elif m == "space-separated":
			tokens = s.strip().split()
			if tokens:
				return tokens
		elif m == "comma-separated":
			tokens = [t.strip() for t in s.split(",") if t.strip()]
			if tokens:
				return tokens

	# Fallback: no tokens
	return []


@st.cache_data(show_spinner=False)
def split_data(df: pd.DataFrame, test_size: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
	train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed)
	return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def train_vectorizer(
	texts: List[str],
	max_features: int,
	ngram_range: Tuple[int, int],
	min_df: int,
	max_df: float,
) -> TfidfVectorizer:
	"""Fit a TF-IDF vectorizer on training texts and return the fitted vectorizer."""
	vec = TfidfVectorizer(
		max_features=int(max_features),
		ngram_range=tuple(ngram_range),
		min_df=int(min_df),
		max_df=float(max_df),
	)
	vec.fit(texts)
	return vec


def _normalize_query_text(text: str, lowercase: bool, splitter: str) -> str:
	"""Normalize user query according to UI settings and return a space-joined string."""
	if lowercase:
		text = text.lower()
	if splitter == "comma":
		tokens = [t.strip() for t in text.split(",") if t.strip()]
	else:
		tokens = text.split()
	return " ".join(tokens)


def find_closest_sentence(
	input_text: str,
	X_matrix,
	names: pd.Series,
	vectorizer: TfidfVectorizer,
	lowercase: bool = True,
	splitter: str = "space",
) -> Tuple[str, float]:
	"""Find closest row (by cosine similarity) to the input_text using TF-IDF space.

	- X_matrix: sparse TF-IDF matrix for the dataset (train or test)
	- names: corresponding names (e.g., BottleName) indexed like X_matrix rows
	- vectorizer: fitted TfidfVectorizer
	"""
	q_text = _normalize_query_text(input_text, lowercase, splitter)
	q_vec = vectorizer.transform([q_text])  # 1 x F
	if X_matrix.shape[0] == 0:
		return "", 0.0
	# cosine_similarity returns 1 x N array
	sims = cosine_similarity(q_vec, X_matrix).ravel()
	best_idx = int(np.argmax(sims))
	best_name = str(names.iloc[best_idx]) if len(names) > 0 else ""
	best_sim_pct = float(f"{sims[best_idx] * 100:.2f}") if sims.size > 0 else 0.0
	return best_name, best_sim_pct


# -----------------------------
# Streamlit UI
# -----------------------------

#EDIT 
st.set_page_config(page_title="Food → Wine Recommender", page_icon="🍷", layout="wide")
st.title("🍷 Food → Wine Recommender")
st.write(
	"Train a TF-IDF model on your dataset and get wine recommendations by matching food descriptions to the closest wine descriptions."
)

# Defaults (no sidebar controls) #EDIT
default_path = Path(__file__).parent / "WLSA_NN.csv"
use_bundled = default_path.exists()
default_token_col = "Notes"
default_name_col = "Wines"
token_parse_mode = "auto"  # attempts python-list then falls back to space split

# TF-IDF defaults
max_features = 5000
ngram_max = 1
min_df = 1
max_df = 1.0
test_size = 0.25
seed = 42

# Query defaults
lowercase = True
splitter = "space"

# Robust CSV reader to handle encodings
def _read_csv_with_fallback(path: Path) -> Tuple[pd.DataFrame, str]:
	last_err = None
	for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
		try:
			return pd.read_csv(path, encoding=enc), enc
		except Exception as e:
			last_err = e
	# Final fallback: python engine with bad line skipping
	try:
		return pd.read_csv(path, encoding="latin-1", engine="python", on_bad_lines="skip"), "latin-1 (python, skip bad lines)"
	except Exception as e:
		raise e if last_err is None else last_err

# Load data from bundled file only
# ===================== DUAL CSV LOAD (REPLACE your single-file section) =====================

# ✅ UPDATE HERE: filenames only (keep them in the repo next to app.py, or in /data or /nigel)
WINE_FILE = "WLSA_NN.csv"         # your main wine notes data
FOOD_FILE = "food_choices.csv"    # your second CSV (food choices / curated queries)

wine_csv_path = _find_csv(WINE_FILE)
food_csv_path = _find_csv(FOOD_FILE)

missing = []
if wine_csv_path is None:
    missing.append(WINE_FILE)
if food_csv_path is None:
    missing.append(FOOD_FILE)

if missing:
    st.error(
        "Couldn’t find these required files:\n"
        + "\n".join(f"- {m}" for m in missing)
        + "\n\nI looked in:\n"
        "- the script folder (and its `data/` and `nigel/` subfolders)\n"
        "- the current working directory (and its `data/` and `nigel/` subfolders)\n"
        "- one folder up"
    )
    st.stop()

try:
    df_food, used_enc_food = _read_csv_with_fallback(food_csv_path)
    st.caption(f"Loaded food dataset: {food_csv_path} (encoding: {used_enc_food})")
except Exception as e:
    st.error(f"Failed to read `{food_csv_path}`: {e}")
    st.stop()

st.caption(f"Working directory: {Path.cwd().resolve()}")
st.subheader("Preview of bundled data")
st.dataframe(df_raw.head(10), use_container_width=True)
st.dataframe(df_food.head(10), use_container_width=True)
# ============================================================================================

# These already exist in your code — keep them
default_token_col = "Notes"   # <- UPDATE if your wine text column is different
default_name_col  = "Wines"   # <- UPDATE if your wine name column is different

# ===== Food CSV column mapping (ADD after the dual-file load) =====
# ✅ UPDATE HERE if your second CSV uses different column names
FOOD_NAME_COL = "Food"          # e.g., "Food", "Label", "Dish"
FOOD_DESC_COL = "Description"   # e.g., "Description", "Query", "Text"

# Validate presence once, fail fast with a clear message
missing_food_cols = [c for c in (FOOD_NAME_COL, FOOD_DESC_COL) if c not in df_food.columns]
if missing_food_cols:
    st.error(
        "Your food CSV is missing required columns:\n"
        + "\n".join(f"- {c}" for c in missing_food_cols)
        + "\n\nUpdate FOOD_NAME_COL / FOOD_DESC_COL in the code to match your file."
    )
    st.stop()
# ================================================================st

# Choose columns using defaults with simple fallbacks
token_col = default_token_col if default_token_col in all_cols else None
name_col = default_name_col if default_name_col in all_cols else None

if token_col is None:
	for cand in ["Description_lemmatized", "Description", "Food Pairing", "Notes"]:
		if cand in all_cols:
			token_col = cand
			break
if token_col is None:
	# fallback to a likely text column
	for c in all_cols:
		if pd.api.types.is_string_dtype(df_raw[c]):
			token_col = c
			break
if token_col is None:
	# last resort: first column
	token_col = all_cols[0]

if name_col is None:
	for cand in ["BottleName", "Wines", "Name", "Title"]:
		if cand in all_cols:
			name_col = cand
			break
if name_col is None:
	# fallback to the first non-token column
	name_col = all_cols[0] if all_cols[0] != token_col else (all_cols[1] if len(all_cols) > 1 else all_cols[0])

st.write(f"Using text column: `{token_col}` and name column: `{name_col}`")

# Parse tokens into a new column and build text for TF-IDF
tokenized_col = f"{token_col}__tokens"
text_col = f"{token_col}__text"
df = df_raw.copy()
df[tokenized_col] = df[token_col].apply(lambda v: _parse_tokens(v, token_parse_mode))
# If the tokenization produced strings (e.g., space-separated), split once more into tokens for consistency
def _to_text(toks_or_str):
	if isinstance(toks_or_str, list):
		return " ".join(map(str, toks_or_str))
	s = "" if pd.isna(toks_or_str) else str(toks_or_str)
	return s

df[text_col] = df[tokenized_col].apply(_to_text)

empty_token_rows = int((df[tokenized_col].apply(len) == 0).sum())
if empty_token_rows == len(df):
	st.warning("No tokens were parsed from the selected column. Check the parse mode and column selection.")

st.write(f"Parsed tokens into column `{tokenized_col}` and prepared TF-IDF text in `{text_col}`. Empty token rows: {empty_token_rows}")

# Split
train_df, test_df = split_data(df[[name_col, tokenized_col, text_col]].copy(), test_size=float(test_size), seed=int(seed))

# Train TF-IDF on training set texts
train_texts: List[str] = train_df[text_col].tolist()
with st.spinner("Fitting TF-IDF vectorizer..."):
	vectorizer = train_vectorizer(
		texts=train_texts,
		max_features=int(max_features),
		ngram_range=(1, int(ngram_max)),
		min_df=int(min_df),
		max_df=float(max_df),
	)

# Compute matrices for train/test
with st.spinner("Vectorizing train/test sets..."):
	X_train = vectorizer.transform(train_df[text_col])
	X_test = vectorizer.transform(test_df[text_col])

st.success("Vectorizer fitted and text vectors computed.")

# Query input
st.subheader("Enter food descriptions")
st.caption("Enter one food description per line, or upload a JSON mapping of name → description below.")
query_lines = st.text_area(
	"Food descriptions (one per line)",
	value="Roast chicken with herbs\nSpicy seafood pasta\nDark chocolate dessert",
	height=120,
)

food_map: Dict[str, str] = {}

# derive from lines
for line in [ln.strip() for ln in query_lines.splitlines() if ln.strip()]:
    # Allow optional label|description format
    if "|" in line:
        label, desc = line.split("|", 1)
        food_map[label.strip()] = desc.strip()
    else:
        # Use truncated text as label
        label = (line[:40] + "…") if len(line) > 40 else line
        food_map[label] = line

num_choices = min(5, len(food_map))
st.write(f"Detected {len(food_map)} food queries. Showing top {num_choices} recommendations.")

if not food_map:
	st.info("Add at least one food description to get recommendations.")
	st.stop()

run_btn = st.button("Find recommendations")

if run_btn:
	closest_wines_test: List[Tuple[str, float]] = []
	closest_wines_train: List[Tuple[str, float]] = []

	for food_name, food_desc in food_map.items():
		wine_test, sim_test = find_closest_sentence(
			food_desc, X_test, test_df[name_col], vectorizer, lowercase=lowercase, splitter=splitter
		)
		closest_wines_test.append((f"{food_name} → {wine_test}", sim_test))

		wine_train, sim_train = find_closest_sentence(
			food_desc, X_train, train_df[name_col], vectorizer, lowercase=lowercase, splitter=splitter
		)
		closest_wines_train.append((f"{food_name} → {wine_train}", sim_train))

	# Sort descending by similarity
	closest_wines_test.sort(key=lambda x: x[1], reverse=True)
	closest_wines_train.sort(key=lambda x: x[1], reverse=True)


#EDIT to combine both train and test dataset pd.concat
	# Display
	st.subheader("Top 5 wine recommendations (Test set)")
	test_df_display = pd.DataFrame(
		[{"Rank": i + 1, "Pairing": name, "Cosine Similarity (%)": sim} for i, (name, sim) in enumerate(closest_wines_test[:5])]
	)
	st.dataframe(test_df_display, use_container_width=True, hide_index=True)

	st.subheader("Top 5 wine recommendations (Train set)")
	train_df_display = pd.DataFrame(
		[{"Rank": i + 1, "Pairing": name, "Cosine Similarity (%)": sim} for i, (name, sim) in enumerate(closest_wines_train[:5])]
	)
	st.dataframe(train_df_display, use_container_width=True, hide_index=True)


st.markdown("---")
st.caption(
	"Developed by WineList Asiatic (https://winelistasia.com) 🍷"
)

