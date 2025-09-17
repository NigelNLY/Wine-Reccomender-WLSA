import ast
from pathlib import Path
from typing import Dict, List, Tuple
import re
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.model_selection import train_test_split
# -----------------------------
# Helpers and cached functions
# -----------------------------

# 1) Load custom stopwords (one token per line) and merge with English
def load_stopwords_file(path: Path) -> set:
    if not path.exists():
        return set()
    words = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        w = line.strip().lower()
        if w:
            words.add(w)
    return words

@st.cache_resource
def build_models(train_df, test_df, food_choices, stopwords_path: Path):
    # wine texts are already lemmatized lists in your dataframes
    train_texts = train_df["Description_lemmatized"].apply(lambda toks: " ".join(toks))
    test_texts  = test_df["Description_lemmatized"].apply(lambda toks: " ".join(toks))
    wine_texts_all = list(train_texts) + list(test_texts)

    # custom stopwords
    custom_sw = load_stopwords_file(stopwords_path)
    # IMPORTANT: do not include domain words in stopwords (fish, seafood, spicy, grilled, smoky, citrus, nutty, creamy, etc.)
    stop_words = "english"  # we’ll pass english + custom via a set below
    # We'll construct a real set after we fit the vectorizer to get its english list
    # (sklearn uses a built-in list when stop_words="english").

    # word-level TF-IDF + char-level TF-IDF
    word_vec = TfidfVectorizer(stop_words="english", min_df=1, max_df=0.95, ngram_range=(1,2), norm="l2")
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=1, norm="l2")

    # Fit on wines first to get vocab + english stop list, then refit with union stopwords and foods included
    word_vec.fit(wine_texts_all)
    english_sw = set(getattr(word_vec, "stop_words_", []))
    full_sw = english_sw | custom_sw

    word_vec = TfidfVectorizer(stop_words=full_sw, min_df=1, max_df=0.95, ngram_range=(1,2), norm="l2")
    word_vec.fit(wine_texts_all + [desc for desc in food_choices.values()])  # align vocab with foods
    char_vec.fit(wine_texts_all)

    # Transform wines
    X_train_word = word_vec.transform(train_texts)
    X_test_word  = word_vec.transform(test_texts)
    X_train_char = char_vec.transform(train_texts)
    X_test_char  = char_vec.transform(test_texts)

    # Store on copies (keep as sparse matrices; no .toarray())
    train_copy = train_df.copy()
    test_copy  = test_df.copy()
    train_copy["vec_word"] = list(X_train_word)
    train_copy["vec_char"] = list(X_train_char)
    test_copy["vec_word"]  = list(X_test_word)
    test_copy["vec_char"]  = list(X_test_char)

    # small synonym map -> only keep synonyms that exist in the wine vocab
    wine_vocab = set(word_vec.get_feature_names_out())
    raw_synonyms = {
        # taste / mouthfeel
        "spicy": ["spice", "pepper", "peppery", "ginger", "clove", "cinnamon", "cardamom"],
        "creamy": ["creamy", "buttery", "silky", "round"],
        "oily":   ["oily", "rich"],
        "sweet":  ["sweet", "honey", "ripe"],
        "sour":   ["acid", "acidity", "tart", "zesty"],
        "bitter": ["bitter", "almond"],
        "umami":  ["savory"],

        # aroma / cooking
        "smoky":  ["smoke", "smoky", "char", "toasty"],
        "grilled":["smoke", "smoky", "char", "toasty"],
        "fried":  ["rich"],
        "curry":  ["spice", "turmeric", "coriander", "cumin"],
        "chili":  ["pepper", "peppery", "spice"],
        "peanut": ["nutty", "hazelnut", "almond"],
        "herbs":  ["herbal", "sage", "rosemary", "thyme", "basil", "mint"],
        "ginger": ["ginger", "spice"],
        "coconut":["tropical", "creamy"],
        "lemon":  ["citrus", "lemon", "zest"],
        "lime":   ["citrus", "lime", "zest"],
        "orange": ["citrus", "orange", "zest"],
        "floral": ["floral", "rose", "jasmine", "acacia"],
        "vanilla":["vanilla"],

        # sea / stock
        "seafood":["saline", "briny", "mineral"],
        "shellfish":["saline", "briny", "oyster"],
        "prawn":  ["saline", "briny"],
        "squid":  ["saline", "briny"],
        "broth":  ["savory", "herbal"],

        # starchy / bready
        "rice":   ["biscuit", "bread", "cereal"],
        "toast":  ["toasty"],
    }
    synonyms = {k: [w for w in v if w in wine_vocab] for k, v in raw_synonyms.items()}

    return word_vec, char_vec, train_copy, test_copy, synonyms, wine_vocab

def _normalize_food(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # tiny stemming-ish tweaks to line up with wine lemmas
    toks = s.split()
    out = []
    for t in toks:
        if t.endswith("ies"): out.append(t[:-3] + "y")
        elif t.endswith("ing"): out.append(t[:-3])
        elif t.endswith("ed"): out.append(t[:-2])
        elif t.endswith("s") and len(t) > 3: out.append(t[:-1])
        else: out.append(t)
    return " ".join(out)

def _translate_food_to_wine_query(s: str, synonyms: dict, wine_vocab: set) -> str:
    s = _normalize_food(s)
    toks = s.split()
    out = []
    i = 0
    while i < len(toks):
        if i+1 < len(toks):  # bigram first
            bigram = f"{toks[i]} {toks[i+1]}"
            if bigram in synonyms:
                out.extend(synonyms[bigram]); i += 2; continue
        t = toks[i]
        if t in wine_vocab:
            out.append(t)
        if t in synonyms:
            out.extend(synonyms[t])
        i += 1
    if not out:
        out = [t for t in toks if t in wine_vocab]
    return " ".join(out)

def _food_vecs(food_desc: str, word_vec, char_vec, synonyms, wine_vocab):
    q = _translate_food_to_wine_query(food_desc, synonyms, wine_vocab)
    vw = word_vec.transform([q])
    vc = char_vec.transform([_normalize_food(food_desc)])
    return vw, vc

def _cosine_ensemble(vw_query, vc_query, vw_doc, vc_doc, w_word=0.65, w_char=0.35):
    sw = cosine_similarity(vw_query, vw_doc)[0][0]
    sc = cosine_similarity(vc_query, vc_doc)[0][0]
    return w_word*sw + w_char*sc

def rank_wines_for_food(food_desc: str, df, word_vec, char_vec, synonyms, wine_vocab, k=5, pct_display=True):
    qw, qc = _food_vecs(food_desc, word_vec, char_vec, synonyms, wine_vocab)
    scores = []
    for _, row in df.iterrows():
        s = _cosine_ensemble(qw, qc, row["vec_word"], row["vec_char"])
        scores.append((row["BottleName"], s))
    scores.sort(key=lambda t: t[1], reverse=True)
    top = scores[:k]
    if pct_display:
        # scale to 0..100 within this query for user-friendly %s
        vals = np.array([s for _, s in top])
        if vals.size and vals.max() > 0:
            disp = 100.0 * (vals / vals.max())
        else:
            disp = np.zeros_like(vals)
        return [(name, float(round(p, 2)), float(score)) for (name, score), p in zip(top, disp)]
    else:
        return [(name, float(score)) for name, score in top]

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
        return pd.read_csv(path, encoding="latin-1", engine="python", on_bad_lines="skip"), \
               "latin-1 (python, skip bad lines)"
    except Exception as e:
        raise e if last_err is None else last_err

# ---------- small cached wrapper so Streamlit doesn't re-read every rerun ----------
@st.cache_data(show_spinner=False)
def load_csv_cached(path_str: str) -> Tuple[pd.DataFrame, str]:
    df, enc = _read_csv_with_fallback(Path(path_str))
    # optional: normalize column names
    df.columns = [c.strip() for c in df.columns]
    return df, enc

# ---------- locate your files (bundle them under ./data next to app.py) ----------
DATA_DIR = Path(__file__).parent / "data"
WINES_PATH = DATA_DIR / "WLSA_NN.csv"
FOOD_PATH  = DATA_DIR / "food_choices.csv"

st.title("Wine Recommender")

# Try local files first; if missing, fall back to uploaders
wines_df = None
food_df  = None
enc_wines = enc_food = None

if WINES_PATH.exists() and FOOD_PATH.exists():
    wines_df, enc_wines = load_csv_cached(str(WINES_PATH))
    food_df,  enc_food  = load_csv_cached(str(FOOD_PATH))
else:
    st.warning("Local CSVs not found. Upload them below:")
    u_wines = st.file_uploader("Upload wines.csv", type="csv", key="wines_up")
    u_food  = st.file_uploader("Upload food_choices.csv", type="csv", key="food_up")
    if u_wines is not None and u_food is not None:
        wines_df = pd.read_csv(u_wines)
        food_df  = pd.read_csv(u_food)
        enc_wines = enc_food = "uploaded"

# Stop early if we still don't have both
if wines_df is None or food_df is None:
    st.stop()

# Show quick status
st.caption(
    f"Loaded wines.csv ({enc_wines}) — {len(wines_df):,} rows • "
    f"food_choices.csv ({enc_food}) — {len(food_df):,} rows"
)

# Example: dropdown for food names from food_choices.csv
# Expecting columns like: name, description
food_name = st.selectbox("Pick a dish", options=food_df["name"].tolist())
food_desc = food_df.loc[food_df["name"] == food_name, "description"].iloc[0]

st.write("Food description:", food_desc)



# ===================== DUAL CSV LOAD (LOCAL ONLY) =====================
# Filenames (ensure they exist locally; e.g., next to app.py or under ./data or ./nigel)
WINE_FILE = "WLSA_NN.csv"         # main wine notes
FOOD_FILE = "food_choices.csv"    # curated queries / food

def _read_local_only(filename: str):
    """Read from local disk using _find_csv and _read_csv_with_fallback."""
    local = _find_csv(filename)
    if local is None:
        raise FileNotFoundError(
            f"Could not find '{filename}'. I looked in the script folder, its 'data/' and 'nigel/' subfolders, "
            f"the working directory (and its 'data/'/'nigel/' subfolders), and one folder up. "
            f"Place the file locally and rerun."
        )
    df, enc = _read_csv_with_fallback(local)
    return df, f"local ({enc})", str(local)

# Read both datasets locally (no uploads)
wines_df, enc_wines, wines_src = _read_local_only(WINE_FILE)
food_df,  enc_food,  food_src  = _read_local_only(FOOD_FILE)

st.caption(f"Wines loaded from {enc_wines}: {wines_src} • Food loaded from {enc_food}: {food_src}")
st.caption(f"Working directory: {Path.cwd().resolve()}" )

st.subheader("Preview of bundled data")
st.dataframe(wines_df.head(10), use_container_width=True)
st.dataframe(food_df.head(10), use_container_width=True)
# =====================================================================

# Use the wines_df we already loaded
df_raw = wines_df.copy()
df_food = food_df.copy()
all_cols = list(df_raw.columns)

all_cols = list(df_raw.columns)

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



def _read_csv_url(url: str) -> tuple[pd.DataFrame, str]:
    """Read CSV from a raw GitHub URL (or any HTTP(S) URL)."""
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return pd.read_csv(io.BytesIO(r.content)), "remote (github/raw)"
