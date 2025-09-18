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

def _find_csv(filename: str) -> Path | None:
    """Search common local locations for a CSV."""
    candidates = []
    try:
        script_dir = Path(__file__).parent.resolve()
        candidates += [
            script_dir / filename,
            script_dir / "data" / filename,
            script_dir / "nigel" / filename,
        ]
    except NameError:
        pass

    cwd = Path.cwd().resolve()
    candidates += [
        cwd / filename,
        cwd / "data" / filename,
        cwd / "nigel" / filename,
        cwd.parent / filename,
        cwd.parent / "data" / filename,
        cwd.parent / "nigel" / filename,
    ]

    for p in candidates:
        if p.exists() and p.is_file():
            return p.resolve()
    return None

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
st.set_page_config(page_title="Wine List Asia Wine Recommender", page_icon="🍷", layout="wide")
st.title("🍷 Food → Wine Recommender")
st.write(
	"Select from our dropdown box a list of food you wish to know a wine suitable for it."
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
# Local files only
wines_df = None
food_df  = None
enc_wines = enc_food = None

if WINES_PATH.exists() and FOOD_PATH.exists():
    wines_df, enc_wines = load_csv_cached(str(WINES_PATH))
    food_df,  enc_food  = load_csv_cached(str(FOOD_PATH))
else:
    # (optional) broader local search using _find_csv
    w = _find_csv("WLSA_NN.csv")
    f = _find_csv("food_choices.csv")
    if w and f:
        wines_df, enc_wines = _read_csv_with_fallback(w)
        food_df,  enc_food  = _read_csv_with_fallback(f)
    else:
        st.error("Local CSVs not found. Place WLSA_NN.csv and food_choices.csv next to app.py or under ./data or ./nigel.")
        st.stop()

# Show quick status
st.caption(
    f"Loaded wines.csv ({enc_wines}) — {len(wines_df):,} rows • "
    f"food_choices.csv ({enc_food}) — {len(food_df):,} rows"
)

# Use the loaded frames everywhere below
df_raw = wines_df.copy()
df_food = food_df.copy()
all_cols = list(df_raw.columns)

# Optional preview
st.subheader("Data Showcase")
st.dataframe(df_raw.head(10), use_container_width=True)


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
food_name_cand = None
food_desc_cand = None
for cand_name, cand_desc in (("Food", "Description"), ("name", "description"), ("Label", "Description")):
    if cand_name in df_food.columns and cand_desc in df_food.columns:
        food_name_cand, food_desc_cand = cand_name, cand_desc
        break
if food_desc_cand is None:
    # fallback: first string column
    str_cols = [c for c in df_food.columns if pd.api.types.is_string_dtype(df_food[c])]
    food_desc_cand = str_cols[1] if len(str_cols) > 1 else str_cols[0]

# Prefer a precomputed normalized column if you added one; else use desc
query_series = (
    df_food["Normalized_Description"]
    if "Normalized_Description" in df_food.columns
    else df_food[food_desc_cand]
).fillna("").astype(str)

with st.spinner("Fitting vectorizers (word + char) on wines + food queries..."):
    vocab_fit = df[text_col].astype(str).tolist() + query_series.tolist()

    # Word n-grams with better settings for short texts
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),      # add bigrams
        min_df=2,                # drop ultra-rare junk
        max_df=0.9,              # drop near-stopwords
        sublinear_tf=True,       # dampen long docs
        smooth_idf=True,         # stabler idf
        max_features=int(max_features) if max_features else None
    )
    vectorizer.fit(vocab_fit)

    # Character n-grams catch morphology/typos (“char_wb” respects word boundaries)
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1
    )
    char_vec.fit(vocab_fit)

# Compute matrices for train/test
with st.spinner("Vectorizing train/test sets (word + char)..."):
    X_train_w = vectorizer.transform(train_df[text_col])
    X_test_w  = vectorizer.transform(test_df[text_col])
    X_train_c = char_vec.transform(train_df[text_col])
    X_test_c  = char_vec.transform(test_df[text_col])


st.success("Vectorizer fitted and text vectors computed.")

# --- Pick a food from your CSV and get recommendations ---
# Try common column names; fail fast if not present
FOOD_NAME_COL = None
FOOD_DESC_COL = None
for cand_name, cand_desc in (("Food", "Description"), ("name", "description"), ("Label", "Description")):
    if cand_name in df_food.columns and cand_desc in df_food.columns:
        FOOD_NAME_COL, FOOD_DESC_COL = cand_name, cand_desc
        break
if FOOD_NAME_COL is None:
    st.error("Your food_choices.csv must contain columns 'Food' + 'Description' (or 'name' + 'description').")
    st.stop()

st.subheader("Select a food choice")
food_options = (
    df_food[FOOD_NAME_COL]
    .dropna()
    .astype(str)
    .drop_duplicates()
    .sort_values()
    .tolist()
)
selected_food = st.selectbox("Food", options=food_options, index=0)

# Get the description that corresponds to the selected food
food_desc = (
    df_food.loc[df_food[FOOD_NAME_COL].astype(str) == str(selected_food), FOOD_DESC_COL]
    .astype(str)
    .iloc[0]
)
st.caption(f"Query used → {food_desc}")

# Helper to get true Top-N for a single query
def _topn_for_query(desc: str,
                    Xw, Xc,
                    names: pd.Series,
                    n: int = 5,
                    w_word: float = 0.7,
                    w_char: float = 0.3):
    # use your existing normalizer
    q_text = _normalize_query_text(desc, lowercase, splitter)
    qw = vectorizer.transform([q_text])  # word n-grams
    qc = char_vec.transform([q_text])    # char n-grams

    sims = w_word * cosine_similarity(qw, Xw).ravel() + \
           w_char * cosine_similarity(qc, Xc).ravel()

    if sims.size == 0:
        return []

    top_idx = np.argsort(-sims)[:n]
    max_s = sims[top_idx[0]] if top_idx.size else 0.0

    rows = []
    for rank, i in enumerate(top_idx, start=1):
        pct = (sims[i] / max_s * 100.0) if max_s > 0 else 0.0
        rows.append({"Rank": rank,
                     "Wine": str(names.iloc[i]),
                     "Cosine Similarity (%)": round(pct, 2)})
    return rows

# Make sure these exist earlier in your code:
# X_train_w, X_test_w, X_train_c, X_test_c  (the word & char matrices)

if st.button("Find recommendations"):
    # compute top hits from each split
    top_test = _topn_for_query(food_desc, X_test_w, X_test_c, test_df[name_col], n=5)
    top_train = _topn_for_query(food_desc, X_train_w, X_train_c, train_df[name_col], n=5)

    # DataFrames (NO "Set" tagging anymore)
    df_test = pd.DataFrame(top_test)
    df_train = pd.DataFrame(top_train)

    # merge test+train and keep the best score per wine
    merged_all = pd.concat([df_test, df_train], ignore_index=True)
    if merged_all.empty:
        st.info("No recommendations found.")
    else:
        collapsed = (
            merged_all
            .sort_values("Cosine Similarity (%)", ascending=False)
            .drop_duplicates(subset=["Wine"], keep="first")
            .reset_index(drop=True)
        )
        collapsed["Rank"] = np.arange(1, len(collapsed) + 1)

        # ---- attach links from WLSA_NN.csv (df_raw) using your wine name column ----
        LINK_COL_CANDS = ["Link", "URL", "Url", "link", "Website", "website", "Product URL", "Product_URL"]
        link_col = next((c for c in LINK_COL_CANDS if c in df_raw.columns), None)

        if link_col is not None:
            # map wine -> link
            link_map = df_raw.set_index(name_col)[link_col]
            collapsed["Link"] = collapsed["Wine"].map(link_map).fillna("")

            st.subheader("Wine Recommendations")
            # Prefer clickable links via LinkColumn if your Streamlit is recent enough
            try:
                st.data_editor(
                    collapsed[["Rank", "Wine", "Cosine Similarity (%)", "Link"]],
                    column_config={
                        "Link": st.column_config.LinkColumn("Link", help="Open wine page")
                    },
                    hide_index=True,
                    use_container_width=True,
                )
            except Exception:
                # fallback: show raw URL text
                st.dataframe(
                    collapsed[["Rank", "Wine", "Cosine Similarity (%)", "Link"]],
                    hide_index=True,
                    use_container_width=True,
                )
        else:
            st.subheader("Wine Recommendations")
            st.dataframe(
                collapsed[["Rank", "Wine", "Cosine Similarity (%)"]],
                hide_index=True,
                use_container_width=True,
            )
            st.warning(
                "No link column found in WLSA_NN.csv. Add one named "
                "'Link' or 'URL' (or update LINK_COL_CANDS) to show clickable links."
            )
