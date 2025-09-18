# 🍷 Wine List Asia — Food → Wine Recommender

A Streamlit app that recommends wines for a selected food. It uses TF‑IDF (word **and** character n‑grams) and cosine similarity to match **food descriptions** to **wine tasting notes**.

---

## ✨ What’s in this repo
- **`winerecommender.py`** – the Streamlit app (UI + recommender).
- **`WLSA_NN.csv`** – wine catalogue with tasting notes (and optional product links).
- **`food_choices.csv`** – curated list of food names and descriptions shown in a **dropdown** (no manual typing).
- **`EDA & Modeling.ipynb`** – notebook for exploratory analysis and experiments that mirrors the app’s logic so results look the same.
- **`WLSA_Data_Dictionary.csv`** – optional field dictionary for the wine dataset.
- **`stopwords.txt`** – (optional) custom stopword list you can extend.

> If you keep `WLSA_NN.csv` and `food_choices.csv` in `./data/` next to `app2.py`, the app will automatically load them. If they are elsewhere, the app also searches a few sensible local paths (see `_find_csv()` inside the code).

---

## 🏁 Quick start

```bash
# 1) Create & activate a virtual environment (optional but recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt
# or minimal set
pip install streamlit pandas scikit-learn numpy

# 3) Project layout (recommended)
# your_project/
# ├── app2.py
# └── data/
#     ├── WLSA_NN.csv
#     ├── food_choices.csv
#     └── stopwords.txt   # optional

# 4) Run the app
streamlit run app2.py
```

Open the URL Streamlit prints (usually http://localhost:8501).

---

## 📦 Data Requirements

### 1) `data/WLSA_NN.csv` (wines)

**Required (or auto-detected) columns**

- **Text column** for TF‑IDF: one of `Notes` (default), `Description_lemmatized`, `Description`, `Food Pairing` (the app auto‑picks the first available).
- **Wine name**: one of `Wines` (default), `BottleName`, `Name`, `Title`.
- **Link (optional)**: any one of `Link`, `URL`, `Url`, `link`, `Website`, `website`, `Product URL`, `Product_URL`.
  - If present, the app shows **clickable links** next to each recommendation.

### 2) `data/food_choices.csv` (foods)

**Required columns** (any one pairing):

- `Food` + `Description`, **or**
- `name` + `description`, **or**
- `Label` + `Description`.

If you create a preprocessed column `Normalized_Description`, the app will use that instead of `Description` automatically.

---

## 🧠 How the recommender works

1. The app loads `WLSA_NN.csv` and picks the best **text column** (prefers `Notes`).
2. It fits **two TF‑IDF vectorizers** on `wines + food queries`:
   - **Word n‑grams**: `TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True, smooth_idf=True)`
   - **Character n‑grams**: `TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))`
3. At query time (when you pick a food from the dropdown):
   - The food description is normalized and vectorized with both models.
   - For every wine, the app computes a **weighted cosine ensemble**:  
     `score = 0.7 * cos(q_word, wine_word) + 0.3 * cos(q_char, wine_char)`
   - Results from **train** and **test** splits are merged and **de‑duplicated per wine**, keeping the best score.
   - If a link column exists, it is displayed as a clickable URL.

> The displayed **Cosine Similarity (%)** is scaled relative to the top match for the query, so the best result is 100% and the rest are shown proportionally. This keeps the UI stable across different queries and datasets.

---

## 🖥️ Using the app

1. Place your CSVs in `./data/`.
2. Start the app: `streamlit run app2.py`.
3. In the UI:
   - Choose a **food** from the dropdown (`food_choices.csv`).
   - Click **Find recommendations**.
   - You’ll see a **single merged table** with columns: Rank, Wine, Cosine Similarity (%), and (optionally) Link.

No uploads or manual typing required.

---

## 🧪 Notebook — EDA & Modeling

The notebook **`EDA & Modeling.ipynb`** mirrors the Streamlit logic so that results match the app. It includes:

- Loading `WLSA_NN.csv` and `food_choices.csv` from `./data/`.
- The same vectorizer settings (word + char models) and the same **weighted cosine ensemble**.
- Helper to compute **Top‑N** recommendations for any food.
- (Optional) data cleaning steps (e.g., normalizing descriptions, removing odd whitespace, deduplicating wines).

> Tip: to keep the notebook aligned with the app, import shared code from a small helper module (e.g. `recommender_utils.py`) and use the same constants (n‑gram ranges, weights, etc.).

---

## 🔧 Tweaks for better matches

- **Improve food descriptions** in `food_choices.csv`: use terms that appear in your wines’ notes (e.g., *citrus, peppery, creamy, smoky, mineral*). You can add a column `Normalized_Description` and the app will use it.
- **Adjust weights** of the ensemble: in `_topn_for_query`, change `w_word` / `w_char` (e.g., `0.8/0.2`).
- **Vectorizer breadth**: increase `max_features` or allow trigrams if your data is large.
- **Add domain synonyms**: if you adopt the optional `build_models` approach in the code, extend the `raw_synonyms` map to translate food terms into the wine vocabulary.
- **Custom stopwords**: see next section.

---

## 📝 Custom stopwords (`data/stopwords.txt`)

You can place a file at `data/stopwords.txt` (one token per line). These will be **added** on top of the built‑in English stopwords *only if you wire it in*. A safe starter list is provided in this repo. **Do not** add domain‑important words like: `spicy`, `peppery`, `smoky`, `citrus`, `mineral`, `creamy`, `tannin`, `oak`, `cherry`, `tropical`, etc.

**How to use (optional):** if you want to merge `stopwords.txt` with the English list, adapt your vectorizer setup like this:

```python
from pathlib import Path

def load_stopwords_file(path: Path) -> set:
    if not path.exists():
        return set()
    return {line.strip().lower() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}

# After fitting once to get english list, union with custom:
word_vec.fit(wine_texts_all)
english_sw = set(getattr(word_vec, "stop_words_", []))
custom_sw = load_stopwords_file(Path("data/stopwords.txt"))
full_sw = english_sw | custom_sw

word_vec = TfidfVectorizer(stop_words=full_sw, ngram_range=(1, 2), min_df=2, max_df=0.9,
                           sublinear_tf=True, smooth_idf=True)
word_vec.fit(vocab_fit)
```

This mirrors the pattern shown in `build_models(...)` inside the code.

---

## 🧩 Troubleshooting

- **“Local CSVs not found”**: make sure both `WLSA_NN.csv` and `food_choices.csv` are in `./data/` or a searched folder (`./nigel`, parent `data/`).
- **`df_raw` is not defined**: ensure the section that sets `df_raw = wines_df.copy()` and `df_food = food_df.copy()` runs **before** any usage.
- **`top_test` / `top_train` undefined**: only reference them **inside** the `if st.button("Find recommendations"):` block that creates them.
- **No links showing**: add a link column to `WLSA_NN.csv` with one of the supported names (`Link`, `URL`, …) or update `LINK_COL_CANDS` in the code.

---

## 📋 Minimal `requirements.txt`

```txt
streamlit>=1.34
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
```

Add `matplotlib` / `ipykernel` if you run the notebook locally.

---

## 🔐 Notes on paths

The app tries `./data/` first:
```python
DATA_DIR = Path(__file__).parent / "data"
WINES_PATH = DATA_DIR / "WLSA_NN.csv"
FOOD_PATH  = DATA_DIR / "food_choices.csv"
```
If missing, `_find_csv()` checks sibling folders like `./nigel` and parent directories. You can hard‑code your paths if you prefer.

---

## ✅ Checklist before running

- [ ] `app2.py` is in your project root.
- [ ] `data/WLSA_NN.csv` exists and has **Notes** (or another text column) + **Wines** (or name column).
- [ ] `data/food_choices.csv` exists and has **Food + Description** (or **name + description**).
- [ ] (Optional) Add a **Link** column to `WLSA_NN.csv` to show clickable URLs in results.
- [ ] (Optional) Adjust `data/stopwords.txt` (but avoid domain‑important words).

---

## 📄 License
Proprietary / internal use for Wine List Asia (adjust as appropriate).
