RhymeLM
====================================
RhymeLM is a lightweight character-level language model designed to generate 16-bar rap verses
using a custom lyric dataset. The aim of this project is to keep the architecture simple, the training loop
transparent, and experimentation straightforward—all while producing coherent bars from a compact
model as a proof of concept of understanding nueral networks.
# Overview
RhymeLM is built inside a single Jupyter notebook.  
It:

- Loads a CSV of raw lyrics  
- Splits each track into 16‑line segments  
- Builds a character‑level corpus  
- Trains a small feed‑forward model  
- Generates new 16‑bar verses  

---

## Dataset
The notebook expects a file named:

```
lyrics_raw.csv
```

with a column:

```
raw_lyrics
```

An example dataset is referenced from:  
**Rap Lyrics for NLP** — https://www.kaggle.com/datasets/ceebloop/rap-lyrics-for-nlp

---

## Dependencies
Install via:

```
pip install -r requirements.txt
```

### Requirements
- torch  
- pandas  
- numpy  
- nltk  
- jupyter  
- tqdm  
- reportlab (optional, for PDF export)

If using NLTK tokenizers:

```python
import nltk
nltk.download('punkt')
```

---

## How to Use
1. Install dependencies  
2. Place `lyrics_raw.csv` in the project directory  
3. Open the notebook:

```
jupyter notebook RhymeLM_v2.ipynb
```

4. Run cells from top to bottom to train  
5. Generate a verse with:

```python
print(generate_16(model))
```

6. Save your model:

```python
save_checkpoint("rhyme_lm.pt")
```

---

## Project Files
```
RhymeLM_v2.ipynb      
README.md             
HOW_TO_USE.md         
lyrics_raw.csv        
```

---

## License
MIT License.  
Dataset license follows the terms provided on Kaggle.
