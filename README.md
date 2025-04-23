# 🎬 MovieLens Recommender System with LightFM

This repository contains a complete hybrid recommendation system built using the [LightFM](https://making.lyst.com/lightfm/docs/home.html) library on the MovieLens 100k dataset.  
It was implemented as a **one-day challenge**, showcasing a full end-to-end pipeline for top-N recommendations.

---

## 🚀 Project Overview

The goal was to build a recommender system that predicts which movies a user is likely to enjoy, based on historical ratings.

**Key aspects:**
- Hybrid approach (matrix factorization with implicit feedback)
- Model: `LightFM` with WARP loss
- Dataset: [MovieLens 100k](https://grouplens.org/datasets/movielens/100k/)
- Output: Python function returning top-N recommendations for a user

---

## 🧠 Technologies Used

- **Python**
- **LightFM** (for hybrid recommendation modeling)
- **Pandas / NumPy** (for data manipulation)
- **scikit-learn** (for evaluation + normalization)

---

## 📊 Evaluation

The model was evaluated using:

| Metric        | Description |
|---------------|-------------|
| `Precision@5` | Measures how relevant the top-5 recommended movies are |
| `AUC Score`   | Measures the model’s ability to score liked items higher than unliked ones |

---

## 🛠️ Pipeline

1. **Preprocessing**
   - Filtered positive interactions (`rating >= 4`)
   - Converted user & item IDs to strings
   - Built sparse interaction matrix using LightFM's internal Dataset

2. **Model training**
   - Used `LightFM` with `loss='warp'`
   - Trained with 20 epochs and evaluated at each epoch

3. **Evaluation**
   - Used `precision_at_k` and `auc_score` to assess performance
   - Normalized prediction scores for interpretability

4. **Recommendation function**
   - A `recommend()` function was implemented to retrieve top-N movies for a given user based on model scores

---

## 📝 Notes

- All results were obtained in a **single working day**, from data prep to evaluation and recommendation interface.
- The project is **local-only** (no web app or API) and runs fully in a Jupyter Notebook or script.

---
## 👤 Author

**Franck Sirguey**  
📧 franck43220@gmail.com  
🌍 Open to relocation in Europe and Canada  
📌 CV available upon request
