# ecommerce-sentiment-analysis-distilbert
End-to-end NLP project that fine-tunes DistilBERT to classify real user reviews (scraped from Google Play Store) for top Indian e-commerce apps into positive and negative sentiments. Includes custom preprocessing, emoji handling, tokenization, model training, evaluation (validation accuracy: ~90%), and Streamlit-based interactive UI for deployment.

📌 Project Overview

This project focuses on binary sentiment analysis (Positive / Negative) of real-time app reviews from the Google Play Store.
Reviews were scraped from 15 popular apps including:

- Flipkart
- Amazon
- Myntra
…and more.

Instead of using pre-made datasets, I built a custom, balanced dataset to better capture authentic user feedback.

# Key Highlights

- Custom dataset creation using google-play-scraper

- Careful data cleaning:
  - Removed noisy 3-star reviews (often neutral or unrelated).
  - Filtered out short or single-word reviews (< 200 characters) to prevent overfitting.

- Balanced dataset:
  - 6,691 reviews per class → Total 26,764 reviews.

- DistilBERT chosen over BERT for faster training with minimal accuracy trade-off.
- Emoji-aware preprocessing to handle tokenization issues like difference in sentiment for "awesome😭" and "awesome 😭".

# Dataset Preparation

| Step          | Description                                                         |
| ------------- | ------------------------------------------------------------------- |
| **Scraping**  | Extracted fresh reviews directly from Google Play Store.            |
| **Filtering** | Removed 3★ neutral reviews and reviews shorter than 200 characters. |
| **Labeling**  | 1★, 2★ → **Negative**; 4★, 5★ → **Positive**.                       |
| **Balancing** | 6,691 reviews per class → **26,764 total reviews**.                 |

# Tech Stack

- Python
- Hugging Face Transformers – DistilBERT (distilbert-base-uncased)
- PyTorch – Model training
- Google Play Scraper – Dataset creation
- Pandas, NumPy – Data handling
- Matplotlib, Seaborn – Visualization
- Regex – Emoji and text preprocessing

# Model Training
| Parameter                  | Value                                                 |
| -------------------------- | ----------------------------------------------------- |
| **Max Sequence Length**    | 200 tokens (with padding up to 300 for preprocessing) |
| **Batch Size**             | 16                                                    |
| **Optimizer**              | AdamW                                                 |
| **Dropout**                | 0.1 (default)                                         |
| **Learning Rate & Warmup** | Tuned iteratively                                     |

# Performance
| Metric        | Negative | Positive | Macro Avg | Weighted Avg |
| ------------- | -------- | -------- | --------- | ------------ |
| **Precision** | 0.88     | 0.91     | 0.90      | 0.90         |
| **Recall**    | 0.91     | 0.88     | 0.89      | 0.89         |
| **F1-score**  | 0.90     | 0.89     | 0.90      | 0.89         |
| **Accuracy**  | -        | -        | **89%**   | **89%**      |

- Training Accuracy: **94.69%**
- Validation Accuracy: **~90%**

# Key Learnings

- The importance of data quality over quantity.
- How noisy labels (like neutral reviews) can drop accuracy by ~20%.
- Debugging NLP edge cases with emojis in real-world datasets.
- Iterative hyperparameter tuning for better performance.

# Future Improvements
- Multi-language support – Extend beyond English reviews.
- Continuous improvement with new real-time datasets.
