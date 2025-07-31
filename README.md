## 📜 Project Overview

This notebook covers a standard NLP workflow:

1.  **Data Loading & EDA**: We'll start by loading the Amazon Reviews dataset and performing exploratory data analysis (EDA) to see how the review scores are distributed.
2.  **Basic NLP with NLTK**: We'll run through some NLP basics like tokenization, part-of-speech tagging, and named entity recognition.
3.  **VADER Sentiment Scoring**: We'll use the Valence Aware Dictionary and sEntiment Reasoner (VADER) to quickly calculate polarity scores for each review.
4.  **RoBERTa Sentiment Scoring**: For a deeper analysis, we'll use a powerful, pre-trained RoBERTa model to predict sentiment.
5.  **Comparative Analysis**: Finally, we'll compare the results from VADER and RoBERTa against the actual star ratings to evaluate their performance.

-----

## 📊 Dataset

This project uses the **Amazon Fine Food Reviews** dataset from Kaggle. For efficiency, our analysis is based on the first **500 reviews** from the dataset. The key data points are the review text and the corresponding star rating (Score).

-----

## 🛠️ Technologies & Libraries Used

  * **Python 3** & **Jupyter Notebook**
  * **Data Handling**: `pandas`, `numpy`
  * **Visualization**: `matplotlib`, `seaborn`
  * **Progress Bar**: `tqdm`
  * **NLP**:
      * `nltk`: For VADER sentiment analysis and foundational NLP tasks.
      * `transformers`: For using the pre-trained RoBERTa model via Hugging Face.
      * `torch` / `tensorflow`: As backends for the Transformers library.
      * `scipy`: For the softmax function to normalize model outputs.

-----

## 🚀 Methodology

### 1\. Exploratory Data Analysis (EDA)

First, we visualized the distribution of review scores (1 to 5 stars) to check the dataset's balance. As expected, 5-star reviews are the most common.

```python
ax = df['Score'].value_counts().sort_index()\
    .plot(kind='bar',
          title='Count of Reviews by Stars',
          figsize=(10, 5),
          color='c')
ax.set_xlabel('Review Stars')
plt.show()
```

### 2\. NLTK's VADER Sentiment Analysis

We used VADER, a lexicon and rule-based sentiment analyzer, to get quick sentiment scores. For each review, it returns four scores: **negative**, **neutral**, **positive**, and a **compound** score that summarizes the overall sentiment. A positive compound score indicates positive sentiment, while a negative one indicates negative sentiment.

```python
from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()
res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['Text']
    myid = row['Id']
    res[myid] = sia.polarity_scores(text)
```

Plotting the compound score against the star ratings revealed a strong positive correlation, showing that VADER provides a solid baseline.

### 3\. Transformer-based Sentiment Analysis (RoBERTa)

For a more advanced analysis, we used the `cardiffnlp/twitter-roberta-base-sentiment` model from the Hugging Face hub. This model is fine-tuned on a massive dataset of tweets and has a more nuanced understanding of language.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

def polarity_scores_roberta(text):
    # This function processes the text through the RoBERTa model
    # and returns a dictionary of negative, neutral, and positive scores.
    encoded_text = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg' : scores[0],
        'roberta_neu' : scores[1],
        'roberta_pos' : scores[2]
    }
    return scores_dict
```

This model outputs separate scores for negative, neutral, and positive sentiment, which we then compared to VADER's results and the original ratings.

-----

## 📈 Results & Comparison

A comparative analysis showed the strengths and weaknesses of each model.

  * **VADER** 💨: It's fast and effective, showing a clear correlation between its compound score and the product's star rating. However, it can be tripped up by sarcasm or complex sentence structures.
  * **RoBERTa** 🤖: It demonstrates a much deeper understanding of sentiment. It correctly identified strong negative sentiment in reviews that VADER scored as neutral or only slightly negative. It also captured nuances, like detecting negative undertones in a mostly positive review (e.g., "This was delicious, but too bad I gained 2 pounds\!").
  * **Overall**: For clearly positive (5-star) or negative (1-star) reviews, both models generally agree. The biggest differences appear in the 2, 3, and 4-star reviews, where sentiment can be mixed and nuanced.

-----

## ⚙️ How to Run this Project

To run this notebook on your local machine, follow these steps:

1.  **Clone the Repository**:

    ```bash
    git clone https://github.com/your-username/sentiment-analysis-amazon-reviews.git
    cd sentiment-analysis-amazon-reviews
    ```

2.  **Set up a Virtual Environment** (Recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies**:
    Create a `requirements.txt` file with the content below and run `pip install -r requirements.txt`.

    ```
    pandas
    numpy
    matplotlib
    seaborn
    nltk
    tqdm
    transformers
    scipy
    torch
    tensorflow
    ```

4.  **Download NLTK Resources**:
    Run the following in a Python interpreter to download the necessary NLTK packages:

    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('vader_lexicon')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')
    ```

5.  **Download the Dataset**:

      * Download the `Reviews.csv` file from the [Amazon Fine Food Reviews dataset on Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews).
      * Place the `Reviews.csv` file in the project's root directory.

6.  **Launch Jupyter Notebook**:

    ```bash
    jupyter notebook
    ```

    Then, open and run the project's `.ipynb` file.

-----

## 🔚 Conclusion

This project highlights the trade-offs between different NLP sentiment analysis techniques. While **VADER is a fast and simple baseline**, Transformer models like **RoBERTa provide a more accurate and nuanced understanding of human language**. The results clearly show the power of large, pre-trained language models for complex NLP tasks.
