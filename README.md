# Film Review Classification - Positive and Negative
<p align="left"> 
</a>   <a href="https://pandas.pydata.org/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/2ae2a900d2f041da66e950e4d48052658d850630/icons/pandas/pandas-original.svg" alt="pandas" width="40" height="40"/> </a> 
</a> <a href="https://www.python.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/> </a> <a href="https://pytorch.org/" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/pytorch/pytorch-icon.svg" alt="pytorch" width="40" height="40"/> </a> <a href="https://scikit-learn.org/" target="_blank" rel="noreferrer"> <img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" alt="scikit_learn" width="40" height="40"/> </a> 

## Project Objective:
To build a machine learning model that classifies movie reviews as either positive or negative based on the sentiment expressed within the text.

## Data and Preprocessing:
```
1 A wonderful little production. <br /><br />The...  positive
2 I thought this was a wonderful way to spend ti...  positive
3 Basically there's a family where a little boy ...  negative
4 Petter Mattei's "Love in the Time of Money" is...  positive
```

- ## Dataset: 
```python
films_data = pd.read_csv("IMDB Dataset.csv")
```
Collected from IMDb movie reviews (or other review datasets).
- ## Text Processing: 
The reviews were cleaned by removing punctuation, converting text to lowercase, removing stopwords, and applying lemmatization.
```python
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("","", string.punctuation))
    text = re.sub(r"\d+","",text)
    stop_words = set(stopwords.words("english"))
    text = " ".join(word for word in text.split() if word not in stop_words)
    lemmatizer = WordNetLemmatizer()
    text = " ".join(lemmatizer.lemmatize(word) for word in text.split())
    return text 
```
- ## Cleaned data:
```
1 wonderful little production br br filming tech...
2 thought wonderful way spend time hot summer we...
3 basically there family little boy jake think t...
4 petter matteis love time money visually stunni...
```
## Vectorization: 
Text data was vectorized using methods like TF-IDF or Count Vectorizer to transform it into numerical format for model training.
```python
vectorized = CountVectorizer(max_features=10000)
X = vectorized.fit_transform(films_data["review"]).toarray()
```
## Model and Evaluation:

- ## Model Choice: 
Tried models like Logistic Regression, Random Forest, and recurrent neural networks (RNN).
```python
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```
- ## Evaluation Metrics: 
Used accuracy, precision, recall, and F1-score to assess model performance.
```
Accuracy: 0.8748
Classification Report:
              precision    recall  f1-score   support

           0       0.88      0.87      0.87      4961
           1       0.87      0.88      0.88      5039

    accuracy                           0.87     10000
   macro avg       0.87      0.87      0.87     10000
weighted avg       0.87      0.87      0.87     10000
```
## Results:
The model achieved an accuracy of around 88%, with balanced performance across precision and recall, indicating reliable classification of sentiment in film reviews.

## Conclusion:
This project showcases how to approach text-based sentiment classification, including essential steps like data preprocessing, model training, and evaluation.
