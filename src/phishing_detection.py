import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Sample dataset
data = {
    'url_length': [20, 85, 15, 120],
    'has_at_symbol': [0, 1, 0, 1],
    'is_phishing': [0, 1, 0, 1]
}
df = pd.DataFrame(data)

X = df[['url_length', 'has_at_symbol']]
y = df['is_phishing']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

new_url = [[100, 1]]  # length=100, has '@'
print("Prediction (1=phishing, 0=safe):", model.predict(new_url))

