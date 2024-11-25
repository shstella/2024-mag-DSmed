import spacy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

nlp = spacy.load("en_core_web_sm")

texts = [
    "The patient was diagnosed with diabetes mellitus type 2.",
    "The doctor prescribed metformin and insulin therapy.",
    "Symptoms included hyperglycemia and neuropathy.",
    "MRI scans showed no abnormalities in the brain.",
    "He has a history of hypertension and chronic kidney disease."
]

annotations = [
    [("The", "O"), ("patient", "O"), ("was", "O"), ("diagnosed", "O"), 
     ("with", "O"), ("diabetes", "B-Disease"), ("mellitus", "I-Disease"), ("type", "I-Disease"), ("2", "I-Disease"), (".", "O")],
    [("The", "O"), ("doctor", "O"), ("prescribed", "O"), ("metformin", "B-Medication"), 
     ("and", "O"), ("insulin", "B-Medication"), ("therapy", "O"), (".", "O")],
]

def extract_features(doc):
    features = []
    for token in doc:
        token_features = {
            "text": token.text,
            "lemma": token.lemma_,
            "pos": token.pos_,
            "tag": token.tag_,
            "is_alpha": token.is_alpha,
            "is_stop": token.is_stop,
        }
        features.append(token_features)
    return features

X, y = [], []
for text, annotation in zip(texts, annotations):
    doc = nlp(text)
    features = extract_features(doc)
    labels = [label for _, label in annotation]
    X.extend(features)
    y.extend(labels)

vectorizer = DictVectorizer(sparse=False)
X_vect = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

new_text = "The patient is taking aspirin for headache."
new_doc = nlp(new_text)
new_features = extract_features(new_doc)
new_vect = vectorizer.transform(new_features)
new_pred = clf.predict(new_vect)

for token, label in zip(new_doc, new_pred):
    print(f"{token.text}: {label}")
