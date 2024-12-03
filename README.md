

# Chatbot Project

This project implements a simple yet effective chatbot using a machine learning-based intent classification model. The chatbot uses a JSON-based dataset containing intents, patterns, and responses and leverages a Naive Bayes classifier to predict user intents.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Data](#data)
3. [Features](#features)
4. [Steps to Build the Chatbot](#steps-to-build-the-chatbot)
   - Import Necessary Libraries
   - Load the JSON Data
   - Extract Patterns, Tags, and Responses
   - Data Preprocessing
   - Split Data into Training and Testing Sets
   - Train the Model
   - Evaluate the Model
   - Build the Chatbot Framework
   - Run the Chatbot
5. [Installation and Setup](#installation-and-setup)
6. [Usage](#usage)
7. [Tools and Libraries](#tools-and-libraries)
8. [Contributors](#contributors)

---

## Project Overview

This project creates a chatbot capable of:
- Recognizing user intents based on input queries.
- Providing dynamic responses based on recognized intents.
- Supporting interaction through a command-line interface.

The chatbot uses a Naive Bayes classifier for intent classification and is trained on a JSON dataset.

---

## Data

The chatbot uses a JSON dataset containing:
- **Intents**: Categories for user queries (e.g., "greeting", "goodbye").
- **Patterns**: Examples of user input queries for each intent.
- **Responses**: Predefined replies for each intent.

Example JSON structure:
```json
{
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["Hi", "Hello", "How are you?"],
            "responses": ["Hello!", "Hi there!", "How can I help you?"]
        },
        {
            "tag": "goodbye",
            "patterns": ["Bye", "See you", "Goodbye"],
            "responses": ["Goodbye!", "Take care!", "See you later!"]
        }
    ]
}
```

---

## Features

- **Intent Classification**: Identifies the intent of user queries.
- **Dynamic Responses**: Randomly selects a response for recognized intents.
- **Real-Time Interaction**: Interacts with users via a command-line interface.

---

## Steps to Build the Chatbot

### 1. Import Necessary Libraries
```python
import json  # To load and parse the JSON file
import random  # For selecting a random response for each intent
from sklearn.feature_extraction.text import CountVectorizer  # To vectorize text data
from sklearn.model_selection import train_test_split  # To split data into training and testing sets
from sklearn.naive_bayes import MultinomialNB  # The Naive Bayes classifier
from sklearn.metrics import accuracy_score  # To evaluate model performance
```

### 2. Load the JSON Data
Load the chatbot intents from a JSON file:
```python
with open('chatbot.json', 'r') as file:
    data = json.load(file)
```

### 3. Extract Patterns, Tags, and Responses
Process the JSON data to extract:
- Patterns: Example queries.
- Tags: Intent labels.
- Responses: Predefined replies for each intent.
```python
patterns = []
tags = []
responses = {}

for intent in data['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])
    responses[intent['tag']] = intent['responses']
```

### 4. Data Preprocessing
Convert text patterns into numerical vectors and encode tags as numerical labels:
```python
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(patterns)  # Vectorize patterns
unique_tags = list(set(tags))
y_encoded = [unique_tags.index(tag) for tag in tags]
```

### 5. Split Data into Training and Testing Sets
Split the dataset into training and testing subsets:
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)
```

### 6. Train the Model
Train a Naive Bayes classifier:
```python
model = MultinomialNB()
model.fit(X_train, y_train)
```

### 7. Evaluate the Model
Evaluate the model on the test set using accuracy as a metric:
```python
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
```

### 8. Build the Chatbot Framework
Implement a function to process user inputs, predict intents, and provide responses:
```python
def chatbot_response(user_input):
    user_input_vectorized = vectorizer.transform([user_input])
    predicted_tag_index = model.predict(user_input_vectorized)[0]
    predicted_tag = unique_tags[predicted_tag_index]
    return random.choice(responses[predicted_tag])
```

### 9. Run the Chatbot
Launch the chatbot and interact via the command line:
```python
print("Chatbot: Hello! I am your assistant. Type 'exit' to end the conversation.")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Chatbot: Goodbye! Have a nice day!")
        break
    response = chatbot_response(user_input)
    print(f"Chatbot: {response}")
```

---

## Installation and Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/chatbot-project.git
   cd chatbot-project
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. Run the chatbot script:
   ```bash
   python chatbot_app.py
   ```

2. Type queries to interact with the chatbot. Type `exit` to quit.

---

## Tools and Libraries

- **Python 3.7+**: Programming language for development.
- **Scikit-learn**: For vectorization and intent classification.
- **JSON**: To structure the dataset.
- **Random**: For dynamic response selection.

---

## Contributors

**Saikiran Barma**  
Email: [saikiranbarma@gmail.com](mailto:saikiranbarma@gmail.com)
