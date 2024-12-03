

# Chatbot Project

This project implements a conversational chatbot capable of understanding user intents and providing relevant responses. The chatbot is built using a JSON-based dataset, employs machine learning for intent classification, and supports real-time interactions through a command-line interface.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Data](#data)
3. [Features](#features)
4. [Steps to Build the Chatbot](#steps-to-build-the-chatbot)
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

The objective of this project is to develop an intelligent chatbot that:
- Recognizes user intents based on input patterns.
- Generates dynamic and relevant responses.
- Supports interaction through a command-line interface.

The chatbot uses a Naive Bayes classifier for intent classification and processes a JSON dataset containing intents, patterns, and responses.

---

## Data

The dataset for this project is stored in a `chatbot.json` file, structured as follows:

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
            "responses": ["Goodbye!", "Take care!", "See you soon!"]
        }
    ]
}
```

Key Components:
- **Tag**: Represents the intent of the user query.
- **Patterns**: Example user queries for the intent.
- **Responses**: Predefined chatbot replies for the intent.

---

## Features

1. **Data Preprocessing**: Vectorizes text patterns and encodes intent tags.
2. **Training**: Trains a Naive Bayes classifier to classify intents.
3. **Dynamic Responses**: Provides random responses for each recognized intent.
4. **Interactive Chat**: Real-time interaction through the command-line interface.

---

## Steps to Build the Chatbot

### 1. Load the JSON Data
Load the `chatbot.json` file and parse its contents into Python:
```python
with open('chatbot.json', 'r') as file:
    data = json.load(file)
```

### 2. Extract Patterns, Tags, and Responses
Process the data to extract:
- **Patterns**: User inputs (e.g., "Hi").
- **Tags**: Corresponding intent labels (e.g., "greeting").
- **Responses**: Replies for each intent.
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

### 3. Data Preprocessing
Convert text patterns into numerical vectors and encode tags as numerical labels:
```python
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(patterns)  # Vectorize patterns
unique_tags = list(set(tags))
y_encoded = [unique_tags.index(tag) for tag in tags]
```

### 4. Split Data into Training and Testing Sets
Divide the dataset into training and testing subsets:
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)
```

### 5. Train the Model
Train a Naive Bayes classifier to predict intents:
```python
model = MultinomialNB()
model.fit(X_train, y_train)
```

### 6. Evaluate the Model
Evaluate the trained model on the test set using accuracy as a metric:
```python
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
```

### 7. Build the Chatbot Framework
Implement a function to process user inputs, predict intents, and provide responses:
```python
def chatbot_response(user_input):
    user_input_vectorized = vectorizer.transform([user_input])
    predicted_tag_index = model.predict(user_input_vectorized)[0]
    predicted_tag = unique_tags[predicted_tag_index]
    return random.choice(responses[predicted_tag])
```

### 8. Run the Chatbot
Launch the chatbot for real-time interaction:
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

- **Python 3.7+**
- **Scikit-learn**: Machine learning and preprocessing.
- **Flask** (optional): For API deployment.
- **JSON**: Data storage for intents, patterns, and responses.

---

## Contributors

- **Saikiran Barma**  
  Email: [saikiranbarma@gmail.com](mailto:saikiranbarma@gmail.com)

---
