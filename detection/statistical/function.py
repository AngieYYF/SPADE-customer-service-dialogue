import pandas as pd
import math
from collections import Counter

def sentence_collect(dialogue):
    lines = dialogue.strip().split("\n")
    collect_lines = pd.DataFrame(columns=["sentence"])
    count = 0

    for line in lines:
        if line.startswith("user:"):
            collect_lines.loc[count] = line
            count += 1
    return collect_lines

def entropy_dia(human_par_dia, len_h, word_freq_counter_h):
    line = 0
    entropy = 0.0
    entropy_list = []
    for j in human_par_dia["dia_no"].unique():
        for i in range(len(human_par_dia[human_par_dia["dia_no"] == j]["sentence"])):
            sen = human_par_dia[human_par_dia["dia_no"] == j]["sentence"][:i+1]
            for key, count in word_freq_counter_h.items():
                if key in ' '.join(sen).split():
                    prob = count / len_h
                    entropy -= prob * math.log2(prob)
            entropy_list.append(entropy)
            human_par_dia.loc[line, "entropy_word_level"] = entropy
            entropy = 0.0
            line += 1
    return human_par_dia

def average_entropy_adding_sen(df, index):
    # df = df for human, gpt or llama
    # index = dia_no
    l1 = []
    count = 0
    for i in df[df["dia_no"] == index]["entropy_word_level"]:
        count += 1
        l1.append(i/count)
    return l1

def word_freq_counter(df_sen, index):
    word_freq_counter_h = Counter()
    total_word_count = 0
    for sentence in df_sen[df_sen["label"] == index]["sentence"]:
        # Split the sentence into words
        words = sentence.split()
        total_word_count += len(words)
        # Update the Counter with the words from the current sentence
        word_freq_counter_h.update(words)
    return total_word_count, word_freq_counter_h

def entropy_cal_word(sen, word_freq, sentence_len):
    entropy = 0.0
    for key, count in word_freq.items():
        if key in sen.split():
            prob = count / sentence_len
            entropy -= prob * math.log2(prob)
        
    return entropy

def process_dataframe_min(df):
    grouped = df.groupby('dia_no').agg({
        'entropy_word_level': 'min',
        'label': 'first'
    })
    return grouped

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

def dt(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Initialize Decision Tree Classifier
    clf = DecisionTreeClassifier(max_depth = 10)

    # Train the classifier
    clf.fit(X_train, y_train)

    # Predict on the test data
    y_pred = clf.predict(X_test)

    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    try: 
        f1 = f1_score(y_test, y_pred, average='macro')
        print(f"f1 score: {f1:.5f}")
    except:
        print("f1 score only available for binary classification")
    # print(f"Accuracy: {accuracy:.2f}")
    # print(cm)
    return accuracy, cm


def process_dataframe(df):
    # find the average entropy for each user response in dia
    grouped = df.groupby('dia_no').agg({
        'entropy_word_level': 'max',
        'label': 'first'
    })
    counts = df['dia_no'].value_counts()
    grouped['entropy_word_level'] = grouped['entropy_word_level'] / counts
    return grouped
