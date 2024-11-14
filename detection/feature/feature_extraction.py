import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
import string
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from textblob import TextBlob
import language_tool_python
import re
import syllables
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import numpy as np
from sentence_transformers.util import cos_sim
import re
import warnings

# load model and tokenier for log prob
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
model = T5ForConditionalGeneration.from_pretrained('t5-small').to(device)
tokenizer = T5Tokenizer.from_pretrained('t5-small', model_max_length=512)

# Install NLTK and download stopwords
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# tool for counting grammer error
tool = language_tool_python.LanguageTool('en-US')

# bert sentence embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
warnings.filterwarnings("ignore", message="huggingface/tokenizers:.*")

def sentiment_scores(text): 
    '''
    Calculate the sentiment polarity and subjectivity of the text.

    Input: 
        * text = concatenated user utterances

    Usage: 
        ```
        sentiment_polarity, sentiment_subjectivity = sentiment_scores(concat_utterances)
        ```
    '''
    wiki = TextBlob(text)
    return wiki.sentiment.polarity, wiki.sentiment.subjectivity

def grammer_error_count(utterances, tool = None): 
    '''
    Count the number of grammar errors in all user utterances.
    
    Input: 
        * utterances = list of user utterances (needed for e.g. uppercase first letter in a sentence, 
                        where each sentence may not be ended by a punctuation mark)
        * tool = grammer checking tool, can be passed in as an argument to 
                avoid overhead in setting up and shutting off a server
    
    Usage: 
        ```
        tool = language_tool_python.LanguageTool('en-US')
        n_grammer_errors = grammer_error_count(list_utetrances, grammer_tool)
        ```
    '''
    # check if a new tool needs to be created
    create_tool = (tool is None) or (not isinstance(tool, language_tool_python.LanguageTool))
    if create_tool: 
        tool = language_tool_python.LanguageTool('en-US')
    
    # count grammar errors
    n_errors = 0
    for text in utterances: 
        matches = tool.check(text)
        n_errors += len(matches)
    
    # if a tool was created, close the server
    if create_tool: 
        tool.close()
    
    return n_errors


def multi_blanks(utterances):
    '''
    Count the number of occurences of multiple blanks in user utterances

    Input: 
        * utterances = list of user utterances (avoid multiblanks when joining two utterances togetehr)
    
    Usage: 
        ```
        n_multi_blanks = multi_blanks(list_utetrances)
        ```
    '''
    multi_blank_pattern = re.compile(r'\s{2,}')
    n_multi_blanks = 0
    for text in utterances: 
        matches = re.findall(multi_blank_pattern, text)
        n_multi_blanks += len(matches)
    return n_multi_blanks


def readability(utterances):
    '''
    Calculate the flesh reading ease (FRE) and flesch–kincaid grade level (FKGL) readability score of combined user utternaces
    
    Input: 
        * utterances = list of user utterances
    
    Usage: 
        ```
        num_sentences, num_words, flesh_reading_ease, flesch_kincaid_grade_level = readability(list_utetrances)
        ```
    '''
    # count number of sentences
    sentences = []
    for utterance in utterances: 
        sentences += sent_tokenize(utterance)
    total_sentences = len(sentences)
    
    # count number of words
    words = [word for utterance in utterances for word in word_tokenize(utterance) if re.search('[a-zA-Z0-9]', word)]
    total_words = len(words)

    if total_words == 0 or total_sentences == 0: return 0
    
    # Count total syllables
    total_syllables = sum(syllables.estimate(word) for word in words)

    # Calculate Flesch Reading Ease Score, bound between 0 and 100
    FRE = 206.835 - 1.015 * (total_words / total_sentences) - 84.6 * (total_syllables / total_words)
    FRE = min(FRE, 100)
    FRE = max(FRE, 0)

    # Calculate flesch–kincaid grade level (FKGL) readability score, bound between 0 and 18
    FKGL = 0.39 * (total_words / total_sentences) + 11.8 * (total_syllables / total_words) - 15.59
    FKGL = min(FKGL, 18)
    FKGL = max(FKGL, 0)

    return total_sentences, total_words, FRE, FKGL


def fit_tfidf_vectorizer(corpus, max_features=500): 
    '''
    Fit a tf-idf vectorizer on a corpus

    Input: 
        * corpus = a list of documents, where each document contains the extracted user 
                    responses of a dialogue
        * max_features = maximum dimension of the tf-idf vectors, features (terms) are selected 
                        according to term frequency across corpus
    
    Usage: 
        ```
        vectorizer = fit_tfidf_vectorizer(corpus, 200)
        ```
    '''
    # obtain a tf-idf vectorizer extracting unigram and bigram, returning the top max_features terms according to term frequency
    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=max_features)
    vectorizer.fit(corpus)
    return vectorizer

def tfidf_vector(texts, vectorizer): 
    '''
    Transform the texts into tf-idf vectors using the vectorizer. Returns tfidf vectors, one vector for each document
    
    Input: 
        * texts = list of documents, where each document contains the extracted and concatenated user responses of a dialogue
    
    Usage: 
        ```
        vectorizer = fit_tfidf_vectorizer(corpus, 200)
        tfidf_vectors = tfidf_vector(list_dialogues, vectorizer)
        ```
    '''
    return vectorizer.transform(texts).todense().tolist()


def _pairwise_similarity_mask(n):
    '''
    Generate a matrix of shape n*n, masking (set to 0) the diagonals and elements below, else set to 1.
    
    Input: 
        * n = shape of the square matrix
    '''
    tensor = np.zeros((n, n), dtype=int)
    for i in range(n - 1):
        tensor[i, i+1:] = 1
    return tensor

def _average_cosine_sim(embeddings): 
    '''
    Calculate the average pairwise cosine similarity of the embeddings 
    
    Input: 
        * embeddings = array of sentence embeddings
    '''
    # a n*n matrix containing pairwise similarities between n embeddings
    similarities = cos_sim(embeddings, embeddings) 
    # masking duplicating similarity scores and the ones between same embeddings
    similarity_mask = _pairwise_similarity_mask(len(similarities)) 
    # obtain the total similarity scores
    total_pairwise_similarities = np.sum(np.multiply(similarities, similarity_mask).tolist()) 
    # obtain the average similarity score
    avg_pairwise_similarities = total_pairwise_similarities / np.sum(similarity_mask) 
    return avg_pairwise_similarities

def sent_bert_embedding_dist(utterances, model = None): 
    '''
    Retrieve the average sentence bert embedding and the average pairwise cosine similarity of the sentence embeddings.
    Embedding has dimension 384 for the model 'all-MiniLM-L6-v2'.

    Input: 
        * utterances = list of user utterances
        * model = SentenceTransformer('all-MiniLM-L6-v2') 
            'all-mpnet-base-v2' has best performance but much larger and slower
            https://www.sbert.net/docs/sentence_transformer/pretrained_models.html
    
    Usage: 
        ```
        model = SentenceTransformer('all-MiniLM-L6-v2')
        avg_embedding, avg_cos_sim = sent_bert_embedding_dist(list_utterances, model)
        ```
    '''
    sentences = []
    for utterance in utterances: 
        sentences += sent_tokenize(utterance)
    if model is None or not isinstance(model, SentenceTransformer): 
        model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # get sentence embeddings and the average sentence embedding
    embeddings = model.encode(sentences)
    avg_embeddings = np.mean(embeddings, axis=0)

    # get sentence embedding distance (cosine similarity)
    avg_sim = _average_cosine_sim(embeddings)
    return avg_embeddings, avg_sim

def extract_utterances(dialogue, role='user'): 
    '''
    Extract all utterances from a dialogue, could be of a particulr role. Any preceding "role:" or "role: " would been removed.
    Returns a list of utterances.

    Input: 
        * dialogue = full dialogue text
        * role = the role whose utterances to be extracted. None or empty string extracts all utterances.
    '''
    # extract utterances of a particular role 
    if role: 
        pattern = re.compile(f'^{role}: ?(.*)\s*(?:\n|$)', re.MULTILINE)
        return [user_utterance.group(1) for user_utterance in re.finditer(pattern, dialogue)]
    # extract all utterances
    else: 
        pattern = re.compile(r'^\S{4,6}: ?(.*)\n', re.MULTILINE)
        return [utterance.group(1) for utterance in re.finditer(pattern, dialogue)]


def sentence_collect(dialogue):
    # output: the number of sentence in a dia
    # output: a string join all user response together with \n
    lines = dialogue.strip().split("\n")
    collect_lines = []
    count = 0

    for line in lines:
        if line.startswith("user:"):
            line = line.replace("\r", "")
            line = line.replace("user:", "")
            collect_lines.append(line)
            count += 1
    return count, ' '.join(collect_lines)

def count_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    stopword_count = sum(1 for word in words if word in stop_words)
    return stopword_count

def count_special_char(text):
    special_char = set(string.punctuation)
    special_char_count = sum(1 for char in text if char in special_char)
    return special_char_count

def count_unique_word(text):
    unique_word = set(text.lower().split(" "))
    return len(unique_word)

def log_prob_cal(sentence):

    model.eval()

    # define some source text and tokenize it
    source_text = sentence
    source_ids = tokenizer(source_text, return_tensors="pt").input_ids.to(device)

    # generate the output using beam search
    # model: t5-small
    gen_outputs = model.generate(
        inputs=source_ids,
        num_beams=2,
        min_length=0,
        max_length=512,
        length_penalty=1,
        output_scores=True,
        return_dict_in_generate=True,
        
    )

    # the loss given by forward()
    log_prob = gen_outputs.sequences_scores.item()

    return log_prob

def sen_length(text):
    words = text.lower().split(" ")
    return len(words)

def sen_density(text):
    # count is the number of sentence in the dia
    words = len(text.lower().split(" "))
    unique_word = len(set(text.lower().split(" ")))
    return unique_word/words

def countverb(sentence):
    text = word_tokenize(sentence)
    tagged = pos_tag(text)
    vb = []
    for word, state in tagged:
        if state == "VB":
            vb.append((word, state))

    return len(vb)

def countnoun(sentence):
    text = word_tokenize(sentence)
    tagged = pos_tag(text)
    nn = []
    for word, state in tagged:
        if state == "NN":
            nn.append((word, state))

    return len(nn)



def feature_extract(dialogue_df, col = 'dia', id_col = 'dia_no', args = None):
    '''
    Extract features for the training and testing dataset.
    Note: make sure to pass in the same vectorizer used to transform the training dataset when transforming the testing dataset
    
    Input: 
        * dialogue_df = a dataframe containing dialogues to extract features from
        * col = column in the dialogue_df containing the dialogue
        * id_col = column in dialogue_df containing the dialogue id
        * args = additional arguments (e.g. 'full_dialogue': True to include system utterances as well (default is user only), 'vectorizer' to provide the tf-idf vectorizer fitted on training dataset)
    
    Usage: 
        ```
        # need to separate training dataset and testing dataset (unless intends to fit the tf-idf vectoriser on entire training and testing dataset)
        args = extract_features(train_df, 'dia')
        args = extract_features(test_df, 'dia', args)
        args['train_features'], args['test_features']
        ```
    '''
    if args is None: 
        args = dict()
        
    assert col in dialogue_df.columns, f'column {col} does not exist in the dataframe'
    
    result = pd.DataFrame()
    result['dia'] = dialogue_df[col]
    result['dia_no'] = dialogue_df[id_col]

    result[["num_utterance", "sentence"]] = dialogue_df[col].apply(lambda x: pd.Series(sentence_collect(x)))

    result["log_prob_cal"] = result["sentence"].apply(lambda x: log_prob_cal(x))

    result["count_stopwords"] = result["sentence"].apply(lambda x: count_stopwords(x))

    result["count_special_char"] = result["sentence"].apply(lambda x: count_special_char(x)) 

    result["count_unique_word"] = result["sentence"].apply(lambda x: count_unique_word(x)) 

    result["sen_len"] = result["sentence"].apply(lambda x: sen_length(x)) 

    result["sen_density"] = result["sentence"].apply(lambda x: sen_density(x)) 

    result["count_verb"] = result["sentence"].apply(lambda x: countverb(x)) 
    result["count_noun"] = result["sentence"].apply(lambda x: countnoun(x)) 

    role = None if 'full_dialogue' in args else 'user'
    utterances_list = dialogue_df[col].apply(lambda x: extract_utterances(x, role))
    
    result[['sentiment_polarity', 'sentiment_subjectivity']] = result["sentence"].apply(lambda x: pd.Series(sentiment_scores(x)))

    result["gramma_error_count"] = utterances_list.apply(lambda x: grammer_error_count(x, tool))

    result["n_multi_blanks"] = utterances_list.apply(lambda x: multi_blanks(x))

    result[['num_sentence', 'num_word', 'flesch_reading_ease', 'flesch_kincaid_grade_level']] = utterances_list.apply(lambda x: pd.Series(readability(x)))

    result[["avg_embedding","avg_cos_sim"]] = utterances_list.apply(lambda x: pd.Series(sent_bert_embedding_dist(x, model)))

    transform_train = False
    if 'vectorizer' not in args: 
        print('[Log] Create new tfidf vectorizer')
        transform_train = True
        args['vectorizer'] = fit_tfidf_vectorizer(result["sentence"])
    result["tfidf"] = tfidf_vector(result["sentence"], args['vectorizer'])

    if transform_train: 
        args['train_features'] = result
    else: 
        args['test_features'] = result
    return args


def calculate_feature(dfs, feature, scale_feature, features):
    '''
    Calculate a new feature by scaling a feature by another scale_feature. Add the new feature to the dfs and features list.

    Input: 
        * dfs = a single dataframe, or a list of dataframes to calculate and append the new features
        * feature = feature value as numerator
        # scale_feature = feature value as denominator
        * features = a list of features that originally exists

    Usage: 
        ```
        calculate_feature([human_df, gpt_df], 'num_word', 'num_sentence', existing_features)
        ```
    '''
    if isinstance(dfs, pd.DataFrame): 
        dfs = [dfs]

    if not feature in features or not scale_feature in features: 
        return
    for df in dfs: 
        if feature in features and scale_feature in features: 
            title= f'{feature} / {scale_feature}'
            if title not in features: 
                features.append(title)
            df[title] = df[feature] / df[scale_feature]


def vec_to_col(dataset, col, colname):
    '''
    Convert a text vector column to multiple columns.

    Input: 
        * dataset = the dataset dataframe containing the text vector column
        * col = the name of the original text vector column in dataset
        * colname = naming of the new columns converted from the original text vector
    
    Usage: 
        ```
        dataset = vec_to_col(dataset, 'tfidf', 'tf')
        ```
    '''
    new_columns = {}
    for i in range(len(dataset[col][0])):
        new_columns[f'col_{colname}_{i+1}'] = dataset[col].apply(lambda x: x[i] if i < len(x) else None)
    new_columns = pd.DataFrame(new_columns)
    dataset = pd.concat([dataset, new_columns], axis=1)
    dataset.drop(col, axis=1, inplace=True)
    return dataset


def preprocess_dataset(dataset): 
    '''
    Append some new features to the dataset, and convert text vector elements to columns.
    '''
    # add new features
    features = list(dataset.columns)
    calculate_feature(dataset, 'gramma_error_count', 'num_word', features)
    calculate_feature(dataset, 'n_multi_blanks', 'num_word', features)
    calculate_feature(dataset, 'count_stopwords', 'num_word', features)
    calculate_feature(dataset, 'count_noun', 'num_word', features)
    calculate_feature(dataset, 'count_verb', 'num_word', features)
    calculate_feature(dataset, 'num_word', 'num_sentence', features)
    calculate_feature(dataset, 'num_word', 'num_utterance', features)
    calculate_feature(dataset, 'num_sentence', 'num_utterance', features)
    
    # fill NA values
    dataset["avg_cos_sim"] = dataset["avg_cos_sim"].fillna(0)

    # convert text vector to columns
    dataset = vec_to_col(dataset, "avg_embedding", "embd")
    dataset = vec_to_col(dataset, "tfidf", "tf")

    return dataset