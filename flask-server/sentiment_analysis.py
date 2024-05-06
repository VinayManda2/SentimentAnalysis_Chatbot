import re
import tensorflow as tf
import pickle
from keras.models import load_model
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer

# from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def preprocess_and_tokenize(input_text, tokenizer, max_sequence_length=20):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    # Remove HTML tags, URLs, and mentions starting with '@'
    processed_input = re.sub(r'<[^>]+>|http\S+|@\w+', '', str(input_text))

    # Remove all special characters, punctuation, and numbers
    processed_input = re.sub(r'\W|\d+', ' ', processed_input)

    # Convert to lowercase
    processed_input = processed_input.lower()

    # Tokenize the text
    words = word_tokenize(processed_input)

    # Remove stopwords and lemmatize
    processed_text = ' '.join(lemmatizer.lemmatize(word) for word in words if word not in stop_words)

    # Tokenize using provided tokenizer
    sequences = tokenizer.texts_to_sequences([processed_text])

    # Pad sequences with maxlen=max_sequence_length
    padded_sequences = pad_sequences(sequences, padding='post', maxlen=max_sequence_length)

    # Convert to NumPy array
    padded_sequences = np.array(padded_sequences)

    return padded_sequences

def load_tokenizer(file_path):
    with open(file_path, 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer


# user interaction

def predict(inp):
#    print(inp)
   best_model = tf.keras.models.load_model('model.h5')
   tokenizer = load_tokenizer('tokenizer.pkl')
   text = preprocess_and_tokenize(input,tokenizer)
   predictions = best_model.predict(text)
   predicted_labels = np.argmax(predictions, axis=1)
   label_encoder = LabelEncoder()
   predicted_labels_text = label_encoder.inverse_transform(predicted_labels)[0]
   return predicted_labels_text
