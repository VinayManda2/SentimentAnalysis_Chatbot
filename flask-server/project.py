import numpy as np
#from keras.src.layers import Bidirectional
import tensorflow
from tensorflow.keras.layers import Bidirectional
from keras_tuner import HyperModel, BayesianOptimization
from keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras_tuner.tuners import RandomSearch
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import nltk
from nltk.corpus import stopwords
import re
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
#from keras_tuner.engine.hypermodel import HyperModel - already imported
from tensorflow.keras.models import Sequential
import pandas as pd

df=pd.read_excel("/content/data.xlsx")
data = df.to_records(index=False).tolist()
df.isnull().values.any()
y_value = df['Sentiment']
sentiment_mapping = {"positive":0,"negative":1,"neutral":2}
label = np.array([sentiment_mapping[sentiment] for sentiment in y_value])

def preprocess(text):

  text=text.lower()

  # removing html tags using Beautifulsoup
  soup = BeautifulSoup(text, 'html.parser')
  text= soup.get_text()

  #removing special characters from text
  text=re.sub(r'[^\w\s]','',text)

  #removing numbers from text
  text = re.sub(r'\d', '', text)

  #removing words which are started with '$'
  pattern = r'\$[a-zA-Z0-9\.]+'
  text = re.sub(pattern, '', text)

  #removing stopwords
  text=[word for word in text.split() if word not in stopwords.words('english')]

  #spell correction
  spell = SpellChecker()
  words = text      #.split()
  misspelled = spell.unknown(words)
  corrected_words = []
  for word in words:
    if word in misspelled:
      corrected_word = spell.correction(word)
      corrected_words.append(corrected_word)
    else:
      corrected_words.append(word)

  text =corrected_words

  #removing None values
  text = [item for item in text if item is not None]

  #lemmatization
  lemmatizer = WordNetLemmatizer()
  lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
  text=lemmatized_words


  # Perform POS tagging
  tagged_tokens = pos_tag(text)

  # function to filter out non-nouns and non-verbs
  def filter_nouns_verbs(tagged_tokens):
    filtered_tokens = [word for word, tag in tagged_tokens if tag.startswith('N') or tag.startswith('V')]
    return filtered_tokens

  filtered_text = filter_nouns_verbs(tagged_tokens)
  text = filtered_text
  print(text)
  #text = ' '.join(text)

  return text

corpus = [preprocess(text) for text, _ in data]

# Create a TfidfVectorizer object
tfidf_vectorizer = TfidfVectorizer()

  #  Fit and transform the documents to compute TF-IDF scores
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

  # Get the feature names (words)
feature_names = tfidf_vectorizer.get_feature_names_out()

  # Convert the TF-IDF matrix to a dense array and display it
tfidf_matrix_dense = tfidf_matrix.toarray()
  #print(tfidf_matrix_dense)

# Apply PCA for dimensionality reduction
n_components = 2
pca = PCA(n_components=n_components)
reduced_tfidf = pca.fit_transform(tfidf_matrix.toarray())

#tokenization and padding
tokenizer = Tokenizer(num_words=5000)    # oov_token='<OOV>')
tokenizer.fit_on_texts(corpus)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(corpus)
padded_sequences = pad_sequences(sequences,maxlen=100,padding='pre')

#split the data into train and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(padded_sequences,y_value,test_size=0.2, random_state=42)

#calculate class weights for imbalanced dataset
class_counts = np.bincount(train_labels)
total_samples = sum(class_counts)
class_weights = {cls: total_samples / count for cls, count in enumerate(class_counts) }

class SentimentHyperModel(HyperModel):

    def build(self, hp):
        model = Sequential()

        # Embedding layer
        model.add(Embedding(len(word_index) + 1, hp.Int('embedding_dim', min_value=64, max_value=256, step=16)))

        # Bidirectional LSTM layer
        model.add(Bidirectional(LSTM(hp.Int('lstm_units', min_value=64, max_value=128, step=16), return_sequences=True)))

        # Optional Conv1D layer
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))

        # Global Max Pooling layer
        model.add(GlobalMaxPooling1D())

        # Additional Dense layers for TF-IDF input
        model.add(Dense(units=64, activation='relu', input_shape=(reduced_tfidf.shape[1],)))
        model.add(Dense(units=32, activation='relu'))
        #model.add(Dense(units=3, activation='softmax'))

        # Dense layers
        model.add(Dense(hp.Int('dense_units', min_value=64, max_value=256, step=32), activation='relu'))
        model.add(Dropout(hp.Float('dense_dropout', min_value=0.2, max_value=0.6, step=0.1)))
        model.add(Dense(3, activation='softmax'))

        # Optimizer and Loss
        optimizer = Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1, sampling='log'))
        loss = SparseCategoricalCrossentropy()
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

        return model


hypermodel = SentimentHyperModel()#3,word_index,reduced_tfidf)
# Define the BayesianOptimization tuner
tuner = BayesianOptimization(hypermodel,
                                objective='val_accuracy',  # Metric to optimize
                                max_trials=100,directory='my_dir',project_name='sentiment_analysis',        # Number of trials
                                num_initial_points=10)       # Number of initial random points

# perform the hyperparameter search using training data
tuner.search(train_texts, train_labels, epochs=100, validation_data=(val_texts, val_labels),
             class_weight=class_weights,callbacks=[
                EarlyStopping(patience=5, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.2, patience=2)
             ])

best_model = hypermodel.build(best_hps)
best_model.fit(train_texts, train_labels, epochs=100, validation_data=(val_texts, val_labels), class_weight=class_weights)

best_model.save("model.h5")

