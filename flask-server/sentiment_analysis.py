import re
from nltk.corpus import stopwords
from spellchecker import SpellChecker
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model


def process(text):

  text=text.lower()
  text=re.sub(r'[^\w\s]','',text)
  text = re.sub(r'\d', '', text)
  text=[word for word in text.split() if word not in stopwords.words('english')]

  #spell correction
  spell = SpellChecker()
  words = text
  misspelled = spell.unknown(words)
  corrected_words = []
  for word in words:
    if word in misspelled:
      corrected_word = spell.correction(word)
      corrected_words.append(corrected_word)
    else:
      corrected_words.append(word)

  text =corrected_words
  text = [item for item in text if item is not None]

  #lemmatization
  lemmatizer = WordNetLemmatizer() #requires tokens
  lemmatized_words = [lemmatizer.lemmatize(text) for text in text if text!=None]
  text=lemmatized_words

  # Perform POS tagging
  tagged_tokens = pos_tag(text)

  # Define a function to filter out non-nouns and non-verbs
  def filter_nouns_verbs(tagged_tokens):
    filtered_tokens = [word for word, tag in tagged_tokens if tag.startswith('N') or tag.startswith('V')]
    return filtered_tokens

  filtered_text = filter_nouns_verbs(tagged_tokens)
  text = filtered_text

  text = ' '.join(text) if text is not None else ''


  return text


def predict_sentiment_with_best_model(user_input,model):
  preprocessed_input = process(user_input)
  sequence = tokenizer.texts_to_sequences([preprocessed_input])
  padded_sequence = pad_sequences(sequence, padding='post', maxlen=max_length+1)
  sentiment_probabilities = model.predict(padded_sequence)[0]
  padded_sequence = pad_sequences(sequence, maxlen=30, padding='post')
  return list(sentiment_mapping.keys())[list(sentiment_mapping.values()).index(predictedt_sentiment)]





# user interaction

def predict(input):
  print(input)
#   best_model = tensorflow.keras.models.load_model("model.h5")
#   sentiment = predict_sentiment_with_best_model(user_input, best_model)
  return "hello from server"