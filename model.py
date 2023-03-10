import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
import unicodedata
import re
import numpy as np
import os
import io
import time
import pickle


import requests
import datetime

def get_random_joke():
    url = "https://dad-jokes.p.rapidapi.com/random/joke"

    headers = {
        "X-RapidAPI-Key": "133ceb241cmsha10b4c1917a04d1p170d50jsnda7cc3792736",
        "X-RapidAPI-Host": "dad-jokes.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers)
    joke_data = response.json()

    joke = joke_data["body"][0]["setup"] + "\n" + joke_data["body"][0]["punchline"]

    return joke




BATCH_SIZE = 128
embedding_dim = 256  # for word embedding
units = 1024  # dimensionality of the output space of RNN
max_length_inp = 141
max_length_targ = 25



class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_size):
    super(Encoder, self).__init__()
    self.batch_size = batch_size
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim) #embedding layer
    self.gru = tf.keras.layers.GRU(self.enc_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform') #GRU layer

  def call(self, x, hidden):
    x = self.embedding(x) #embedding layer
    output, state = self.gru(x, initial_state = hidden) #GRU layer
    return output, state
    
  def initialize_hidden_state(self):
    return tf.zeros((self.batch_size, self.enc_units)) #initializing the hidden state
  
 

class DotProductAttention(tf.keras.layers.Layer):
  def call(self, query, values):
    query_with_time_axis = tf.expand_dims(query, 1)
    score = query_with_time_axis * values
    score = tf.reduce_sum(score, axis=2)
    score = tf.expand_dims(score, 2)
    attention_weights = tf.nn.softmax(score, axis=1)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights


class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    query_with_time_axis = tf.expand_dims(query, 1)
    score = self.V(tf.nn.tanh( self.W1(query_with_time_axis) + self.W2(values)))
    attention_weights = tf.nn.softmax(score, axis=1)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights


class DecoderWithAttention(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_size, attention_layer = None):
    super(DecoderWithAttention, self).__init__()
    self.batch_size = batch_size  
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim) #embedding layer
    self.gru = tf.keras.layers.GRU(self.dec_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform') #GRU layer
    self.fc = tf.keras.layers.Dense(vocab_size) #dense layer

    self.attention = attention_layer #attention layer

  def call(self, x, hidden, enc_output):
    x = self.embedding(x) #embedding layer
    attention_weights = None
    if self.attention: #if attention layer is not None
      context_vector, attention_weights = self.attention(hidden, enc_output) #calculate the context vector and the attention weights
      x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1) #concatenate the context vector with the input

    output, state = self.gru(x, initial_state = hidden) #GRU layer

    output = tf.reshape(output, (-1, output.shape[2])) #reshape the output

    x = self.fc(output) #dense layer
 
    return x, state, attention_weights #return the output, the hidden state and the attention weights
    


def unicode_to_ascii(s: str) -> str:
  """
  convert unicode to ascii and replaces all accented characters with their equivalent unaccented character (for example, é becomes e)
  """
  return ''.join(c for c in unicodedata.normalize('NFD', s)
      if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w: str) -> str:
  """
  adds spaces around punctuation and removes all non-alphanumeric characters and adds start and end tokens
  """
  w = unicode_to_ascii(w.lower().strip()).replace(")"," )").replace("(","( ")
  w = re.sub(r"([?.!,¿])", r" \1 ", w)
  w = re.sub(r'[" "]+', " ", w)
  w = re.sub(r"[^a-zA-Z()[]?.!,¿]+", " ", w)
  w = w.strip()
  w = '<sos> ' + w + ' <eos>'
  return w


def load_tokenizers():
    with open('C:\\Users\\orime\\school\\shinebox_virtual_assistent\\inp_lang.pickle', 'rb') as handle:
        inp_lang = pickle.load(handle)
    with open('C:\\Users\\orime\\school\\shinebox_virtual_assistent\\targ_lang.pickle', 'rb') as handle:
        targ_lang = pickle.load(handle)
    return inp_lang, targ_lang







def translate(sentence, encoder, decoder):
  attention_plot = np.zeros((max_length_targ, max_length_inp)) #creating an attention plot

  sentence = preprocess_sentence(sentence) #preprocessing the sentence

  inputs = [inp_lang.word_index[i] for i in sentence.split(' ')] #creating the inputs using the input language tokenizer
  inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding='post')  #padding the inputs
  inputs = tf.convert_to_tensor(inputs) #converting the inputs to a tensor

  result = ''

  hidden = [tf.zeros((1, units))] #creating the hidden state
  enc_out, enc_hidden = encoder(inputs, hidden) #getting the encoder output and the encoder hidden state

  dec_hidden = enc_hidden #setting the decoder hidden state to the encoder hidden state
  dec_input = tf.expand_dims([targ_lang.word_index['<sos>']], 0) #setting the decoder input to the start of sentence token

  for t in range(max_length_targ): #for each word in the target language
    predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out) #getting the predictions, the decoder hidden state and the attention weights

    predicted_id = tf.argmax(predictions[0]).numpy() #getting the predicted id

    result += targ_lang.index_word[predicted_id] + ' ' #adding the predicted word to the result

    if targ_lang.index_word[predicted_id] == '<eos>': #if the predicted word is the end of sentence token then return the result and the sentence
      return result, sentence

    dec_input = tf.expand_dims([predicted_id], 0) #setting the decoder input to the predicted id

  return result, sentence #returning the result and the sentence if the end of sentence token is not predicted after the maximum length of the target language

def outputTriggerToOutputStruct(outputTrigger) -> dict[str, str, list[str]]:
        actionType = outputTrigger.split("[")[0]
        api = outputTrigger.split("[")[1].split("]")[0]
        parameters = outputTrigger.split("]")[1].split("(")[1].split(")")[0]
        return {'actionType': actionType, 'api': api, 'parameters': parameters}


def handleResponse(request):
    response = translate(request, encoder, decoder)[0]
    responseStruct = outputTriggerToOutputStruct(response)
    print(responseStruct)
    if responseStruct['actionType'] == 'qanda':
        return responseStruct['parameters']
    elif responseStruct['actionType'] == 'apiusage ':
      print("API CALL")
      if responseStruct['api'].replace(" ","") == 'jokes':
        return get_random_joke()
      return responseStruct['parameters'].split(",")
    else:
        return responseStruct['parameters']


def getResponse(input_text: str) -> str:
    return handleResponse(input_text)


def bootstrap(): # automatically called when the module is loaded
    # load tokenizers
    global inp_lang, targ_lang
    inp_lang, targ_lang = load_tokenizers()
    vocab_inp_size = len(inp_lang.word_index)+1
    vocab_tar_size = len(targ_lang.word_index)+1
    # load models
    global encoder, decoder
    encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
    decoder = DecoderWithAttention(vocab_tar_size, embedding_dim, units, BATCH_SIZE, DotProductAttention())
    encoder.load_weights('C:\\Users\\orime\\school\\shinebox_virtual_assistent\\DP\\encoder_dp')
    decoder.load_weights('C:\\Users\\orime\\school\\shinebox_virtual_assistent\\DP\\decoder_dp')


bootstrap()

if __name__ == "__main__":
    print(getResponse(f"input . ( {input()} )" ))