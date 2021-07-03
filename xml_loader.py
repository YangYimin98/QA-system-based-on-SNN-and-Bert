import  xml.dom.minidom
import pandas as pd
import numpy as np
import tensorflow as tf
from discord import Client
from util import question_embeddings
from util import zero_padding
from util import ManDist
from tensorflow.keras import layers
# from transformers import BertTokenizer, TFBertModel
from gensim.models import KeyedVectors
from SquadProcess import *
import keras

class Chat_bot:
    def __init__(self, path):
        self.path = path
        self.knowledge_database = {}
        self.embedding_dim = 300
        self.max_seq_length = 20
        self.SiameseModel = tf.keras.models.load_model('./data/SiameseLSTM.h5', custom_objects={'ManDist': ManDist})
        self.BertModel = keras.models.load_model("./data/model.h5")
        #self.word2vec = KeyedVectors.load_word2vec_format("./data/GoogleNews-vectors-negative300.bin.gz", binary=True)
        self.vocab = np.load('vocabs.npy', allow_pickle=True).item()
        self.question_list, self.answer_list = self.load_data()
        questions_vector = question_embeddings(self.question_list, self.vocab, True)
        self.questions_vector = zero_padding(questions_vector, self.max_seq_length)



    def load_data(self):
        dom = xml.dom.minidom.parse(self.path)
        root = dom.documentElement
        question_list = root.getElementsByTagName('subject')
        answer_list = root.getElementsByTagName('bestanswer')
        return question_list, answer_list

    def compare_similarity(self, questions):
        ask_question = question_embeddings(questions, self.vocab, False)
        duplicate_questioins = np.full(self.questions_vector.shape, zero_padding(ask_question, self.max_seq_length))
        prediction = self.SiameseModel.predict([duplicate_questioins, self.questions_vector])
        answer_index = np.argmax(prediction)
        similarity = prediction[answer_index]
        return answer_index, similarity

    def load_answer(self, answer_index):
        answer =self.answer_list[answer_index].childNodes[0].data
        question = self.question_list[answer_index].childNodes[0].data
        packed_data = {"data":
            [
                {"title": question,
                 "paragraphs": [
                     {
                         "context": answer,
                         "qas": [
                             {"question": question,
                              "id": "5737aafd1c456719005744fb",
                              "answers": [{"text": "kilogram-force", "answer_start": 82}]},
                         ]}]}]}

        test_samples = create_squad_examples(packed_data)
        x_test, _ = create_inputs_targets(test_samples)
        pred_start, pred_end = self.BertModel.predict(x_test)
        for idx, (start, end) in enumerate(zip(pred_start, pred_end)):
            test_sample = test_samples[idx]
            offsets = test_sample.context_token_to_char
            start = np.argmax(start)
            end = np.argmax(end)
            pred_ans = None
            if start >= len(offsets):
                continue
            pred_char_start = offsets[start][0]
            if end < len(offsets):
                pred_char_end = offsets[end][1]
                pred_ans = test_sample.context[pred_char_start:pred_char_end]
            else:
                pred_ans = test_sample.context[pred_char_start:]

            print("Question: " + test_sample.question)
            print("Predict answer: " + pred_ans)

            return pred_ans


    def online_answering(self, question):
        answer_index, similarity = self.compare_similarity([question])
        print("similarity " + similarity)
        answer = self.load_answer(answer_index)
        return answer



if __name__ == '__main__':
    data_loader = Chat_bot("manner.xml")
    # answer_index,similarity = data_loader.compare_similarity(['How to choose good garden tools?'])
    # print(answer_index)
    # print(similarity)
    #data_loader.load_answer(32)
    while True:
        question = input()
        print("What's you question?")
        answer = data_loader.online_answering(question)
        print(answer)