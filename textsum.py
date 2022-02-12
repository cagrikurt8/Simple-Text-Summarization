import math
import statistics
import string
from bs4 import BeautifulSoup
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd


class TextSummarizer:
    def __init__(self, xml):
        self.soup = BeautifulSoup(xml, "xml")
        self.stop_words = stopwords.words('english')
        self.lemmatizer = WordNetLemmatizer()

        self.headers = self.soup.find_all("value", {"name": "head"})
        self.texts = self.soup.find_all("value", {"name": "text"})
        self.text_sentences = self.text_sentence_tokenize()

        self.sqrt_lengths = self.calculate_sqrt_lengths()
        self.tokenized_words = self.tokenize()

        self.lemmatized_words = self.lemmatize()
        self.lemma_numbers = self.count_lemmas()
        self.lemma_probs = self.count_probability()

        self.sentence_probs = self.calculate_sentence_probs()
        self.sorted_sentences = self.sort_sentences()

    def text_sentence_tokenize(self):
        text_sentences = []

        for text in self.texts:
            text = sent_tokenize(text.text)
            text_sentences.append(text)

        return text_sentences

    def calculate_sqrt_lengths(self):
        sqrt_lengths = []

        for text in self.text_sentences:
            sqrt_lengths.append(round(math.sqrt(len(text))))

        return sqrt_lengths

    def tokenize(self):
        tokenized_words = []

        for text in self.text_sentences:
            text_words = []

            for sentence in text:
                text_words.append(word_tokenize(sentence))

            for sentence in text_words:
                for i in range(len(sentence)):
                    sentence[i] = sentence[i].lower()

            tokenized_words.append(text_words)

        return tokenized_words

    def lemmatize(self):
        lemmatized_words = []

        for text in self.tokenized_words:
            text_words = []

            for sentence in text:
                sentence_words = []

                for word in sentence:
                    if word not in string.punctuation and word not in self.stop_words:
                        sentence_words.append(self.lemmatizer.lemmatize(word))
                text_words.append(sentence_words)
            lemmatized_words.append(text_words)

        return lemmatized_words

    def count_lemmas(self):
        lemma_numbers = []

        for text in self.lemmatized_words:
            length = 0

            for sentence in text:
                length += len(sentence)
            lemma_numbers.append(length)
        return lemma_numbers

    def count_probability(self):
        lemma_probs = {"text1": {}, "text2": {}, "text3": {}, "text4": {}, "text5": {},
                       "text6": {}, "text7": {}, "text8": {}, "text9": {}, "text10": {}}
        lemmatized_texts = []

        for text in self.lemmatized_words:
            news_text = []

            for sentence in text:
                for word in sentence:
                    news_text.append(word)
            lemmatized_texts.append(news_text)

        for text, idx in zip(lemmatized_texts, range(len(lemmatized_texts))):
            text_idx = f"text{idx + 1}"

            for word in text:
                prob = text.count(word) / self.lemma_numbers[idx]
                lemma_probs[text_idx][word] = prob

        return lemma_probs

    def calculate_sentence_probs(self):
        sentence_probs = []

        for text, idx in zip(self.lemmatized_words, range(len(self.lemmatized_words))):
            text_probs = []
            text_idx = f"text{idx + 1}"

            for sentence in text:
                prob_list = []

                for word in sentence:
                    prob_list.append(self.lemma_probs[text_idx][word])

                text_probs.append(statistics.mean(prob_list))
            sentence_probs.append(text_probs)

        return sentence_probs

    def sort_sentences(self):
        sorted_text_sentences = []
        sorted_text_sentences_index = self.sorted_sentences_index()

        for text, idx in zip(sorted_text_sentences_index, range(len(sorted_text_sentences_index))):
            text_key = f"text{idx + 1}"

            sorted_sentence = []
            i = 0

            while len(sorted_sentence) < self.sqrt_lengths[idx]:
                highest_prob_word = self.find_highest_word(text_key)
                sentence_idx = sorted_text_sentences_index[idx][i]
                sentence = self.lemmatized_words[idx][sentence_idx]

                if highest_prob_word in sentence:
                    sorted_sentence.append(self.text_sentences[idx][sentence_idx])
                    self.update_weights(text_key, idx, sentence_idx)
                    sorted_text_sentences_index = self.sorted_sentences_index()
                    i = 0

                else:
                    i += 1

            sorted_text_sentences.append(sorted_sentence)

        return self.order_selected_sentences(sorted_text_sentences)

    def sorted_sentences_index(self):
        sorted_sentences_index = []

        for text in self.sentence_probs:
            sentences = text.copy()
            sorted_sentences = []

            for i in range(len(sentences)):
                idx = sentences.index(max(sentences))
                sorted_sentences.append(idx)
                sentences[idx] = 0

            sorted_sentences_index.append(sorted_sentences)

        return sorted_sentences_index

    def find_highest_word(self, text_key):
        highest_lemma = ""
        highest_prob = 0

        for lemma, lemma_prob in self.lemma_probs[text_key].items():
            if lemma_prob > highest_prob:
                highest_prob = lemma_prob
                highest_lemma = lemma

        return highest_lemma

    def update_weights(self, text_key, text_idx, sentence_idx):
        for word in self.lemmatized_words[text_idx][sentence_idx]:
            self.lemma_probs[text_key][word] *= self.lemma_probs[text_key][word]

        self.sentence_probs = self.calculate_sentence_probs()

    def order_selected_sentences(self, sorted_sentences):
        final_text_sentences = []

        for text, text_idx in zip(sorted_sentences, range(len(sorted_sentences))):
            idx_list = []

            for sentence in text:
                idx = self.text_sentences[text_idx].index(sentence)
                idx_list.append(idx)
            idx_list.sort()

            text_sentences = []
            for i in idx_list:
                text_sentences.append(self.text_sentences[text_idx][i])

            final_text_sentences.append(text_sentences)

        return final_text_sentences


class Vectorizer:
    def __init__(self, xml):
        self.model = TfidfVectorizer(tokenizer=lambda x: word_tokenize(x))
        self.soup = BeautifulSoup(xml, "xml")
        self.stop_words = stopwords.words('english')
        self.lemmatizer = WordNetLemmatizer()

        self.headers = self.soup.find_all("value", {"name": "head"})
        self.texts = self.soup.find_all("value", {"name": "text"})

        self.text_sentences = [sent_tokenize(text.text) for text in self.texts]
        self.sqrt_lengths = self.calculate_sqrt_lengths()

        self.tokenized_words = self.tokenize_words()
        self.lemmatized_words = self.lemmatize()

        self.datasets = self.extract_data()
        self.sentence_probs = self.count_sentence_probabilities()

        self.selected_sentences = self.select_sentences()

        self.modified_probabilities = self.count_modified_probabilities()
        self.modified_model_sentences = self.modified_model_select_sentences()

    def calculate_sqrt_lengths(self):
        sqrt_lengths = []

        for text in self.text_sentences:
            sqrt_lengths.append(round(math.sqrt(len(text))))

        return sqrt_lengths

    def tokenize_words(self):
        tokenized_words = []

        for text in self.text_sentences:
            text_sentences = []

            for sentence in text:
                words = []

                for word in word_tokenize(sentence):
                    if word.lower() not in string.punctuation and word.lower() not in self.stop_words:
                        words.append(word.lower())

                text_sentences.append(words)

            tokenized_words.append(text_sentences)

        return tokenized_words

    def lemmatize(self):
        lemmatized_words = []

        for text in self.tokenized_words:
            text_sentences = []

            for sentence in text:
                words = []

                for word in sentence:
                    words.append(self.lemmatizer.lemmatize(word))

                text_sentences.append(words)

            lemmatized_words.append(text_sentences)

        return lemmatized_words

    def extract_data(self):
        datasets = []

        for text in self.lemmatized_words:
            dataset = []

            for sentence in text:
                dataset.append(" ".join(sentence))
            datasets.append(dataset)

        return datasets

    def count_sentence_probabilities(self):
        text_sentence_probs = []

        for dataset, idx in zip(self.datasets, range(len(self.datasets))):
            sentence_probs = []

            tf_idf_matrix = self.model.fit_transform(dataset)
            dframe = pd.DataFrame(tf_idf_matrix.toarray(), columns=self.model.get_feature_names_out())

            for sentence_idx in range(len(self.text_sentences[idx])):
                sentence_prob = []

                for prob, word_idx in zip(dframe.iloc[sentence_idx], range(len(dframe.iloc[sentence_idx]))):
                    if prob > 0:
                        sentence_prob.append(prob)

                sentence_probs.append(np.mean(sentence_prob))

            text_sentence_probs.append(sentence_probs)

        return text_sentence_probs

    def select_sentences(self):
        selected_sentence_indexes = []
        selected_sentences = []

        for text, idx in zip(self.sentence_probs, range(len(self.sentence_probs))):
            selected_index = []

            for i in range(self.sqrt_lengths[idx]):
                index = text.index(min(text))
                selected_index.append(index)
                text[index] = 1
            selected_sentence_indexes.append(sorted(selected_index))

        for text, idx in zip(self.text_sentences, range(len(self.text_sentences))):
            sentences = []

            for i in selected_sentence_indexes[idx]:
                sentences.append(text[i])

            selected_sentences.append(sentences)

        return selected_sentences

    def count_modified_probabilities(self):
        text_sentence_probs = []
        extra_weight = 3

        for dataset, idx, header in zip(self.datasets, range(len(self.datasets)), self.headers):
            sentence_probs = []
            header_words = []

            for word in word_tokenize(header.text):
                if word.lower() not in string.punctuation and word.lower() not in self.stop_words:
                    header_words.append(word.lower())

            for header_word_idx in range(len(header_words)):
                header_words[header_word_idx] = self.lemmatizer.lemmatize(header_words[header_word_idx])

            tf_idf_matrix = self.model.fit_transform(dataset)
            dframe = pd.DataFrame(tf_idf_matrix.toarray(), columns=self.model.get_feature_names_out())

            for sentence_idx in range(len(self.text_sentences[idx])):
                sentence_prob = []

                for prob, word_idx in zip(dframe.iloc[sentence_idx], range(len(dframe.iloc[sentence_idx]))):

                    if prob > 0:
                        if dframe.iloc[sentence_idx].index[word_idx] in header_words:
                            sentence_prob.append(prob / extra_weight)
                        else:
                            sentence_prob.append(prob)

                sentence_probs.append(np.mean(sentence_prob))

            text_sentence_probs.append(sentence_probs)

        return text_sentence_probs

    def modified_model_select_sentences(self):
        selected_sentence_indexes = []
        selected_sentences = []

        for text, idx in zip(self.modified_probabilities, range(len(self.modified_probabilities))):
            selected_index = []

            for i in range(self.sqrt_lengths[idx]):
                index = text.index(min(text))
                selected_index.append(index)
                text[index] = 1
            selected_sentence_indexes.append(sorted(selected_index))

        for text, idx in zip(self.text_sentences, range(len(self.text_sentences))):
            sentences = []

            for i in selected_sentence_indexes[idx]:
                sentences.append(text[i])

            selected_sentences.append(sentences)

        return selected_sentences


xml_file = open("news.xml", 'r').read()

"""
SumBasic Algorithm Result

summarizer = TextSummarizer(xml_file)

for header, txt in zip(summarizer.headers, summarizer.sorted_sentences):
    final_text = " \n".join(txt)
    print(f"HEADER: {header.text}")
    print(f"TEXT: {final_text}")
    print()
"""

"""
TF-IDF Vectorizer Result

vectorizer = Vectorizer(xml_file)

for header, txt in zip(vectorizer.headers, vectorizer.selected_sentences):
    final_text = " \n".join(txt)
    print(f"HEADER: {header.text}")
    print(f"TEXT: {final_text}")
    print()
"""

# Modified TF-IDF Vectorizer Result
vectorizer = Vectorizer(xml_file)

for hdr, txt in zip(vectorizer.headers, vectorizer.modified_model_sentences):
    final_text = " \n".join(txt)
    print(f"HEADER: {hdr.text}")
    print(f"TEXT: {final_text}")
    print()
