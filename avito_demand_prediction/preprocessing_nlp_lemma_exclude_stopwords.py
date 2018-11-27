# -*- coding: utf-8 -*-

import pandas as pd

__author__ = "ujihirokazuya"


pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999


train_df = pd.read_csv("preprocessed_nlp_lemma_train.csv")
test_df = pd.read_csv("preprocessed_nlp_lemma_test.csv")

stopwords = {'какая', 'меня', 'можно', 'нас', 'от', 'тогда', 'ему', 'им', 'ничего', 'один', 'впрочем', 'чем', 'если',
             'чтоб', 'другой', 'нет', 'ведь', 'разве', 'а', 'только', 'еще', 'было', 'конечно', 'все', 'с', 'будет',
             'там', 'потому', 'так', 'сейчас', 'об', 'ей', 'при', 'этой', 'вдруг', 'чуть', 'такой', 'сам', 'надо',
             'него', 'лучше', 'иногда', 'была', 'эти', 'нельзя', 'нее', 'даже', 'ли', 'тем', 'когда', 'мой', 'был',
             'что', 'может', 'них', 'никогда', 'по', 'из', 'ней', 'ж', 'эту', 'он', 'со', 'много', 'мне', 'опять',
             'тот', 'про', 'над', 'наконец', 'себе', 'она', 'моя', 'уже', 'нибудь', 'были', 'ни', 'ну', 'между',
             'вот', 'они', 'вы', 'больше', 'за', 'на', 'не', 'теперь', 'же', 'этого', 'есть', 'этот', 'раз', 'где',
             'вам', 'два', 'совсем', 'в', 'для', 'во', 'какой', 'чтобы', 'или', 'бы', 'чего', 'том', 'потом', 'мы',
             'ты', 'будто', 'кто', 'у', 'тебя', 'себя', 'да', 'хоть', 'быть', 'его', 'их', 'тоже', 'всех', 'то',
             'вас', 'и', 'этом', 'куда', 'как', 'после', 'я', 'более', 'но', 'ее', 'без', 'тут', 'хорошо', 'перед',
             'здесь', 'под', 'три', 'того', 'всегда', 'о', 'свою', 'почти', 'через', 'всего', 'до', 'ним', 'уж',
             'всю', 'зачем', 'к'}


def _replace_word(word: str):
    words = word.split(" ")
    new_words = [word for word in words if word not in stopwords]
    return " ".join(new_words)


def _remove(data_frame):
    data_frame["title_description"] = data_frame["title_description"].map(_replace_word)
    return data_frame


train_df = _remove(train_df)
test_df = _remove(test_df)


train_df.to_csv("preprocessed_nlp_lemma_stop_train.csv", index=False)
test_df.to_csv("preprocessed_nlp_lemma_stop_test.csv", index=False)
