import os
import re
import csv
import string
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering


def computeTF(wordDict, bagOfWords):
    tfDict = {}
    bagOfWordsCount = len(bagOfWords)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bagOfWordsCount)
    return tfDict


def computeIDF(documents):
    N = len(documents)
    idfDict = dict.fromkeys(documents[0].keys(), 0)
    for document in documents:
        for word, val in document.items():
            if val > 0:
                idfDict[word] += 1

    for word, val in idfDict.items():
        idfDict[word] = math.log(N / float(val))
    return idfDict


def computeTFIDF(tfBagOfWords, idfs):
    tfidf = {}
    for word, val in tfBagOfWords.items():
        tfidf[word] = val * idfs[word]
    return tfidf


def remove_digits(list):
    pattern = '[0-9]'
    list = [re.sub(pattern, '', i) for i in list]
    return list


if __name__ == '__main__':


    # --------------------------------------- GENERAL STATS---------------------------------------


    # load dataset:
    df = pd.read_csv('COVID19.csv')

    # count infected:
    inf_count = df[['country', 'id']].groupby(['country']).agg(['count'])

    # avg. age:
    avg_age = df[['country', 'id', 'age']].fillna(df.mean()).groupby(['country']).agg(['mean']).\
        drop(['id'], axis=1)

    # megre data & rename columns:
    stats = pd.merge(inf_count, avg_age, on='country', how='inner')
    stats = stats.reset_index().rename(columns={"id": "infected_count",
                                                "age": "average_age"})
    stats.columns = stats.columns.droplevel(1)

    # leave only highly infected countries:
    stats = stats[stats['infected_count'] > 30]

    # sort by inf_count:
    stats = stats.sort_values(by=['infected_count'], ascending=False)

    # add male-female ratio to stats:
    female_count = df.groupby('country')['gender'].apply(lambda x: (x == 'female').sum()). \
        reset_index(name='female_count')
    stats = pd.merge(stats, female_count, on='country', how='inner')
    stats['female_percent'] = 100 * (stats.female_count / stats.infected_count)
    stats = stats.drop(columns=['female_count'])

    # write stats to txt file:
    project_path = os.getcwd()
    with open('country_stats.txt', 'w') as file:
        file.write(stats.to_string())


    # ---------------------------------------LABLES PER COUNTRY---------------------------------------


    # vector rep. for 'summary' column:
    exclude = set(string.punctuation)
    stop_path = 'stopwords.txt'
    stop_list = open(stop_path, 'r')
    reader = csv.reader(stop_list)
    stop = []
    for row in reader:
        if len(row) > 0:
            stop.append(row[0])

    df['summary_upd'] = df['summary'].astype(str) \
        .apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

    uniqueWords = {}
    summaries = []
    for i, row in df.iterrows():
        row['summary_upd'] = ''.join(ch for ch in row['summary_upd'] if ch not in exclude)
        new_bag = row['summary_upd'].split(' ')
        summaries.append(new_bag)
        uniqueWords = set(new_bag).union(uniqueWords)

    vectors = []
    for s in summaries:
        vec = dict.fromkeys(uniqueWords, 0)
        for word in s:
            vec[word] += 1
        vectors.append(vec)

    tf_vector = []
    for i in range(len(vectors)):
        tf_vector.append(computeTF(vectors[i], summaries[i]))

    idfs = computeIDF(vectors)

    tfidf = []
    for i in range(len(vectors)):
        tfidf.append(computeTFIDF(tf_vector[i], idfs))

    df_tfidf = pd.DataFrame(tfidf)

    # foreach sum, top k labels:
    k = 5
    c = []
    for i in range(1, k+1):
        c.append(str(i) + ' Max')
    sum_lables = (df_tfidf
          .apply(lambda x: pd.Series(x.nlargest(k).index, index=c), axis=1)
          .reset_index())

    # foreach country, create a list of it's all summeries top labels:
    ctry_lbl_dct = {}
    for country in stats['country']:
        idx = df.index[df['country'] == country].tolist()
        bag = []
        for i, row in sum_lables.iloc[idx].iterrows():
            bag += list(row)
        bag = list(set(bag))
        bag = [str(i) for i in bag]
        bag = list(set(remove_digits(bag)))
        if "" in bag: bag.remove("")

        bag_copy = bag.copy()
        for s in bag_copy:
            if s[0].isupper():
                bag.remove(s)

        ctry_lbl_dct[country] = bag

    # tfidf on ctry_lbl_dct:
    uniqueWords = []

    for k, v in ctry_lbl_dct.items():
        uniqueWords += v
    uniqueWords = set(uniqueWords)

    vectors = {}
    for k, v in ctry_lbl_dct.items():
        vec = dict.fromkeys(uniqueWords, 0)
        for word in v:
            vec[word] += 1
        vectors[k] = vec

    tf_vector = {}
    for k, v in ctry_lbl_dct.items():
        tf_vector[k] = computeTF(vectors[k], ctry_lbl_dct[k])

    idfs = computeIDF(list(vectors.values()))

    tfidf = {}
    for k, v in vectors.items():
        tfidf[k] = computeTFIDF(tf_vector[k], idfs)

    # foreach country, top t labels:
    t = 15
    c = []
    for i in range(1, t+1):
        c.append(str(i) + ' Max')

    country_lables = {}
    for k, v in tfidf.items():
        country_lables[k] = list(pd.Series(v).nlargest(t).index)

    # write results to txt file:
    project_path = os.getcwd()
    with open('country_lables.txt', 'w') as file:
        for k, v in country_lables.items():
            file.write(k + ":\n")
            file.write(str(v) + "\n")


    # ---------------------------------------SUMMARIES CLUSTERING---------------------------------------


    # summaries clustering:
    k = 10
    cluster = AgglomerativeClustering(n_clusters=k, affinity='cosine', linkage='complete')
    cluster.fit(df_tfidf)

    # create a dict of k: cluster number , v: idx list
    cluster_dict = {}
    for i in range(k):
        cluster_dict[i] = []
        for j, val in enumerate(cluster.labels_):
            if val == i:
                cluster_dict[i].append(j)

    cluster_dict_df = {}
    for k, v in cluster_dict.items():
        cluster_dict_df[k] = df.loc[v,'summary']

    # write clusters to txt file:
    project_path = os.getcwd()
    with open('summaries_clusters.txt', 'w') as file:
        for k, v in cluster_dict_df.items():
            file.write('---------------------Clustr no. ' + str(k + 1) + '---------------------')
            file.write('\n\n')
            file.write(v.to_csv(header=False, index=False))
            file.write('(Items in cluster: ' + str(v.shape[0]) + ')')
            file.write('\n\n')


    # ---------------------------------------LABLES PER COUNTRY---------------------------------------


    # foreach cluster create list of lables (using tf idf):
    cluster_lbl_dict = {}
    for cluster_num, idx_lst in cluster_dict.items():
        bag = []
        for i, row in sum_lables.iloc[idx_lst].iterrows():
            bag += list(row)
        bag = list(set(bag))
        bag = [str(i) for i in bag]
        bag = list(set(remove_digits(bag)))
        if "" in bag: bag.remove("")

        bag_copy = bag.copy()
        for s in bag_copy:
            if s[0].isupper():
                bag.remove(s)

        cluster_lbl_dict[cluster_num] = bag

    # tfidf on cluster_lbl_dct:
    uniqueWords = []

    for k, v in cluster_lbl_dict.items():
        uniqueWords += v
    uniqueWords = set(uniqueWords)

    vectors = {}
    for k, v in cluster_lbl_dict.items():
        vec = dict.fromkeys(uniqueWords, 0)
        for word in v:
            vec[word] += 1
        vectors[k] = vec

    tf_vector = {}
    for k, v in cluster_lbl_dict.items():
        tf_vector[k] = computeTF(vectors[k], cluster_lbl_dict[k])

    idfs = computeIDF(list(vectors.values()))

    tfidf = {}
    for k, v in vectors.items():
        tfidf[k] = computeTFIDF(tf_vector[k], idfs)

    # foreach cluster, top t labels:
    t = 5
    c = []
    for i in range(1, t + 1):
        c.append(str(i) + ' Max')

    cluster_lables = {}
    for k, v in tfidf.items():
        cluster_lables[k] = list(pd.Series(v).nlargest(t).index)

    # write results to txt file:
    project_path = os.getcwd()
    with open('cluster_lables.txt', 'w') as file:
        for k, v in cluster_lables.items():
            file.write("Cluster no. " + str(k+1) + ":\n")
            file.write(str(v) + "\n")
