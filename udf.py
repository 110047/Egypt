import matplotlib.cm as cm
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn import cluster
from sklearn.metrics import silhouette_samples, silhouette_score
from wordcloud import WordCloud
from sklearn.feature_extraction.text  import TfidfVectorizer
import udf


class transform:
    def __init__(self):
        return
# list to string
    def ListToString(list):
        string_words = ' '.join(list)
        return string_words

# string to list
    def StringToList(string):
        listRes = list(string.split(" "))
        return listRes




from nltk import WordNetLemmatizer
StopWords = stopwords.words("english")
StopWords.extend(["u","from"])


class clean:
    def __init__(self):
        return

    def clean_text(text):
        """
        This function takes as input a text on which several
        NLTK algorithms will be applied in order to preprocess it
        """
        tokens = word_tokenize(text)
        # Remove the punctuations
        tokens = [word for word in tokens if word.isalpha()]
        # Lower the tokens
        tokens = [word.lower() for word in tokens]
        # Remove stopword
        tokens = [word for word in tokens if not word in StopWords]
        # Lemmatize
        lemma = WordNetLemmatizer()
        tokens = [lemma.lemmatize(word, pos="v") for word in tokens]
        tokens = [lemma.lemmatize(word, pos="n") for word in tokens]
        # list to string
        text = " ".join(tokens)
        return text

    # Extract nouns from speeches
    def nouns_extract(cont):
        nouns = []  # empty to array to hold all nouns
        cont = udf.transform.StringToList(cont)
        for word, pos in nltk.pos_tag(cont):
            if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'):
                nouns.append(word)
                string_nouns = udf.transform.ListToString(nouns)
        return string_nouns


class scrab:
    def __init__(self):
        return

    def extract_text(url):
        headers = {
            'User-Agent': "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.6; rv:2.0.1) Gecko/20100101 Firefox/4.0.1"
        }
        resp = requests.get(url, headers=headers)
        s = BeautifulSoup(resp.text, "html.parser")
        title = s.title
        text = s.get_text(strip=True)
        return title, text

    def title(title):
        title = str(title)
        title = title[title.find("Obama -") + len("Obama -"):title.find("</title>")]
        title = title[:title.find("(transcript-audio-video)")]
        title = title[:title.find("(text-audio-video)")].strip()
        return title

    ## delete something like cd, pdf ...
    def allowed(speech):
        allowed = speech[speech.find("transcribed directly from audio]")
                         + len("transcribed directly from audio]"):speech.find(
            "Book/CDs by Michael E. Eidenmuller,")].strip()
        return allowed


class k_means:
    def __init__(self):
        return

    def run_KMeans(max_k, data):
        max_k += 1
        kmeans_results = dict()
        for k in range(2, max_k):
            kmeans = cluster.KMeans(n_clusters=k
                                    , init='k-means++'
                                    , n_init=10
                                    , tol=0.0001
                                    , random_state=1
                                    , algorithm='lloyd')

            kmeans_results.update({k: kmeans.fit(data)})

        return kmeans_results

    def printAvg(avg_dict):
        for avg in sorted(avg_dict.keys(), reverse=True):
            print("Avg: {}\tK:{}".format(avg.round(4), avg_dict[avg]))

    def plotSilhouette(df, n_clusters, kmeans_labels, silhouette_avg):
        fig, ax1 = plt.subplots(1)
        fig.set_size_inches(8, 6)
        ax1.set_xlim([-0.2, 1])
        ax1.set_ylim([0, len(df) + (n_clusters + 1) * 10])

        ax1.axvline(x=silhouette_avg, color="red",
                    linestyle="--")  # The vertical line for average silhouette score of all the values
        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
        plt.title(("Silhouette analysis for K = %d" % n_clusters), fontsize=10, fontweight='bold')

        y_lower = 10
        sample_silhouette_values = silhouette_samples(df,
                                                      kmeans_labels)  # Compute the silhouette scores for each sample
        for i in range(n_clusters):
            ith_cluster_silhouette_values = sample_silhouette_values[kmeans_labels == i]
            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color,
                              edgecolor=color, alpha=0.7)

            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i,
                     str(i))  # Label the silhouette plots with their cluster numbers at the middle
            y_lower = y_upper + 10  # Compute the new y_lower for next plot. 10 for the 0 samples
        plt.show()

    def silhouette(kmeans_dict, df, plot=False):
        df = df.to_numpy()
        avg_dict = dict()
        for n_clusters, kmeans in kmeans_dict.items():
            kmeans_labels = kmeans.predict(df)
            silhouette_avg = silhouette_score(df, kmeans_labels)  # Average Score for all Samples
            avg_dict.update({silhouette_avg: n_clusters})

            if (plot): udf.k_means.plotSilhouette(df, n_clusters, kmeans_labels, silhouette_avg)

#    def get_top_features_cluster(tf_idf_array, prediction, n_feats):
#        labels = np.unique(prediction)
#        dfs = []
#        for label in labels:
#            id_temp = np.where(prediction == label)  # indices for each cluster
#            x_means = np.mean(tf_idf_array[id_temp], axis=0)  # returns average score across cluster
#            sorted_means = np.argsort(x_means)[::-1][:n_feats]  # indices with top 20 scores
#            vectorizer = TfidfVectorizer()
#            features = vectorizer.get_feature_names()
#            best_features = [(features[i], x_means[i]) for i in sorted_means]
#            df = pd.DataFrame(best_features, columns=['features', 'score'])
#            dfs.append(df)
#        return dfs

#    def plotWords(dfs, n_feats):
#        plt.figure(figsize=(8, 4))
#        for i in range(0, len(dfs)):
#            plt.title(("Most Common Words in Cluster {}".format(i)), fontsize=10, fontweight='bold')
#            sns.barplot(x='score', y='features', orient='h', data=dfs[i][:n_feats])
#            plt.show()

    # Transforms a centroids dataframe into a dictionary to be used on a WordCloud.
    def centroidsDict(centroids, index):
        a = centroids.T[index].sort_values(ascending=False).reset_index().values
        centroid_dict = dict()

        for i in range(0, len(a)):
            centroid_dict.update({a[i, 0]: a[i, 1]})

        return centroid_dict

    def generateWordClouds(centroids):
        wordcloud = WordCloud(max_font_size=100, background_color='white')
        for i in range(0, len(centroids)):
            centroid_dict = udf.k_means.centroidsDict(centroids, i)
            wordcloud.generate_from_frequencies(centroid_dict)

            plt.figure()
            plt.title('Cluster {}'.format(i))
            plt.imshow(wordcloud)
            plt.axis("off")
            plt.show()


