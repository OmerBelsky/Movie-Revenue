import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn
seaborn.set()
from ast import literal_eval
from functools import partial
import xgboost as xgb
from time import time
from gensim.models import KeyedVectors
import string
import pickle
from collections import Counter

def rmsle(y_true, y_pred):
    """
    Calculates Root Mean Squared Logarithmic Error between two input vectors
    :param y_true: 1-d array, ground truth vector
    :param y_pred: 1-d array, prediction vector
    :return: float, RMSLE score between two input vectors
    """
    assert y_true.shape == y_pred.shape, \
        ValueError("Mismatched dimensions between input vectors: {}, {}".format(y_true.shape, y_pred.shape))
    return np.sqrt((1/len(y_true)) * np.sum(np.power(np.log(y_true + 1) - np.log(y_pred + 1), 2)))

def important_words(X_words, target):
    """
    Meant for text fields.
    Trains a linear regression model over the tf-idf vectors of the text
    and returns top 10 words w.r.t feature importance.
    """

    word_importance_model = LinearRegression()
    word_importance_model.fit(X_words, target)
    abs_coef = np.absolute(word_importance_model.coef_)
    top_words = np.argsort(abs_coef)[:10]
    return top_words

def get_count_occur(row, names):
    """
    Meant for crew and cast columns apply.
    Given a list of names (oscar winners) returns the number of
    occurences of those names
    """

    row_list = literal_eval(row)
    count = 0
    for person_dict in row_list:
        count += int(person_dict["name"] in names)
    return count

def keep_train_words():
    """
    Downloaded a massive model trained on all of wikipedia, this function
    keeps only the words that appear in the train set. so that
    loading the model doesn't take 5 minutes.
    """

    w2v_model = KeyedVectors.load_word2vec_format("wiki-news-300d-1M-subword.vec", binary=False)
    features = ["original_title", "overview", "Keywords"]
    data = pd.read_csv("train.tsv", sep="\t")[features]

    translator = str.maketrans({punc: None for punc in string.punctuation})
    words = set()
    data["overview"].fillna("").apply(lambda x: words.update([word.translate(translator).lower() for word in x.split()]))
    data["Keywords"] = data["Keywords"].fillna("[]").apply(lambda x: " ".join([word_dict["name"] for word_dict in literal_eval(x)]))
    data["Keywords"].apply(lambda x: words.update([word.translate(translator).lower() for word in x.split()]))
    data["original_title"].apply(lambda x: words.update([word.translate(translator).lower() for word in x.split()]))

    train_word_vecs = {word: w2v_model[word] for word in words if word in w2v_model.vocab}

    with open("train_word_vecs.pkl", 'wb') as f:
        pickle.dump(train_word_vecs, f, pickle.HIGHEST_PROTOCOL)

def create_clusters(w2v_dict, K=10):
    """
    Creates K clusters from a word2vec dictionary of {word: vec} pairs.
    """

    km = KMeans(K)
    km.fit(list(w2v_dict.values()))
    word_to_cluster = {word: km.predict([vec])[0] for word, vec in w2v_dict.items()}
    return word_to_cluster


def main():

    """
    Feature engineering and some commented EDA.
    """

    start = time()
    features = ["revenue", "original_language", "spoken_languages", "belongs_to_collection", "original_title", "overview", "Keywords", "popularity", "vote_average", "vote_count", "cast", "crew", "release_date", "budget"]  # , "genres"]
    data = pd.read_csv("train.tsv", sep="\t")[features]
    target = data["revenue"]
    # W2V_K = 10
    # translator = str.maketrans({punc: None for punc in string.punctuation})
    # with open("train_word_vecs.pkl", 'rb') as f:
    #     w2v_dict = pickle.load(f)

    # genres = Counter()
    # data["genres"].apply(lambda x: genres.update([genre["name"] for genre in literal_eval(x)]))
    # print(genres.items())
    # seaborn.heatmap(data.drop("revenue", axis=1).corr())
    # plt.show()
    # for q in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    #     print(data["budget"].quantile(q), data["revenue"].quantile(q))
    # exit()

    ###############
    # Preprocessing
    ###############

    data["spoken_languages"] = data['spoken_languages'].apply(lambda x: literal_eval(x))
    data["overview"] = data["overview"].fillna("")
    data["Keywords"] = data["Keywords"].fillna("[]").apply(lambda x: " ".join([word_dict["name"] for word_dict in literal_eval(x)]))
    data["no_budget"] = (data["budget"] == 0).astype(int)

    # data["original_title"] = data["original_title"].apply(lambda x: [word.translate(translator).lower() for word in x.split()])
    # data["overview"] = data["overview"].apply(lambda x: [word.translate(translator).lower() for word in x.split()])
    # data["Keywords"] = data["Keywords"].apply(lambda x: [word.translate(translator).lower() for word in x.split()])

    ########################
    # Transform release_date
    ########################

    seasons = {1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring", 6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn", 12: "Winter"}
    data["release_month"] = data["release_date"].apply(lambda x: int(x.split("-")[1]))
    data["release_season"] = data["release_month"].apply(lambda x: seasons[x])
    data.drop("release_date", inplace=True, axis=1)

    #######################
    # Transform cast & crew
    #######################

    award_data = pd.read_csv("oscars-demographics.csv", sep=";")
    cast_awards = ["Best Supporting Actor", "Best Supporting Actress", "Best Actor", "Best Actress"]
    best_cast = set(award_data[award_data["Award"].isin(cast_awards)]["Person"])
    best_crew = set(award_data[~award_data["Award"].isin(cast_awards)]["Person"])
    cast_occur = partial(get_count_occur, names=best_cast)
    crew_occur = partial(get_count_occur, names=best_crew)
    data["best_cast_count"] = data["cast"].apply(cast_occur)
    data["best_crew_count"] = data["crew"].apply(crew_occur)
    data.drop("cast", inplace=True, axis=1)
    data.drop("crew", inplace=True, axis=1)

    ######################
    # Transform vote_count
    ######################

    data["log_vote_count"] = np.log(data["vote_count"] + 1)
    data.drop("vote_count", inplace=True, axis=1)

    #############################
    # Transform original_language
    #############################

    languages = ["en", "fr", "hi", "ja", "es"]
    data["filtered_lang"] = data["original_language"].apply(lambda x: x if x in languages else "other")
    data.drop("original_language", inplace=True, axis=1)

    #################################
    # Transform belongs_to_collection
    #################################

    data["belongs_to_collection__missing"] = data["belongs_to_collection"].isna().astype(int)
    data.drop("belongs_to_collection", inplace=True, axis=1)

    ############################
    # Transform spoken_languages
    ############################

    data["num_spoken"] = data['spoken_languages'].apply(lambda x: len(x))
    data.drop("spoken_languages", inplace=True, axis=1)

    print(f"Finished Global feature engineering in: {round((time() - start) / 60.0, 2)} minutes.")
    
    train_rmsle = []
    test_rmsle = []
    for i in range(5):
        cv_start = time()
        X_train, X_val, y_train, y_val = train_test_split(data.drop("revenue", axis=1), target, random_state=i)
        normal_revenue_idx = np.where(y_train > 10000)[0]
        X_train, y_train = X_train.iloc[normal_revenue_idx], y_train.iloc[normal_revenue_idx]
        # train_words = set()
        # X_train["original_title"].apply(lambda x: train_words.update(set(x)))
        # X_train["overview"].apply(lambda x: train_words.update(set(x)))
        # X_train["Keywords"].apply(lambda x: train_words.update(set(x)))
        # train_w2v_dict = {word: w2v_dict[word] for word in train_words if word in w2v_dict}
        # word_clusters = create_clusters(train_w2v_dict, W2V_K)
        
        # seaborn.heatmap(X_train.drop(["original_title", "overview", "Keywords"], axis=1).corr(), annot=True)
        # plt.show()

        ##########################
        # Transform original_title
        ##########################

        # tfidf
        tfidf = TfidfVectorizer(decode_error="ignore", strip_accents="unicode")
        tfidf.fit(X_train["original_title"])

        # get important words, transform train.
        X_words = tfidf.transform(X_train["original_title"])
        i_words = important_words(X_words, y_train)
        word_list = tfidf.get_feature_names()
        word_df = pd.DataFrame(X_words[:, i_words].toarray(), columns=["original_title__" + word_list[i] for i in i_words], index=X_train.index)
        X_train = pd.concat([X_train, word_df], axis=1)
        
        # transform test.
        X_words = tfidf.transform(X_val["original_title"])
        word_df = pd.DataFrame(X_words[:, i_words].toarray(), columns=["original_title__" + word_list[i] for i in i_words], index=X_val.index)
        X_val = pd.concat([X_val, word_df], axis=1)

        # # Word2Vec
        # clusters_train = X_train["original_title"].apply(lambda x: Counter([word_clusters[word] if word in word_clusters else W2V_K for word in x]))
        # clusters_train = pd.DataFrame(clusters_train.apply(lambda counter: [counter[i] if i in counter else 0 for i in range(W2V_K + 1)]).values.tolist(), columns=["original_title_cluster_" + str(i) for i in range(W2V_K + 1)], index=X_train.index)
        # X_train = pd.concat([X_train, clusters_train], axis=1)

        # clusters_val = X_val["original_title"].apply(lambda x: Counter([word_clusters[word] if word in word_clusters else W2V_K for word in x]))
        # clusters_val = pd.DataFrame(clusters_val.apply(lambda counter: [counter[i] if i in counter else 0 for i in range(W2V_K + 1)]).values.tolist(), columns=["original_title_cluster_" + str(i) for i in range(W2V_K + 1)], index=X_val.index)
        # X_val = pd.concat([X_val, clusters_val], axis=1)

        X_train.drop("original_title", inplace=True, axis=1)
        X_val.drop("original_title", inplace=True, axis=1)

        ####################
        # Transform overview
        ####################
        
        # tfidf
        tfidf = TfidfVectorizer(decode_error="ignore", strip_accents="unicode")
        tfidf.fit(X_train["overview"])


        # get important words, transform train.
        X_words = tfidf.transform(X_train["overview"])
        i_words = important_words(X_words, y_train)
        word_list = tfidf.get_feature_names()
        word_df = pd.DataFrame(X_words[:, i_words].toarray(), columns=["overview__" + word_list[i] for i in i_words], index=X_train.index)
        X_train = pd.concat([X_train, word_df], axis=1)
        
        # transform test.
        X_words = tfidf.transform(X_val["overview"])
        word_df = pd.DataFrame(X_words[:, i_words].toarray(), columns=["overview__" + word_list[i] for i in i_words], index=X_val.index)
        X_val = pd.concat([X_val, word_df], axis=1)

        # # Word2Vec
        # clusters_train = X_train["overview"].apply(lambda x: Counter([word_clusters[word] if word in word_clusters else W2V_K for word in x]))
        # clusters_train = pd.DataFrame(clusters_train.apply(lambda counter: [counter[i] if i in counter else 0 for i in range(W2V_K + 1)]).values.tolist(), columns=["overview_cluster_" + str(i) for i in range(W2V_K + 1)], index=X_train.index)
        # X_train = pd.concat([X_train, clusters_train], axis=1)

        # clusters_val = X_val["overview"].apply(lambda x: Counter([word_clusters[word] if word in word_clusters else W2V_K for word in x]))
        # clusters_val = pd.DataFrame(clusters_val.apply(lambda counter: [counter[i] if i in counter else 0 for i in range(W2V_K + 1)]).values.tolist(), columns=["overview_cluster_" + str(i) for i in range(W2V_K + 1)], index=X_val.index)
        # X_val = pd.concat([X_val, clusters_val], axis=1)

        X_train.drop("overview", inplace=True, axis=1)
        X_val.drop("overview", inplace=True, axis=1)

        ####################
        # Transform Keywords
        ####################
        
        # tfidf
        tfidf = TfidfVectorizer(decode_error="ignore", strip_accents="unicode")
        tfidf.fit(X_train["Keywords"])

        # get important words, transform train.
        X_words = tfidf.transform(X_train["Keywords"])
        i_words = important_words(X_words, y_train)
        word_list = tfidf.get_feature_names()
        word_df = pd.DataFrame(X_words[:, i_words].toarray(), columns=["Keywords__" + word_list[i] for i in i_words], index=X_train.index)
        X_train = pd.concat([X_train, word_df], axis=1)
        
        # transform test.
        X_words = tfidf.transform(X_val["Keywords"])
        word_df = pd.DataFrame(X_words[:, i_words].toarray(), columns=["Keywords__" + word_list[i] for i in i_words], index=X_val.index)
        X_val = pd.concat([X_val, word_df], axis=1)

        # #Word2Vec
        # clusters_train = X_train["Keywords"].apply(lambda x: Counter([word_clusters[word] if word in word_clusters else W2V_K for word in x]))
        # clusters_train = pd.DataFrame(clusters_train.apply(lambda counter: [counter[i] if i in counter else 0 for i in range(W2V_K + 1)]).values.tolist(), columns=["Keywords_cluster_" + str(i) for i in range(W2V_K + 1)], index=X_train.index)
        # X_train = pd.concat([X_train, clusters_train], axis=1)

        # clusters_val = X_val["Keywords"].apply(lambda x: Counter([word_clusters[word] if word in word_clusters else W2V_K for word in x]))
        # clusters_val = pd.DataFrame(clusters_val.apply(lambda counter: [counter[i] if i in counter else 0 for i in range(W2V_K + 1)]).values.tolist(), columns=["Keywords_cluster_" + str(i) for i in range(W2V_K + 1)], index=X_val.index)
        # X_val = pd.concat([X_val, clusters_val], axis=1)

        X_train.drop("Keywords", inplace=True, axis=1)
        X_val.drop("Keywords", inplace=True, axis=1)

        #######
        # Model
        #######

        ohe = OneHotEncoder(sparse=False)
        ohe.fit(X_train[["release_season", "filtered_lang"]])
        X_train_ohe = pd.DataFrame(ohe.transform(X_train[["release_season", "filtered_lang"]]), columns=[col.replace("x0", "release_season_").replace("x1", "filtered_lang_") for col in ohe.get_feature_names()], index=X_train.index)
        X_train.drop(["release_season", "filtered_lang"], inplace=True, axis=1)
        X_train = pd.concat([X_train, X_train_ohe], axis=1)
        X_val_ohe = pd.DataFrame(ohe.transform(X_val[["release_season", "filtered_lang"]]), columns=[col.replace("x0", "release_season_").replace("x1", "filtered_lang_") for col in ohe.get_feature_names()], index=X_val.index)
        X_val.drop(["release_season", "filtered_lang"], inplace=True, axis=1)
        X_val = pd.concat([X_val, X_val_ohe], axis=1)

        # plt.figure(figsize=(23, 13))
        # seaborn.heatmap(X_train.corr(), annot=True)
        # plt.show()

        params = {
            "n_estimators": [300],
            "colsample_bytree": [0.8],
            "max_depth": [5, 6],
            "learning_rate": [0.05, 0.1],
            "random_state": [0],
            "reg_alpha": [0.3],
            "reg_lambda": [0.3],
            "gamma": [0, 0.1, 0.2],
            "min_child_weight": [1, 10, 20]
        }

        def neg_rmsle(y_true, y_pred):
            y_pred[y_pred < 0] = 0
            return rmsle(y_true, y_pred)

        regressor = GridSearchCV(xgb.XGBRegressor(), scoring=make_scorer(neg_rmsle, greater_is_better=False), param_grid=params, cv=3)
        # regressor = xgb.XGBRegressor(n_estimators=300, colsample_bytree=0.8, max_depth=5, learning_rate=0.01, random_state=0, reg_alpha=0.5, reg_lambda=0.6)
        regressor.fit(X_train, y_train)
        print(regressor.best_params_)
        print(regressor.best_score_)
        y_pred_train = regressor.best_estimator_.predict(X_train)
        y_pred_train[y_pred_train < 0] = 0
        train_rmsle.append(rmsle(y_train, y_pred_train))

        y_pred_val = regressor.best_estimator_.predict(X_val)
        y_pred_val[y_pred_val < 0] = 0
        test_rmsle.append(rmsle(y_val, y_pred_val))
        f_imp = regressor.best_estimator_.feature_importances_
        print("\n".join([str(x) for x in sorted(zip(X_train.columns, f_imp), key=lambda tmp: tmp[1]) if abs(x[1]) > 0]))
        
        y_train, y_pred_train = zip(*sorted(zip(y_train, y_pred_train), key=lambda x: x[0]))
        # plt.plot(y_train, y_pred_train)
        # plt.title("True Revenue vs. Predicted Revenue In Train Set")
        # plt.xlabel("True Revenue")
        # plt.ylabel("Predicted Revenue")
        # plt.show()

        y_val, y_pred_val = zip(*sorted(zip(y_val, y_pred_val), key=lambda x: x[0]))
        # plt.plot(y_val, y_pred_val)
        # plt.title("True Revenue vs. Predicted Revenue In Val Set")
        # plt.xlabel("True Revenue")
        # plt.ylabel("Predicted Revenue")
        # plt.show()

        print(f"Finished CV Fold {i} in: {round((time() - cv_start) / 60.0, 2)} minutes.")
    print(train_rmsle)
    print(test_rmsle)


if __name__ == "__main__":
    main()
    # keep_train_words()
