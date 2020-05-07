import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from ast import literal_eval
from functools import partial
import xgboost as xgb
import pickle
import os
os.chdir("./")
from main import get_count_occur, important_words

def make_model():
    features = ["revenue", "original_language", "spoken_languages", "belongs_to_collection", "original_title", "overview", "Keywords", "popularity", "vote_average", "vote_count", "cast", "crew", "release_date", "budget"]  # , "genres"]
    data = pd.read_csv("data/train.tsv", sep="\t")[features]
    target = data["revenue"]

    ###############
    # Preprocessing
    ###############

    data["spoken_languages"] = data['spoken_languages'].apply(lambda x: literal_eval(x))
    data["overview"] = data["overview"].fillna("")
    data["Keywords"] = data["Keywords"].fillna("[]").apply(lambda x: " ".join([word_dict["name"] for word_dict in literal_eval(x)]))
    data["no_budget"] = (data["budget"] == 0).astype(int)


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

    award_data = pd.read_csv("data/oscars-demographics.csv", sep=";")
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

    normal_revenue_idx = np.where(target > 10000)[0]
    data, target = data.iloc[normal_revenue_idx], target.iloc[normal_revenue_idx]

    ##########################
    # Transform original_title
    ##########################

    # tfidf
    tfidf = TfidfVectorizer(decode_error="ignore", strip_accents="unicode")
    tfidf.fit(data["original_title"])
    with open("text_transformers/original_title_tfidf.pkl", 'wb') as f:
        pickle.dump(tfidf, f, pickle.HIGHEST_PROTOCOL)

    # get important words, transform train.
    X_words = tfidf.transform(data["original_title"])
    i_words = important_words(X_words, target)
    with open("text_transformers/original_title_i_words.pkl", 'wb') as f:
        pickle.dump(i_words, f, pickle.HIGHEST_PROTOCOL)
    word_list = tfidf.get_feature_names()
    word_df = pd.DataFrame(X_words[:, i_words].toarray(), columns=["original_title__" + word_list[i] for i in i_words], index=data.index)
    data = pd.concat([data, word_df], axis=1)

    data.drop("original_title", inplace=True, axis=1)

    ####################
    # Transform overview
    ####################
    
    # tfidf
    tfidf = TfidfVectorizer(decode_error="ignore", strip_accents="unicode")
    tfidf.fit(data["overview"])
    with open("text_transformers/overview_tfidf.pkl", 'wb') as f:
        pickle.dump(tfidf, f, pickle.HIGHEST_PROTOCOL)

    # get important words, transform train.
    X_words = tfidf.transform(data["overview"])
    i_words = important_words(X_words, target)
    with open("text_transformers/overview_i_words.pkl", 'wb') as f:
        pickle.dump(i_words, f, pickle.HIGHEST_PROTOCOL)
    word_list = tfidf.get_feature_names()
    word_df = pd.DataFrame(X_words[:, i_words].toarray(), columns=["overview__" + word_list[i] for i in i_words], index=data.index)
    data = pd.concat([data, word_df], axis=1)

    data.drop("overview", inplace=True, axis=1)

    ####################
    # Transform Keywords
    ####################
    
    # tfidf
    tfidf = TfidfVectorizer(decode_error="ignore", strip_accents="unicode")
    tfidf.fit(data["Keywords"])
    with open("text_transformers/Keywords_tfidf.pkl", 'wb') as f:
        pickle.dump(tfidf, f, pickle.HIGHEST_PROTOCOL)

    # get important words, transform train.
    X_words = tfidf.transform(data["Keywords"])
    i_words = important_words(X_words, target)
    with open("text_transformers/Keywords_i_words.pkl", 'wb') as f:
        pickle.dump(i_words, f, pickle.HIGHEST_PROTOCOL)
    word_list = tfidf.get_feature_names()
    word_df = pd.DataFrame(X_words[:, i_words].toarray(), columns=["Keywords__" + word_list[i] for i in i_words], index=data.index)
    data = pd.concat([data, word_df], axis=1)

    data.drop("Keywords", inplace=True, axis=1)

    #######
    # Model
    #######

    ohe = OneHotEncoder(sparse=False)
    ohe.fit(data[["release_season", "filtered_lang"]])
    with open("text_transformers/text_ohe_transformer.pkl", 'wb') as f:
        pickle.dump(ohe, f, pickle.HIGHEST_PROTOCOL)
    data_ohe = pd.DataFrame(ohe.transform(data[["release_season", "filtered_lang"]]), columns=[col.replace("x0", "release_season_").replace("x1", "filtered_lang_") for col in ohe.get_feature_names()], index=data.index)
    data.drop(["release_season", "filtered_lang"], inplace=True, axis=1)
    data = pd.concat([data, data_ohe], axis=1)
        
    regressor = xgb.XGBRegressor(colsample_bytree=0.8, learning_rate=0.05, max_depth=5, min_child_weight=10, n_estimators=300, random_state=0, reg_lambda=0.3, reg_alpha=0.3)
    regressor.fit(data, target)

    with open("model.pkl", 'wb') as f:
        pickle.dump(regressor, f, pickle.HIGHEST_PROTOCOL)

make_model()