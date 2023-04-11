from psycop_feature_generation.text_models.preprocessing import tfidf_preprocessing

if __name__ == "__main__":
    df = tfidf_preprocessing(
        text_sfi_names="Aktuelt psykisk", n_rows=10, split_name=["train", "val"],
    )

    df

revert 29b442b39be78071614b584f420d6cdde0fa80d7
d2861665585bcb9d8cf5f3e9023f59ddd038787b
