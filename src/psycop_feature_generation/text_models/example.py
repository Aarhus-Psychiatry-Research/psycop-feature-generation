from psycop_feature_generation.text_models.preprocessing import tfidf_preprocessing

if __name__ == "__main__":
    df = tfidf_preprocessing(
        text_sfi_names="Aktuelt psykisk", n_rows=10, split_name="train"
    )

    df
