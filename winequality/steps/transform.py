def transform(dataset):
    ## Standarize the dataset
    dataset_standard = (dataset - dataset.mean()) / dataset.std()

    ## Normalize to 0-1
    dataset_standard_norm = (dataset_standard - dataset_standard.min()) / (dataset_standard.max() - dataset_standard.min())

    return dataset_standard_norm
