import zipfile

ZIP_PATH ="brainTumorDataPublic_1766.zip"

def dataset_extraction(file_name = ZIP_PATH, outpath='./dataset'):
    with zipfile.ZipFile(file_name) as file:
        file.extractall(outpath)

dataset_extraction()