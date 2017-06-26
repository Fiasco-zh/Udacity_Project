import os
import pickle
import csv
import zipfile  

def un_zip(file_name):  
    """unzip zip file"""  
    zip_file = zipfile.ZipFile(file_name)  
    if os.path.isdir(file_name + "_files"):  
        pass  
    else:  
        os.mkdir(file_name + "_files")  
    for names in zip_file.namelist():  
        zip_file.extract(names,file_name + "_files/")  
    zip_file.close()  

def load_data(path):
    """
    Load Dataset from File
    """
    print("Load Udacity dataset")
    input_file = os.path.join(path)
    with open(input_file, "r") as f:
        data = f.read()

    return data[81:]

def load_csv_data(path):
    """
    Load Dataset from csvFile
    """
    print("Load kaggle dataset")
    with open(path) as f:
        text = ''
        f_csv = csv.reader(f)
        for row in f_csv:
            text += row[3]
            text += '\n'
        f.close()
        
    return text[9:]


def preprocess_and_save_data(dataset_path, token_lookup, create_lookup_tables):
    """
    Preprocess Text Data
    """
    if dataset_path[-3:] == csv:
        text = load_csv_data(dataset_path)
    else:
        text = load_data(dataset_path)
    # Ignore notice, since we don't use it for analysing the data
    # text = text[81:]

    token_dict = token_lookup()
    for key, token in token_dict.items():
        text = text.replace(key, ' {} '.format(token))

    text = text.lower()
    text = text.split()

    vocab_to_int, int_to_vocab = create_lookup_tables(text)
    int_text = [vocab_to_int[word] for word in text]
    pickle.dump((int_text, vocab_to_int, int_to_vocab, token_dict), open('preprocess.p', 'wb'))


def load_preprocess():
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    return pickle.load(open('preprocess.p', mode='rb'))


def save_params(params):
    """
    Save parameters to file
    """
    pickle.dump(params, open('params.p', 'wb'))


def load_params():
    """
    Load parameters from file
    """
    return pickle.load(open('params.p', mode='rb'))
