import gensim
import numpy as np
import pickle

model_path = "crawl-300d-2M.vec"
model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=False)

vocab = model.index_to_key


def get_vec(text_list: list, save_file_name=''):
    vector_len = 300
    vector = np.zeros((len(text_list), vector_len))
    for i, l in enumerate(text_list):
        vector[i] = np.mean([model[word] for word in l.split()], axis=0)
    # np.savetxt(save_file_name, vector)
    return vector


labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']

# get_vec(labels, 'CheXpert.txt')
pickle.dump(get_vec(labels), open('CheXpert_glove_wordEmbedding.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)

labels = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis',
    'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Pleural Thickening', 'Pneumonia', 'Pneumothorax'
]
# get_vec(labels, 'CXR8.txt')
pickle.dump(get_vec(labels), open('CXR8_glove_wordEmbedding.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)

labels = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged',
    'Fracture', 'Lung Lesion', 'Lung Opacity', 'Pleural Effusion', 'Pneumonia',
    'Pneumothorax', 'Pleural Other', 'Support Devices',
    'No Finding'
]
# get_vec(labels, 'MIMIC.txt')
pickle.dump(get_vec(labels), open('MIMIC_glove_wordEmbedding.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
