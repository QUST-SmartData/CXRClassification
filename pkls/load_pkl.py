import pickle

if __name__ == '__main__':
    # vec = pickle.load(open('../w2v/coco_glove_word2vec.pkl', 'rb'))
    # adj = pickle.load(open('../w2v/coco_adj.pkl', 'rb'))
    # vec = pickle.load(open('CheXpert_glove_wordEmbedding.pkl', 'rb'))
    adj = pickle.load(open('CheXpert_adj.pkl', 'rb'))
    # print(vec)
    print(adj)
