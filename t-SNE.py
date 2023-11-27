import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn import manifold

def draw_tsne(emb_filename):
    model = json.load(open(emb_filename, 'r'))
    X = np.array(model['features'])
    y = np.array(model['label'])
    '''t-SNE'''
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(X)
    print(X.shape)
    print(X_tsne.shape)
    print(y.shape)
    print("Org data dimension is {}.Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

    '''visualize'''
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化

    plt.figure(figsize=(8, 8))
    for i in range(X_norm.shape[0]):
        plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]),
                 fontdict={'weight': 'bold', 'size': 10})
    plt.xticks([])
    plt.yticks([])
    plt.suptitle('Roberta-SPC CE', fontsize=24)
    plt.show()


emb_filename = ("/home/disk2/jye/ABSA_Curriculum_Learning/roberta-absa/roberta_spc_total_test1.json")
# emb_filename = ("/home/disk2/jye/ABSA_Curriculum_Learning/roberta-absa/roberta_spc_explicit_test1.json")
# emb_filename = ("/home/disk2/jye/ABSA_Curriculum_Learning/roberta-absa/roberta_spc_implicit_test1.json")
# emb_filename = ("/home/disk2/jye/ABSA/111/roberta-absa/roberta_spc_implicit_triplet_loss.json")
draw_tsne(emb_filename)
