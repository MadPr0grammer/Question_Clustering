import get_sentence_embeddings_gensim
import utils
import visualize
import time
import k_means
import numpy as np
import pandas as pd


def main():
    X_50  = get_sentence_embeddings_gensim.load_model_get_question_vector_matrix("question_corpus.model")
    X_300  = get_sentence_embeddings_gensim.load_model_get_question_vector_matrix("question_corpus_300.model")
    utils.write_matrix_to_file(X_50, "X_50.txt")
    utils.write_matrix_to_file(X_300, "X_300.txt")

    X_50_2_pca = visualize.get_2d_matrix_pca(X_50)
    utils.write_matrix_to_file(X_50_2_pca, "X_50_2_pca.txt")

    X_300_2_pca = visualize.get_2d_matrix_pca(X_300)
    utils.write_matrix_to_file(X_300_2_pca, "X_300_2_pca.txt")

    X_50_2_tsne = visualize.get_2d_matrix_tsne(X_50)
    utils.write_matrix_to_file(X_50_2_tsne, "X_50_2_tsne.txt")

    X_300_2_tsne = visualize.get_2d_matrix_tsne(X_300)
    utils.write_matrix_to_file(X_300_2_tsne, "X_300_2_tsne.txt")

    visualize.visualize_2d_matrix_without_labels(X_50_2_pca, "X_50_2_pca.png")
    visualize.visualize_2d_matrix_without_labels(X_300_2_pca, "X_300_2_pca.png")
    visualize.visualize_2d_matrix_without_labels(X_50_2_tsne, "X_50_2_tsne.png")
    visualize.visualize_2d_matrix_without_labels(X_300_2_tsne, "X_300_2_tsne.png")

def main2():
    X_50 = np.loadtxt("X_50.txt", delimiter=',')
    X_300 = np.loadtxt("X_300.txt", delimiter=',')

    # labels_50_5k = k_means.get_labels(X_50, 5) # 5 clusters
    # utils.write_1d_matrix_to_file(labels_50_5k, "labels_50_5k.txt")
    #
    # labels_300_5k = k_means.get_labels(X_300, 5) # 5 clusters
    # utils.write_1d_matrix_to_file(labels_300_5k, "labels_300_5k.txt")

    # labels_50_8k = k_means.get_labels(X_50, 8) # 8 clusters
    # utils.write_1d_matrix_to_file(labels_50_8k, "labels_50_8k.txt")

    # labels_300_8k = k_means.get_labels(X_300, 8) # 8 clusters
    # utils.write_1d_matrix_to_file(labels_300_8k, "labels_300_8k.txt")

    X_50_2_pca = np.loadtxt("X_50_2_pca.txt", delimiter=',')
    X_300_2_pca = np.loadtxt("X_300_2_pca.txt", delimiter=',')
    labels = np.loadtxt("labels_300_5k.txt", delimiter=',')
    visualize.visualize_2d_matrix_without_labels(X_50_2_pca, "X_50_2_pca.png")
    #data = pd.read_csv("one_question_per_line.txt", sep=None, header=None)
    # with open("label_0.txt", 'w+', encoding="utf-8") as f0, open("label_1.txt", 'w+', encoding="utf-8") as f1, open("label_2.txt", 'w+', encoding="utf-8") as f2, open("label_3.txt", 'w+', encoding="utf-8") as f3, open("label_4.txt", 'w+', encoding="utf-8") as f4, open("one_question_per_line.txt", encoding="utf-8") as file:
    #     for i, line in enumerate(file):
    #         if(labels[i] == 0):
    #             f0.write(line)
    #             f0.write('\n')
    #         elif(labels[i] == 1):
    #             f1.write(line)
    #             f1.write('\n')
    #         elif (labels[i] == 2):
    #             f2.write(line)
    #             f2.write('\n')
    #         elif (labels[i] == 3):
    #             f3.write(line)
    #             f3.write('\n')
    #         else:
    #             f4.write(line)
    #             f4.write('\n')

    visualize.visualize_2d_matrix_with_labels(X_300_2_pca, labels, "X_50_2_pca_labeled_5k.png")
    # visualize.visualize_2d_matrix_with_labels(X_300_2_pca, labels_300_5k, "X_300_2_pca_labeled_5k.png")
    #
    # visualize.visualize_2d_matrix_with_labels(X_50_2_pca, labels_50_8k, "X_50_2_pca_labeled_8k.png")
    #visualize.visualize_2d_matrix_with_labels(X_300_2_pca, labels_300_8k, "X_300_2_pca_labeled_8k.png")

start_time = time.time()
main2()
end_time = time.time()
print("Plotting time with K-means(pca):", (end_time-start_time)/60)