import os
from itertools import cycle

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve

import config


def draw(epochs, value_list, label_list, title, x_label, y_label, color_list, save_path):
    fig = plt.figure()
    x = range(1, epochs + 1)
    for y, l, c in zip(value_list, label_list, color_list):
        plt.plot(x, y, color=c, label=l)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc="best")
    plt.xlim(0.5, epochs + 0.5)

    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f'{title}.png'))
    plt.close(fig)


def drawPR(y_true, y_score, n_classes, class_labels, save_path):
    colors = cycle(["#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30"])

    precision = dict()
    recall = dict()

    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_score[:, i])

        np.savetxt(os.path.join(save_path, f"{class_labels[i]}_pr_data.txt"),
                   np.transpose([recall[i], precision[i]], (1, 0)), '%.4f')

    for i, color in zip(range(n_classes), colors):
        fig = plt.figure()
        plt.plot(recall[i], precision[i], color=color, label=f"PR curve of class {class_labels[i]}")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall (Sensitivity)")
        plt.ylabel("Precision")
        plt.title(class_labels[i])
        plt.legend(loc="lower right")
        # plt.show()
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f'{class_labels[i]}_pr.png'))
        plt.close(fig)

    fig = plt.figure()
    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[i], precision[i], color=color, label=f"PR curve of class {class_labels[i]}")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall (Sensitivity)")
    plt.ylabel("Precision")
    plt.title('PR')
    plt.legend(loc="lower right")
    # plt.show()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, 'all_pr.png'))
    plt.close(fig)


def drawROC(y_true, y_score, n_classes, class_labels, save_path):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        np.savetxt(os.path.join(save_path, f"{class_labels[i]}_roc_data.txt"),
                   np.transpose([fpr[i], tpr[i]], (1, 0)), '%.4f')

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    colors = cycle(["#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30"])
    # lw = 2

    # Plot every class's ROC curves
    for i, color in zip(range(n_classes), colors):
        fig = plt.figure()

        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            # lw=lw,
            label="ROC curve of class {0} (area = {1:0.2f})".format(class_labels[i], roc_auc[i]),
        )

        plt.plot([0, 1], [0, 1], "k--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(class_labels[i])
        plt.legend(loc="lower right")
        # plt.show()
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f'{class_labels[i]}_roc.png'))
        plt.close(fig)

    # Plot all ROC curves
    fig = plt.figure()
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
        color="#4DBEEE",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
        color="#A2142F",
        linestyle=":",
        linewidth=4,
    )
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            # lw=lw,
            label="ROC curve of class {0} (area = {1:0.2f})".format(class_labels[i], roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC")
    plt.legend(loc="lower right")
    # plt.show()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, 'all_roc.png'))
    plt.close(fig)


if __name__ == '__main__':
    draw(10, [np.random.rand(10), np.random.rand(10), np.random.rand(10)], ['a', 'b', 'c'],
         'test_all', 'epoch', 'acc', ['red', 'blue', 'green'], '.')
