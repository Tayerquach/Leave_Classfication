import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0, os.getcwd())
from src.helper.helpers import load_features, load_labels, num_folds, __index_kfold__, __result_files__, __visualisation_files__
import pickle
from src.visualization.performance import Performance
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
from collections import Counter
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# import argparse
# parser = argparse.ArgumentParser(description='Train encoders.')

# parser.add_argument('-p', '--predicted_data', type=str,
# 					help='Check predicted data file in results folder.')

# parser.add_argument('-a', '--actual_data', type=str,
# 					help='Check actual data file in results folder.')

# parser.add_argument('-f', '--false_data', type=str,
# 					help='Check false data file in results folder.')


labels = load_labels()
counter = Counter(labels)
class_names = sorted(list(counter.keys()))

#Read data
image_labels = pd.read_csv(__index_kfold__)
images = load_features('image')

def create_result_table():
    val_accs = np.load(__result_files__["kfold_val_acc"])
    test_accs = np.load(__result_files__["kfold_test_acc"])
    fold = ["Fold_" + str(i) for i in range(1,num_folds + 1,1)]

    
    df = pd.DataFrame([list(val_accs), list(test_accs)], columns = fold)
    df["dataset"] = ["Validation accuracy", "Test accuract"]
    val_result = "{:.4f} +- {:.4f}".format(np.mean(val_accs), np.std(val_accs))
    test_result = "{:.4f} +- {:.4f}".format(np.mean(test_accs), np.std(test_accs))
    df["Final result"] = [val_result, test_result]
    df.set_index(['dataset'], inplace = True)

    df.to_csv(__visualisation_files__['result_table'], index=False)

def create_image_from_name(name):
  image_id = image_labels[image_labels['Flower'] == name].index.values[2]
  arr = images[image_id]
  return arr

def generate_confusion_matrix(cnf_matrix, classes, normalize=False, title='Confusion matrix'):
    if normalize:
        cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig = plt.figure(figsize=(10,20))
    plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title(title)
    # plt.colorbar()
    plt.tight_layout()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cnf_matrix.max() / 2.

    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, format(cnf_matrix[i, j], fmt), horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return cnf_matrix

def plot_confusion_matrix(predicted_labels_list, y_test_list):
    cnf_matrix = confusion_matrix(y_test_list, predicted_labels_list)

    fig = plt.figure(figsize=(20,20))
    sns.set(font_scale=2)
    cfm_plot = sns.heatmap(cnf_matrix, annot=True, cmap="Blues", fmt='d')
    cfm_plot.figure.savefig(__visualisation_files__["confusion_matrix"])
    
    # np.set_printoptions(precision=2)

    # # Plot non-normalized confusion matrix
    # cnf_matrix_without_norm = generate_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix, without normalization')
    # plt.show()
    # cnf_matrix_without_norm.savefig('results/confusion_matrix_without_normalization.png')

    # # Plot normalized confusion matrix
    # cnf_matrix_with_norm = generate_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Normalized confusion matrix')
    # plt.show()
    # cnf_matrix_with_norm.savefig('results/confusion_matrix_with_normalization.png')

def plot_misclassified_images(false_data):
    fig, ax = plt.subplots(len(false_data),3, figsize=(20,40))
    for id in range(len(false_data)):
        test_image, true_label, predicted_label = false_data[id][0], false_data[id][1], false_data[id][2]
        predicted_class = create_image_from_name(predicted_label)
        true_class = create_image_from_name(true_label)
        ax[id, 0].imshow(test_image, interpolation='nearest', aspect='auto')
        ax[id, 0].set_title(f'Predicted class: {predicted_label}', fontsize=18)
        ax[id, 1].imshow(true_class, interpolation='nearest')
        ax[id, 1].set_title(f'Actual class: {true_label}', fontsize=18)
        ax[id, 2].imshow(predicted_class, interpolation='nearest')
        ax[id, 2].set_title(f'Misclassified_class: {predicted_label}', fontsize=18) 
    fig.savefig(__visualisation_files__["misclassified_images"])


if __name__ == "__main__":
    # args = parser.parse_args()
    # predicted_data_file = args.predicted_data
    # actual_data_file    = args.actual_data
    # false_data_file     = args.false_data

    predicted_data = np.load(__result_files__['prediction'], allow_pickle=True)
    actual_data    = np.load(__result_files__['true'], allow_pickle=True)
    false_data     = np.load(__result_files__['false'], allow_pickle=True)

    create_result_table()
    plot_confusion_matrix (predicted_data, actual_data)
    plot_misclassified_images(false_data)
    print("Please check the results in reports")