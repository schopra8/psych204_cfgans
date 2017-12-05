import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Causal Structure')
    plt.xlabel('Generated World')

def construct_cm(y_test, y_pred):
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    class_names = ['Correct_A_C_D', 'Correct_Not_A_Or_Not_C', 'Incorrect_A_C_D', 'Incorrect_Not_A_Or_Not_C']
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')

    # # Plot normalized confusion matrix
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
    #                       title='Normalized confusion matrix')

    plt.show()


if __name__ == '__main__':
    # 100,000
    # 0.3
    y_test = ['Correct_A_C_D'] * 13 + ['Correct_Not_A_Or_Not_C'] * 37
    y_pred = ['Correct_A_C_D'] * 10 + ['Incorrect_A_C_D'] * 3 + ['Correct_Not_A_Or_Not_C'] * 31 + ['Incorrect_Not_A_Or_Not_C'] * 6
    construct_cm(y_test, y_pred)


    # 0.1

    # 10,0000
    # 0.3
    # 0.1