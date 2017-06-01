import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',
                          cmap=plt.cm.Blues):
    import itertools
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        normalized_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, normalized_cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def plot_roc(fpr, tpr, auc):
    color_palette = ['darkorange', 'red', 'blue', 'green', 'purple'
                    'yellow', 'grey', 'aqua', 'pink', 'darkblue']
    plt.figure()
    plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
    for i in range(len(tpr)):
        plt.plot(fpr[i], tpr[i], color=color_palette[i], lw=2,
             label='ROC curve (area = %0.2f)' % auc[i])
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
    plt.show()

def plot_pr(prec, rec, auc):
    color_palette = ['darkorange', 'red', 'blue', 'green', 'purple'
                    'yellow', 'grey', 'aqua', 'pink', 'darkblue']
    plt.figure()
    for i in range(len(prec)):
        plt.plot(prec[i], rec[i], color=color_palette[i], lw=2,
             label='PR curve (area = %0.2f)' % auc[i])
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Precision')
        plt.ylabel('Recall')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower right')
    plt.show()

def cross_val_method(models, X, Y, class_names, params=None, nested=False):
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import cross_val_score, cross_val_predict
    from sklearn.model_selection import KFold, StratifiedKFold
    from sklearn import metrics

    fpr = [None]*len(models)
    tpr = [None]*len(models)
    auc = [None]*len(models)
    precision = [None]*len(models)
    recall = [None]*len(models)
    auc_pr = [None]*len(models)
    count = 0
    cv = StratifiedKFold(10)
    for model in models:
        print ('Starting Iteration: ', count)
        scores = cross_val_score(model, X, Y, scoring="roc_auc", cv=cv)
        predicted = cross_val_predict(model, X, Y)

        accuracy = metrics.accuracy_score(Y, predicted)
        roc_auc = metrics.roc_auc_score(Y, predicted)
        f1_score = metrics.f1_score(Y, predicted)
        kappa = metrics.cohen_kappa_score(Y, predicted)
        print ("Cross Validation Scores: ", scores.mean())
        print ("Accuracy Score: ", accuracy)
        print ("AUC Score: ", roc_auc)
        print ("F1 Score: ", f1_score)
        print ("Kappa: ", kappa)
        confusion_matrix = metrics.confusion_matrix(Y, predicted)
        plot_confusion_matrix(confusion_matrix, normalize=True,
                              classes=class_names,
                              title='Confusion Matrix without Normalization')
        fpr[count], tpr[count], _ = metrics.roc_curve(Y, predicted)
        precision[count], recall[count], _ = metrics.precision_recall_curve(Y, predicted)
        # fpr = false positive rate and tpr is true positive rate.
        auc[count] = metrics.auc(fpr[count], tpr[count])
        auc_pr[count] = metrics.auc(precision[count], recall[count])
        count += 1
    plot_roc(fpr, tpr, auc)
    plot_pr(precision, recall, auc_pr)