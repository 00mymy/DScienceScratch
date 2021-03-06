from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import random

def main():
    true_labels = [random.randint(1, 10) for i in range(100)]
    predicted_labels = [random.randint(1, 10) for i in range(100)]
    cm = confusion_matrix(true_labels, predicted_labels)
    plot = getConfusionMatrixPlot(true_labels, predicted_labels, cm, [i for i in range(10)])
    plot.show()

def getConfusionMatrixPlot(true_labels, predicted_labels, cm, class_names):
    # Compute confusion matrix
    #cm = confusion_matrix(true_labels, predicted_labels)

    # create figure
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(cm, cmap=plt.cm.Blues,
                    interpolation='nearest', vmin=0, vmax=len(true_labels)/5)

    # add color bar
    plt.colorbar(res)

    # annotate confusion entries
    width = len(cm)
    height = len(cm[0])

    for x in range(width):
        for y in range(height):
            ax.annotate(str(cm[x][y]), xy=(y, x), horizontalalignment='center',
                        verticalalignment='center', color=getFontColor(cm[x][y]))

    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.xticks(range(width), class_names[:width], rotation=30)
    plt.yticks(range(height), class_names[:height])
    return plt

def getFontColor(value):
    if value < 5:
        return "black"
    else:
        return "white"

if __name__ == "__main__":
    main()