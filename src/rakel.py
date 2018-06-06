import numpy as np


def example_output(true_labels):
    # We expect that the different models have an activation between 0 and 1 in the last layer.
    output = np.random.rand(5, 228)/10
    # We also expect that the models do an ok-ish job, thus there must be some columns that have a value near 1
    for label in true_labels:
        output[:, label] = 1 - np.random.rand(1,5) / 10
    # Finally, we expect that there is some noise in the output
    for i in range(100):
        x = np.random.randint(0,5)
        y = np.random.randint(0,228)
        output[x,y] = np.random.rand()

    return output


def precision(Y, T):
    TP = 0
    for y in Y:
        if y in T:
            TP += 1
    if TP == 0: return 0
    return TP / len(Y)


def recall(Y, T):
    TP = 0
    for y in Y:
        if y in T:
            TP += 1
    if TP == 0: return 0
    return TP / len(T)


def F1(Y,T):
    if precision(Y,T) + recall(Y,T) == 0: return 0
    return 2 * (precision(Y,T) * recall(Y,T))/(precision(Y,T) + recall(Y,T))


def CV(t, X, Y, nr_folds = 10):
    #shuffle the data
    import random
    l = [a for a in zip(X, Y)]
    random.shuffle(l)
    X, Y = list(zip(*l))
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=nr_folds)
    accuracies = []
    fold = 1
    for train_index, test_index in kf.split(X):
        X_train = [X[x] for x in train_index]
        X_test = [X[x] for x in test_index]
        Y_train = [Y[x] for x in train_index]
        Y_test = [Y[x] for x in test_index]

        # get the outputs of the RAKEL model
        outputs = []
        for o in X_train:
            labels, new_output = RAKEL(o, threshold=t)
            outputs.append(labels)

        # calculate the accuracy:

        fold += 1

    print(accuracies)
    print("avg", sum(accuracies)/len(accuracies))
    return accuracies, sum(accuracies)/len(accuracies)


def learn_RAKEL(data, true_labels, steps = 1000, plot = False):
    """
    Here, the algorithm empirically tries to find the perfect value for the threshold
    :param data: an array containing the outputs of the models (NxMxL) N=#trainingData, M=#models, L=#labels
    :param true_labels: an array with the true training labels
    :return: the perfect threshold
    """

    # take a starting point for the threshold:
    t = 0.9
    performance = []
    for t in np.linspace(0.2, 1, steps):
        Y = []
        FOnes = []
        for index, output in enumerate(data):
            labels, new_output = RAKEL(output, threshold=t)
            FOnes.append(F1(labels,true_labels[index]))
            Y.append(new_output)
        # print("avg F1 with t =", t, ":", sum(FOnes)/len(FOnes))
        performance.append(sum(FOnes)/len(FOnes))
    print("thus, t =", np.linspace(0.2,1,steps)[performance.index(max(performance))], "with F1 =", max(performance), "is the best option!")
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(np.linspace(0.2,1,steps), performance)
        plt.show()
    return np.linspace(0.2,1,steps)[performance.index(max(performance))]




def RAKEL(output, threshold = 0.9):
    """
    RAKEL is an multi label ensemble method
    :param output:the output of the models (an array with m (number models) arrays that contain l (number possible labels) elements)
    :return: the labels as a list and a combined output of the models
    """
    # first, avg across the models:
    output = np.mean(output, axis=0)
    labels = [i for i, val in enumerate(output) if val > threshold]
    new_output = [1 if val in labels else 0 for val in range(228)]
    return labels, new_output



if __name__ == '__main__':
    # for testing, we create a possible output of the 5 models
    output = []
    TL = []
    for i in range(100):
        true_labels = list(set([np.random.randint(0,228) for j in range(5)]))
        TL.append(true_labels) #[1 if val in true_labels else 0 for val in range(228)])
        output.append(example_output(true_labels))
    threshold = learn_RAKEL(output, TL)


