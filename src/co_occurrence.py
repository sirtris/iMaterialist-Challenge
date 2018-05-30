import json
import numpy as np
import matplotlib.pyplot as plt


"""
use get_co-occurrence_matrix to get the normalized co-occurrence matrix. Note that row 0 and column 0 exists, but is 
empty, because label 0 does not exist. Thus you can access the co-occurrence probability between label 1 and 5 
with co_oc_matrix[1][5]
"""
def get_cooccurrence_matrix():
    M = [line.replace("\n","").split(";") for line in open("co_occurrence_matrix.txt", 'r').readlines()]
    co_oc_matrix = np.zeros(shape=(229,229))
    for i, row in enumerate(M):
        for j, e in enumerate(row):
            co_oc_matrix[i][j] = float(e)
    return co_oc_matrix

def __plot(matrix):
    print("start plotting...")
    fig, axis = plt.subplots()  # il me semble que c'est une bonne habitude de faire supbplots
    heatmap = axis.pcolor(co_oc, cmap=plt.cm.Reds)  # heatmap contient les valeurs

    plt.imshow(co_oc, cmap='hot', interpolation='nearest')
    plt.colorbar(heatmap)
    plt.show()

if __name__ == '__main__':
    data = json.load(open("train.json", 'r'))

    # count co-occurrences:
    print("start counting...")
    co_oc = np.zeros(shape=(229, 229))
    for image in data['annotations']:
        labels = image['labelId']
        for i in range(len(labels)):
            for j in range(i+1, len(labels)):
                co_oc[int(labels[i])][int(labels[j])] += 1
    #plt.figure(1)
    #__plot(co_oc)
    print("start normalizing...")
    # normalize co-occurrences:
    for i, row in enumerate(co_oc):
        s = (sum(row)+1)
        for j, e in enumerate(row):
            co_oc[i][j] = e/s

    __plot(co_oc)
    out = open("co_occurrence matrix.txt", 'w')
    for row in co_oc:
        out.write(";".join([str(e) for e in row[1:]][1:]) + "\n")
    print(co_oc)




