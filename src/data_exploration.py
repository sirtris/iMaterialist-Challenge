import json
import urllib.request as rq
import os
from PIL import Image
import matplotlib.pyplot as plt

def download_images(data, n = 100):
    for i in range(n):
        url = data['images'][i]['url']
        suffix = url.split('/')[-1]
        try:
            rq.urlretrieve(url, 'data/' + suffix + '.jpg')
        except:
            print("Problem with", url)


def reorganize_data(database, N):
    # make a list of all images of class 1:
    for label in range(300):
        print(label)
        for id, name, labels in database:
            if str(label) in labels:
                if not os.path.exists("data_buckets/" + str(label)):
                    os.makedirs("data_buckets/" + str(label))

                try:
                    file = Image.open("data/" + name + ".jpg")
                    file.save("data_buckets/" + str(label) + "/" + name + ".jpg")

                except:
                    print("skip")


if __name__ == '__main__':
    N = 10000
    data = json.load(open("train.json", 'r'))
    data['images'] = data['images'][0:N]
    data['annotations'] = data['annotations'][0:N]

    # comment if you do have the data downloaded already
    # download_images(data, n=N)

    # join the 2 db:
    database = []
    for i in range(N):
        database.append([data['images'][i]['imageId'], data['images'][i]['url'].split("/")[-1], data['annotations'][i]['labelId']])

    # label distribution
    distr = [0]*228
    for id, name, labels in database:
        for l in labels:
            distr[int(l)-1] += 1
    print(distr)
    plt.bar(range(1,229), distr)
    plt.xlabel("label")
    plt.ylabel("amount")
    plt.show()


    # reorganize_data(database, N)
    #print(data)

