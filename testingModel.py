import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

def showImage(i, arr_predicts, real_labels, images, classNames):
    arr_predicts, real_labels, img = arr_predicts[i], real_labels[i], images[i]
    plt.grid(False)
    plt.xticks([])  
    plt.yticks([])  
    plt.imshow(img.reshape((128, 128)), cmap=plt.cm.binary)

    predict_label = np.argmax(arr_predicts)
    color = 'blue' if predict_label == real_labels else 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(
        classNames[predict_label],
        100 * np.max(arr_predicts),  
        classNames[real_labels]), color=color
    )

def showArray(i, arr_predicts, real_labels, classNames):
    arr_predicts, real_labels = arr_predicts[i], real_labels[i]
    plt.grid(False)
    plt.xticks([])  
    plt.yticks([])  
    graph = plt.bar(range(len(classNames)), arr_predicts, color='#777777')
    plt.ylim([0, 1])  
    predict_label = np.argmax(arr_predicts)

    graph[predict_label].set_color('red')
    graph[real_labels].set_color('blue')

def evaluate_model(model, testData, imagesTest, className=['Benigna', 'Maligna'], max_images_to_show=200):
    all_predictions = []
    all_labels = []
    all_images = []


    for imagesT, labelT in testData:

        imagesT = imagesT.numpy()
        labelT = labelT.numpy()


        predictions = model.predict(imagesT)

        all_predictions.append(predictions)
        all_labels.append(labelT)
        all_images.append(imagesT)

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_images = np.concatenate(all_images, axis=0)

    rows = 10
    columns = 10
    num_img = min(len(all_predictions), max_images_to_show)

    rows = (num_img // columns) + (1 if num_img % columns != 0 else 0)

    plt.figure(figsize=(2 * 2 * columns, 2 * rows))
    for i in range(num_img):
        plt.subplot(rows, 2 * columns, 2 * i + 1)
        showImage(i, all_predictions, all_labels, all_images, className)
        plt.subplot(rows, 2 * columns, 2 * i + 2)
        showArray(i, all_predictions, all_labels, className)

    print(classification_report(all_labels, np.argmax(all_predictions, axis=1), target_names=className))
    plt.tight_layout()
    plt.show()
