import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from model import Model
from dataset import MnistDataset

# Load the data
#train = pd.read_csv("./data/train.csv")
#test = pd.read_csv("./data/test.csv")

#test = test / 255.0

DATASET_PATH = './data/train.csv'


# hyper parameters
learning_rate = 0.001
training_epochs = 1
batch_size = 1000

def _batch_loader(iterable, n=1):
    length = len(iterable)
    for n_idx in range(0, length, n):
        yield iterable[n_idx:min(n_idx + n, length)]

# initialize
sess = tf.Session()

models = []
num_models = 1
for m in range(num_models):
    models.append(Model(sess, "model" + str(m), learning_rate))

sess.run(tf.global_variables_initializer())

print('Learning Started!')

# train my model
for epoch in range(training_epochs):
    dataset = MnistDataset(DATASET_PATH, is_training=True)
    dataset_len = len(dataset)
    one_batch_size = dataset_len // batch_size
    if dataset_len % batch_size != 0:
        one_batch_size += 1
    
    avg_cost_list = np.zeros(len(models))
    for i, (data, labels) in enumerate(_batch_loader(dataset, batch_size)):
        onehot_labels = sess.run(tf.one_hot(labels, 10, dtype=tf.float32))
        x_train, x_test, y_train, y_test = train_test_split(data, onehot_labels, random_state = 42)

        # train each model
        for m_idx, m in enumerate(models):
            c, _ = m.train(x_train, y_train)
            avg_cost_list[m_idx] += c / one_batch_size
        
        for m_idx, m in enumerate(models):
            print(m_idx, 'Accuracy:', m.get_accuracy(x_test, y_test))
            p = m.predict(x_test)
            predictions += p

        ensemble_correct_prediction = tf.equal(
            tf.argmax(predictions, 1), tf.argmax(y_test, 1))
        ensemble_accuracy = tf.reduce_mean(
            tf.cast(ensemble_correct_prediction, tf.float32))
        print('Ensemble accuracy:', sess.run(ensemble_accuracy))



    print('Epoch:', '%04d' % (epoch + 1), 'cost =', avg_cost_list)

print('Learning Finished!')


#---- submission ----
TESTSET_PATH = './data/test.csv'
test_data = pd.read_csv(TESTSET_PATH)
x_test = test_data / 255.0
test_dataset_len = len(test_data)

predictions = np.zeros([test_dataset_len, 10])

for m_idx, m in enumerate(models):
    p = m.predict(x_test)
    predictions += p

# select the indix with the maximum probability
results = np.argmax(predictions,axis = 1)

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_mnist_datagen.csv",index=False)

