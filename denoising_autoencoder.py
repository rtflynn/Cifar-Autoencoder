import numpy as np
from keras.layers import Conv2D, UpSampling2D, BatchNormalization
from keras.models import Sequential,
from keras.datasets import cifar10
import matplotlib.pyplot as plt


(x_train, _), (x_test, _) = cifar10.load_data()
x_train = x_train/255
x_test = x_test/255


# The next three methods to visualize input/output of our model side-by-side
def hstackimgs(min, max, images):
    return np.hstack(images[i] for i in range(min, max))

def sqstackimgs(length, height, images):
    return np.vstack(hstackimgs(i*length, (i+1)*length, images) for i in range(height))

def sbscompare(images1, images2, length, height):
    A = sqstackimgs(length, height, images1)
    B = sqstackimgs(length, height, images2)
    C = np.ones((A.shape[0], 32, 3))
    return np.hstack((A, C, B))



model = Sequential()

model.add(Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(BatchNormalization())     # 32x32x32
model.add(Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu'))      # 16x16x32
model.add(Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu'))      # 16x16x32
model.add(BatchNormalization())     # 16x16x32
model.add(UpSampling2D())
model.add(Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu'))      # 32x32x32
model.add(BatchNormalization())
model.add(Conv2D(3,  kernel_size=1, strides=1, padding='same', activation='sigmoid'))   # 32x32x3

model.compile(optimizer='adam', metrics=['accuracy'], loss='mean_squared_error')
model.summary()


# We want to add different noise vectors for each epoch
num_epochs = 3
for i in range(num_epochs):
    noise = np.random.normal(0, 0.3, x_train.shape)
    model.fit(x_train + noise, x_train, epochs=1, batch_size=100)


x_test = x_test[:400]
noise = np.random.normal(0, 0.3, x_test.shape)
pred_imgs = model.predict(x_test + noise)

plt.imshow(sbscompare(x_test + noise, pred_imgs, 20, 20))
plt.axis('off')
plt.rcParams["figure.figsize"] = [60,60]
plt.show()