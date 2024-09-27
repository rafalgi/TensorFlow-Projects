# Intel_Image_Classification

## Exploratory Data Analysis + Data Visualization + Deep Learning Modelling 

### 1 - Abstract

In this project I made Exploratory Data Analysis, Data Visualisation and lastly Modelling. Intel Image Dataset contains 25 images in 3 different sets. Each example is 150x150 image and associated with __6__ labels(targets). After examining the dataset I used ImageDataGenerator for example rescaling images and increasing the artifical training and test datasets. In modelling part, with a InceptionResNetV2 and several other layers implemented. Model trained with __10__ Epochs for training the data. Also for long epochs time I implemented callback for time saving. Overally, model gives __0.9023__ accuracy. Furthermore with hyperparameter tuning model can give higher accuracy or using GPU for training it will reduce time and number of epochs can be increased.


### 2 - Data
Intel Image Dataset contains __25,000__ examples,train set of __14,000__ test set of __3,000__ examples and validation set of __7,000__ . Each example is a __150x150__  image, associated with a label from 6 labels.

Each training and test example is assigned to one of the following labels:

* __0 Buildings__
* __1 Forest__
* __2 Glacier__
* __3 Mountain__
* __4 Sea__
* __5 Street__


<p align="center">
  <img width="500" height="300" src="https://github.com/HalukSumen/Image_Classification_Intel/blob/main/images/trainset.png">
</p>
<p align="center">
     <b>Train Dataset Example</b>
</p>

<p align="center">
  <img width="500" height="300" src="https://github.com/HalukSumen/Image_Classification_Intel/blob/main/images/testset.png">
</p>
<p align="center">
   <b>Test Dataset Example</b>
</p>

### 3 - Exploratory Data Analysis

Firstly, I checked data, which came three different dataset which are train, test and validation. Later I checked distribution of labels in datasets moreover I see all the classes(labels) equally distributed. So luckily I dont need to do Oversampling or Undersampling.

Number of images in Train Directory: 
* __Buildings:  2191__
* __Street:     2382__
* __Mountain:   2512__
* __Glacier:    2404__
* __Sea:        2274__
* __Forest:     2271__

### 4 - Data Preprocessing

For preparing datasets to the model I used ImageDataGenerator for rescaling which is __1/255.0__. Also I defined batch size __64__, and for creating artifical images I used rotating that takes maximum __40__ degree rotation. In the below code every parameters are visible with explanation. 
```
train_datagen = ImageDataGenerator(horizontal_flip=True,
                                  vertical_flip=True,
                                  rescale=1.0/255,
                                  rotation_range=40,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2,
                                  fill_mode='nearest',
                                  zoom_range=0.2
                                 )

```

```
train_generator = train_datagen.flow_from_directory(directory=train_path,
                                                    target_size=(IMAGE_SIZE , IMAGE_SIZE),
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='categorical')   
```

### 5 - Modelling 
I created my own model. 

```
model = keras.models.Sequential([
        keras.layers.Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(IMAGE_SIZE,IMAGE_SIZE,3)),
        keras.layers.MaxPool2D(2,2),
        keras.layers.Conv2D(64,kernel_size=(3,3),activation='relu'),
        keras.layers.MaxPool2D(2,2),
        keras.layers.Conv2D(128,kernel_size=(3,3),activation='relu'),
        keras.layers.MaxPool2D(2,2),
        keras.layers.Conv2D(256,kernel_size=(3,3),activation='relu'),
        keras.layers.MaxPool2D(2,2),
        keras.layers.Conv2D(512,kernel_size=(3,3),activation='relu'),
        keras.layers.MaxPool2D(2,2),
        keras.layers.Flatten() ,    
        keras.layers.Dense(128,activation='relu') ,            
        keras.layers.Dropout(rate=0.5) ,            
        keras.layers.Dense(6,activation='softmax') ,    
        ])
```
Finally I am compiling model according these parameters, I used RMSprop class and I gave learning rate 0.001. 
```
model.compile(optimizer ='adam',loss='categorical_crossentropy',metrics=['accuracy'])
```

I used pretrained ResNet50 model. The ResNet50 model is already trained more than 1 million images. Then I added these parameters over the pre-trained model.
```
base_model = ResNet50(include_top=False , weights='imagenet', input_shape=(IMAGE_SIZE ,IMAGE_SIZE ,3))
base_model.trainable = False

model_2 = Sequential()
model_2.add(base_model)
model_2.add(layers.Flatten())
model_2.add(layers.Dense(units=128 , activation='relu' ))
model_2.add(layers.Dense(units=6 , activation='softmax'))

model_2.summary()
```
I then compiled the model and used these parameters
```
model_2.compile(optimizer=optimizers.RMSprop(learning_rate=0.001) , loss='categorical_crossentropy' , metrics=['accuracy'])
```
