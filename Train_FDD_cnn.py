
def main():
    
    #Import some packages to use
    import numpy as np
    
    #To see our directory
    import os
    import gc   #Gabage collector for cleaning deleted data from memory
    from PIL import Image
    #import tqdm as tqdm
    
    img_cols, img_rows = 64,64
    
    
    
    vid_dir=[]
    Fall_dir=r"D:/Project/suspicios_activity/videos/our videos"
    
    vid_dir = []
    for root, dirs, files in os.walk(Fall_dir):
        for i in range(len(files)):
            vid_dir.append(root + '/' + files[i])
    
    
    
    
    immatrix = np.array([np.array(Image.open(im2).convert('L').resize((img_cols, img_rows))).flatten()
                  for im2 in vid_dir],'f')
    
    ######FOR NOT event22222222222222222222222222222222222222222222222222222222222222222222222222222222
    del vid_dir
    gc.collect()
    
    vid_dir=[]
    NotFall_dir=r"D:/Project/suspicios_activity/videos/our videos"
    
    vid_dir = []
    for root, dirs, files in os.walk(NotFall_dir):
        for i in range(len(files)):
            vid_dir.append(root + '/' + files[i])
    
    
    
    
    immatrix2 = np.array([np.array(Image.open(im2).convert('L').resize((img_cols, img_rows))).flatten()
                  for im2 in vid_dir],'f')
    
    del vid_dir
    gc.collect()
    
    ####################################################################################################################
    
    Mainmatrix=np.vstack((immatrix,immatrix2))
    
    num_samples=Mainmatrix.shape[0]
    
    label=np.ones((num_samples,),dtype = int)
    
    label[0:27760]=1
    label[27761:]=0
    
        #splits array randomly into train and tests subset
    from sklearn.model_selection import train_test_split

        #shuffle array in consistent manner
    from sklearn.utils import shuffle
    
        #keras.util= it provides numpy utilities library to perform actions on numpy array.
        #numpy arrays = grid values of same type nd indexed.
        #np_utils = numpy utility
        
    from keras.utils import np_utils

    
    
    #random state determines random no. generation for shuffling data
    #mainmatrix is an ndarray
    #ndarray = multidimensional array of fixed size
    data,Label = shuffle(Mainmatrix,label, random_state=2)

# list of train data
    train_data = [data,Label]
        
    nb_classes = 2
    
    (X, y) = (train_data[0],train_data[1])
    
    
    # STEP 1: split X and y into training and testing sets
    # here x_train nd y_train contains training dataset
    # while x_test nd Y_test contains testing dataset
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    
    #reshape = it transforms the tensor ie; image from 2D to 1D. so that the metrix can be reperesented by array.
    X_train = X_train.reshape(X_train.shape[0], img_cols, img_rows, 1)
    X_test = X_test.reshape(X_test.shape[0], img_cols, img_rows, 1)
    
    #astype = allows us to convert or cast entire datatype with existing data column
    #conversion of data takes place as float then targets to int
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    #pixels are represented in array of no.s whose values ranges from [0,255]
    #so it should be scaled to values of type (float32) within [0,1] interval.
    X_train /= 255
    X_test /= 255

    
    #shape = it is tuple of integers that describes how many dimensions the tensor has along with each axis 
    print('y_train shape:', Y_train.shape)
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')# x_train.shape[0]: returns length of 1st row
    print(X_test.shape[0], 'test samples')
        
    # convert class vectors to binary class matrices
    #to_categorical = Converts a class vector (integers) to binary class matrix
    Y_train = np_utils.to_categorical(Y_train, nb_classes)
    Y_test = np_utils.to_categorical(Y_test, nb_classes)
#######################################################################################################
#sequential = groups a linear stack of layers into keras model 
# and provides training and inference features on this model

    from keras.models import Sequential
    from keras.models import Sequential
    from keras.layers import Convolution2D #it is instrument for general image processing like blurring,sharpening etc.
    from keras.layers import MaxPooling2D 
    from keras.layers import Flatten #Flattens the input. Does not affect the batch size.
    from keras.layers import Dense, Dropout 
    # dense = it feeds all outputs from the previous layers.
    #dropout = deactivates or ignoring neurons of the n\w. applied in training phase to reduce overfitting effects. 
    from keras import optimizers # optimizers : selects the best options among the choices
    
    from keras.layers import Dense, Conv2D, Flatten
    
    MODEL_NAME="abnormalevent.h5" #.h5 file is used to store structured data.
    
    
    #####################CNN basic MODEL#############################################
    model = Sequential()
    
    # relu = recitifed kinear unit = activation function in deep learning models.
    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(img_cols, img_rows,1)))
    
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    
    model.add(Flatten())
    # softmax = used to normalize output to fit between zero and one.
    model.add(Dense(2, activation='softmax'))
    
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10)
    
    model.save(MODEL_NAME)
    
    predictions = model.predict(X_train)
    
    ##################################################################################
    
    
    #  # Initialing the CNN
    #classifier = Sequential()
    
    # # Step 1 - Convolutio Layer 
    # classifier.add(Convolution2D(32, 3,  3, input_shape = (64, 64, 1), activation = 'relu'))
    
    # #step 2 - Pooling
    # classifier.add(MaxPooling2D(pool_size =(2,2)))
    
    # # Adding second convolution layer
    # classifier.add(Convolution2D(32, 3,  3, activation = 'relu'))
    # classifier.add(MaxPooling2D(pool_size =(2,2)))
    
    # #Adding 3rd Concolution Layer
    # classifier.add(Convolution2D(64, 3,  3, activation = 'relu'))
    # classifier.add(MaxPooling2D(pool_size =(2,2)))
    
    
    # #Step 3 - Flattening
    # classifier.add(Flatten())
    
    # #Step 4 - Full Connection
    # classifier.add(Dense(256, activation = 'relu'))
    # classifier.add(Dropout(0.2))
    # classifier.add(Dense(2, activation = 'softmax'))  #change class no.
    
    # #Compiling The CNN
    # classifier.compile(
    #               optimizer = optimizers.SGD(lr = 0.01),
    #               loss = 'categorical_crossentropy',
    #               metrics = ['accuracy'])
        
    
    
    # classifier.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10)
    
    
    # classifier.save(MODEL_NAME)
    # predictions = classifier.predict(X_train)
    
    ############################################################################################################
    
    accuracy = 0
    for prediction, actual in zip(predictions, Y_train):
        predicted_class = np.argmax(prediction)
        actual_class = np.argmax(actual)
        if(predicted_class == actual_class):
            accuracy+=1
    
    accuracy =( accuracy / len(Y_train))*100
    
    A = "Training Accuracy is {0}".format(accuracy)
        
    return A
    
    ##########################################################################################################
    
    
    
    # from keras.models import load_model
    
    # img_cols, img_rows = 64,64
    
    # FALLModel=load_model(r'D:\Alka_python_2019_20\FDD\Fall-Detection-with-CNNs-and-Optical-Flow-master\25APRFALLDtection.h5')
                         
    # # vid_dir="D:/Alka_python_2019_20/FDD/Fall-Detection-with-CNNs-and-Optical-Flow-master/URFD_images/Falls/fall_fall-01"
    # # vid_dir=r'D:\Alka_python_2019_20\FDD\Fall-Detection-with-CNNs-and-Optical-Flow-master\URFD_images\NotFalls\notfall_fall-01_pre'
    
    # vid_dir=r"C:\alka\FULL_FDD_Data\RGB\fall-01-cam0-rgb"
    # # vid_dir=r"D:\Alka_python_2019_20\FDD\Fall-Detection-with-CNNs-and-Optical-Flow-master\URFD_images\NotFalls_full\notfall_adl-40"
     
    # imlist = os.listdir(vid_dir)
    
    # immatrix = np.array([np.array(Image.open(vid_dir + '\\' + im2).convert('L').resize((img_cols, img_rows))).flatten()
    #               for im2 in imlist],'f')
    
    
    # X_img = immatrix.reshape(immatrix.shape[0], img_cols, img_rows, 1)
    
    # X_img = X_img.astype('float32')
    
    # X_img /= 255
    
    # predicted =FALLModel.predict(X_img)
    
    
    # for i in range(len(predicted)):
    #     if predicted[i][0] < 0.5:
    #         predicted[i][0] = 0
    #         predicted[i][1] = 1
    
    #     else:
            
    #         predicted[i][0] = 1
    #         predicted[i][1] = 0
        
    #     # Array of predictions 0/1
    
    # predicted = np.asarray(predicted).astype(int) 
    
