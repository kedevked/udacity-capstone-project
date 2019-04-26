import numpy as np
from keras.applications.resnet50 import preprocess_input, decode_predictions, ResNet50
import cv2  
from keras.preprocessing import image                  
from tqdm import tqdm
from extract_bottleneck_features import extract_InceptionV3
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Dense
import pickle
from PIL import Image
from keras import backend as K

#ResNet50_model = ResNet50(weights='imagenet')
face_cascade = cv2.CascadeClassifier('face_detector/haarcascade_frontalface_alt.xml')


def path_to_tensor(img_path):
    #try:
    # loads RGB image as PIL.Image.Image type
    # img = Image.open(img_path)
    #img = img.resize((224,224))
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    print('xshape', x.shape)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)
    #except:
    #    pass

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

def face_detector(img_path):
    
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    ResNet50_model = ResNet50(weights='imagenet')
    img = preprocess_input(path_to_tensor(img_path))
    print('prediction resnet', ResNet50_model.predict(img))
    return np.argmax(ResNet50_model.predict(img))

def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))

# def extract_features() :
    
#     train_VGG19 = bottleneck_features['train']
#     valid_VGG19 = bottleneck_features['valid']
#     test_VGG19 = bottleneck_features['test']
#     return train_VGG19, valid_VGG19, test_VGG19

def predict_breed(img_path):
    '''
        args:
            img_path: string path of the image
        returns
            breed: string name of the breed of dog
    '''
    is_dog = dog_detector(img_path)
    is_human = face_detector(img_path)
    if is_dog or is_human:
        breed = VGG19_predict_breed(img_path)
        result = {}
        result["type"] = 'dog' if is_dog else 'human'
        result["breed"] = breed.rsplit('.')[-1]
        return result
    else:
        raise ValueError('The file selected does not contain either a human or dog picture')

def VGG19_predict_breed(img_path):
    # extract bottleneck features
    K.clear_session()
    VGG19_model = get_model()
    bottleneck_feature = extract_InceptionV3(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = VGG19_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    with open ('dog_names', 'rb') as fp:
        dog_names = pickle.load(fp)
    print('dog name', dog_names[np.argmax(predicted_vector)])
    return dog_names[np.argmax(predicted_vector)]

def get_model():
    VGG19_model = Sequential()
    VGG19_model.add(GlobalAveragePooling2D(input_shape=(5, 5, 2048)))

    VGG19_model.add(Dense(133, activation='softmax'))
    VGG19_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    VGG19_model.load_weights('saved_models/weights.best.VGG19.hdf5')
    return VGG19_model