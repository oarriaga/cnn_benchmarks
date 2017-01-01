from functools import wraps
from keras.applications import InceptionV3,ResNet50,VGG16,VGG19
from keras.applications.inception_v3 import preprocess_input as inceptionv3_input
from keras.applications.vgg16 import preprocess_input as vgg16_input
from keras.applications.vgg19 import preprocess_input as vgg19_input
from keras.applications.resnet50 import preprocess_input as resnet50_input
from keras import backend as K
from keras.preprocessing import image as keras_preprocessor
import numpy as np
import pickle
import time

DATA_PROFILE = {}
def profile(function):
    @wraps(function)
    def profiling_function(*args,**kwargs):
        start_time = time.monotonic()
        function_return = function(*args,**kwargs)
        elapsed_time = time.monotonic() - start_time
        if function.__name__ not in DATA_PROFILE:
            DATA_PROFILE[function.__name__] = []
        DATA_PROFILE[function.__name__].append(elapsed_time)
        return function_return
    return profiling_function

def print_data_profile():
    for function_name, data in DATA_PROFILE.items():
        max_time = np.max(data)
        min_time = np.min(data)
        avg_time = np.mean(data)
        var_time = np.var(data)
        print('Function %s called %d times.' % (function_name, len(data)))
        print('Execution time min: %.3f' % min_time)
        print('Execution time max: %.3f' % max_time)
        print('Execution time average: %.3f' % avg_time)
        print('Execution time variance: %.3f' % var_time)

def save_data_profile(file_name,pickle_data=True):
    for function_name, data in DATA_PROFILE.items():
        max_time = np.max(data)
        min_time = np.min(data)
        avg_time = np.mean(data)
        var_time = np.var(data)
        log_file = open(file_name+'log','w')
        log_file.write('Function %s called %d times. \n' % (function_name, data[0]))
        log_file.write('Execution time min: %.3f \n' % min_time)
        log_file.write('Execution time max: %.3f \n' % max_time)
        log_file.write('Execution time average: %.3f \n' % avg_time)
        log_file.write('Execution time variance: %.3f \n' % var_time)
        log_file.close()
        if pickle_data == True:
            pickle.dump(DATA_PROFILE,open(file_name+'.p','wb'))

def clear_data_profile():
    global DATA_PROFILE
    DATA_PROFILE = {}

def test_cnns(num_iterations, log_filename = 'test_results.log'):

    @profile
    def load_resnet50():
        model = ResNet50(weights = 'imagenet')
        return model

    @profile
    def load_inceptionv3():
        model = InceptionV3(weights = 'imagenet')
        return model

    @profile
    def load_vgg16():
        model = VGG16(weights = 'imagenet')
        return model

    @profile
    def load_vgg19():
        model = VGG19(weights = 'imagenet')
        return model

    @profile
    def forward_pass_resnet50(model, x):
        predictions = model.predict(x)
        return predictions

    @profile
    def forward_pass_inceptionv3(model, x):
        predictions = model.predict(x)
        return predictions

    @profile
    def forward_pass_vgg16(model, x):
        predictions = model.predict(x)
        return predictions

    @profile
    def forward_pass_vgg19(model, x):
        predictions = model.predict(x)
        return predictions

    def preprocess_input_resnet50(image_filename):
        image = keras_preprocessor.load_img(image_filename,
                                            target_size = (224, 224))
        x = keras_preprocessor.img_to_array(image)
        x = np.expand_dims(x, axis=0)
        x = resnet50_input(x)
        return x

    def preprocess_input_inceptionv3(image_filename):
        image = keras_preprocessor.load_img(image_filename,
                                            target_size = (299, 299))
        x = keras_preprocessor.img_to_array(image)
        x = np.expand_dims(x, axis=0)
        x = inceptionv3_input(x)
        return x
    def preprocess_input_vgg16(image_filename):
        image = keras_preprocessor.load_img(image_filename,
                                            target_size = (224, 224))
        x = keras_preprocessor.img_to_array(image)
        x = np.expand_dims(x, axis=0)
        x = vgg16_input(x)
        return x

    def preprocess_input_vgg19(image_filename):
        image = keras_preprocessor.load_img(image_filename,
                                            target_size = (224, 224))
        x = keras_preprocessor.img_to_array(image)
        x = np.expand_dims(x, axis=0)
        x = vgg19_input(x)
        return x

    test_functions = {
        'resnet50':[load_resnet50, forward_pass_resnet50,
                   preprocess_input_resnet50],
        'inceptionv3':[load_inceptionv3, forward_pass_inceptionv3,
                      preprocess_input_inceptionv3],
        'vgg16':[load_vgg16,forward_pass_vgg16,preprocess_input_vgg16],
        'vgg19':[load_vgg19,forward_pass_vgg19,preprocess_input_vgg19]
        }

    image_filename = 'images/test_image.jpg'
    for iteration in range(num_iterations):
        for model_name, model_functions in test_functions.items():
            print(model_name)
            load_function = model_functions[0]
            forward_pass_function = model_functions[1]
            preprocess_input = model_functions[2]
            model = load_function()
            image_input = preprocess_input(image_filename)
            predictions = forward_pass_function(model,image_input)
            K.clear_session()

    print_data_profile()
    save_data_profile(log_filename)

if __name__ == '__main__':

    test_cnns(num_iterations=5,log_filename='test_results')



