from functools import wraps
from keras import backend as K
import numpy as np
import time

PROFILE_DATA = {}
def profile(function):
    @wraps(function)
    def with_profiling(*args,**kwargs):
        start_time = time.monotonic()

        function_return = function(*args,**kwargs)

        elapsed_time = time.monotonic() - start_time

        if function.__name__ not in PROFILE_DATA:
            PROFILE_DATA[function.__name__] = [0,[]]

        PROFILE_DATA[function.__name__][0] += 1
        PROFILE_DATA[function.__name__][1].append(elapsed_time)

        return function_return

    return with_profiling

def print_profile_data():
    for function_name, data in PROFILE_DATA.items():
        max_time = np.max(data[1])
        min_time = np.min(data[1])
        avg_time = np.mean(data[1])
        var_time = np.var(data[1])
        print('Function %s called %d times.' % (function_name, data[0]))
        print('Execution time min: %.3f' % min_time)
        print('Execution time max: %.3f' % max_time)
        print('Execution time average: %.3f' % avg_time)
        print('Execution time variance: %.3f' % var_time)

def save_profile_data(file_name):
    for function_name, data in PROFILE_DATA.items():
        max_time = np.max(data[1])
        min_time = np.min(data[1])
        avg_time = np.mean(data[1])
        var_time = np.var(data[1])
        log_file = open(file_name,'w')
        log_file.write('Function %s called %d times.' % (function_name, data[0]))
        log_file.write('Execution time min: %.3f' % min_time)
        log_file.write('Execution time max: %.3f' % max_time)
        log_file.write('Execution time average: %.3f' % avg_time)
        log_file.write('Execution time variance: %.3f' % var_time)

def clear_profile_data():
    global PROFILE_DATA
    PROFILE_DATA = {}

if __name__ == '__main__':

    from keras.applications import InceptionV3,ResNet50,VGG16,VGG19

    @profile
    def load_inceptionv3():
        model = InceptionV3(weights = 'imagenet')
        return model

    @profile
    def load_resnet50():
        model = ResNet50(weights = 'imagenet')
        return model

    @profile
    def load_vgg16():
        model = VGG16(weights = 'imagenet')
        return model

    @profile
    def load_vgg19():
        model = VGG19(weights = 'imagenet')
        return model

    models_dictionary = {
        'vgg16':load_vgg16(),
        'vgg19':load_vgg19(),
        'inceptionv3':load_inceptionv3(),
        'resnet50':load_resnet50() }

    num_iterations = 1
    for iteration in range(num_iterations):
        for model_name,model_function in models_dictionary.items():
            model = model_function()
            K.clear_session()

    print_profile_data()
    save_profile_data('loading_times.log')


