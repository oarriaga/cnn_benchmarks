import matplotlib.pyplot as plt
import numpy as np
import pickle

test_results = pickle.load(open('test_results.p','rb'))
num_functions, num_points = np.asarray(list(test_results.values())).shape
forward_pass_data = np.zeros((num_points,num_functions//2))
loading_data = np.zeros((num_points,num_functions//2))
loading_functions = []
forward_pass_functions = []
forward_pass_arg = 0
loading_arg = 0
for function_name, function_results in test_results.items():
    if 'forward_pass' in function_name:
        forward_pass_functions.append(function_name)
        forward_pass_data[:,forward_pass_arg] = function_results
        forward_pass_arg = forward_pass_arg + 1
    if 'load' in function_name:
        loading_functions.append(function_name)
        loading_data[:,loading_arg] = function_results
        loading_arg = loading_arg + 1

def make_box_plot(data,labels,image_name,save_image=False):
    plt.figure(figsize=(12,5))
    plt.boxplot(data,labels=labels,
                showmeans=False,
                meanline=True,
                showbox=True)
    plt.title('CNN benchmarks')
    plt.ylabel('time (s)')
    if save_image == True:
        plt.savefig(image_name)
    else:
        plt.show()

make_box_plot(forward_pass_data,
            forward_pass_functions,
            image_name = 'forward_pass',
            save_image=True)
make_box_plot(loading_data,
            loading_functions,
            image_name = 'loading_time',
            save_image=True)
