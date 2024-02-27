import os
import numpy as np
import time

def show_iter_progress(
        i,
        iters,
        error,
        accuracy_check_span,
        accurate_predictions_train,
        all_data_train,
        accuracy_ratio_train,
        accurate_predictions_test,
        all_data_test,
        accuracy_ratio_test,
        accuracy_x_axis,
    ):

    os.system('cls' if os.name == 'nt' else 'clear')  
    
    percent = ( i / iters ) * 100

    print("======== progress ========")
    print(" ")
    print(f"iter  : {i} / {iters}   ( {percent:.2f} % done )")
    print(f"error : {error}")
    print(" ")
    print(" ")
    print(" ")
    print("======== accuracy ========")
    print(" ")
    print(f"accuracy check span :  {accuracy_check_span}")
    print(" ")
    print(f"check point : {i // accuracy_check_span} / {iters // accuracy_check_span}")
    print(" ")
    try:
        if( len(accuracy_x_axis) != 0 ):
            print(f"accuracy (train) : {accurate_predictions_train} / {all_data_train}   ( {accuracy_ratio_train:.2f} % )")
            print(f"accuracy (test)  : {accurate_predictions_test} / {all_data_test}   ( {accuracy_ratio_test:.2f} % )")
    except:
        pass

def show_iter_progress_for_gun(i, iters, picture_check_span, test_input):

    os.system('cls' if os.name == 'nt' else 'clear')  
    
    percent = ( i / iters ) * 100

    np.set_printoptions(precision=2)

    print("======== progress ========")
    print(" ")
    print(f"iter  : {i} / {iters}   ( {percent:.2f} % done )")
    print(" ")
    print(" ")
    print(" ")
    print("======== check picture ========")
    print(" ")
    print(f"picture check span :  {picture_check_span}")
    print(" ")
    print(f"check point : {i // picture_check_span} / {iters // picture_check_span}")
    print(" ")
    print(f"test input :  {test_input}")




 
