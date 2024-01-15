import os
import time

def show_iter_progress(
        i,
        iters,
        error,
        accuracy_check_span,
        train_data_length,
        test_data_length,
        accuracy_result_train_data,
        accuracy_result_test_data,
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
            percent_tr = ( accuracy_result_train_data[-1] / train_data_length ) * 100
            print(f"accuracy (train) : {accuracy_result_train_data[-1]} / {train_data_length}   ( {percent_tr:.2f} % )")
            percent_te = ( accuracy_result_test_data[-1] / test_data_length ) * 100
            print(f"accuracy (test)  : {accuracy_result_test_data[-1]} / {test_data_length}   ( {percent_te:.2f} % )")
    except:
        pass




 
