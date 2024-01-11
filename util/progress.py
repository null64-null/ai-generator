import time

def variable(value):
    formatted_value = f"{value:.1f}"
    return formatted_value

def show_progress(done, all):
    percent = 100*done/all

    if percent == 100:
        print(str(done) + "/" + str(all) + " | " + variable(percent) + "% : done")
    else:
        print(str(done) + "/" + str(all) + " | " + variable(percent) + "% : done", end='\r')