should_print = True

def set_should_print(bool):
    global should_print
    should_print = bool

def log(message):
    global should_print
    if should_print:
        print(message)