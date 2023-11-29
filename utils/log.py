should_print = True

def set_should_print(bool):
    global should_print
    should_print = bool

def log(message):
    global should_print
    if should_print:
        print(message)

def show_running_info(params):
    log("Running XRL Test with the following parameters:")
    for key, value in params.items():
        log("\t{}: {}".format(key, value))