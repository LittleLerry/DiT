import os
def AIGenerated(f):
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapper

class f_lock():
    def __init__(self, path_to_lock):
        self.lock = path_to_lock

    def acquire(self,):
        if os.path.exists(self.lock):
            return False
        return True