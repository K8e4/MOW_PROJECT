import os

def dprint(func):
    def wrapper(*args, **kwargs):
        ret = func(*args, **kwargs)

        if os.environ.get('DEBUG') == '1':
            print("\n>> S " + func.__name__  + " <<\n")
            print(ret)
            print("\n>> E " + func.__name__ + " <<")

            return ret

        return ret

    return wrapper
