
def derive(f, x, h=0.0001):
    # Basic value tests
    if f is None or x is None or h is None:
        raise ValueError("Arguments 'f', 'x', and 'h' must not be None.")
    if not isinstance(h, (int, float)):
        raise ValueError("'h' must be a numeric value.")

    # Making sure the function f() is callable
    if not callable(f):
        raise ValueError("'f' must be a callable function.")

    return (f(x+h)-f(x))/h
