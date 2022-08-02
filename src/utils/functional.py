def function_on_pair(function):
    return lambda x, y: (function(x), function(y))


def function_on_triplet(function):
    return lambda x, y, z: (function(x), function(y), function(z))
