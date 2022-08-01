def function_on_pair(function):
    return lambda x, y: function(x), function(y)
