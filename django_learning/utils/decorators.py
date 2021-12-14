from pewtils import is_null


def require_model(func):
    """
    Decorator that gets used on functions belonging to classes that inherit from ``LearningModel``. Forces the
    loading of the model if it's not already been loaded.
    :param func:
    :return:
    """

    def wrapper(self, *args, **options):
        if is_null(self.model, empty_lists_are_null=True):
            self.load_model(**options)
        return func(self, *args, **options)

    return wrapper


def require_training_data(func):
    """
    Decorator that gets used on functions belonging to classes that inherit from ``LearningModel``. Forces the loading
    of the training dataset if it's not already been loaded.
    :param func:
    :return:
    """

    def wrapper(self, *args, **options):
        if is_null(self.training_data, empty_lists_are_null=True):
            self.load_training_data(**options)
        return func(self, *args, **options)

    return wrapper


def temp_cache_wrapper(func):
    """
    Decorator that gets used on functions belonging to classes that inherit from ``LearningModel``. Clears out the
    temporary row-level cache before and after the function is run, unless ``clear_temp_cache=False`` is explicitly
    passed in the keyword arguments.
    :param func:
    :return:
    """

    def wrapper(self, *args, **options):
        if "clear_temp_cache" in options.keys():
            clear_temp_cache = options.pop("clear_temp_cache")
        else:
            clear_temp_cache = True
        if clear_temp_cache:
            self.temp_cache.clear()
        results = func(self, *args, **options)
        if clear_temp_cache:
            self.temp_cache.clear()
        return results

    return wrapper
