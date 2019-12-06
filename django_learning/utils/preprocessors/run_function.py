import itertools

from django_learning.utils.preprocessors import BasicPreprocessor


class Preprocessor(BasicPreprocessor):

    def __init__(self, *args, **kwargs):

        self.name = "run_function"
        super(Preprocessor, self).__init__(*args, **kwargs)

    def run(self, text):

        return self.params["function"](text)