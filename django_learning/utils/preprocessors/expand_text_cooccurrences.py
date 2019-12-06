import itertools

from django_learning.utils.preprocessors import BasicPreprocessor


class Preprocessor(BasicPreprocessor):

    def __init__(self, *args, **kwargs):

        self.name = "clean_text"
        super(Preprocessor, self).__init__(*args, **kwargs)

    def run(self, text):

        return " ".join([" ".join([a, b]) for a, b in itertools.combinations(text.split(), 2)])