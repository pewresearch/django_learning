import os

with open(os.path.join(os.path.dirname(__file__), "VERSION"), "rb") as version_file:
    __version__ = version_file.read().strip()

# print "Using django_learning from: {}".format(os.path.dirname(__file__))