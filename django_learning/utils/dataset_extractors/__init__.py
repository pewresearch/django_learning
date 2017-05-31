import os, numpy, pandas, inspect

from tqdm import tqdm

from django.conf import settings

from pewtils import classproperty, is_not_null, is_null, decode_text, extract_attributes_from_folder_modules, extract_json_from_folder
from pewtils.django import CacheHandler, reset_django_connection_wrapper, get_model, get_app_settings_folders

from django_learning.utils.dataset_code_filters import dataset_code_filters
from django_learning.utils.dataset_coder_filters import dataset_coder_filters
from django_learning.utils.dataset_document_filters import dataset_document_filters


class DatasetExtractor(object):

    def __init__(self, outcome_column=None, **kwargs):

        self.outcome_column = outcome_column
        self.cache_hash = None
        self.cache = CacheHandler(os.path.join(settings.CACHE_PATH, "datasets"),
            hash=False,
            use_s3=True,
            bucket=settings.S3_BUCKET,
            aws_access=settings.AWS_ACCESS_KEY_ID,
            aws_secret=settings.AWS_SECRET_ACCESS_KEY
        )
        self.dataset = None

        self.kwargs = kwargs

    def get_hash(self, **kwargs):

        hash_key = str(inspect.getsourcelines(self._get_dataset)) + str({k: v for k, v in kwargs.iteritems() if k != "refresh"})

        return self.cache.file_handler.get_key_hash(hash_key)

    def set_outcome_column(self, outcome_col):

        self.outcome_column = outcome_col

    def extract(self, refresh=False, only_get_existing=False, **kwargs):

        cache_data = None

        if not refresh:
            if not self.cache_hash:
                self.cache_hash = self.get_hash(**kwargs)
            cache_data = self.cache.read(self.cache_hash)
            if is_not_null(cache_data):
                for k, v in cache_data.iteritems():
                    if k != "dataset":
                        setattr(self, k, v)

        if is_null(cache_data) and not only_get_existing:
            updated_hashstr = self.get_hash(**kwargs)
            cache_data = {"dataset": self._get_dataset(**kwargs)}
            cache_data.update(self._get_preserved_state())
            self.cache.write(updated_hashstr, cache_data)
            self.cache_hash = updated_hashstr

        return cache_data["dataset"]

    def _get_preserved_state(self, **kwargs):

        return {}

    def _get_dataset(self, **kwargs):

        raise NotImplementedError


# TODO: add some sort of gold_standard adjudication system, and add gold_standard_only as an additional dataset code filter
# # write a generic function that computes reliability stats within or between two datasets, based off of document_id and coder_id overlaps
# # you'll need to be able to compute:
#     # - two coders against each other
#     # - all coder combinations against each other (if they have a minimum overlap)
#     # - turkers vs. experts
#     # - consolidated metaturker vs. gold standard expert
#     # - model predictions vs. training data
#     # - model predictions vs. expert gold standard
# # there will be two different ways of doing a lot of these e.g. compute model vs. turk by:
#     # A) having to separate datasets and passing them together into a function, or
#     # B) concatenating them into a dataframe as aggregates (coder_id = 'model' or 'experts') and running it within a dataframe
#
# # you'll also need to detect if the outcome column is a categorical or numeric outcome
# # and restrict functions accordingly (classification vs. regression)

for mod_category, attribute_name in [
    ("dataset_extractors", "Extractor")
]:
    mods = extract_attributes_from_folder_modules(
        os.path.join(__path__[0]),
        attribute_name,
        include_subdirs=True,
        concat_subdir_names=True
    )
    conf_var = "DJANGO_LEARNING_{}".format(mod_category.upper())
    for folder in get_app_settings_folders(conf_var):
        mods.update(
            extract_attributes_from_folder_modules(
                folder,
                attribute_name,
                include_subdirs=True,
                concat_subdir_names=True
            )
        )
    globals()[mod_category] = mods

