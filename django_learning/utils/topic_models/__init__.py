import os

from pewtils import is_not_null, decode_text, extract_attributes_from_folder_modules, extract_json_from_folder
from django_pewtils import CacheHandler, reset_django_connection_wrapper, get_model, get_app_settings_folders
from django_learning.utils import get_param_repr


for mod_category, attribute_name in [
    ("topic_models", "get_parameters")
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


