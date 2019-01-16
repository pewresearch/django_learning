import re, os

from pewtils import extract_json_from_folder
from django_pewtils import get_app_settings_folders


for json_category in [
    "project_qualification_tests"
]:
    mods = extract_json_from_folder(
        os.path.join(__path__[0], json_category),
        include_subdirs=True,
        concat_subdir_names=True
    )
    conf_var = "DJANGO_LEARNING_{}".format(json_category.upper())
    for folder in get_app_settings_folders(conf_var):
        mods.update(
            extract_json_from_folder(
                folder,
                include_subdirs=True,
                concat_subdir_names=True
            )
        )
    globals()[json_category] = mods