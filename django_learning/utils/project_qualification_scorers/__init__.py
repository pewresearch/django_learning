import re, os

from pewtils import extract_attributes_from_folder_modules
from django_pewtils import get_app_settings_folders


for mod_category, attribute_name in [("project_qualification_scorers", "scorer")]:
    mods = extract_attributes_from_folder_modules(
        os.path.join(__path__[0]),
        attribute_name,
        include_subdirs=True,
        concat_subdir_names=True,
    )
    conf_var = "DJANGO_LEARNING_{}".format(mod_category.upper())
    for folder in get_app_settings_folders(conf_var):
        mods.update(
            extract_attributes_from_folder_modules(
                folder, attribute_name, include_subdirs=True, concat_subdir_names=True
            )
        )
    globals()[mod_category] = mods
