# from django_pewtils import get_model
#
#
# # def filter(documents, variable_name=None, code_value=None):
# #
# #     if variable_name and code_value:
# #
# #         code = get_model("Code").objects.get(
# #             value=code_value,
# #             variable=get_model("CodeVariable").objects.get(name=variable_name)
# #         )
# #         documents = documents.filter(classifier_codes__code=code)
# #
# #     return documents
#
#
#
# def filter(documents, classifier_name=None, code_value=None):
#
#     if classifier_name and code_value:
#
#         c = get_model("DocumentClassificationModel").objects.get(name=classifier_name)
#         code_pk = c.variable.codes.get(value=code_value).pk
#         h = c.handler
#         h.load_model()
#         df = h.apply_model(documents)
#         valid_ids = df[df[h.outcome_variable]==code_pk]['pk'].values
#         documents = documents.filter(pk__in=valid_ids)
#
#     return documents
