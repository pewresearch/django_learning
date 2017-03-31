# def filter(codes, **kwargs):
    #
    #     # TODO: convert from django queries to a pandas filtering looper (since you don't have objects any more, and consolidated dicts dont reflect actual DB objects)
    #     coder_stats = []
    #
    #     for coder_id in tqdm(codes.values_list("coder_id", flat=True).distinct()):
    #
    #         for code_variable_id in codes.filter(coder_id=coder_id).values_list("code__variable_id", flat=True).distinct():
    #
    #             doc_ids = codes.filter(coder_id=coder_id).filter(code__variable_id=code_variable_id).values_list("document_id", flat=True).distinct()
    #
    #             coder_codes = codes.filter(coder_id=coder_id).filter(code__variable_id=code_variable_id).filter(document_id__in=doc_ids)
    #             other_codes = codes.exclude(coder_id=coder_id).filter(code__variable_id=code_variable_id).filter(document_id__in=doc_ids)
    #
    #             coder_total = float(coder_codes.count())
    #             for code_id in other_codes.values_list("code_id", flat=True).distinct():
    #
    #                 stat = {
    #                     "coder_id": coder_id,
    #                     "code_variable_id": code_variable_id,
    #                     "code_id": code_id
    #                 }
    #
    #                 stat["coder_pct"] = float(coder_codes.filter(code_id=code_id).count()) / coder_total
    #
    #                 other_coder_pcts = []
    #                 for other_coder_id in other_codes.values_list("coder_id", flat=True):
    #                     other_coder_pcts.append(
    #                         float(other_codes.filter(coder_id=other_coder_id).filter(code_id=code_id).count()) /
    #                         float(other_codes.filter(coder_id=other_coder_id).count())
    #                     )
    #                 stat["other_pct_avg"] = numpy.average(other_coder_pcts)
    #
    #                 stat["coder_diff"] = stat["coder_pct"] - stat["other_pct_avg"]
    #
    #                 coder_stats.append(stat)
    #
    #     # return codes
    #
    #     return pandas.DataFrame(coder_stats)