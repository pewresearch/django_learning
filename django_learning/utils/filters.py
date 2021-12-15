from __future__ import print_function

from django_pewtils import get_model


def filter_hits(
    project=None,
    sample=None,
    turk_only=False,
    experts_only=False,
    finished_only=False,
    unfinished_only=False,
    assignments=None,
    exclude_coders=None,
    filter_coders=None,
    documents=None,
):
    """
    Returns a query set of HITs based on various filtering criteria.

    :param project: Optionally filter to HITs that belong to a particular project
    :param sample: Optionally filter to HITs that belong to a particular sample
    :param turk_only: (default is False) if True, filters to HITs that have been deployed on Mechanical Turk
    :param experts_only: (default is False) if True, filters to in-house HITs
    :param finished_only: (default is False) if True, filters to HITs that have been finished
    :param unfinished_only: (default is False) if True, filters to HITs that have not been finished
    :param assignments: Optionally filters to HITs associated with a particular query set of assignments
    :param exclude_coders: Optionally filters to HITs that have not been completed by a particular query set of coders
    :param filter_coders: Optionally filters to HITs that have been completed by a particular query set of coders
    :param documents: Optionally filters to HITs associated with a particular query set of documents
    :return: A query set of HITs
    """

    hits = get_model("HIT", app_name="django_learning").objects.all()
    if project:
        hits = hits.filter(sample__project=project)
    if sample:
        hits = hits.filter(sample=sample)
    if turk_only:
        hits = hits.filter(turk=True)
    elif experts_only:
        hits = hits.filter(turk=False)
    if finished_only:
        hits = hits.filter(finished=True)
    elif unfinished_only:
        hits = hits.filter(finished=False)
    if exclude_coders != None:
        hits = hits.exclude(assignments__coder__in=exclude_coders)
    if filter_coders != None:
        hits = hits.filter(assignments__coder__in=filter_coders)
    if assignments != None:
        hits = hits.filter(assignments__in=assignments)
    if documents != None:
        hits = hits.filter(sample_unit__document__in=documents)

    return hits.distinct()


def filter_assignments(
    project=None,
    sample=None,
    turk_only=False,
    experts_only=False,
    coder_min_hit_count=None,
    coder=None,
    completed_only=False,
    incomplete_only=False,
    hits=None,
    exclude_coders=None,
    filter_coders=None,
    documents=None,
    **kwargs
):
    """
    Returns a query set of assignments based on various filtering criteria.

    :param project: Optionally filter to assignmennts that belong to a particular project
    :param sample: Optionally filter to assignments that belong to a particular sample
    :param turk_only: (default is False) if True, filters to assignments that have been completed on Mechanical Turk
    :param experts_only: (default is False) if True, filters to in-house assignments
    :param coder_min_hit_count: Optionally filter to assignments attached to coders that have completed a minimum
        number of HITs on the project (and sample, if applicable)
    :param coder: Optionally filter to assignments associated with a particular coder
    :param completed_only: (default is False) if True, filters to assignments that have been finished
    :param incomplete_only: (default is False) if True, filters to assignments that have not been finished
    :param hits: Optionally filters to assignments associated with a particular query set of HITs
    :param exclude_coders: Optionally filters to assignments that are not associated with a particular query set of coders
    :param filter_coders: Optionally filters to assignments that are associated with a particular query set of coders
    :param documents: Optionally filters to assignments associated with a particular query set of documents
    :return: A query set of assignments
    """

    assignments = get_model("Assignment", app_name="django_learning").objects.all()
    if project:
        assignments = assignments.filter(hit__sample__project=project)
    if sample:
        assignments = assignments.filter(hit__sample=sample)
    if turk_only:
        assignments = assignments.filter(hit__turk=True)
    elif experts_only:
        assignments = assignments.filter(hit__turk=False)
    if coder_min_hit_count:
        assignments = assignments.filter(
            coder__in=filter_coders(
                project, sample=sample, min_hit_count=coder_min_hit_count
            )
        )
    if coder:
        assignments = assignments.filter(coder=coder)
    if completed_only:
        assignments = assignments.filter(time_finished__isnull=False)
    elif incomplete_only:
        assignments = assignments.filter(time_finished__isnull=True)
    if exclude_coders != None:
        assignments = assignments.exclude(coder__in=exclude_coders)
    if filter_coders != None:
        assignments = assignments.filter(coder__in=filter_coders)
    if hits != None:
        assignments = assignments.filter(hit__in=hits)
    if documents != None:
        assignments = assignments.filter(hit__sample_unit__document__in=documents)

    return assignments.distinct()


def filter_coders(project=None, sample=None, min_hit_count=None):
    """
    Returns a query set of coders based on various filtering criteria.

    :param project: Optionally filters to coders assigned to a particular project
    :param sample: Optionally filters to coders that have completed assignments belonging to a particular sample
    :param min_hit_count: Optionally filters to coders that have completed a minimum number of HITs on the project \
        (and sample, if specified)
    :return: A query set of coders
    """

    good_coder_ids = []
    coders = get_model("Coder", app_name="django_learning").objects.all()
    if project:
        coders = project.coders.all()
    if sample:
        coders = coders.filter(assignments__hit__sample=sample)
    for c in coders:
        a = c.assignments.filter(hit__sample__project=project).filter(
            time_finished__isnull=False
        )
        if sample:
            a = a.filter(hit__sample=sample)
        if not min_hit_count or a.count() >= min_hit_count:
            good_coder_ids.append(c.pk)

    return (
        get_model("Coder", app_name="django_learning")
        .objects.filter(pk__in=good_coder_ids)
        .distinct()
    )
