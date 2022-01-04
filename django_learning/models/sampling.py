from __future__ import print_function
import itertools
import math
import os
import pandas
import random

from django.conf import settings
from django.db import models
from django.conf import settings
from tqdm import tqdm

from django_commander.models import LoggedExtendedModel
from django_learning.utils import filter_queryset_by_params
from django_learning.utils.regex_filters import regex_filters
from django_learning.utils.sampling_frames import sampling_frames
from django_learning.utils.sampling_methods import sampling_methods
from pewtils import is_null, decode_text, is_not_null
from django_pewtils import get_model, CacheHandler
from pewanalytics.stats.sampling import SampleExtractor
from pewtils import get_hash
from pewanalytics.stats.sampling import compute_sample_weights_from_frame


class SamplingFrame(LoggedExtendedModel):

    """
    Sampling frames define a population of documents about which you want to learn something. They should represent
    the full scope of documents you want to study, and from this frame you can draw representative samples and train
    classifiers to make inferences about the population. They are defined in configurations files of the same name.
    """

    name = models.CharField(
        max_length=200,
        unique=True,
        help_text="Name of the sampling frame (must correspond to a config file)",
    )
    documents = models.ManyToManyField(
        "django_learning.Document",
        related_name="sampling_frames",
        help_text="The documents that belong to the sampling frame",
    )

    def __str__(self):

        return self.name

    def save(self, *args, **kwargs):

        if self.name not in sampling_frames.keys():
            raise Exception(
                "Sampling frame '{}' is not defined in any of the known folders".format(
                    self.name
                )
            )
        super(SamplingFrame, self).save(*args, **kwargs)

    @property
    def config(self):
        """
        Returns the configuration for the sampling frame from the config file
        :return:
        """
        return sampling_frames[self.name]()

    def extract_documents(self, refresh=False):
        """
        Uses the parameters in the config file to select a subset of documents for the frame.
        :param refresh: (default is False) if True, will refresh the documents associated with the sampling frame and \
            overwrite any existing relations
        :return:
        """

        print("Extracting sample frame '{}'".format(self.name))
        if self.documents.count() == 0 or refresh:

            if self.samples.count() > 0:
                print(
                    "Warning: you already have samples extracted from this sampling frame"
                )
                print(
                    "Updating the frame will require re-syncing the existing samples and may result in data loss."
                )
                print("Press 'c' to continue, otherwise 'q' to quit and abort")
                import pdb

                pdb.set_trace()
            params = self.config
            if params:
                objs = get_model("Document").objects.all()
                objs = list(
                    filter_queryset_by_params(objs, params)
                    .distinct()
                    .values_list("pk", flat=True)
                )
                self.documents.set(objs)
                self.save()
                print(
                    "Extracted {} documents for frame '{}'".format(
                        self.documents.count(), self.name
                    )
                )
            else:
                print("Error!  No frame named '{}' was found".format(self.name))

            self.get_sampling_flags(refresh=True)
            for s in self.samples.all():
                s.sync_with_frame()

        else:

            print(
                "If you want to overwrite the current frame, you need to explicitly declare refresh=True"
            )

    def get_sampling_flags(self, refresh=False, sampling_search_subset=None):
        """
        Iterates over samples that have been drawn from the sampling frame, notes any stratification or keyword
        oversampling variables that were used to pull the samples, and computes a dataframe with all of the required
        variables (binary flags) for sampling and weighting. This gets saved to the cache.

        :param refresh: (default is False) if True, existing cached data will be ignored and everything will be recomputed
        :param sampling_search_subset: (Optional) by default, this function will loop over all samples and compute binary
            columns for all of the ``regex_filter`` functions that were used in oversampling. Based on all of these searches,
            a ``search_none`` column is then computed to flag documents that don't match to any of the regexes used in
            sampling. If you wish to compute the ``search_none`` function off of a subset of regex flags (e.g. just the
            flags that were used to pull a particular sample) you can pass a list of those ``regex_filters`` to this
            variable, and the ``search_none`` column will be computed accordingly, effectively ignoring other regex
            searches that may have been for sampling in other samples linked to the frame. This has no effect on the
            cached data, it gets computed after the cached data is loaded.
        :return: A dataframe with binary columns for all of the variables that have been used to sample from the frame
        """

        cache = CacheHandler(
            os.path.join(
                settings.DJANGO_LEARNING_S3_CACHE_PATH, "sampling_frame_flags"
            ),
            hash=False,
            use_s3=settings.DJANGO_LEARNING_USE_S3,
            bucket=settings.S3_BUCKET,
        )

        frame = None
        if not refresh:
            frame = cache.read(self.name)
            # if is_not_null(frame):
            #     print("Loaded frame sampling flags from cache")

        sampling_searches = []
        stratification_variables = []
        additional_variables = {}
        for sample in self.samples.all():
            params = sample.get_params()
            stratify_by = params.get("stratify_by", None)
            if stratify_by and stratify_by not in stratification_variables:
                stratification_variables.append(stratify_by)
            for search in params.get("sampling_searches", []):
                if search["regex_filter"] not in sampling_searches:
                    sampling_searches.append(search["regex_filter"])
            additional_variables.update(params.get("additional_weights", {}))

        if is_null(frame) or refresh:

            # print("Recomputing frame sampling flags")

            vals = (
                ["pk", "text", "date"]
                + stratification_variables
                + list(set([a["field_lookup"] for a in additional_variables.values()]))
            )
            frame = pandas.DataFrame.from_records(self.documents.values(*vals))

            if len(sampling_searches) > 0:
                regex_patterns = {
                    search_name: regex_filters[search_name]()
                    for search_name in sampling_searches
                }

                for search_name, search_pattern in regex_patterns.items():
                    frame["search_{}".format(search_name)] = frame["text"].str.contains(
                        search_pattern
                    )
                frame["flag_counts"] = (
                    frame[
                        [
                            "search_{}".format(search_name)
                            for search_name in regex_patterns.keys()
                        ]
                    ]
                    .astype(int)
                    .sum(axis=1)
                )
                frame["search_none"] = frame["flag_counts"] == 0
                del frame["flag_counts"]

            for name, additional in additional_variables.items():
                frame[name] = frame[additional["field_lookup"]].map(
                    additional["mapper"]
                )
            for name, additional in additional_variables.items():
                try:
                    del frame[additional["field_lookup"]]
                except KeyError:
                    pass

            cache.write(self.name, frame, timeout=None)

        if len(sampling_searches) > 0:
            if sampling_search_subset and len(sampling_search_subset) > 0:
                sampling_searches = sampling_search_subset
            frame["search_none"] = ~frame[
                ["search_{}".format(s) for s in sampling_searches]
            ].max(axis=1)

        return frame


class Sample(LoggedExtendedModel):
    """
    A sample of documents that have been drawn from a sampling frame.
    """

    name = models.CharField(
        max_length=100,
        help_text="A name for the sample (must be unique within the project)",
    )
    project = models.ForeignKey(
        "django_learning.Project",
        related_name="samples",
        on_delete=models.CASCADE,
        help_text="The coding project the sample belongs to",
    )

    frame = models.ForeignKey(
        "django_learning.SamplingFrame",
        related_name="samples",
        on_delete=models.CASCADE,
        help_text="The sampling frame the sample was drawn from",
    )
    sampling_method = models.CharField(
        max_length=200,
        null=True,
        default="random",
        help_text="The method used for sampling (must correspond to a sampling method file)",
    )

    parent = models.ForeignKey(
        "django_learning.Sample",
        related_name="subsamples",
        null=True,
        on_delete=models.SET_NULL,
        help_text="The parent sample, if it's a subsample",
    )

    documents = models.ManyToManyField(
        "django_learning.Document",
        through="django_learning.SampleUnit",
        related_name="samples",
        help_text="Documents in the sample",
    )

    # AUTO-FILLED RELATIONS
    qualification_tests = models.ManyToManyField(
        "django_learning.QualificationTest",
        related_name="samples",
        help_text="Qualification tests required to code the sample (set automatically)",
    )

    class Meta:
        unique_together = ("project", "name")

    def __str__(self):

        if not self.parent:
            return "{}: sample '{}', {} documents using '{}' from {}".format(
                self.project,
                self.name,
                self.documents.count(),
                self.sampling_method,
                self.frame,
            )
        else:
            return "Subsample '{}' from {}, {} documents".format(
                self.name, self.parent, self.documents.count()
            )

    def save(self, *args, **kwargs):

        super(Sample, self).save(*args, **kwargs)
        self.qualification_tests.set(self.project.qualification_tests.all())

    def get_params(self):
        """
        Loads the sampling method from the config file
        :return:
        """

        try:
            return sampling_methods.get(self.sampling_method, None)()
        except TypeError:
            return {}

    def sync_with_frame(self, override_doc_ids=None):
        """
        Syncs the sample with the frame, adding or removing documents accordingly. Will raise warnings and confirmation
        prompts if this will result in data loss.
        :param override_doc_ids: (Optional) pass a specific list of Document primary keys to set the sample to
        :return:
        """

        if not override_doc_ids:
            override_doc_ids = list(self.documents.values_list("pk", flat=True))
        frame_ids = self.frame.documents.values_list("pk", flat=True)
        bad_ids = [id for id in override_doc_ids if id not in frame_ids]
        if len(bad_ids) > 0:
            print(
                "Warning: sample is out of sync with its frame; {} documents are now out-of-scope".format(
                    len(bad_ids)
                )
            )
            override_doc_ids = [id for id in override_doc_ids if id in frame_ids]
            print(
                "{} documents will remain in the sample".format(len(override_doc_ids))
            )
            print(
                "Press 'c' to continue and remove them (and any associated codes) from the sample, otherwise 'q' to quit"
            )
            import pdb

            pdb.set_trace()
            SampleUnit.objects.filter(sample=self, document_id__in=bad_ids).delete()

    def extract_documents(
        self,
        size=None,
        recompute_weights=False,
        override_doc_ids=None,
        allow_overlap_with_existing_project_samples=False,
        clear_existing_documents=False,
        skip_weighting=False,
        seed=None,
    ):
        """
        Pulls a sample from the frame using ``sampling_method``. If documents already exist for the specified sample,
        the existing sample will be expanded with ``size`` additional documents using the sampling method.

        :param size: Size of the sample to pull
        :param recompute_weights: (default is False) if True, recomputes the weights on the sample units
        :param override_doc_ids: (default is None) pass a specific list of Document primary keys to set the sample to
        :param allow_overlap_with_existing_project_samples: (default is False) if True, documents that are already
            associated with other samples in the project will be avaialble for sampling and inclusion in this sample;
            by default, documents can only be included in one sample for each project
        :param clear_existing_documents: (default is False) if True, existing documents will be removed and a fresh
            sample will be pulled; otherwise new documents will be added to the existing sample
        :param skip_weighting: (default is False) if True, sampling weights won't be computed
        :param seed: (Optional) random seed to be used during sampling
        :return:
        """

        if clear_existing_documents:
            print("Clearing out existing documents")
            self.document_units.all().delete()

        if recompute_weights:
            override_doc_ids = list(self.documents.values_list("pk", flat=True))
            if len(override_doc_ids) == 0:
                override_doc_ids = None
            self.sync_with_frame(override_doc_ids=override_doc_ids)

        params = self.get_params()
        if params:

            print("Extracting sampling frame")

            docs = self.frame.documents.all()
            if size and size < 1 and size > 0:
                size = int(float(docs.count()) * size)
            if (
                not allow_overlap_with_existing_project_samples
                and not recompute_weights
            ):
                existing_doc_ids = SampleUnit.objects.filter(
                    sample__project=self.project
                ).values_list("document_id", flat=True)
                docs = docs.exclude(pk__in=existing_doc_ids)
            if self.documents.count() > 0:
                docs = docs.exclude(pk__in=self.documents.values_list("pk", flat=True))

            stratify_by = params.get("stratify_by", None)
            try:
                frame = self.frame.get_sampling_flags(
                    sampling_search_subset=params.get("sampling_searches", None)
                )
                if stratify_by and stratify_by not in frame.columns:
                    raise KeyError()
            except KeyError:
                frame = self.frame.get_sampling_flags(
                    sampling_search_subset=[
                        s["regex_filter"] for s in params.get("sampling_searches", [])
                    ],
                    refresh=True,
                )
            frame = frame[frame["pk"].isin(list(docs.values_list("pk", flat=True)))]

            weight_vars = []
            use_keyword_searches = False
            if (
                "sampling_searches" in params.keys()
                and len(params["sampling_searches"]) > 0
            ):
                weight_vars.append("search_none")
                weight_vars.extend(
                    [
                        "search_{}".format(search["regex_filter"])
                        for search in params["sampling_searches"]
                    ]
                )
                use_keyword_searches = True

            if is_null(override_doc_ids):

                print("Extracting sample")

                sample_chunks = []
                if not use_keyword_searches:

                    sample_chunks.append(
                        SampleExtractor(frame, "pk", seed=seed).extract(
                            min([len(frame), int(size)]),
                            sampling_strategy=params["sampling_strategy"],
                            stratify_by=stratify_by,
                        )
                    )

                else:

                    sample_chunks = []
                    non_search_sample_size = 1.0 - sum(
                        [search["proportion"] for search in params["sampling_searches"]]
                    )
                    if non_search_sample_size > 0:
                        subset = frame[frame["search_none"] == 1]
                        sample_size = min(
                            [
                                len(subset),
                                int(math.ceil(float(size) * non_search_sample_size)),
                            ]
                        )
                        sample_chunks.append(
                            SampleExtractor(subset, "pk", seed=seed).extract(
                                sample_size,
                                sampling_strategy=params["sampling_strategy"],
                                stratify_by=stratify_by,
                            )
                        )

                    for search in params["sampling_searches"]:
                        subset = frame[
                            frame["search_{}".format(search["regex_filter"])] == 1
                        ]
                        sample_size = min(
                            [len(subset), int(math.ceil(size * search["proportion"]))]
                        )
                        sample_chunks.append(
                            SampleExtractor(subset, "pk", seed=seed).extract(
                                sample_size,
                                sampling_strategy=params["sampling_strategy"],
                                stratify_by=stratify_by,
                            )
                        )

                sample_ids = list(set(list(itertools.chain(*sample_chunks))))
                if len(sample_ids) < size:
                    fill_ids = list(docs.values_list("pk", flat=True))
                    if seed:
                        random.seed(seed)
                    random.shuffle(fill_ids)
                    while len(sample_ids) < size:
                        try:
                            sample_ids.append(fill_ids.pop())
                            sample_ids = list(set(sample_ids))
                        except IndexError:
                            break

            else:

                print("Override document IDs passed, skipping sample extraction")
                sample_ids = override_doc_ids

            if stratify_by:
                dummies = pandas.get_dummies(frame[stratify_by], prefix=stratify_by)
                weight_vars.extend(dummies.columns)
                frame = frame.join(dummies)

            weight_vars = list(set(weight_vars))
            df = frame.loc[frame["pk"].isin(sample_ids)].copy()
            if not skip_weighting:
                print("Computing weights")
                df["weight"] = list(
                    compute_sample_weights_from_frame(frame, df, weight_vars)
                )
            else:
                df["weight"] = 1.0

            print("Saving documents")

            if self.documents.count() > 0:
                new_docs = set(df["pk"].values).difference(
                    set(list(self.documents.values_list("pk", flat=True)))
                )
                if len(new_docs) > 0:
                    print(
                        "Warning: you're going to add {} additional documents to a sample that already has {} documents".format(
                            len(list(new_docs)), self.documents.count()
                        )
                    )
                    print(
                        "Please press 'c' to continue, or 'q' to cancel the operation"
                    )
                    import pdb

                    pdb.set_trace()

            for index, row in tqdm(df.iterrows(), desc="Updating sample documents"):
                SampleUnit.objects.create_or_update(
                    {"document_id": row["pk"], "sample": self},
                    {"weight": row["weight"]},
                    return_object=False,
                    save_nulls=False,
                )

            self.save()

        else:

            print("Error!  No module named '{}' was found".format(self.name))


class SampleUnit(LoggedExtendedModel):
    """
    A document that belongs to a sample. Contains a ``weight`` with the sampling weight designed to make the
    sample representative of the frame from which it was drawn.
    """

    document = models.ForeignKey(
        "django_learning.Document",
        related_name="sample_units",
        on_delete=models.CASCADE,
        help_text="The document that was sampled",
    )
    sample = models.ForeignKey(
        "django_learning.Sample",
        related_name="document_units",
        on_delete=models.CASCADE,
        help_text="The sample the document belongs to",
    )
    weight = models.FloatField(
        default=1.0,
        help_text="Sampling weight based on the sample and sampling frame the document belongs to",
    )

    class Meta:
        unique_together = ("document", "sample")

    def __str__(self):
        return "{}, {}, {}".format(self.sample, self.document, self.weight)

    def decoded_document_text(self):
        return decode_text(self.document.text)
