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

    name = models.CharField(max_length=200, unique=True)
    documents = models.ManyToManyField(
        "django_learning.Document", related_name="sampling_frames"
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
        return sampling_frames[self.name]()

    def extract_documents(self, refresh=False):

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

        cache = CacheHandler(
            os.path.join(
                settings.DJANGO_LEARNING_S3_CACHE_PATH, "sampling_frame_flags"
            ),
            hash=False,
            use_s3=settings.DJANGO_LEARNING_USE_S3,
            aws_access=settings.AWS_ACCESS_KEY_ID,
            aws_secret=settings.AWS_SECRET_ACCESS_KEY,
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
            for search_name in params.get("sampling_searches", {}).keys():
                if search_name not in sampling_searches:
                    sampling_searches.append(search_name)
            additional_variables.update(params.get("additional_weights", {}))

        if is_null(frame) or refresh:

            # print("Recomputing frame sampling flags")

            vals = (
                ["pk", "text", "date"]
                + stratification_variables
                + [a["field_lookup"] for a in additional_variables.values()]
            )
            frame = pandas.DataFrame.from_records(self.documents.values(*vals))

            if len(sampling_searches) > 0:
                regex_patterns = {
                    search_name: regex_filters[search_name]().pattern
                    for search_name in sampling_searches
                }
                frame["search_none"] = ~frame["text"].str.contains(
                    r"|".join(regex_patterns.values())
                )
                for search_name, search_pattern in regex_patterns.items():
                    frame["search_{}".format(search_name)] = frame["text"].str.contains(
                        search_pattern
                    )

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

    DISPLAY_CHOICES = (
        ("article", "Article"),
        ("image", "Image"),
        ("audio", "Audio"),
        ("video", "Video"),
    )

    name = models.CharField(max_length=100)
    project = models.ForeignKey(
        "django_learning.Project", related_name="samples", on_delete=models.CASCADE
    )

    frame = models.ForeignKey(
        "django_learning.SamplingFrame",
        related_name="samples",
        on_delete=models.CASCADE,
    )
    sampling_method = models.CharField(max_length=200, null=True, default="random")

    parent = models.ForeignKey(
        "django_learning.Sample",
        related_name="subsamples",
        null=True,
        on_delete=models.SET_NULL,
    )

    documents = models.ManyToManyField(
        "django_learning.Document",
        through="django_learning.SampleUnit",
        related_name="samples",
    )

    display = models.CharField(max_length=20, choices=DISPLAY_CHOICES)

    # AUTO-FILLED RELATIONS
    qualification_tests = models.ManyToManyField(
        "django_learning.QualificationTest", related_name="samples"
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

        try:
            return sampling_methods.get(self.sampling_method, None)()
        except TypeError:
            return {}

    def sync_with_frame(self, override_doc_ids=None):

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
            except KeyError:
                frame = self.frame.get_sampling_flags(
                    sampling_search_subset=params.get("sampling_searches", None),
                    refresh=True,
                )
            frame = frame[frame["pk"].isin(list(docs.values_list("pk", flat=True)))]

            weight_vars = []
            use_keyword_searches = False
            if (
                "sampling_searches" in params.keys()
                and len(params["sampling_searches"].keys()) > 0
            ):
                weight_vars.append("search_none")
                weight_vars.extend(
                    [
                        "search_{}".format(name)
                        for name in params["sampling_searches"].keys()
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
                        [
                            p["proportion"]
                            for search, p in params["sampling_searches"].items()
                        ]
                    )
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

                    for search, p in params["sampling_searches"].items():
                        subset = frame[frame["search_{}".format(search)] == 1]
                        sample_size = min(
                            [len(subset), int(math.ceil(size * p["proportion"]))]
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

    document = models.ForeignKey(
        "django_learning.Document",
        related_name="sample_units",
        on_delete=models.CASCADE,
    )
    sample = models.ForeignKey(
        "django_learning.Sample",
        related_name="document_units",
        on_delete=models.CASCADE,
    )
    weight = models.FloatField(default=1.0)

    class Meta:
        unique_together = ("document", "sample")

    def __str__(self):
        return "{}, {}, {}".format(self.sample, self.document, self.weight)

    def decoded_document_text(self):
        return decode_text(self.document.text)
