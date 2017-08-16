import pandas, math, random, itertools, os

from django.db import models
from django.conf import settings

from pewtils import is_null, decode_text, is_not_null
from pewtils.nlp import get_hash
from pewtils.sampling import compute_sample_weights_from_frame
from pewtils.django import get_model, CacheHandler
from pewtils.django.sampling import SampleExtractor

from tqdm import tqdm

from django_commander.models import LoggedExtendedModel
from django_learning.utils.sampling_frames import sampling_frames
from django_learning.utils.regex_filters import regex_filters
from django_learning.utils.sampling_methods import sampling_methods
from django_learning.settings import S3_CACHE_PATH, LOCAL_CACHE_PATH


class SamplingFrame(LoggedExtendedModel):

    name = models.CharField(max_length=200, unique=True)
    documents = models.ManyToManyField("django_learning.Document", related_name="sampling_frames")

    def __str__(self):

        # return "{}, {} documents".format(self.name, self.documents.count())
        return self.name

    def save(self, *args, **kwargs):

        if self.name not in sampling_frames.keys():
            raise Exception("Sampling frame '{}' is not defined in any of the known folders".format(self.name))
        super(SamplingFrame, self).save(*args, **kwargs)

    @property
    def config(self):
        return sampling_frames[self.name]()

    def extract_documents(self, refresh=False):

        print "Extracting sample frame '{}'".format(self.name)
        if self.documents.count() == 0 or refresh:

            params = self.config
            if params:
                objs = get_model("Document").objects.all()
                if "filter_dict" in params.keys() and params["filter_dict"]:
                    objs = objs.filter(**params["filter_dict"])
                if "exclude_dict" in params.keys() and params["exclude_dict"]:
                    objs = objs.exclude(**params["exclude_dict"])
                self.documents = objs
                self.save()
                print "Extracted {} documents for frame '{}'".format(self.documents.count(), self.name)
            else:
                print "Error!  No frame named '{}' was found".format(self.name)

        else:

            print "If you want to overwrite the current frame, you need to explicitly declare refresh=True"


    def get_sampling_flags(self, refresh=True):

        cache = CacheHandler(os.path.join(S3_CACHE_PATH, "sampling_frame_flags"),
            hash=False,
            use_s3=True,
            aws_access=settings.AWS_ACCESS_KEY_ID,
            aws_secret=settings.AWS_SECRET_ACCESS_KEY,
            bucket=settings.S3_BUCKET
        )

        frame = None
        if not refresh:
            frame = cache.read(self.name)
            if is_not_null(frame):
                print "Loaded frame sampling flags from cache"

        if is_null(frame) or refresh:

            print "Recomputing frame sampling flags"
            stratification_variables = []
            sampling_searches = []
            for sample in self.samples.all():
                params = sample.get_params()
                stratify_by = params.get("stratify_by", None)
                if stratify_by and stratify_by not in stratification_variables:
                    stratification_variables.append(stratify_by)
                for search_name in params.get("sampling_searches", {}).keys():
                    if search_name not in sampling_searches:
                        sampling_searches.append(search_name)

            vals = ["pk", "text"] + stratification_variables
            frame = pandas.DataFrame.from_records(self.documents.values(*vals))

            if len(sampling_searches) > 0:
                regex_patterns = {search_name: regex_filters[search_name]().pattern for search_name in sampling_searches}
                frame['search_none'] = ~frame["text"].str.contains(r"|".join(regex_patterns.values()))
                for search_name, search_pattern in regex_patterns.iteritems():
                    frame["search_{}".format(search_name)] = frame["text"].str.contains(search_pattern)

            cache.write(self.name, frame)

        return frame


class Sample(LoggedExtendedModel):

    DISPLAY_CHOICES = (
        ('article', 'Article'),
        ('image', 'Image'),
        ('audio', 'Audio'),
        ('video', 'Video')
    )

    name = models.CharField(max_length=100)
    project = models.ForeignKey("django_learning.Project", related_name="samples")

    hit_type = models.ForeignKey("django_learning.HITType", related_name="samples")
    frame = models.ForeignKey("django_learning.SamplingFrame", related_name="samples")
    sampling_method = models.CharField(max_length=200, null=True, default="random")

    parent = models.ForeignKey("django_learning.Sample", related_name="subsamples", null=True)

    documents = models.ManyToManyField("django_learning.Document", through="django_learning.SampleUnit", related_name="samples")

    display = models.CharField(max_length=20, choices=DISPLAY_CHOICES)

    # AUTO-FILLED RELATIONS
    qualification_tests = models.ManyToManyField("django_learning.QualificationTest", related_name="samples")

    class Meta:
        unique_together = ("project", "name")

    def __str__(self):

        if not self.parent:
            return "{}: sample '{}', {} documents using '{}' from {}, HITType {}".format(
                self.project,
                self.name,
                self.documents.count(),
                self.sampling_method,
                self.frame,
                self.hit_type
            )
        else:
            return "Subsample '{}' from {}, {} documents".format(
                self.name,
                self.parent,
                self.documents.count()
            )

    def save(self, *args, **kwargs):

        super(Sample, self).save(*args, **kwargs)
        self.qualification_tests = self.project.qualification_tests.all()

    def get_params(self):

        return sampling_methods.get(self.sampling_method, None)()

    def extract_documents(self,
        size=None,
        recompute_weights=False,
        override_doc_ids=None,
        allow_overlap_with_existing_project_samples=False,
        clear_existing_documents=False,
        skip_weighting=False
    ):

        if clear_existing_documents:
            print "Clearing out existing documents"
            self.document_units.all().delete()

        if recompute_weights:
            override_doc_ids = list(self.documents.values_list("pk", flat=True))
            frame_ids = self.frame.documents.values_list("pk", flat=True)
            bad_ids = [id for id in override_doc_ids if id not in frame_ids]
            if len(bad_ids) > 0:
                print "Warning: sample is out of sync with its frame; {} documents are now out-of-scope".format(
                    len(bad_ids))
                override_doc_ids = [id for id in override_doc_ids if id in frame_ids]
                print "{} documents will remain in the sample".format(len(override_doc_ids))
                print "Press 'c' to continue and remove them from the sample, otherwise 'q' to quit"
                import pdb
                pdb.set_trace()
                SampleUnit.objects.filter(sample=self, document_id__in=bad_ids).delete()

        params = self.get_params()
        if params:

            print "Extracting sampling frame"

            docs = self.frame.documents.all()
            if size and size < 1 and size > 0:
                size = int(len(docs)*size)
            if not allow_overlap_with_existing_project_samples and not recompute_weights:
                existing_doc_ids = SampleUnit.objects.filter(sample__project=self.project).values_list("document_id", flat=True)
                docs = docs.exclude(pk__in=existing_doc_ids)

            stratify_by = params.get("stratify_by", None)
            frame = self.frame.get_sampling_flags()
            frame = frame[frame["pk"].isin(list(docs.values_list("pk", flat=True)))]

            weight_vars = []
            use_keyword_searches = False
            if "sampling_searches" in params.keys() and len(params["sampling_searches"].keys()) > 0:
                weight_vars.append("search_none")
                weight_vars.extend(["search_{}".format(name) for name in params["sampling_searches"].keys()])
                use_keyword_searches = True

            if is_null(override_doc_ids):

                print "Extracting sample"

                sample_chunks = []
                if not use_keyword_searches:

                    sample_chunks.append(
                        SampleExtractor(
                            sampling_strategy=params["sampling_strategy"],
                            id_col="pk",
                            stratify_by=stratify_by
                        ).extract(
                            frame,
                            sample_size=int(size)
                        )
                    )

                else:

                    sample_chunks = []
                    non_search_sample_size = 1.0 - sum([p['proportion'] for search, p in params["sampling_searches"].iteritems()])
                    sample_chunks.append(
                        SampleExtractor(
                            sampling_strategy=params["sampling_strategy"],
                            id_col="pk",
                            stratify_by=stratify_by
                        ).extract(
                            frame[frame["search_none"] == 1],
                            sample_size=int(math.ceil(float(size) * non_search_sample_size))
                        )
                    )

                    for search, p in params["sampling_searches"].iteritems():
                        sample_chunks.append(
                            SampleExtractor(
                                sampling_strategy=params["sampling_strategy"],
                                id_col="pk",
                                stratify_by=stratify_by
                            ).extract(
                                frame[frame["search_{}".format(search)] == 1],
                                sample_size=int(math.ceil(size * p["proportion"]))
                            )
                        )

                sample_ids = list(set(list(itertools.chain(*sample_chunks))))
                if len(sample_ids) < size:
                    # fill_ids = list(frame[~frame["pk"].isin(sample_ids)]["pk"].values)
                    fill_ids = list(docs.values_list("pk", flat=True))
                    random.shuffle(fill_ids)
                    while len(sample_ids) < size:
                        try:
                            sample_ids.append(fill_ids.pop())
                            sample_ids = list(set(sample_ids))
                        except IndexError: break

            else:

                print "Override document IDs passed, skipping sample extraction"
                sample_ids = override_doc_ids

            if stratify_by:
                dummies = pandas.get_dummies(frame[stratify_by], prefix=stratify_by)
                weight_vars.extend(dummies.columns)
                frame = frame.join(dummies)

            weight_vars = list(set(weight_vars))
            df = frame[frame['pk'].isin(sample_ids)]
            if not skip_weighting:
                print "Computing weights"
                df['weight'] = compute_sample_weights_from_frame(frame, df, weight_vars)
            else:
                df['weight'] = 1.0

            print "Saving documents"

            if self.documents.count() > 0:
                new_docs = set(df["pk"].values).difference(set(list(self.documents.values_list("pk", flat=True))))
                if len(new_docs) > 0:
                    print "Warning: you're going to add {} additional documents to a sample that already has {} documents".format(
                        len(list(new_docs)),
                        self.documents.count()
                    )
                    print "Please press 'c' to continue, or 'q' to cancel the operation"
                    import pdb
                    pdb.set_trace()

            for index, row in tqdm(df.iterrows(), desc="Updating sample documents"):
                SampleUnit.objects.create_or_update(
                    {"document_id": row["pk"], "sample": self},
                    {"weight": row["weight"]},
                    return_object=False,
                    save_nulls=False
                )

            self.save()

        else:

            print "Error!  No module named '{}' was found".format(self.name)

    # def extract_subsample(self, pct, training=False, override_doc_ids=None):

    #     frame = pandas.DataFrame.from_records(self.documents.values("pk"))

    #     if is_not_null(override_doc_ids):

    #         print "Override document IDs passed, skipping sample extraction"
    #         sample_ids = override_doc_ids

    #     else:

    #         size = int(float(self.documents.count()) * pct)
    #         sample_ids = SampleExtractor(sampling_strategy="random", id_col="pk").extract(frame, sample_size=int(size))

    #     print "Computing weights"

    #     df = frame[frame['pk'].isin(sample_ids)]
    #     df['weight'] = compute_sample_weights_from_frame(frame, df, [])

    #     print "Creating subsample"

    #     subsample = get_model("DocumentSample").objects.create(
    #         parent=self,
    #         method="random",
    #         frame=None,
    #         training=training
    #     )
    #     subsample.code_variables = self.code_variables.all()
    #     subsample.save()

    #     print "Saving documents"

    #     for index, row in df.iterrows():
    #         DocumentSampleDocument.objects.create_or_update(
    #             {"document_id": row["pk"], "sample": subsample},
    #             {"weight": row["weight"]},
    #             return_object=False,
    #             save_nulls=False
    #         )


class SampleUnit(LoggedExtendedModel):

    document = models.ForeignKey("django_learning.Document", related_name="sample_units")
    sample = models.ForeignKey("django_learning.Sample", related_name="document_units")
    weight = models.FloatField(default=1.0)

    class Meta:
        unique_together = ("document", "sample")

    def __str__(self):
        return "{}, {}, {}".format(self.sample, self.document, self.weight)

    def decoded_document_text(self):
        return decode_text(self.document.text)