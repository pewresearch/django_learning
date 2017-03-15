import pandas, math, random, itertools

from django.db import models

from pewtils import is_null, decode_text
from pewtils.sampling import compute_sample_weights_from_frame
from pewtils.django import get_model
from pewtils.django.sampling import SampleExtractor

from tqdm import tqdm

from django_commander.models import LoggedExtendedModel
from django_learning.utils.sampling_frames import sampling_frames
from django_learning.utils.sampling_methods import sampling_methods


class SamplingFrame(LoggedExtendedModel):

    name = models.CharField(max_length=200, unique=True)
    documents = models.ManyToManyField("django_learning.Document", related_name="sampling_frames")

    def __str__(self):

        return "{}, {} documents".format(self.name, self.documents.count())

    def save(self, *args, **kwargs):

        if self.name not in sampling_frames.keys():
            raise Exception("Sampling frame '{}' is not defined in any of the known folders".format(self.name))
        super(SamplingFrame, self).save(*args, **kwargs)

    @property
    def config(self):
        return sampling_frames[self.name]()

    def extract_documents(self, refresh=False):

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

    def extract_documents(self, size=None, recompute_weights=False, override_doc_ids=None,
                          allow_overlap_with_existing_project_samples=False):

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
                SampleUnit.objects.filter(sample=self, document_id__in=bad_ids).delete()

        params = self.get_params()
        if params:

            print "Extracting sampling frame"

            docs = self.frame.documents.all()
            if not allow_overlap_with_existing_project_samples and not recompute_weights:
                existing_doc_ids = SampleUnit.objects.filter(sample__project=self.project).values_list("document_id", flat=True)
                docs = docs.exclude(pk__in=existing_doc_ids)
                # for s in self.frame.samples.filter(project=self.project):
                #     docs = docs.exclude(pk__in=s.documents.values_list("pk", flat=True))

            if not "stratify_by" in params.keys() or not params["stratify_by"]:
                frame = pandas.DataFrame.from_records(docs.values("text", "pk"))
            else:
                frame = pandas.DataFrame.from_records(docs.values("text", "pk", params["stratify_by"]))

            weight_vars = []
            use_keyword_searches = False
            if "sampling_searches" in params.keys() and len(params["sampling_searches"].keys()) > 0:

                print "Extracting sampling search flags"

                frame['none'] = ~frame["text"].str.contains(
                    r"|".join([s['pattern'] for s in params["sampling_searches"].values()]))
                for search_name, p in params["sampling_searches"].items():
                    frame[search_name] = frame["text"].str.contains(p['pattern'])
                use_keyword_searches = True
                weight_vars.extend(params["sampling_searches"].keys() + ['none'])

            if is_null(override_doc_ids):

                print "Extracting sample"

                sample_chunks = []
                if not use_keyword_searches:

                    sample_chunks.append(
                        SampleExtractor(sampling_strategy=params["sampling_strategy"], id_col="pk").extract(
                            frame,
                            sample_size=int(size)
                        )
                    )

                else:

                    sample_chunks = []
                    non_search_sample_size = 1.0 - sum([s['proportion'] for s in params["sampling_searches"].values()])
                    sample_chunks.append(
                        SampleExtractor(sampling_strategy=params["sampling_strategy"], id_col="pk").extract(
                            frame[frame['none'] == 1],
                            sample_size=int(math.ceil(size * non_search_sample_size))
                        )
                    )

                    for search, p in params["sampling_searches"].iteritems():
                        sample_chunks.append(
                            SampleExtractor(sampling_strategy=params["sampling_strategy"], id_col="pk").extract(
                                frame[frame[search] == 1],
                                sample_size=int(math.ceil(size * p["proportion"]))
                            )
                        )

                sample_ids = list(set(list(itertools.chain(*sample_chunks))))
                if len(sample_ids) < size:
                    fill_ids = list(self.frame.documents.values_list("pk", flat=True))
                    random.shuffle(fill_ids)
                    while len(sample_ids) < size:
                        sample_ids.append(fill_ids.pop())
                        sample_ids = list(set(sample_ids))

            else:

                print "Override document IDs passed, skipping sample extraction"
                sample_ids = override_doc_ids

            print "Computing weights"

            df = frame[frame['pk'].isin(sample_ids)]
            df['weight'] = compute_sample_weights_from_frame(frame, df, weight_vars)

            print "Saving documents"

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