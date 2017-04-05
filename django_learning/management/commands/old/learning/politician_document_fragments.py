import random, pandas, numpy, importlib

from tqdm import tqdm

from sklearn.metrics.pairwise import cosine_similarity

from django.db.models import Count

from logos.models import Document, DocumentFragment
from pewtils.django import get_model
from pewtils.nlp import has_fragment, get_hash, TextHelper
from logos.learning.supervised import ClassificationHandler

from django_commander.commands import DownloadIterateCommand


class Command(DownloadIterateCommand):

    """
    """

    option_defaults = [
        {"name": "only_use_existing_fragments", "default": False, "help": ""},
        {"name": "scan_top_n_matches_per_doc", "default": 10, "help": ""},
        {"name": "min_fragment_length", "default": 40, "help": ""},
        {"name": "test", "default": False, "help": ""}
    ]
    parameter_defaults = [
        {"name": "document_type", "default": None, "help": ""}
    ]
    dependencies = []

    def __init__(self, **options):

        self.options = {o['name']: options.get(o['name'], o['default']) for o in self.option_defaults}
        self.parameters = {p['name']: options.get(p['name'], p['default']) for p in self.parameter_defaults}
        self.log = None

    def download(self):

        # cleaning_model = importlib.import_module("logos.learning.text_cleaning.{0}".format(self.parameters["document_type"]))\
        #     .get_model(min_token_length=self.options["min_fragment_length"])

        # cleaning_model = ClassificationHandler(
        #     "clean_scraper_press_releases",
        #     "boilerplate",
        #     pipeline="clean_scraper_press_releases",
        #     num_cores=1
        # )
        # cleaning_model.load_model()

        return [ ]

    def iterate(self):

        pol_ids = list(get_model(self.parameters["document_type"], app_name="logos").objects.values_list("politician_id", flat=True))
        random.shuffle(pol_ids)
        for pol_id in pol_ids:
            yield [pol_id, ]
            if self.options["test"]: break

    def parse_and_save(self, pol_id):

        print "Politician {0}".format(pol_id)

        search_scope = {"{0}__politician_id".format(self.parameters["document_type"]): pol_id}

        docs = Document.objects\
            .filter(**{"{0}__isnull".format(self.parameters["document_type"]): False})\
            .filter(**search_scope)\
            .filter(is_clean=True)

        print "Resetting document fragment links"

        docs.reset_document_fragments()

        if not self.options["only_use_existing_fragments"]:

            print "Extracting fragments from scratch"

            doc_df = pandas.DataFrame(list(docs.values("pk", "text", "date")))

            h = TextHelper(doc_df, "text")

            fragments = h.extract_corpus_fragments(
                scan_top_n_matches_per_doc=self.options["scan_top_n_matches_per_doc"],
                min_fragment_length=self.options["min_fragment_length"]
            )
            if len(fragments) > 0:

                frag_tfidf = h.vectorizer.transform(fragments)
                similarity_matrix = cosine_similarity(frag_tfidf, h.tfidf)

                frag_df = pandas.DataFrame(fragments, columns=['text'])
                for index, row in frag_df.iterrows():
                    frag_df.loc[index, 'avg'] = numpy.average(similarity_matrix[index])
                    frag_df.loc[index, 'std'] = numpy.std(similarity_matrix[index])
                    frag_df.loc[index, 'above_25_pct'] = float(len([s for s in similarity_matrix[index] if s >= .25]))/len(similarity_matrix[index])

                # TODO: currently restricting this to clean documents, which shouldn't need any boilerplate filtering!
                frag_df["boilerplate"] = 0.0
                # frag_df['boilerplate'] = cleaning_model.predict(frag_df)
                # frag_df["boilerplate"] = pandas.DataFrame(cleaning_model.apply_model(frag_df))["boilerplate"]

                for index, row in frag_df[frag_df['boilerplate'] == 0.0].iterrows():
                    ssdeep = get_hash(row['text'].strip(), hash_function="ssdeep")
                    DocumentFragment.objects.create_or_update(
                        {
                            "hash": ssdeep,
                            "document_type": self.parameters["document_type"],
                            "scope": search_scope
                        },
                        {
                            "text": row['text'].strip()
                        },
                        save_nulls=True,
                        empty_lists_are_null=False,
                        return_object=False
                    )

        print "Applying fragments"

        fragments = DocumentFragment.objects.filter(document_type=self.parameters["document_type"], scope=search_scope)
        for f in tqdm(fragments, desc="Scanning for fragments"):
            for d in tqdm(docs, desc="Iterating over documents"):
                if has_fragment(d.text, f.text):
                    d.document_fragments.add(f)

        print "Removing old fragments"

        DocumentFragment.objects\
            .filter(document_type=self.parameters["document_type"], scope=search_scope)\
            .annotate(c=Count("documents"))\
            .filter(c__lte=1)\
            .delete()

    def cleanup(self):

        pass