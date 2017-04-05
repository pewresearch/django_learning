import re, random

from tqdm import tqdm
from django.db.models import Count
from django.db import transaction

from logos.models import Document, NgramSet, DocumentNgramSet
from pewtils import decode_text

from django_commander.commands import DownloadIterateCommand


class Command(DownloadIterateCommand):

    """
    """

    option_defaults = [
        {"name": "search_existing", "default": False, "help": ""},
        {"name": "only_clean", "default": False, "help": ""},
        {"name": "test", "default": False, "help": ""}
    ]
    parameter_defaults = [
        {"name": "document_type", "default": None, "help": ""},
        {"name": "dictionary", "default": None, "help": ""}
    ]
    dependencies = []

    def __init__(self, *args, **kwargs):

        super(Command, self).__init__(*args, **kwargs)

        self.regexes = {}
        for cat in NgramSet.objects.filter(dictionary=self.parameters["dictionary"]):
            full_words = []
            wildcards = []
            for word in cat.words:
                if word.endswith("*"):
                    search_word = word.replace("*", "")
                    if search_word != '':
                        wildcards.append(re.escape(search_word))
                else:
                    search_word = word
                    if search_word != '':
                        full_words.append(re.escape(search_word))
            # self.regexes[cat] = re.compile(
            #     r"((?<=\s)" + "|".join(wildcards) + "(?=\w)|" + \
            #     r"(?<=\s)" + "|".join(full_words) + "(?=\W))",
            #     re.IGNORECASE
            # )
            self.regexes[cat] = re.compile(
                r"\W(" + "|".join(["{}\w*".format(w) for w in wildcards]) + "|" + "|".join(full_words) + ")\W",
                re.IGNORECASE
            )

    def download(self):

        return [None, ]

    def iterate(self, data):

        if self.parameters["document_type"] and self.parameters["dictionary"]:
            docs = Document.objects.filter(**{"{0}_id__isnull".format(self.parameters["document_type"]): False})
            if self.options["only_clean"]:
                docs = docs.filter(is_clean=True)
            if not self.options["search_existing"]:
                existing = set(NgramSet.objects.filter(dictionary=self.parameters['dictionary']).values_list("documents__document_id", flat=True))
                docs = list(set(docs.values_list("pk", flat=True)) - existing)
                existing = None # force immediate garbage collection
                # ngram_sets = NgramSet.objects.filter(dictionary=self.parameters['dictionary']).values_list("pk", flat=True)
                # docs = docs.exclude(ngram_sets__ngram_set_id__in=ngram_sets)
                # # existing_docs = DocumentNgramSet.objects\
                # #     .filter(ngram_set_id__in=ngram_sets)\
                # #     .values_list("document_id", flat=True)
                # # docs = docs.exclude(pk__in=existing_docs)
            else:
                docs = list(docs.values_list("pk", flat=True))
            random.shuffle(docs)
            for id in tqdm(docs, desc="Extracting document {0} categories ({1})".format(self.parameters["dictionary"], self.parameters["document_type"])):
                yield [id, ]
                if self.options["test"]: break

    def parse_and_save(self, id):

        try:
            doc = Document.objects.get(pk=id)

            # if self.options["search_existing"] or \
            #         doc.ngram_sets.filter(ngram_set__dictionary=self.parameters['dictionary']).count() == 0:

            text = decode_text(doc.text)
            text_len = float(len(text.split()))
            # doc.ngram_sets.filter(ngram_set__dictionary=self.parameters['dictionary']).delete()
            # don't need to do the above, right?  since they get overwritten anyway

            if text_len > 0:

                with transaction.atomic():

                    for cat in self.regexes.keys():
                        matches = self.regexes[cat].findall(" {0} ".format(text))
                        val = float(len([m for m in matches if m != '']))
                        if val > 0:
                            DocumentNgramSet.objects.create_or_update(
                                {
                                    "ngram_set": cat,
                                    "document": doc
                                },
                                {
                                    "percent": (val / text_len) * 100.0,
                                    "count": val
                                },
                                command_log=self.log,
                                return_object=False
                            )
        except Exception as e:
            print e

    def cleanup(self):

        pass