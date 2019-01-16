import os, random, ner, subprocess, time

from tqdm import tqdm
from django.db.models import Count

from django_learning.models import Document, Entity
from pewanalytics.internal.ner import EntityExtractor
from django.conf import settings

from django_commander.commands import DownloadIterateCommand


class Command(DownloadIterateCommand):

    """
    """

    parameter_names = ["document_type"]
    dependencies = []

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("document_type", type=str)
        parser.add_argument("--search_existing", default=False, action="store_true")
        parser.add_argument("--only_clean", default=False, action="store_true")
        return parser

    def __init__(self, *args, **kwargs):

        super(Command, self).__init__(*args, **kwargs)

        ner_path = os.path.join(settings.DJANGO_LEARNING_EXTERNAL_PACKAGE_DIR, "stanford-ner-2016-10-31/")
        self.server = subprocess.Popen(
            ["java", "-mx1000m", "-cp", "stanford-ner.jar", "edu.stanford.nlp.ie.NERServer", "-loadClassifier", "classifiers/english.all.3class.distsim.crf.ser.gz", "-port", "9191"],
            shell=False,
            cwd=ner_path
        )
        time.sleep(5)
        print "Connecting to extractor"
        self.extractor = EntityExtractor(
            ner_path,
            corenlp_tagger=ner.SocketNER(host='localhost', port=9191, output_format='slashTags')
        )
        print "Connected"

    def download(self):

        return [None, ]

    def iterate(self, data):

        print "Extracting document IDs"
        if self.parameters["document_type"]:
            docs = Document.objects.filter(**{"{0}__isnull".format(self.parameters["document_type"]): False})
            if self.options["only_clean"]:
                docs = docs.filter(is_clean=True)
            if not self.options["search_existing"]:
                docs = docs.annotate(c=Count("entities")).filter(c=0)
            docs = list(docs.values_list("pk", flat=True))
            print "Document IDs extracted"
            random.shuffle(docs)
            for id in tqdm(docs, desc="Extracting document entities ({0})".format(self.parameters["document_type"])):
                yield [id, ]
                if self.options["test"]: break

    def parse_and_save(self, id):

        doc = Document.objects.get(pk=id)

        if not self.options["search_existing"] or doc.entities.count() == 0:

            try: raw_entities = self.extractor.extract(doc.text)
            except Exception as e:
                print e
                raw_entities = []

            # with transaction.atomic():

            entities = []
            for tag, e in raw_entities:
                try:
                    entities.append(
                        Entity.objects.create_or_update({
                            "tag": tag,
                            "name": e
                        }, command_log=self.log)
                    )
                except Exception as e:
                    print e

            doc.entities = entities
            doc.save()

            if self.log:
                doc.commands.add(self.log.command)
                doc.command_logs.add(self.log)

    def cleanup(self):

        print "Shutting down CoreNLP server..."
        self.server.kill()