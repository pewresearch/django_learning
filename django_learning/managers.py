from __future__ import print_function
from builtins import object
import gensim, random, pandas, os

from collections import defaultdict
from tqdm import tqdm
from gensim.models import Word2Vec, Doc2Vec

from django.db import transaction
from django.db.models import Count, F
from django.conf import settings

from pewtils import chunk_list, decode_text
from pewanalytics.text import TextCleaner, SentenceTokenizer
from django_pewtils import CacheHandler, get_model
from django_pewtils.managers import BasicExtendedManager
from django_queries.managers import QueryModelManager

from django_learning.utils.dataset_extractors import dataset_extractors


class QuestionManager(BasicExtendedManager):
    def create_from_config(self, owner_model_name, owner, q, i):

        labels = q.get("labels", [])
        examples = q.get("examples", [])
        question = self.create_or_update(
            {owner_model_name: owner, "name": q["name"]},
            {
                "prompt": decode_text(q["prompt"]),
                "display": q["display"],
                "multiple": q.get("multiple", False),
                "tooltip": decode_text(q["tooltip"])
                if q.get("tooltip", None)
                else None,
                "priority": i,
                "optional": q.get("optional", False),
                "show_notes": q.get("show_notes", False),
            },
            save_nulls=True,
        )
        if q.get("dependency", None):
            dep = q.get("dependency", None)
            other_question = self.model.objects.filter(**{owner_model_name: owner}).get(
                name=dep["question_name"]
            )
            label = other_question.labels.get(value=dep["label_value"])
            question.dependency = label
            question.save()

        label_ids = []
        for j, l in enumerate(labels):
            label = get_model("Label").objects.create_or_update(
                {"question": question, "value": decode_text(l["value"])},
                {
                    "label": decode_text(l["label"]),
                    "priority": j,
                    "pointers": [decode_text(p) for p in l.get("pointers", [])],
                    "select_as_default": l.get("select_as_default", False),
                },
            )
            label_ids.append(label.pk)
            label.pointers = [decode_text(p) for p in l.get("pointers", [])]
            label.save()
        for l in question.labels.all():
            if l.pk not in label_ids:
                l.delete()

        example_ids = []
        for e in examples:
            example_ids.append(
                get_model("Example")
                .objects.create_or_update(
                    {
                        "question": question,
                        "quote": decode_text(e["quote"]),
                        "explanation": decode_text(e["explanation"]),
                    }
                )
                .pk
            )
        for e in question.examples.all():
            if e.pk not in example_ids:
                e.delete()

        owner.questions.add(question)


class DocumentManager(QueryModelManager):
    def document_types(self):

        return [f.name for f in self.model.get_parent_relations()]

    def reset_text_to_original(self):
        @transaction.atomic
        def reset_chunk(docs):
            for doc in docs:
                doc.text = doc.original_text
                doc.is_clean = False
                doc.document_fragments = []
                doc.save()

        counter = 0
        docs = []
        for doc in tqdm(
            self.exclude(text=F("original_text")),
            desc="Resetting documents to original text",
            total=self.count(),
        ):
            docs.append(doc)
            counter += 1
            if counter == 10:
                reset_chunk(docs)
                docs = []
                counter = 0
        if len(docs) > 0:
            reset_chunk(docs)
        # self.update(text=F("original_text"))
        # for doc in self.all():
        #     doc.save()

    def reset_document_fragments(self):
        @transaction.atomic
        def reset_chunk(docs):
            for doc in docs:
                doc.document_fragments = []
                doc.save()

        counter = 0
        docs = []
        for doc in tqdm(
            self.annotate(c=Count("document_fragments")).filter(c__gte=1),
            desc="Unlinking existing documents fragments",
            total=self.count(),
        ):
            docs.append(doc)
            counter += 1
            if counter == 10:
                reset_chunk(docs)
                docs = []
                counter = 0
        if len(docs) > 0:
            reset_chunk(docs)

    def word2vec(
        self,
        document_type,
        refresh=False,
        window_size=5,
        use_skipgrams=False,
        chunk_size=20000,
        workers=2,
        use_sentences=False,
        dimensions=300,
    ):

        cleaner = TextCleaner(process_method=None, strip_html=True)
        tokenizer = SentenceTokenizer()
        w2v_model = None

        cache = CacheHandler(
            os.path.join(settings.DJANGO_LEARNING_S3_CACHE_PATH, "word2vec"),
            use_s3=settings.DJANGO_LEARNING_USE_S3,
            bucket=settings.S3_BUCKET,
            aws_access=settings.AWS_ACCESS_KEY_ID,
            aws_secret=settings.AWS_SECRET_ACCESS_KEY,
        )
        if not refresh:

            w2v_model = cache.read(
                "word2vec_{}_{}_{}_{}_{}".format(
                    document_type, dimensions, use_sentences, window_size, use_skipgrams
                )
            )

        if not w2v_model:

            doc_ids = list(
                self.model.objects.filter(
                    **{"{0}__isnull".format(document_type): False}
                ).values_list("pk", flat=True)
            )
            random.shuffle(doc_ids)
            w2v_model = None
            for i, chunk in enumerate(chunk_list(doc_ids, chunk_size)):
                sentences = []
                for text in tqdm(
                    self.model.objects.filter(pk__in=chunk).values_list(
                        "text", flat=True
                    ),
                    desc="Loading documents {0} - {1}".format(
                        (i) * chunk_size, (i + 1) * chunk_size
                    ),
                ):
                    if use_sentences:
                        sentences.extend(
                            [cleaner.clean(s).split() for s in tokenizer.tokenize(text)]
                        )
                    else:
                        sentences.append(cleaner.clean(text).split())
                print("Transforming and training")
                bigram_transformer = gensim.models.Phrases(sentences)
                if not w2v_model:
                    w2v_model = Word2Vec(
                        bigram_transformer[sentences],
                        size=dimensions,
                        sg=1 if use_skipgrams else 0,
                        window=window_size,
                        min_count=5,
                        workers=workers,
                    )
                else:
                    w2v_model.train(bigram_transformer[sentences])
                print("{0} documents loaded".format((i + 1) * chunk_size))
            w2v_model.init_sims(replace=True)

            cache.write(
                "word2vec_{}_{}_{}_{}_{}".format(
                    document_type, dimensions, use_sentences, window_size, use_skipgrams
                ),
                w2v_model,
            )

        return w2v_model

    def doc2vec(
        self, document_type, refresh=False, chunk_size=20000, workers=2, dimensions=300
    ):
        class DocumentWrapper(object):
            def __init__(self, words):
                self.words = words
                self.tags = []

        cleaner = TextCleaner(process_method=None, strip_html=True)
        d2v_model = None

        cache = CacheHandler(
            os.path.join(settings.DJANGO_LEARNING_S3_CACHE_PATH, "doc2vec"),
            use_s3=settings.DJANGO_LEARNING_USE_S3,
            bucket=settings.S3_BUCKET,
            aws_access=settings.AWS_ACCESS_KEY_ID,
            aws_secret=settings.AWS_SECRET_ACCESS_KEY,
        )
        if not refresh:

            d2v_model = cache.read("doc2vec_{0}_{1}".format(document_type, dimensions))

        if not d2v_model:

            doc_ids = list(
                self.model.objects.filter(
                    **{"{0}__isnull".format(document_type): False}
                ).values_list("pk", flat=True)
            )
            random.shuffle(doc_ids)
            d2v_model = None
            for i, chunk in enumerate(chunk_list(doc_ids, chunk_size)):
                docs = []
                for text in tqdm(
                    self.model.objects.filter(pk__in=chunk).values_list(
                        "text", flat=True
                    ),
                    desc="Loading documents {0} - {1}".format(
                        (i) * chunk_size, (i + 1) * chunk_size
                    ),
                ):
                    docs.append(DocumentWrapper(words=cleaner.clean(text).split()))
                print("Transforming and training")
                if not d2v_model:
                    d2v_model = Doc2Vec(
                        docs, size=dimensions, window=5, min_count=5, workers=workers
                    )
                else:
                    d2v_model.train(docs)
                print("{0} documents loaded".format((i + 1) * chunk_size))
            # d2v_model.init_sims(replace=True)

            cache.write("doc2vec_{0}_{1}".format(document_type, dimensions), d2v_model)

        return d2v_model


# class CoderDocumentCodeManager(BasicExtendedManager):
#
#     pass
#
#     # NOTE: commenting the code below out because it looks... strange and outdated
#     # we'll figure out what it's for if it breaks something, but Pycharm can't find any invocations
#
#     # def update_sample_weights(self, sample_name=None, refresh_frame=False):
#     #
#     #     if sample_name:
#     #
#     #         h = DocumentSampleManager(name=sample_name)
#     #         if refresh_frame: frame = h.extract_sample_frame(refresh=True)
#     #         df = pandas.DataFrame.from_records(self.values("document_id", "document__text"))
#     #         df = h._add_sample_weights(df, text_col="document__text")
#     #         for index, row in tqdm(df.iterrows(), desc="Updating code weights"):
#     #             c = self.model.get(pk=row['pk'])
#     #             c.weight = row['weight']
#     #             c.save()


class NgramSetManager(BasicExtendedManager):
    def get_dictionary_word_map(self, dictionary=None):

        word_map = defaultdict(list)
        if dictionary:
            for cat in self.filter(dictionary=dictionary):
                for word in cat.words:
                    word_map[word].append(cat)

        return word_map


class CodeManager(BasicExtendedManager):

    pass
