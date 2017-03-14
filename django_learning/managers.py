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
        if q["display"] != "number":
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


class DocumentManager(BasicExtendedManager):
    def document_types(self):

        return [f.name for f in self.model.get_parent_relations()]


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
