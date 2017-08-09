import time, random

from django_commander.commands import BasicCommand

from django_learning.models import Project, Sample, Assignment
from django_learning.mturk import MTurk


class Command(BasicCommand):

    parameter_names = ["project_name", "sample_name"]
    dependencies = []

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("project_name", type=str)
        parser.add_argument("sample_name", type=str)
        parser.add_argument("--time_sleep", default=300, type=int)
        parser.add_argument("--prod", default=False, action="store_true")
        parser.add_argument("--loop", default=False, action="store_true")
        parser.add_argument("--probability", default=.1, type=float)
        return parser

    def run(self):

        project = Project.objects.get(name=self.parameters["project_name"])

        sample = Sample.objects.get(
            name=self.parameters["sample_name"],
            project=project
        )

        mturk = MTurk(sandbox=(not self.options["prod"]))

        while True:
            approved = 0
            assignments = Assignment.objects\
                    .filter(hit__sample=sample)\
                    .filter(turk_id__isnull=False)\
                    .filter(turk_approved=False)
            for a in assignments:
                if random.random() <= self.options["probability"]:
                    mturk.conn.approve_assignment(a.turk_id)
                    a.turk_approved = True
                    a.save()
                    approved += 1
            print "{} of {} assignments approved".format(approved, assignments.count())
            if not self.options["loop"]:
                break
            time.sleep(self.options["time_sleep"])

    def cleanup(self):

        pass