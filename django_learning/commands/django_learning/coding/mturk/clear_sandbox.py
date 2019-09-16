from django_commander.commands import BasicCommand

from django_learning.models import Project, Sample
from django_learning.mturk import MTurk


class Command(BasicCommand):

    parameter_names = []
    dependencies = []

    @staticmethod
    def add_arguments(parser):
        return parser

    def run(self):

        from django_commander.commands import commands

        mturk = MTurk(sandbox=True)
        commands["django_learning_coding_mturk_expire_all_hits"](sandbox=True).run()
        commands["django_learning_coding_mturk_delete_all_hits"](sandbox=True).run()
        for qual_test in mturk.paginate_endpoint("list_qualification_types", 'QualificationTypes',
                                                     MustBeRequestable=True,
                                                     MustBeOwnedByCaller=True):
            mturk.conn.delete_qualification_type(QualificationTypeId=qual_test['QualificationTypeId'])

    def cleanup(self):

        pass