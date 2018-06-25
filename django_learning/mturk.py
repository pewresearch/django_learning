from boto.mturk.connection import MTurkConnection
from boto.mturk.qualification import (
    LocaleRequirement,
    NumberHitsApprovedRequirement,
    PercentAssignmentsApprovedRequirement,
    Qualifications,
    Requirement
)
from boto.mturk.question import Question as TurkQuestion
from boto.mturk.question import (
    AnswerSpecification,
    FreeTextAnswer,
    HTMLQuestion,
    LengthConstraint,
    NumericConstraint,
    Overview,
    QuestionContent,
    QuestionForm,
    RegExConstraint,
    SelectionAnswer
)
from django_learning.models import *
from django.conf import settings
from django.template.loader import render_to_string
from pewtils import is_not_null, decode_text
from tqdm import tqdm
import re


class MTurk(object):
    def __init__(self, sandbox=True):
        self.sandbox = sandbox

        if sandbox:
            mturk_host = "mechanicalturk.sandbox.amazonaws.com"
        else:
            mturk_host = "mechanicalturk.amazonaws.com"

        mturk_params = { 'host': mturk_host }

        if getattr(settings, 'MTURK_API_ACCESS', None) is not None \
            and getattr(settings, 'MTURK_API_SECRET', None) is not None:
                mturk_params['aws_access_key_id'] = settings.MTURK_API_ACCESS
                mturk_params['aws_secret_access_key'] = settings.MTURK_API_SECRET

        self.conn = MTurkConnection(**mturk_params)

    def sync_hit_type(self, hit_type):

        print "Compiling qualifications"

        quals = Qualifications()

        if not self.sandbox:

            quals.add(MasterRequirement(sandbox=False))

            quals.add(
                PercentAssignmentsApprovedRequirement(
                    comparator="GreaterThan",
                    integer_value=str(int(hit_type.min_approve_pct * 100)),
                    required_to_preview=True
                )
            )
            quals.add(
                NumberHitsApprovedRequirement(
                    comparator="GreaterThan",
                    integer_value=str(hit_type.min_approve_cnt),
                    required_to_preview=True
                )
            )
            quals.add(
                LocaleRequirement(
                    "EqualTo",
                    "US",
                    required_to_preview=True
                )
            )

        hit_type_qual_tests = hit_type.project.qualification_tests.all() | hit_type.qualification_tests.all()

        qual_id = None
        for qualification_test in hit_type_qual_tests.distinct():
            # if hit_type.project.has_qualification:

            print "Creating qualification test"

            qual_test = QuestionForm()
            for q in qualification_test.questions.all():
                content = QuestionContent()
                content.append_field("Title", decode_text(q.prompt))
                if q.display == "radio":
                    labels = [(l.label, l.pk) for l in q.labels.all()]
                    form = AnswerSpecification(
                        SelectionAnswer(
                            min=1,
                            max=1,
                            selections=labels,
                            type='text',
                            style='radiobutton',
                            other=False
                        )
                    )
                elif q.display == "number":
                    form = AnswerSpecification(
                        FreeTextAnswer(
                            constraints=[
                                LengthConstraint(min_length=1, max_length=100),
                                RegExConstraint(r'[0-9]+', error_text="Please enter a number.")
                                # NumericConstraint(min_value=1, max_value=100)
                            ],
                            default="",
                            num_lines=1
                        )
                    )
                turk_question = TurkQuestion(
                    identifier=q.name,
                    content=content,
                    is_required=True,
                    answer_spec=form
                )
                qual_test.append(turk_question)

            existing_quals = [q for q in self.conn.search_qualification_types(qualification_test.name) if
                              q.Name == qualification_test.name]
            if len(existing_quals) > 0:
                qual_id = existing_quals[0].QualificationTypeId
                self.conn.update_qualification_type(
                    qual_id,
                    description="You must answer a few quick questions before qualifying for these HITs.",
                    status="Active",
                    retry_delay=None,
                    test_duration=qualification_test.duration_minutes * 60,
                    auto_granted=False,
                    test=qual_test
                )
            else:
                qual_id = self.conn.create_qualification_type(
                    qualification_test.name,
                    "You must answer a few quick questions before qualifying for these HITs.",
                    "Active",
                    keywords=qualification_test.keywords,
                    retry_delay=None,
                    test_duration=qualification_test.duration_minutes * 60,
                    auto_granted=False,
                    test=qual_test
                )[0].QualificationTypeId

            quals.add(
                Requirement(
                    qual_id,
                    comparator="EqualTo",
                    integer_value=1,
                    required_to_preview=False
                )
            )

        print "Registering HIT Type"

        hit_type_id = self.conn.register_hit_type(
            decode_text(hit_type.title),
            decode_text(hit_type.description),
            hit_type.price,
            hit_type.duration_minutes * 60,
            keywords=hit_type.keywords,
            approval_delay=hit_type.approval_wait_hours * 60 * 60,
            qual_req=quals
        )[0].HITTypeId

        if is_not_null(hit_type.turk_id) and hit_type.turk_id != hit_type_id:

            existing_hits = HIT.objects.filter(sample__hit_type=hit_type).filter(turk=True).filter(
                turk_id__isnull=False)
            if existing_hits.count() > 0:
                print "Existing HITs detected, syncing and then migrating to new type"
                for s in Sample.objects.filter(pk__in=existing_hits.values_list("sample_id", flat=True)).distinct():
                    self.sync_sample_hits(s)
                for h in tqdm(existing_hits, desc="Migrating existing HITs"):
                    print "{}, {}".format(h.turk_id, hit_type.turk_id)
                    self.conn.change_hit_type_of_hit(h.turk_id, hit_type.turk_id)

        hit_type.turk_id = hit_type_id
        hit_type.save()

    def create_sample_hits(self, sample, num_coders=1, template_name=None):

        if template_name:
            # template_path = "custom_hits/{}.html".format(template_name)
            # should be able to use the DJANGO_LEARNING_HIT_TEMPLATE_DIRS settings
            template_path = "{}.html".format(template_name)
        else:
            template_path = "django_learning/hit.html"

        for su in tqdm(sample.document_units.all(), desc="Creating HITs", total=sample.document_units.count()):

            existing = HIT.objects.get_if_exists({
                "sample_unit": su, "turk": True
            })
            if existing and existing.turk_id:
                print "Skipping existing HIT: {}".format(existing)
            else:

                hit = HIT.objects.create_or_update(
                    {"sample_unit": su, "turk": True},
                    {
                        "template_name": template_name,
                        "num_coders": num_coders
                    }
                )

                try:
                    html = render_to_string(template_path, {
                        "project": sample.project,
                        "sample": sample,
                        "hit": hit,
                        "questions": hit.sample.project.questions.order_by("priority"),
                        "django_learning_template": "django_learning/_template.html" # settings.DJANGO_LEARNING_BASE_TEMPLATE
                    })
                    # html = re.sub("\t{2,}", " ", html)
                    # html = re.sub("\n{2,}", "\n\n", html)
                    # html = re.sub("\r{2,}", "\r\r", html)
                    html = re.sub("[^\S\r\n]{2,}", " ", html)
                    turk_hit = HTMLQuestion(html, "1000")
                    response = self.conn.create_hit(
                        question=turk_hit,
                        max_assignments=num_coders,
                        lifetime=sample.hit_type.lifetime_days * 60 * 60 * 24,
                        hit_type=sample.hit_type.turk_id
                    )
                    hit.turk_id = response[0].HITId
                    hit.save()
                except Exception as e:
                    import pdb
                    pdb.set_trace()

    def sync_sample_hits(self, sample, resync=False, approve=True):

        sample_qual_tests = sample.project.qualification_tests.all() | sample.hit_type.qualification_tests.all()
        for qual_test in sample_qual_tests.distinct():
            self.sync_qualification_test(qual_test)

        for hit in tqdm(self._update_and_yield_sample_hits(sample), desc="Syncing HITs"):

            if hit.turk_id and (hit.assignments.filter(time_finished__isnull=False).count() < hit.num_coders or resync):

                for a in self._get_hit_assignments(str(hit.turk_id)):

                    coder = Coder.objects.create_or_update({"name": a.WorkerId}, {"is_mturk": True})
                    sample.project.coders.add(coder)
                    # time_spent = (datetime.datetime.strptime(a.SubmitTime, "%Y-%m-%dT%H:%M:%SZ") - datetime.datetime.strptime(a.AcceptTime, "%Y-%m-%dT%H:%M:%SZ")).seconds
                    # if time_spent >= 10:
                    assignment = Assignment.objects.get_if_exists({"hit": hit, "coder": coder})
                    if not assignment or not assignment.time_finished or resync:
                        assignment = Assignment.objects.create_or_update(
                            {"hit": hit, "coder": coder},
                            {
                                "turk_id": a.AssignmentId,
                                "time_started": a.AcceptTime,
                                "turk_status": a.AssignmentStatus
                            }
                        )
                        for answer in a.answers[0]:
                            question = answer.qid
                            code_ids = answer.fields
                            if question == "hit_id":
                                pass
                            elif question == "notes":
                                assignment.notes = code_ids[0]
                                assignment.save()
                            elif question == "uncodeable" and "1" in code_ids:
                                assignment.uncodeable = True
                                assignment.save()
                            else:
                                try:
                                    q = hit.sample.project.questions.get(name=question)
                                except:
                                    q = None
                                if q:
                                    if len(code_ids) < 2 and not q.multiple:
                                        code_ids = code_ids[0]
                                    q.update_assignment_response(assignment, code_ids)

                        form_questions = [ans.qid for ans in a.answers[0]]
                        for q in hit.sample.project.questions\
                                .exclude(name__in=form_questions)\
                                .exclude(display="header"):
                            q.update_assignment_response(assignment, None)

                        if not assignment.time_finished:
                            assignment.time_finished = a.SubmitTime
                            assignment.save()
                        else:
                            assignment.save()

                    if assignment and assignment.turk_status == "Approved":
                        assignment.turk_approved = True
                        assignment.save()

                    if approve and assignment and not assignment.turk_approved:
                        try:
                            self.conn.approve_assignment(assignment.turk_id) # a.AssignmentId)
                        except Exception as e:
                            print e
                            print "Couldn't approve assignment (enter 'c' to mark as approved and continue)"
                            import pdb
                            pdb.set_trace()
                        assignment.turk_approved = True
                        assignment.save()

    def sync_qualification_test(self, qual_test):

        for a in self._get_qualification_requests(qual_test):

            coder = Coder.objects.create_or_update({"name": a.SubjectId}, {"is_mturk": True})
            assignment = QualificationAssignment.objects.create_or_update(
                {"test": qual_test, "coder": coder},
                {
                    "turk_id": a.QualificationRequestId,
                    "time_finished": a.SubmitTime
                }
            )
            for answer in a.answers[0]:
                question = answer.qid
                code_ids = answer.fields
                try:
                    q = qual_test.questions.get(name=question)
                except:
                    q = None
                if q:
                    if len(code_ids) < 2 and not q.multiple:
                        code_ids = code_ids[0]
                    q.update_assignment_response(assignment, code_ids)

            if coder.is_qualified(qual_test):  # and coder not in sample.project.inactive_coders.all():
                self.conn.grant_qualification(a.QualificationRequestId)

    def print_account_balance(self):

        print self.conn.get_account_balance()

    def clear_hits(self):

        for hit in HIT.objects.filter(turk=True):
            try:
                self.conn.expire_hit(hit.turk_id)
            except:
                pass
            try:
                self.conn.dispose_hit(hit.turk_id)
            except:
                pass

        # for hit_type_id in HITType.objects.values_list("turk_id", flat=True).distinct():
        #     pass

        # for sample in Sample.objects.all():
        #     qual_identifier = "{}_{}_{}".format(sample.project.name, sample.name, "qualification")
        #     existing_quals = mturk.search_qualification_types(qual_identifier)
        #     for q in existing_quals:
        #         mturk.dispose_qualification_type(q.QualificationTypeId)

    def revoke_qualification(self, qual_test, coder):

        self.conn.update_qualification_score(qual_test.turk_id, coder.name, 0)
        # qual_identifier = "{}_{}".format(sample.project.name, sample.hit_type.name)
        # existing_quals = self.conn.search_qualification_types(qual_identifier)
        # for q in existing_quals:
        #     self.conn.update_qualification_score(q.QualificationTypeId, coder.name, 0)
        #     # self.conn.revoke_qualification(coder.name, q.QualificationTypeId, reason="Sorry, your responses were to inconsistent with our own - thank you for the assignments that you've already completed, and keep an eye out for new samples on other projects.")

    def _get_hit_assignments(self, hit_id):

        assignments = self.conn.get_assignments(hit_id, page_size=10)
        for a in assignments: yield a
        for page_num in range(2, int(int(assignments.TotalNumResults) / 10) + 1):
            for a in self.conn.get_assignments(str(self.conn.turk_id), page_size=10, page_number=page_num):
                yield a

    def _update_and_yield_sample_hits(self, sample):

        for hit in sample.hits.filter(turk=True).filter(turk_id__isnull=False):
            try:
                h = self.conn.get_hit(str(hit.turk_id))[0]
                hit.turk_status = h.HITStatus
                hit.save()
            except Exception as e:
                if e.error_code == "AWS.MechanicalTurk.HITDoesNotExist":
                    hit.turk_id = None
                    hit.turk_status = "Deleted"
                    hit.save()
                else:
                    raise
            yield hit

    def _get_qualification_requests(self, qualification_test):

        existing_quals = self.conn.search_qualification_types(qualification_test.name)
        for q in existing_quals:
            if q.Name == qualification_test.name:
                requests = self.conn.get_qualification_requests(q.QualificationTypeId, page_size=10)
                for r in requests: yield r
                for page_num in range(2, int(int(requests.TotalNumResults) / 10) + 1):
                    for r in self.conn.get_qualification_requests(q.QualificationTypeId, page_size=10, page_number=page_num):
                        yield r

    # def sync_qualification_test(self, qual_test):
    #
    #     for c in tqdm(hit_type.project.coders.filter(turk=True), desc="Updating qualification for all existing coders"):
    #         if c.is_qualified(hit_type.project) and c not in hit_type.project.inactive_coders.all():
    #             try: self.conn.assign_qualification(hit_type.qualification_turk_id, c.name, 1)
    #             except: self.conn.update_qualification_score(hit_type.qualification_turk_id, c.name, 1)
    #         else:
    #             try: self.conn.assign_qualification(hit_type.qualification_turk_id, c.name, 0)
    #             except: self.conn.update_qualification_score(hit_type.qualification_turk_id, c.name, 0)


class MasterRequirement(Requirement):

    def __init__(self, sandbox=False, required_to_preview=False):
        comparator = "Exists"
        sandbox_qualification_type_id = "2ARFPLSP75KLA8M8DH1HTEQVJT3SY6"
        production_qualification_type_id = "2F1QJWKUDD8XADTFD2Q0G6UTO95ALH"
        qualification_type_id = production_qualification_type_id if not sandbox else sandbox_qualification_type_id
        super(MasterRequirement, self).__init__(qualification_type_id=qualification_type_id, comparator=comparator, required_to_preview=required_to_preview)
