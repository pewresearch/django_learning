# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import print_function
from builtins import str
from builtins import range
from builtins import object

from django_learning.models import *
from django.conf import settings
from django.template.loader import render_to_string
from pewtils import is_not_null, decode_text
from tqdm import tqdm
from collections import defaultdict
from xml.dom.minidom import parseString
import re, datetime, boto3, random, time
from pewtils import decode_text


class MTurk(object):
    """
    A handler for interacting with the Mechanical Turk API and syncing data to the database
    """

    def __init__(self, sandbox=True):
        """
        Initializes the handler. Requires that ``MTURK_API_ACCESS`` and ``MTURK_API_SECRET`` are specified in your
        application's settings.

        :param sandbox: (default is True) if False, uses the live API rather than the sandbox
        """

        self.sandbox = sandbox

        if sandbox:
            mturk_host = "https://mturk-requester-sandbox.us-east-1.amazonaws.com"
        else:
            mturk_host = "https://mturk-requester.us-east-1.amazonaws.com"

        mturk_params = {"endpoint_url": mturk_host}

        if (
            getattr(settings, "MTURK_API_ACCESS", None) is not None
            and getattr(settings, "MTURK_API_SECRET", None) is not None
        ):
            mturk_params["aws_access_key_id"] = settings.MTURK_API_ACCESS
            mturk_params["aws_secret_access_key"] = settings.MTURK_API_SECRET

        self.conn = boto3.client("mturk", **mturk_params)

    def paginate_endpoint(self, endpoint, object_key, **kwargs):
        """
        Loops over an endpoint until the pagination ends, and compiles all of the results into a list.
        :param endpoint: API endpoint to hit
        :param object_key: String representing the key to extract from each result
        :param kwargs: Keyword arguments to pass to the ``.paginate`` function
        :return: A list of results
        """

        results = []
        for page in self.conn.get_paginator(endpoint).paginate(**kwargs):
            results.extend(page[object_key])
        return results

    def sync_hit_type(self, hit_type):
        """
        Creates or updates a HIT type in the Mechanical Turk API. Creates qualification tests as needed. Certain
        requirements are hard-coded: Turkers must be in the United States, and they must be experts. After this
        function is run, the HIT type will have a value in ``turk_id``.
        :param hit_type: A HITType instance
        :return:
        """

        print("Compiling qualifications")

        if self.sandbox:
            requirements = []
        else:
            requirements = [
                {
                    "QualificationTypeId": "2ARFPLSP75KLA8M8DH1HTEQVJT3SY6"
                    if self.sandbox
                    else "2F1QJWKUDD8XADTFD2Q0G6UTO95ALH",
                    "Comparator": "Exists",
                    "ActionsGuarded": "DiscoverPreviewAndAccept"
                    if not self.sandbox
                    else "Accept",
                },
                {
                    # Worker_Locale
                    "QualificationTypeId": "00000000000000000071",
                    "Comparator": "EqualTo",
                    "LocaleValues": [{"Country": "US"}],
                    "ActionsGuarded": "DiscoverPreviewAndAccept"
                    if not self.sandbox
                    else "Accept",
                },
                {  # Worker_​PercentAssignmentsApproved
                    "QualificationTypeId": "000000000000000000L0",
                    "Comparator": "GreaterThan",
                    "IntegerValues": [int(hit_type.min_approve_pct * 100)],
                    "ActionsGuarded": "DiscoverPreviewAndAccept"
                    if not self.sandbox
                    else "Accept",
                },
                {
                    # Worker_NumberHITsApproved
                    "QualificationTypeId": "00000000000000000040",
                    "Comparator": "GreaterThan",
                    "IntegerValues": [hit_type.min_approve_cnt],
                    "ActionsGuarded": "DiscoverPreviewAndAccept"
                    if not self.sandbox
                    else "Accept",
                },
            ]

        for qualification_test in hit_type.project.qualification_tests.all():

            print("Creating qualification test")

            all_question_xml = []
            for q in qualification_test.questions.all():

                answer_xml = None
                if q.display == "radio":
                    selection_xml = "\n".join(
                        [
                            """
                                            <Selection>
                                                <SelectionIdentifier>{pk}</SelectionIdentifier>
                                                <Text>{label}</Text>
                                            </Selection>
                                        """.format(
                                pk=l.pk, label=l.label
                            )
                            for l in q.labels.all()
                        ]
                    )
                    answer_xml = """
                        <SelectionAnswer>
                            <StyleSuggestion>radiobutton</StyleSuggestion>
                            <Selections>
                                {selections}
                            </Selections>
                        </SelectionAnswer>
                    """.format(
                        selections=selection_xml
                    )

                elif q.display == "number":
                    answer_xml = """
                        <SelectionAnswer>
                            <FreeTextAnswer>
                                <Constraints>
                                    <Length minLength="1" maxLength="100"/>
                                    <AnswerFormRegex regex="[0-9]+" errorText="Please enter a number."/>
                                </Constraints>
                                <DefaultText></DefaultText>
                                <NumberOfLinesSuggestion>1</NumberOfLinesSuggestion>
                            </FreeTextAnswer>
                        </SelectionAnswer>
                    """

                if answer_xml:
                    question_xml = """
                                    <Question>
                                        <QuestionIdentifier>{identifier}</QuestionIdentifier>
                                        <DisplayName></DisplayName>
                                        <IsRequired>true</IsRequired>
                                        <QuestionContent>
                                            <Text>{prompt}</Text>
                                        </QuestionContent>
                                        <AnswerSpecification>
                                            {answer_spec}
                                        </AnswerSpecification>
                                    </Question>
                                    """.format(
                        identifier=q.name,
                        prompt=decode_text(q.prompt),
                        answer_spec=answer_xml,
                    )
                    all_question_xml.append(question_xml)

            test_xml = """
                <QuestionForm xmlns="http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2017-11-06/QuestionForm.xsd">
                    {all_question_xml}
                </QuestionForm>

            """.format(
                all_question_xml="".join(all_question_xml)
            )

            results = self.paginate_endpoint(
                "list_qualification_types",
                "QualificationTypes",
                MustBeRequestable=True,
                Query=qualification_test.name,
                MustBeOwnedByCaller=True,
            )
            existing_quals = [
                q
                for q in results
                if q["Name"] == qualification_test.name
                and q["QualificationTypeStatus"] != "Disposing"
            ]
            if len(existing_quals) > 0:
                qual_id = existing_quals[0]["QualificationTypeId"]
                self.conn.update_qualification_type(
                    QualificationTypeId=qual_id,
                    Description="You must answer a few quick questions before qualifying for these HITs.",
                    QualificationTypeStatus="Active",
                    TestDurationInSeconds=qualification_test.duration_minutes * 60,
                    AutoGranted=False,
                    Test=test_xml,
                )
            else:
                qual_id = self.conn.create_qualification_type(
                    Name=qualification_test.name,
                    Description="You must answer a few quick questions before qualifying for these HITs.",
                    QualificationTypeStatus="Active",
                    Keywords=",".join(qualification_test.keywords),
                    TestDurationInSeconds=qualification_test.duration_minutes * 60,
                    AutoGranted=False,
                    Test=test_xml,
                )["QualificationType"]["QualificationTypeId"]
            qualification_test.turk_id = qual_id
            qualification_test.save()

            requirements.append(
                {
                    "QualificationTypeId": qual_id,
                    "Comparator": "Exists",
                    "ActionsGuarded": "Accept",
                }
            )

        print("Registering HIT Type")

        hit_type_id = self.conn.create_hit_type(
            Title="{}{}".format(
                decode_text(hit_type.title), " (SANDBOX)" if self.sandbox else ""
            ),
            Description=decode_text(hit_type.description),
            Reward=str(hit_type.price),
            AssignmentDurationInSeconds=hit_type.duration_minutes * 60,
            Keywords=",".join(hit_type.keywords),
            AutoApprovalDelayInSeconds=hit_type.approval_wait_hours * 60 * 60,
            QualificationRequirements=requirements,
        )["HITTypeId"]
        # From the Boto3 docs: "If you register a HIT type with values that match an existing HIT type, the HIT type ID of the existing type will be returned."

        if is_not_null(hit_type.turk_id) and hit_type.turk_id != hit_type_id:

            existing_hits = (
                HIT.objects.filter(hit_type=hit_type)
                .filter(turk=True)
                .filter(turk_id__isnull=False)
            )
            if existing_hits.count() > 0:
                print("Existing HITs detected, syncing and then migrating to new type")
                print("TRIGGERED")
                for s in Sample.objects.filter(
                    pk__in=existing_hits.values_list("sample_id", flat=True)
                ).distinct():
                    self.sync_sample_hits(s)
                for h in tqdm(existing_hits, desc="Migrating existing HITs"):
                    print(
                        "{}, {} -> {}".format(h.turk_id, hit_type.turk_id, hit_type_id)
                    )
                    self.conn.update_hit_type_of_hit(
                        HITId=h.turk_id, HITTypeId=hit_type_id
                    )

        hit_type.turk_id = hit_type_id
        hit_type.save()

    def create_sample_hits(self, sample, hit_type, num_coders=1, template_name=None):
        """
        Creates HITs for the specified sample, using the specified HIT type.
        :param sample: a Sample instance
        :param hit_type: a HITType instance
        :param num_coders: (default is 1) number of coders to complete each HIT
        :param template_name: (Optional) name of the HTML template to use
        :return:
        """

        if template_name:
            # template_path = "custom_hits/{}.html".format(template_name)
            # should be able to use the DJANGO_LEARNING_HIT_TEMPLATE_DIRS settings
            template_path = "{}.html".format(template_name)
        else:
            template_path = "django_learning/hit.html"

        for su in tqdm(
            sample.document_units.all(),
            desc="Creating HITs",
            total=sample.document_units.count(),
        ):

            existing = HIT.objects.get_if_exists({"sample_unit": su, "turk": True})
            if existing and existing.turk_id:
                print("Skipping existing HIT: {}".format(existing))
            else:

                hit = HIT.objects.create_or_update(
                    {"sample_unit": su, "turk": True},
                    {
                        "template_name": template_name,
                        "num_coders": num_coders,
                        "hit_type": hit_type,
                    },
                )

                html = render_to_string(
                    template_path,
                    {
                        "project": sample.project,
                        "sample": sample,
                        "hit": hit,
                        "questions": hit.sample.project.questions.order_by("priority"),
                        "django_learning_template": "django_learning/_template.html"
                        # settings.DJANGO_LEARNING_BASE_TEMPLATE
                    },
                )
                # html = re.sub("\t{2,}", " ", html)
                # html = re.sub("\n{2,}", "\n\n", html)
                # html = re.sub("\r{2,}", "\r\r", html)
                html = re.sub("[^\S\r\n]{2,}", " ", html)
                turk_hit = """
                    <HTMLQuestion xmlns="http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2011-11-11/HTMLQuestion.xsd">
                        <HTMLContent><![CDATA[
                            <!DOCTYPE html>
                                {}
                            </html>
                        ]]></HTMLContent>
                        <FrameHeight>0</FrameHeight>
                    </HTMLQuestion>
                """.format(
                    html
                )
                try:
                    response = self.conn.create_hit_with_hit_type(
                        Question=turk_hit,
                        MaxAssignments=num_coders,
                        LifetimeInSeconds=hit_type.lifetime_days * 60 * 60 * 24,
                        HITTypeId=hit_type.turk_id,
                    )
                    hit.turk_id = response["HIT"]["HITId"]
                    hit.save()
                except Exception as e:
                    try:
                        turk_hit = decode_text(turk_hit)
                        response = self.conn.create_hit_with_hit_type(
                            Question=turk_hit,
                            MaxAssignments=num_coders,
                            LifetimeInSeconds=hit_type.lifetime_days * 60 * 60 * 24,
                            HITTypeId=hit_type.turk_id,
                        )
                        hit.turk_id = response["HIT"]["HITId"]
                        hit.save()
                    except Exception as e:
                        print(e)
                        import pdb

                        pdb.set_trace()

    def _parse_answer_xml(self, a, qual_test=False):
        """
        Parses the XML returned from the Mechanical Turk API for a given assignment response
        :param a: the XML for a completed assignment
        :param qual_test: (default is False) if True, indicates that the assignment belongs to a qualification test
        :return: a dictionary of questions and responses
        """

        if qual_test:
            answer_tag = "SelectionIdentifier"
        else:
            answer_tag = "FreeText"

        answer_xml = parseString(a)
        answer_dict = {}
        for answer in answer_xml.getElementsByTagName("QuestionFormAnswers")[
            0
        ].childNodes:
            try:
                keynode = answer.getElementsByTagName("QuestionIdentifier")[
                    0
                ].firstChild
                key = keynode.nodeValue if keynode else None
            except:
                key = None
            if key:
                try:
                    valuenode = answer.getElementsByTagName(answer_tag)[0].firstChild
                    value = valuenode.nodeValue if valuenode else None
                except:
                    import pdb

                    pdb.set_trace()
                answer_dict[key] = value

        return answer_dict

    def _save_assignment(self, hit, a, resync=False):
        """
        Saves the responses for a particular assignment that's been returned from the API.

        :param hit: the HIT the assignment belongs to
        :param a: the XML for a completed assignment
        :param resync: (default is False) if True, updates the assignment and codes in the database even if it's
            already been synced; by default, existing finished assignments won't be updated
        :return:
        """

        coder = Coder.objects.create_or_update(
            {"name": a["WorkerId"]}, {"is_mturk": True}
        )
        hit.sample.project.coders.add(coder)
        # time_spent = (datetime.datetime.strptime(a.SubmitTime, "%Y-%m-%dT%H:%M:%SZ") - datetime.datetime.strptime(a.AcceptTime, "%Y-%m-%dT%H:%M:%SZ")).seconds
        # if time_spent >= 10:
        assignment = Assignment.objects.get_if_exists({"hit": hit, "coder": coder})
        if not assignment or not assignment.time_finished or resync:
            assignment = Assignment.objects.create_or_update(
                {"hit": hit, "coder": coder},
                {
                    "turk_id": a["AssignmentId"],
                    "time_started": a["AcceptTime"],
                    "turk_status": a["AssignmentStatus"],
                },
            )

            answer = self._parse_answer_xml(a["Answer"])
            if "notes" in answer.keys():
                assignment.notes = answer["notes"]
                assignment.save()
            if "uncodeable" in answer.keys() and answer["uncodeable"] == "1":
                assignment.uncodeable = True
                assignment.save()
            for q in hit.sample.project.questions.exclude(display="header"):
                value = answer.get(q.name, None)
                if q.multiple:
                    value = value.split("|")
                notes = answer.get("{}_notes".format(q.name), None)
                q.update_assignment_response(assignment, value, notes=notes)

            if not assignment.time_finished:
                assignment.time_finished = a["SubmitTime"]
                assignment.save()
            else:
                assignment.save()

        if assignment and assignment.turk_status == "Approved":
            assignment.turk_approved = True
            assignment.save()

        return assignment

    def _approve_assignment(self, assignment):
        """
        Approves an assignment and issues payment.
        :param assignment: an Assignment instance
        :return:
        """

        if assignment and not assignment.turk_approved:
            try:
                self.conn.approve_assignment(
                    AssignmentId=assignment.turk_id, OverrideRejection=True
                )
                status = self.conn.get_assignment(AssignmentId=assignment.turk_id)[
                    "Assignment"
                ]["AssignmentStatus"]
                assignment.turk_status = status
                assignment.turk_approved = status == "Approved"
                assignment.save()
            except Exception as e:
                print(e)
                print(
                    "Couldn't approve assignment (enter 'c' to mark as approved and continue)"
                )
                import pdb

                pdb.set_trace()

    def sync_sample_hits(
        self, sample, resync=False, approve=True, approve_probability=1.0
    ):
        """
        Syncs all of the qualification tests, HITs and assignments associated with a sample. Auto-detects HITs in the
        API that may have failed to sync with the database, using all of the HIT types associated with the sample in the
        database.
        :param sample: the sample to sync
        :param resync: (default is False) if True, finished HITs and assignments will be re-synced even if they already
            exist in the database
        :param approve: (default is False) if True, approves and issues payments for finished assignments that have
            not already been approved
        :param approve_probability: (default is 1.0) if ``approve=True`` this specifies the probability that
            unapproved assignments will be approved and paid
        :return:
        """

        for hit_type_id in set(sample.hits.values_list("hit_type__turk_id", flat=True)):
            self._find_missing_hits(hit_type_id)

        for qual_test in sample.project.qualification_tests.all().distinct():
            self.sync_qualification_test(qual_test)

        for hit in tqdm(
            self._update_and_yield_sample_hits(sample), desc="Syncing HITs"
        ):
            if hit.turk_id and (
                hit.assignments.filter(time_finished__isnull=False).count()
                < hit.num_coders
                or resync
            ):
                for a in self.paginate_endpoint(
                    "list_assignments_for_hit", "Assignments", HITId=str(hit.turk_id)
                ):
                    assignment = self._save_assignment(hit, a, resync=resync)
                    if not assignment.turk_approved and approve:
                        if random.random() >= (1.0 - approve_probability):
                            self._approve_assignment(assignment)

    def print_account_balance(self):
        """Prints the account balance"""

        print(self.conn.get_account_balance())

    def _expire_hits(self, hit_ids):
        """
        Given a set of Mechanical Turk HIT IDs, expires them in the API. They will no longer be available to coders
        after this.
        :param hit_ids: A list of Mechanical Turk HIT IDs (can be found in the ``HIT.turk_id`` field in the database
        :return:
        """

        if len(hit_ids) > 0:
            if not self.sandbox:
                print(
                    "WARNING: you are about to expire {} HITs, are you sure?".format(
                        len(hit_ids)
                    )
                )
                import pdb

                pdb.set_trace()
            for hit_id in hit_ids:
                self.conn.update_expiration_for_hit(
                    HITId=hit_id, ExpireAt=datetime.datetime.now()
                )

    def expire_all_hits(self):
        """
        Emergency function that expires all active HITs across all projects.
        :return:
        """

        hit_ids = [hit["HITId"] for hit in self.paginate_endpoint("list_hits", "HITs")]
        self._expire_hits(hit_ids)

    def expire_sample_hits(self, sample):
        """
        Expires all HITs associated with a particular sample. Syncs the HITs first, gets all of their IDs, and then
        expires everything.
        :param sample: a Sample instance
        :return:
        """

        self.sync_sample_hits(sample)
        hit_ids = (
            sample.hits.filter(turk=True)
            .filter(turk_id__isnull=False)
            .values_list("turk_id", flat=True)
        )
        self._expire_hits(hit_ids)

    def _delete_hits(self, hit_ids):
        """
        Deletes HITs in the API. This cannot be reversed, and a warning prompt will be issued if you try to do this
        in the live API. Preserves the HITs in the database, though their status will be updated to "Deleted".
        :param hit_ids: a list of Mechanical Turk HIT IDs
        :return:
        """

        if len(hit_ids) > 0:
            if not self.sandbox:
                print(
                    "WARNING: you are about to delete {} HITs, are you sure?".format(
                        len(hit_ids)
                    )
                )
                print(
                    "If there are any unapproved assignments, they will be automatically approved before the HIT is deleted"
                )
                import pdb

                pdb.set_trace()

            for hit_id in hit_ids:

                for a in self.paginate_endpoint(
                    "list_assignments_for_hit", "Assignments", HITId=str(hit_id)
                ):
                    if a["AssignmentStatus"] == "Submitted":
                        self.conn.approve_assignment(
                            AssignmentId=a["AssignmentId"], OverrideRejection=True
                        )
                try:
                    self.conn.update_expiration_for_hit(
                        HITId=hit_id, ExpireAt=datetime.datetime.now()
                    )
                    try:
                        self.conn.delete_hit(HITId=hit_id)
                    except:
                        time.sleep(30)
                        self.conn.delete_hit(HITId=hit_id)
                    hit = HIT.objects.get_if_exists({"turk_id": hit_id})
                    if hit:
                        hit.turk_id = None
                        hit.turk_status = "Deleted"
                        hit.save()
                except Exception as e:
                    print(e)
                    import pdb

                    pdb.set_trace()

    def delete_all_hits(self):
        """
        Emergency function that deletes all HITs in the Mechanical Turk API. Preserves the HITs in the database,
        though their status will be updated to "Deleted".
        :return:
        """

        hit_ids = [hit["HITId"] for hit in self.paginate_endpoint("list_hits", "HITs")]
        self._delete_hits(hit_ids)

    def delete_sample_hits(self, sample):
        """
        Deletes all HITs associated with a particular sample in the Mechanical Turk API. Preserves the HITs in the
        database, though their status will be updated to "Deleted".
        :param sample: a Sample instance
        :return:
        """

        self.sync_sample_hits(sample)
        hit_ids = (
            sample.hits.filter(turk=True)
            .filter(turk_id__isnull=False)
            .values_list("turk_id", flat=True)
        )
        self._delete_hits(hit_ids)

    def get_annual_worker_compensation(self, year=None, include_pending=False):
        """
        Get the total amount that's been paid to every Mechanical Turker across all projects, even ones not in the
        database.
        :param year: (Optional) specify a particular year
        :param include_pending: (default is False) if True, includes completed assignments that have not yet been approved and paid
        :return: a dictionary of worker MTurk IDs and their compensation
        """

        workers = defaultdict(float)
        if not year:
            year = datetime.datetime.now().year
        for hit in self.paginate_endpoint("list_hits", "HITs"):
            for a in self.paginate_endpoint(
                "list_assignments_for_hit", "Assignments", HITId=str(hit["HITId"])
            ):
                if (
                    a["AssignmentStatus"] == "Approved"
                    and a["ApprovalTime"].year == year
                ) or (
                    include_pending
                    and a["AssignmentStatus"] == "Submitted"
                    and a["SubmitTime"].year == year
                ):
                    workers[a["WorkerId"]] += float(hit["Reward"])

        return workers

    def update_worker_blocks(self, notify=False, max_comp=500):
        """
        Given a maximum compensation limit (default 500), loops over all workers in the API and blocks them
        from completing assignments if they've exceeded this limit in the current calendar year. Automatically unblocks
        workers when it rolls over into a new year.

        Using this function is not recommended, it pisses Mechanical Turkers off because they get penalized for blocks.
        You should revoke their qualifications instead.


        :param notify: (default is False) if True, sends a notification to the worker thanking them and explaining the block
        :param max_comp: (default is 500) specifies the maximum amount in dollars that we want to pay any given worker in a calendar year
        :return:
        """

        worker_comp = self.get_annual_worker_compensation(
            year=datetime.datetime.now().year, include_pending=True
        )
        blocks = self.get_worker_blocks()
        blocks = blocks.get("Maximum annual compensation", [])

        over_limit = []
        for worker_id, comp in worker_comp.items():
            if comp >= max_comp:
                over_limit.append(worker_id)

        to_add = list(set(over_limit).difference(set(blocks)))
        title = "Maximum number of HITs reached"
        message = """
            Hello - we just wanted to let you know that you've reached the maximum number of HITs that we can offer you \
            for {}.  Thank you so much for helping us with our research, and we hope you'll work with us again in the future! \
            We'll automatically lift the block on your account in the new year.  In the meantime, feel free to reach out \
            to us if you have questions.  
        """.format(
            datetime.datetime.now().year
        )
        if notify and len(to_add) > 0:
            self.conn.notify_workers(
                Subject=title, MessageText=message, WorkerIds=to_add
            )
        for worker_id in to_add:
            self.conn.create_worker_block(
                WorkerId=worker_id, Reason="Maximum annual compensation"
            )

        to_remove = list(set(blocks).difference(set(over_limit)))
        for worker_id in to_remove:
            self.conn.delete_worker_block(WorkerId=worker_id, Reason="")

    def clear_all_worker_blocks(self):
        """
        Clears all worker blocks in the API
        :return:
        """

        for k, blocks in self.get_worker_blocks().items():
            for worker_id in blocks:
                self.conn.delete_worker_block(WorkerId=worker_id, Reason="")

    def get_worker_blocks(self):
        """
        Returns a dictionary of blocked workers grouped by the reason for the block
        :return:
        """

        blocks = defaultdict(list)
        for block in self.paginate_endpoint("list_worker_blocks", "WorkerBlocks"):
            blocks[block["Reason"]].append(block["WorkerId"])

        return blocks

    def sync_qualification_test(self, qual_test):
        """
        Hits the API and syncs completed assignments for a specified qualification test. Runs the scoring function and
        approves workers if they pass the qualification test.

        :param qual_test: a QualificationTest instance
        :return:
        """

        existing_quals = self.paginate_endpoint(
            "list_qualification_types",
            "QualificationTypes",
            MustBeRequestable=True,
            Query=qual_test.name,
            MustBeOwnedByCaller=True,
        )
        for q in existing_quals:
            if q["Name"] == qual_test.name:
                requests = self.paginate_endpoint(
                    "list_qualification_requests",
                    "QualificationRequests",
                    QualificationTypeId=q["QualificationTypeId"],
                )

                for a in requests:

                    coder = Coder.objects.create_or_update(
                        {"name": a["WorkerId"]}, {"is_mturk": True}
                    )
                    assignment = QualificationAssignment.objects.create_or_update(
                        {"test": qual_test, "coder": coder},
                        {
                            "turk_id": a["QualificationRequestId"],
                            "time_finished": a["SubmitTime"],
                        },
                    )
                    answer = self._parse_answer_xml(a["Answer"], qual_test=True)
                    for question, value in answer.items():
                        try:
                            q = qual_test.questions.get(name=question)
                        except:
                            q = None
                        if q:
                            if q.multiple:
                                value = value.split("|")
                            q.update_assignment_response(assignment, value)
                    for q in qual_test.questions.exclude(name__in=answer.keys()):
                        q.update_assignment_response(assignment, None)

                    if coder.is_qualified(qual_test):
                        self.conn.accept_qualification_request(
                            QualificationRequestId=a["QualificationRequestId"]
                        )
                    else:
                        try:
                            self.revoke_qualification(qual_test, coder)
                        except Exception as e:
                            print(e)

    def revoke_qualification(self, qual_test, coder):
        """
        Revokes a qualification for a specified qualification test and coder
        :param qual_test: a QualificationTest instance
        :param coder: a Coder instance (must be a Mechanical Turk coder)
        :return:
        """

        self.conn.disassociate_qualification_from_worker(
            QualificationTypeId=qual_test.turk_id, WorkerId=coder.name
        )

    def delete_qualification_test(self, qual_test):
        """
        Deletes a qualification test in the API (it will remain in the database though)
        :param qual_test: a QualificationTest instance
        :return:
        """

        self.conn.delete_qualification_type(QualificationTypeId=qual_test.turk_id)

    def _find_missing_hits(self, hit_type_id):
        """
        Given a Mechanical Turk HIT type ID, scans the API for any missing HITs that do not currently exist in the
        database, and creates them if any are found.
        :param hit_type_id: a Mechanical Turk HIT type ID
        :return:
        """

        good_hits = (
            HIT.objects.filter(turk=True)
            .filter(turk_id__isnull=False)
            .values_list("turk_id", flat=True)
        )
        bad_hits = HIT.objects.filter(turk=True).filter(turk_id__isnull=True)
        actual_hits = []
        for hit in self.paginate_endpoint("list_hits", "HITs"):
            if hit["HITId"] not in good_hits and hit["HITTypeId"] == hit_type_id:
                actual_hits.append(hit)
        counter = 0
        for hit in actual_hits:
            hit_id = None
            for a in self.paginate_endpoint(
                "list_assignments_for_hit", "Assignments", HITId=hit["HITId"]
            ):
                answer = self._parse_answer_xml(a["Answer"])
                hit_id = answer.get("hit_id", None)
                if hit_id:
                    break
            if hit_id:
                db_hit = bad_hits.get(pk=hit_id)
                db_hit.turk_id = hit["HITId"]
                db_hit.save()
                counter += 1
        if counter > 0:
            print("Found {} missing HITs and restored their turk_ids".format(counter))

    def _update_and_yield_sample_hits(self, sample):
        """
        Loops over all of the Mechanical Turk HITs for a specified sample, updates their status, and
        yields them as a generator.
        :param sample: a Sample instance
        :return: generator of HIT instances
        """

        for hit in sample.hits.filter(turk=True).filter(turk_id__isnull=False):
            try:
                h = self.conn.get_hit(HITId=str(hit.turk_id))["HIT"]
                hit.turk_status = h["HITStatus"]
                hit.save()
            except Exception as e:
                if e.error_code == "AWS.MechanicalTurk.HITDoesNotExist":
                    hit.turk_id = None
                    hit.turk_status = "Deleted"
                    hit.save()
                else:
                    raise
            yield hit
