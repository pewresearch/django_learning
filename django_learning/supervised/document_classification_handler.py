import itertools, numpy, pandas, copy, importlib

from django.db.models import Q
from django.conf import settings
from nltk.metrics.agreement import AnnotationTask
from collections import defaultdict
from sklearn.metrics import classification_report
from sklearn.cross_validation import cross_val_predict
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, brier_score_loss
from statsmodels.stats.inter_rater import cohens_kappa
from scipy.stats import ttest_ind
from multiprocessing import Pool

from pewtils import is_not_null, is_null, chunker, decode_text
from pewtils.sampling import compute_sample_weights_from_frame, compute_balanced_sample_weights
from pewtils.stats import wmom
from pewtils.django import get_model

from django_learning.utils.decorators import require_model, require_training_data, temp_cache_wrapper
from django_learning.supervised.classification_handler import ClassificationHandler


class DocumentClassificationHandler(ClassificationHandler):

    pipeline_folders = ["classification", "documents"]

    def __init__(self,
        document_types,
        code_variable_name,
        pipeline,
        saved_name=None,
        params=None,
        num_cores=2,
        verbose=True,
        **kwargs
    ):

        if isinstance(document_types, str):
            document_types = [document_types]
        else:
            document_types = sorted(document_types)

        super(DocumentClassificationHandler, self).__init__(
            "_".join(["_".join(document_types), code_variable_name]),
            pipeline,
            outcome_variable="code_id",
            params=params,
            num_cores=num_cores,
            verbose=verbose,
            **kwargs
        )

        self._document_types = document_types
        self._code_variable = get_model("CodeVariable").objects.get(name=code_variable_name)
        self._frames = None

        if saved_name:

            self.saved_model = get_model("DocumentClassificationModel").objects.get_if_exists({"name": saved_name})
            if self.saved_model:
                if verbose:
                    print "You passed the name of a saved CodeVaraibleClassifier record; all parameters passed (except for optional ones) will be overridden by the saved values in the database"
                    if sorted(self.saved_model.document_types) != document_types:
                        print "Document types overridden from saved database value: {0} to {1}".format(document_types, self.saved_model.document_types)
                    if self.saved_model.variable.name != self._code_variable.name:
                        print "Code variable overridden from saved database value: {0} to {1}".format(self._code_variable.name, self.saved_model.variable.name)
                    if self.saved_model.pipeline_name != pipeline and is_not_null(pipeline):
                        print "Named pipeline overridden from saved database value: {0} to {1}".format(pipeline, self.saved_model.pipeline_name)
                self._document_types = self.saved_model.document_types
                self._code_variable = self.saved_model.variable
                self._frames = self.saved_model.frames.all()
                self._parameters = self.saved_model.parameters
                self.pipeline_name = self.saved_model.pipeline_name
            else:
                print "No classifier '{0}' found in the database".format(saved_name)

        else:

            self.saved_model = None

        if not self._frames:
            if "frames" in self.parameters["documents"].keys():
                self._frames = get_model("DocumentSampleFrame").objects.filter(name__in=self.parameters["documents"]["frames"])
            else:
                self._frames = get_model("DocumentSampleFrame").objects.filter(name__in=[])

        if self.saved_model and verbose:
            print "Currently associated with a saved classifier: {0}".format(self.saved_model)

        # Always pre-compute unnecessary/bad code parameters and delete them, to keep cache keys consistent:

        def has_all_params(p):
            return all([k in p.keys() for k in [
                "code_filters",
                "consolidation_threshold",
                "consolidation_min_quantile"
            ]])

        has_experts = self._has_raw_codes(turk=False)
        self.use_expert_codes = False
        if "experts" in self.parameters["codes"].keys():
            if is_not_null(self.parameters["codes"]["experts"]) and has_all_params(self.parameters["codes"]["experts"]):
                if has_experts:
                    self.use_expert_codes = True
                else: del self.parameters["codes"]["experts"]
            else: del self.parameters["codes"]["experts"]

        has_mturk = self._has_raw_codes(turk=True)
        self.use_mturk_codes = False
        if "mturk" in self.parameters["codes"].keys():
            if is_not_null(self.parameters["codes"]["mturk"]) and has_all_params(self.parameters["codes"]["mturk"]):
                if has_mturk:
                    self.use_mturk_codes = True
                else: del self.parameters["codes"]["mturk"]
            else: del self.parameters["codes"]["mturk"]

        fallback = False
        if self.use_expert_codes and self.use_mturk_codes:
            if not has_all_params(self.parameters["codes"]):
                fallback = True
                for k in ["code_filters", "consolidation_threshold", "consolidation_min_quantile", "mturk_to_expert_weight"]:
                    if k in self.parameters["codes"]:
                        del self.parameters["codes"][k]
        else:
            for k in ["code_filters", "consolidation_threshold", "consolidation_min_quantile", "mturk_to_expert_weight"]:
                if k in self.parameters["codes"]:
                    del self.parameters["codes"][k]

        if fallback:
            if has_experts and self.use_expert_codes:
                self.use_mturk_codes = False
                if "mturk" in self.parameters["codes"]:
                    del self.parameters["codes"]["mturk"]
            elif has_mturk and self.use_mturk_codes:
                self.use_expert_codes = False
                if "experts" in self.parameters["codes"]:
                    del self.parameters["codes"]["experts"]
            else:
                self.use_mturk_codes = False
                self.use_expert_codes = False


    @property
    def parameters(self):
        if self.saved_model: return self.saved_model.parameters
        else: return self._parameters


    @property
    def code_variable(self):
        # Wrapping this and document_types/parameters as properties ensures that handlers associated with a saved classifier make it difficult to modify these values once saved
        if self.saved_model: return self.saved_model.variable
        else: return self._code_variable


    @property
    def document_types(self):
        if self.saved_model: return self.saved_model.document_types
        else: return self._document_types


    @property
    def frames(self):
        if self.saved_model: return self.saved_model.frames.all()
        else: return self._frames

    def _get_training_data(self, validation=False, **kwargs):

        from django_learning.utils import code_filters

        expert_codes = None
        if self.use_expert_codes:

            print "Extracting expert codes"
            expert_codes = self._get_raw_codes(turk=False, training=(not validation))
            print "{} expert codes extracted".format(len(expert_codes))

            for filter_name, filter_params in self.parameters["codes"]["experts"]["code_filters"]:
                if is_not_null(expert_codes, empty_lists_are_null=True):
                    print "Applying expert code filter: %s" % filter_name
                    expert_codes = code_filters[filter_name](expert_codes, **filter_params)
                    # filter_module = importlib.import_module("logos.learning.utils.code_filters.{0}".format(filter_name))
                    # expert_codes = filter_module.filter(expert_codes, **filter_params)

            if is_not_null(expert_codes):

                if self.parameters["codes"]["experts"]["consolidation_threshold"]:
                    print "Consolidating expert codes at threshold {}".format(self.parameters["codes"]["experts"]["consolidation_threshold"])
                    expert_codes = self._consolidate_codes(
                        expert_codes,
                        threshold=self.parameters["codes"]["experts"]["consolidation_threshold"],
                        keep_quantile=self.parameters["codes"]["experts"]["consolidation_min_quantile"],
                        fake_coder_id="experts"
                    )

        mturk_codes = None
        if self.use_mturk_codes:

            print "Extracting MTurk codes"
            mturk_codes = self._get_raw_codes(turk=True, training=(not validation))
            print "{} MTurk codes extracted".format(len(mturk_codes))

            if is_not_null(mturk_codes, empty_lists_are_null=True):

                for filter_name, filter_params in self.parameters["codes"]["mturk"]["code_filters"]:
                    print "Applying MTurk code filter: %s" % filter_name
                    mturk_codes = code_filters[filter_name](mturk_codes, **filter_params)
                    # filter_module = importlib.import_module("logos.learning.utils.code_filters.{0}".format(filter_name))
                    # mturk_codes = filter_module.filter(mturk_codes, **filter_params)

                if is_not_null(mturk_codes):

                    if self.parameters["codes"]["mturk"]["consolidation_threshold"]:
                        print "Consolidating Mturk codes at threshold {}".format(self.parameters["codes"]["mturk"]["consolidation_threshold"])
                        mturk_codes = self._consolidate_codes(
                            mturk_codes,
                            threshold=self.parameters["codes"]["mturk"]["consolidation_threshold"],
                            keep_quantile=self.parameters["codes"]["mturk"]["consolidation_min_quantile"],
                            fake_coder_id="mturk"
                        )

        if self.use_expert_codes and self.use_mturk_codes:

            codes = pandas.concat([c for c in [expert_codes, mturk_codes] if is_not_null(c)])

            for filter_name, filter_params in self.parameters["codes"]["code_filters"]:
                print "Applying global code filter: %s" % filter_name
                codes = code_filters[filter_name](codes, **filter_params)
                # filter_module = importlib.import_module("logos.learning.utils.code_filters.{0}".format(filter_name))
                # codes = filter_module.filter(codes, **filter_params)

            print "Consolidating all codes"

            if "mturk_to_expert_weight" in self.parameters["codes"]:
                df = self._consolidate_codes(
                    codes,
                    threshold=self.parameters["codes"]["consolidation_threshold"],
                    keep_quantile=self.parameters["codes"]["consolidation_min_quantile"],
                    mturk_to_expert_weight=self.parameters["codes"]["mturk_to_expert_weight"]
                )
            else:
                df = self._consolidate_codes(
                    codes,
                    threshold=self.parameters["codes"]["consolidation_threshold"],
                    keep_quantile=self.parameters["codes"]["consolidation_min_quantile"]
                )

        elif self.use_expert_codes:
            df = expert_codes
        elif self.use_mturk_codes:
            df = mturk_codes
        else:
            df = pandas.DataFrame()

        df = df[df['code_id'].notnull()]
        df = self._add_code_labels(df)

        input_documents = self.filter_documents(
            get_model("Document").objects.filter(pk__in=df["document_id"])
        )

        training_data = df.merge(
            input_documents,
            how='inner',
            left_on='document_id',
            right_on="pk"
        )

        if self.parameters["documents"].get("include_frame_weights", False):
            training_data = self._add_frame_weights(training_data)

        training_data = self._add_balancing_weights(training_data)

        return {
            "training_data": training_data
        }


    def _get_code_query(self, turk=False, training=False, coder_names=None, use_consensus_ignore_flag=True):

        query = get_model("CoderDocumentCode").objects

        if len(self.frames) > 0:
            frame_query = Q()
            for f in self.frames:
                frame_query.add(Q(**{"sample_unit__sample__frame": f}), Q.OR)
            query = query.filter(frame_query)

        if coder_names:
            query = query.filter(coder__name__in=coder_names)

        if use_consensus_ignore_flag:
            query = query.exclude(consensus_ignore=True)

        filter_query = Q()
        for doc_type in self.document_types:
            filter_query.add(Q(**{"sample_unit__document__{}_id__isnull".format(doc_type): False}), Q.OR)

        return query \
            .filter(filter_query) \
            .filter(code__variable=self.code_variable) \
            .filter(coder__is_mturk=turk) \
            .filter(sample_unit__sample__training=training)

    def _has_raw_codes(self, turk=False):

        return self._get_code_query(turk=turk, training=True).count() > 0

    def _get_raw_codes(self, turk=False, training=True, coder_names=None, use_consensus_ignore_flag=True):

        query = self._get_code_query(turk=turk, training=training, coder_names=coder_names, use_consensus_ignore_flag=use_consensus_ignore_flag)
        columns = [
            "sample_unit__document_id",
            "code_id",
            "coder_id",
            "sample_unit__weight"
        ]
        for doc_type in self.document_types:
            columns.append("sample_unit__document__{}_id".format(doc_type))
        codes = pandas.DataFrame(list(query.values(*columns)))
        if len(codes) > 0:
            codes = codes.rename(columns={"sample_unit__weight": "sampling_weight", "sample_unit__document_id": "document_id"})
            codes["sampling_weight"] = codes["sampling_weight"].fillna(1.0)
            codes["is_mturk"] = turk
            for doc_type in self.document_types:
                codes.ix[~codes["sample_unit__document__{}_id".format(doc_type)].isnull(), "document_type"] = doc_type
            codes = pandas.concat([codes, pandas.get_dummies(codes["code_id"], prefix="code")], axis=1)
            # codes = pandas.get_dummies(codes, prefix="code", columns=["code_id"])
        else:
            codes = None
        if not self.parameters["documents"].get("include_sample_weights", False):
            codes["sampling_weight"] = 1.0
        codes["training_weight"] = codes["sampling_weight"]

        return codes


    @staticmethod
    def _add_code_labels(codes_df):
        return codes_df.merge(
            pandas.DataFrame(
                list(
                    get_model("Code").objects\
                        .filter(id__in=codes_df['code_id'].unique())\
                        .values('variable_id', 'variable__name', 'id', 'value', 'label')
                )
            ),
            left_on = 'code_id',
            right_on='id'
        )


    # def _compute_alpha(self, codes, min_rows=10):
    #
    #     alphas = []
    #     combo_count = 0
    #     for coder1, coder2 in itertools.permutations(codes["coder_id"].unique(), 2):
    #         doc_ids = codes[codes["coder_id"] == coder1]["document_id"].unique()
    #         code_subset = codes[(codes["coder_id"].isin([coder1, coder2])) & (codes["document_id"].isin(doc_ids))]
    #         if len(code_subset) >= min_rows:
    #             alpha = AnnotationTask(data=code_subset[["coder_id", "document_id", "code_id"]].as_matrix())
    #             try: alphas.append(alpha.alpha())
    #             except ZeroDivisionError: pass
    #         combo_count += 1
    #     avg_alpha = numpy.average(alphas)
    #     print "Average alpha: {0} ({1} of {2} permutations successfully computed)".format(avg_alpha, len(alphas), combo_count)
    #     return avg_alpha


    def _consolidate_codes(self, codes, threshold=0.6, mturk_to_expert_weight=1.0, keep_quantile=None, fake_coder_id="consolidated"):

        def translate(value, left_min, left_max, right_min, right_max):
            # https://stackoverflow.com/questions/1969240/mapping-a-range-of-values-to-another
            left_span = left_max - left_min
            right_span = right_max - right_min
            value_scaled = float(value - left_min) / float(left_span)
            return right_min + (value_scaled * right_span)

        code_cols = [c for c in codes.columns if c.startswith("code_") and c != "code_id"]

        base_code = None
        if len(codes['code_id'].unique()) == 2:
            base_code = codes['code_id'].value_counts(ascending=False).index[0]
        # NOTE: for binary variables, edge cases will not be omitted, but rather be set to the most frequent class

        def _consolidate(doc_id, group, base_code=None):

            has_mturk = True in group["is_mturk"].unique()
            has_experts = False in group["is_mturk"].unique()

            avgs = {}
            for col in code_cols:

                if has_mturk and has_experts:

                    turkers = group[group["is_mturk"]==True]["coder_id"].unique()
                    experts = group[group["is_mturk"]==False]["coder_id"].unique()
                    total_coders = float(len(turkers) + len(experts))

                    expert_ratio = float(len(experts)) / total_coders
                    adj_expert_ratio = expert_ratio * mturk_to_expert_weight
                    adj_turk_ratio = 1.0 - min([1.0, adj_expert_ratio])

                    avg = (adj_turk_ratio * group[group["coder_id"].isin(turkers)][col].mean()) + \
                        (adj_expert_ratio * group[group["coder_id"].isin(experts)][col].mean())

                else:

                    avg = round(group[col].mean(), 3)

                avgs[col] = avg

            new_threshold = translate(threshold, 0.0, 1.0, min(codes[col]), max(codes[col]))

            if base_code:

                highest = sorted([k for k in avgs.keys() if k != "code_{}".format(base_code)], key=lambda x: avgs[x], reverse=True)[0]
                if avgs[highest] >= new_threshold:

                    consolidated_row = {
                        "coder_id": fake_coder_id,
                        "code_id": int(highest.replace("code_", "")),
                        "document_type": group["document_type"].iloc[0],
                        "document_id": doc_id,
                        "%s" % highest: 1,
                        "is_mturk": (has_mturk and not has_experts),
                        "avg_{0}".format(highest): avgs[highest],
                        "sampling_weight": group['sampling_weight'].mean() if 'sampling_weight' in group.keys() else None,
                        "training_weight": group['training_weight'].mean() if 'training_weight' in group.keys() else None
                    }

                else:

                    consolidated_row = {
                        "coder_id": fake_coder_id,
                        "code_id": base_code,
                        "document_type": group["document_type"].iloc[0],
                        "document_id": doc_id,
                        "code_{}".format(base_code): 1,
                        "is_mturk": (has_mturk and not has_experts),
                        "avg_code_{}".format(base_code): avgs["code_{}".format(base_code)],
                        "sampling_weight": group['sampling_weight'].mean() if 'sampling_weight' in group.keys() else None,
                        "training_weight": group['training_weight'].mean() if 'training_weight' in group.keys() else None
                    }

            else:

                highest = sorted(avgs.keys(), key=lambda x: avgs[x], reverse=True)[0]
                if avgs[highest] >= new_threshold:

                    consolidated_row = {
                        "coder_id": fake_coder_id,
                        "code_id": int(highest.replace("code_", "")),
                        "document_type": group["document_type"].iloc[0],
                        "document_id": doc_id,
                        "%s" % highest: 1,
                        "is_mturk": (has_mturk and not has_experts),
                        "avg_{0}".format(highest): avgs[highest],
                        "sampling_weight": group['sampling_weight'].mean() if 'sampling_weight' in group.keys() else None,
                        "training_weight": group['training_weight'].mean() if 'training_weight' in group.keys() else None
                    }

                else:

                    consolidated_row = {
                        "coder_id": None,
                        "code_id": None,
                        "document_id": None,
                        "document_type": None
                    }

            for col in code_cols:
                if col not in consolidated_row:
                    consolidated_row[col] = 0
                    consolidated_row["avg_{0}".format(col)] = avgs[col]

            df = pandas.DataFrame([consolidated_row])
            return df

        new_df = pandas.concat([_consolidate(i, group, base_code=base_code) for i, group in codes.groupby("document_id")])

        if self.parameters["codes"].get("use_consensus_weights", False):
            print "Applying consensus weights"
            consensus_weights = {}
            avg_codes = codes.groupby("document_id").mean()
            num_codes = float(len(codes['code_id'].unique()))
            balanced_ratio = 1.0 / num_codes
            for code in codes['code_id'].unique():
                consensus_weights[code] = float(len(avg_codes[avg_codes['code_{}'.format(code)] == 1.0])) / float(len(avg_codes))
            total_weights = sum(consensus_weights.values())
            for k, v in consensus_weights.items():
                if v > 0.0:
                    consensus_weights[k] = num_codes*balanced_ratio / (v / total_weights)
                else:
                    consensus_weights[k] = 1.0
            print "Consensus weights: {}".format(consensus_weights)
            def apply_consensus_weights(x):
                for code in codes['code_id'].unique():
                    if x['avg_code_{}'.format(code)]==new_df['avg_code_{}'.format(code)].max():
                        return x["training_weight"] * consensus_weights[code]
                return x["training_weight"]
            new_df['training_weight'] = new_df.apply(apply_consensus_weights, axis=1)

        if keep_quantile:
            final_dfs = []
            for col in code_cols:
                quantile = new_df[new_df[col] == 1.0]['avg_{0}'.format(col)].quantile(keep_quantile)
                final_dfs.append(new_df[(new_df[col] == 1.0) & (new_df['avg_{0}'.format(col)] >= quantile)])
            new_df = pandas.concat(final_dfs)

        for col in code_cols:
            del new_df['avg_{0}'.format(col)]

        return new_df

    def _apply_document_filters(self, documents):

        from django_learning.utils import document_filters
        for filter_name, filter_params in self.parameters["documents"].get("filters", []):
            # print "Applying document filter: %s" % filter_name
            # filter_module = importlib.import_module("logos.learning.utils.document_filters.{0}".format(filter_name))
            documents = document_filters[filter_name](documents, **filter_params)
            # documents = filter_module.filter(documents, **filter_params)
            # documents = getattr(document_filters, f)(documents)
        return documents

    def filter_documents(self, documents, additional_columns=None):

        documents = self._apply_document_filters(documents)
        if not additional_columns: additional_columns = []
        if documents.count() > 0:
            df = pandas.DataFrame(list(documents.values("pk", "text", "date", *additional_columns)))
            df['text'] = df['text'].apply(lambda x: decode_text(x))
            df = df.dropna(subset=['text'])
            df['text'] = df['text'].astype(str)
        else:
            df = pandas.DataFrame(columns=["pk", "text", "date"])
        # This should now be happening when codes are first extracted
        # if len(self.frames) > 0:
        #     df = df[df['pk'].isin(self.frame.documents.values_list("pk", flat=True))]

        return df


    def _add_frame_weights(self, training_data):

        # print "Adding sample weights"

        # base_weights = self.code_variable.samples\
        #     .filter(training=True)\
        #     .filter(document_weights__document_id__in=training_data["pk"])\
        #     .values("document_weights__document_id", "document_weights__weight")
        # base_weights = pandas.DataFrame.from_records(base_weights)
        # base_weights = base_weights.rename(columns={
        #     "document_weights__document_id": "pk",
        #     "document_weights__weight": "weight"
        # })
        # base_weights = base_weights.groupby("pk").apply(numpy.prod)
        # training_data = training_data.merge(base_weights, on="pk", how="left")
        # training_data['weight'] = training_data['weight'].fillna(1.0)

        for frame in self.frames.all():
            code_weights = frame.get_params().get("code_weights", [])
            if len(code_weights) > 0:

                weight_var_names, weight_var_functions = zip(*code_weights)

                print "Adding additional frame weights: {}".format(weight_var_names)

                frame = self.filter_documents(frame.documents.all(), additional_columns=weight_var_names)
                valid_partition = False
                for i, var in enumerate(weight_var_names):
                    frame[var] = frame[var].map(weight_var_functions[i])
                    if len(frame[var].value_counts()) > 1:
                        valid_partition = True

                if valid_partition:
                    weight_vars = []
                    for var in weight_var_names:
                        var_frame = frame.dropna(subset=[var])[["pk", var]]
                        dummies = pandas.get_dummies(var_frame, prefix=var, columns=[var])
                        weight_vars.extend([d for d in dummies.columns if d.startswith(var)])
                        frame = frame.merge(dummies, on="pk", how="left")

                    training_sample = frame[frame['pk'].isin(training_data['pk'].values)]
                    training_sample['weight'] = compute_sample_weights_from_frame(frame, training_sample, weight_vars)
                    training_data['frame_weight'] = \
                    training_data.merge(training_sample[["pk", "weight"]], on="pk", how="left")["weight"]
                    training_data['frame_weight'] = training_data['frame_weight'].fillna(1.0)
                    training_data['training_weight'] = training_data['training_weight'] * training_data['frame_weight']  # remember, training_weight was set initially by sampling_weight
                    # TODO: for documents that belong to multiple frames, this will NOT work as intended and could multiply some rows into the extremes
                    del training_data["frame_weight"]

        return training_data


    def _add_balancing_weights(self, training_data):

        from django_learning.utils import balancing_variables

        sample = copy.copy(training_data)

        weight_var_names = []
        for mapper_name in self.parameters["documents"].get("balancing_weights", []):
            # balancing_module = importlib.import_module(
            #     "logos.learning.utils.balancing_variables.{0}".format(mapper_name))
            sample[mapper_name] = sample.apply(balancing_variables[mapper_name].var_mapper, axis=1)
            # sample[mapper_name] = sample.apply(balancing_module.var_mapper, axis=1)
            weight_var_names.append(mapper_name)

        if len(self.document_types) > 1 and self.parameters["documents"].get("balance_document_types", False):
            weight_var_names.append("document_type")

        if len(weight_var_names) > 0:

            print "Applying balancing variables: {}".format(weight_var_names)

            weight_vars = []
            for var in weight_var_names:
                var_sample = sample.dropna(subset=[var])[["pk", var]]
                dummies = pandas.get_dummies(var_sample, prefix=var, columns=[var])
                weight_vars.extend([d for d in dummies.columns if d.startswith(var)])
                sample = sample.merge(dummies, on="pk", how="left")

            sample['weight'] = compute_balanced_sample_weights(sample, weight_vars, weight_column="sampling_weight")
            training_data['balancing_weight'] = training_data.merge(sample[["pk", "weight"]], on="pk", how="left")[
                "weight"]
            training_data["balancing_weight"] = training_data["balancing_weight"].fillna(1.0)
            training_data["training_weight"] = training_data["training_weight"] * training_data["balancing_weight"]
            del training_data["balancing_weight"]

            # print "Balancing document types"
            #
            # balanced_ratio = 1.0 / float(len(self.document_types))
            #
            # if use_frames:
            #     frame = self.filter_documents(get_model("Document").objects.filter(sample_frames__in=self.frames.all()), additional_columns=["{}_id".format(d) for d in self.document_types])
            #     for doc_type in self.document_types:
            #         weight = balanced_ratio / (float(len(frame[~frame["{}_id".format(doc_type)].isnull()])) / float(len(frame)))
            #         print "Balancing {} with weight {}, based on frames".format(doc_type, weight)
            #         training_data.ix[training_data['document_type'] == doc_type, "training_weight"] = training_data["training_weight"] * weight
            #
            # else:
            #     for doc_type in self.document_types:
            #         weight = balanced_ratio / (float(len(training_data[training_data['document_type'] == doc_type])) / float(len(training_data)))
            #         print "Balancing {} with weight {}, based on sample".format(doc_type, weight)
            #         training_data.ix[training_data['document_type'] == doc_type, "training_weight"] = training_data["training_weight"] * weight

        return training_data


    @require_model
    def show_top_features(self, n=10):

        print "Top features: "

        if hasattr(self.model.best_estimator_, "named_steps"): steps = self.model.best_estimator_.named_steps
        else: steps = self.model.best_estimator_.steps
        
        feature_names = self.get_feature_names(self.model.best_estimator_)
        class_labels = steps['model'].classes_

        top_features = {}
        if hasattr(steps['model'], "coef_"):
            if len(class_labels) == 2:
                try: coefs = steps['model'].coef_.toarray()[0]
                except: coefs = steps['model'].coef_[0]
                values = list(self.code_variable.codes.values_list("value", flat=True).order_by("pk"))
                top_features["{0} ({1})".format(self.code_variable.codes.get(value=values[0]).label, values[0])] = sorted(zip(coefs, feature_names))[:n]
                top_features["{0} ({1})".format(self.code_variable.codes.get(value=values[1]).label, values[1])] = sorted(zip(coefs, feature_names))[:-(n+1):-1]
            else:
                for i, class_label in enumerate(class_labels):
                    try: coefs = steps['model'].coef_.toarray()[i]
                    except: coefs = steps['model'].coef_[i]
                    top_features["{0} ({1})".format(self.code_variable.codes.get(pk=class_label).label, class_label)] = sorted(zip(coefs, feature_names))[-n:]
        elif hasattr(steps['model'], "feature_importances_"):
            top_features["n/a"] = sorted(zip(
                steps['model'].feature_importances_,
                feature_names
            ))[:-(n+1):-1]

        for class_label, top_n in top_features.iteritems():
            print class_label
            for c, f in top_n:
                try: print "\t%.4f\t\t%-15s" % (c, f)
                except:
                    print "Error: {}, {}".format(c, f)


    @require_training_data
    @require_model
    def print_report(self):

        super(DocumentClassificationHandler, self).print_report()


    @require_training_data
    def _get_model(self, pipeline_steps, params, **kwargs):

        print "woot"
        import pdb
        pdb.set_trace()

        results = super(DocumentClassificationHandler, self)._get_model(pipeline_steps, params, **kwargs)

        if not results['predict_y']:
            print "No test data was provided, scanning database for validation data instead"
            validation_df = self._get_training_data(validation=True)['training_data']
            if is_not_null(validation_df):
                print "Validation data found, computing predictions"
                results['test_y'] = validation_df[self.outcome_variable]
                X_cols = validation_df.columns.tolist()
                X_cols.remove(self.outcome_variable)
                results['test_x'] = validation_df[X_cols]
                results['test_ids'] = results['test_y'].index
                results['predict_y'] = results['model'].predict(results['test_x'])

        return results


    @require_training_data
    @require_model
    def get_report_results(self):

        rows = []
        # report = classification_report(self.test_y, self.predict_y, sample_weight=self.test_x['sampling_weight'] if self.parameters["model"]["use_sample_weights"] else None)

        predict_y = pandas.Series(self.predict_y, index=self.test_x.index)
        test_x = self.test_x[~self.test_x['document_id'].isin(self.train_x['document_id'])]
        test_y = self.test_y.ix[test_x.index]
        predict_y = predict_y[test_x.index].values

        report = classification_report(test_y, predict_y, sample_weight=test_x['sampling_weight'])
        for row in report.split("\n"):
            row = row.strip().split()
            if len(row) == 7:
                row = row[2:]
            if len(row) == 5:
                rows.append({
                    "class": row[0],
                    "precision": row[1],
                    "recall": row[2],
                    "f1-score": row[3],
                    "support": row[4]
                })
        return rows


    @require_model
    def save_to_database(self, name, compute_cv_folds=False):

        if not self.saved_model:

            params = copy.copy(self.parameters)
            # if "frames" in self.parameters["documents"]:
            #     params["documents"]["frames"] = [f['name'] for f in self.parameters["documents"]["frames"]]

            print "Saving model '{}' to database".format(name)

            frames = self.frames.all()
            self.saved_model = get_model("DocumentClassificationModel").objects.create_or_update(
                {"name": name},
                {
                    "variable": self.code_variable,
                    "document_types": self.document_types,
                    "pipeline_name": self.pipeline_name,
                    "parameters": params,
                    "handler_class": self.__class__.__name__
                },
                save_nulls=True,
                empty_lists_are_null=True
            )
            self.saved_model.frames = frames
            self.saved_model.save()
            if compute_cv_folds:
                print "Computing CV folds on training set"
                self.saved_model.compute_cv_folds(num_folds=5)
                # print "Computing CV scores on test set"
                # self.saved_model.compute_cv_folds(use_test_data=True, num_folds=10)

        else:

            print "Classifier already saved to database as '{}'!".format(self.saved_model.name)


    @require_model
    @temp_cache_wrapper
    def compute_cv_folds(self, use_test_data=False, num_folds=5, clear_temp_cache=True):

        if use_test_data:
            train_x = self.test_x[~self.test_x['document_id'].isin(self.train_x['document_id'])]
            train_y = self.test_y.ix[train_x.index]
        else:
            train_x = self.train_x
            train_y = self.train_y

        final_model = self.model.best_estimator_
        fold_preds = cross_val_predict(final_model, train_x, train_y,
            keep_folds_separate=True,
            cv=num_folds,
            fit_params={
               'model__sample_weight': [x for x in train_x["sampling_weight"].values]
            }
        )

        return fold_preds


    @require_model
    def get_code_cv_training_scores(self, fold_preds, X, y, code_value="1", partition_by=None, restrict_document_type=None, min_support=0):

        code = self.code_variable.codes.get(value=code_value).pk

        scores = {}
        full_index = X.index
        if restrict_document_type and 'document_type' in X.columns:
            full_index = X[X['document_type']==restrict_document_type].index

        if len(full_index) > 0:
            if partition_by:
                X['partition'] = X.apply(
                    lambda x: get_model("Document").objects.filter(pk=x['document_id']).values_list(partition_by, flat=True)[0],
                    axis=1
                )
                for partition in X['partition'].unique():
                    index = full_index.intersection(X[X['partition'] == partition].index)
                    rows = []
                    for fold, preds in enumerate(fold_preds):
                        preds, indices = preds
                        new_indices = list(set(indices).intersection(set(index.values)))
                        preds = preds[map(map(lambda x: (x), indices).index, new_indices)]
                        indices = new_indices
                        if len(indices) > min_support:
                            weights = X['sampling_weight'][indices]
                            code_preds = [1 if x == code else 0 for x in preds]
                            code_true = [1 if x == code else 0 for x in y[indices]]
                            if sum(code_preds) > 0 and sum(code_true) > 0:
                                rows.append(self._get_scores(code_preds, code_true, weights=weights))
                    if len(rows) > 0:
                        means = {"{}_mean".format(k): v for k, v in pandas.DataFrame(rows).mean().to_dict().iteritems()}
                        stds = {"{}_std".format(k): v for k, v in pandas.DataFrame(rows).std().to_dict().iteritems()}
                        errs = {"{}_err".format(k): v for k, v in pandas.DataFrame(rows).sem().to_dict().iteritems()}
                        scores[partition] = {}
                        scores[partition].update(means)
                        scores[partition].update(stds)
                        scores[partition].update(errs)
                        if len(scores[partition].keys()) == 0:
                            del scores[partition]

            else:
                rows = []
                for fold, preds in enumerate(fold_preds):
                    preds, indices = preds
                    new_indices = list(set(indices).intersection(set(full_index.values)))
                    preds = preds[map(map(lambda x: (x), indices).index, new_indices)]
                    indices = new_indices
                    if len(indices) > min_support:
                        weights = X['sampling_weight'][indices]
                        code_preds = [1 if x == code else 0 for x in preds]
                        code_true = [1 if x == code else 0 for x in y[indices]]
                        if sum(code_preds) > 0 and sum(code_true) > 0:
                            rows.append(self._get_scores(code_preds, code_true, weights=weights))
                if len(rows) > 0:
                    means = {"{}_mean".format(k): v for k, v in pandas.DataFrame(rows).mean().to_dict().iteritems()}
                    stds = {"{}_std".format(k): v for k, v in pandas.DataFrame(rows).std().to_dict().iteritems()}
                    errs = {"{}_err".format(k): v for k, v in pandas.DataFrame(rows).sem().to_dict().iteritems()}
                    scores.update(means)
                    scores.update(stds)
                    scores.update(errs)

        if len(scores.keys()) > 0:
            return scores
        else:
            return None


    def _get_expert_consensus_dataframe(self, code_value, coders=None, use_consensus_ignore_flag=True):

        if not coders:
            coders = ['ahughes', 'pvankessel']

        df = self._get_raw_codes(turk=False, training=False, coder_names=coders, use_consensus_ignore_flag=use_consensus_ignore_flag)

        df = df[df['code_id'].notnull()]
        df = self._add_code_labels(df)

        input_documents = self.filter_documents(
            get_model("Document").objects.filter(pk__in=df["document_id"])
        )

        df = df.merge(
            input_documents,
            how='inner',
            left_on='document_id',
            right_on="pk"
        )

        if "training_weight" in df.columns:
            del df['training_weight']

        code = self.code_variable.codes.get(value=code_value).pk
        df['code'] = df['code_id'].map(lambda x: 1 if int(x) == int(code) else 0)
        df_mean = df.groupby("document_id").mean().reset_index()
        df_mean = df_mean[df_mean['code'].isin([0.0, 1.0])]

        return df_mean


    def get_code_validation_test_scores(self, code_value="1", partition_by=None, restrict_document_type=None, use_expert_consensus_subset=False, compute_for_experts=False, min_support=0):

        self.predict_y = pandas.Series(self.predict_y, index=self.test_x.index)
        self.test_x = self.test_x[~self.test_x['document_id'].isin(self.train_x['document_id'])]
        self.test_y = self.test_y.ix[self.test_x.index]
        self.predict_y = self.predict_y[self.test_x.index].values

        X = self.test_x
        test_y = self.test_y
        predict_y = pandas.Series(self.predict_y, index=X.index)
        # predict_y = pandas.DataFrame()
        # predict_y['predict_y'] = self.predict_y
        # predict_y = pandas.DataFrame(self.predict_y, columns=["predict_y"], index=X.index)["predict_y"]

        code = self.code_variable.codes.get(value=code_value).pk

        if restrict_document_type:
            X = X[X['document_type']==restrict_document_type]

        abort = False
        if use_expert_consensus_subset:
            expert_consensus_df = self._get_expert_consensus_dataframe(code_value)
            if len(expert_consensus_df) > 0:
                # X = pandas.merge(X, expert_consensus_df, on='document_id')
                old_index = X.index
                X = X.merge(expert_consensus_df[['code', 'code_id', 'document_id']], how='left', on='document_id')
                X.index = old_index
                X = X[~X['code'].isnull()]
                if compute_for_experts:
                    test_y = X['code_id']
                del X['code_id']
                del X['code']
            else:
                abort = True

        if not abort:

            test_y = test_y.ix[X.index].values
            predict_y = predict_y.ix[X.index].values

            scores = {}
            if len(X) > 0:
                if partition_by:
                    scores = {}
                    X['partition'] = X.apply(lambda x: get_model("Document").objects.filter(pk=x['document_id']).values_list(partition_by, flat=True)[0], axis=1)
                    for partition in X['partition'].unique():
                        index = X[X['partition']==partition].index
                        if len(index) > min_support:
                            weights = X['sampling_weight'][index]
                            code_preds = [1 if x == code else 0 for x in predict_y[index]]
                            code_true = [1 if x == code else 0 for x in test_y[index]]
                            if sum(code_preds) > 0 and sum(code_true) > 0:
                                scores[partition] = self._get_scores(code_preds, code_true, weights=weights)
                                if len(scores[partition].keys()) == 0:
                                    del scores[partition]

                elif len(X) > min_support:
                    weights = X['sampling_weight']
                    code_preds = [1 if x == code else 0 for x in predict_y]
                    code_true = [1 if x == code else 0 for x in test_y]
                    if sum(code_preds) > 0 and sum(code_true) > 0:
                        scores = self._get_scores(code_preds, code_true, weights=weights)

            if len(scores.keys()) > 0:
                return scores
            else:
                return None

        else:

            return None


    def _get_scores(self, code_preds, code_true, weights=None):

        row = {
            "matthews_corrcoef": matthews_corrcoef(code_true, code_preds, sample_weight=weights),
            "accuracy": accuracy_score(code_true, code_preds, sample_weight=weights),
            "f1": f1_score(code_true, code_preds, pos_label=1, sample_weight=weights),
            "precision": precision_score(code_true, code_preds, pos_label=1, sample_weight=weights),
            "recall": recall_score(code_true, code_preds, pos_label=1, sample_weight=weights),
            "roc_auc": roc_auc_score(code_true, code_preds, sample_weight=weights) if len(numpy.unique(code_preds)) > 1 and len(numpy.unique(code_true)) > 1 else None
        }

        for codesetname, codeset in [
            ("pred", code_preds),
            ("true", code_true)
        ]:
            # unweighted = wmom(codeset, [1.0 for x in codeset], calcerr=True, sdev=True)
            weighted = wmom(codeset, weights, calcerr=True, sdev=True)
            # for valname, val in zip(["mean", "err", "std"], list(unweighted)):
            #     row["{}_{}".format(codesetname, valname)] = val
            for valname, val in zip(["mean", "err", "std"], list(weighted)):
                row["{}_{}".format(codesetname, valname)] = val

        row["ttest_t"], row["ttest_p"] = ttest_ind(code_preds, code_true)
        if row["ttest_p"] > .05: row["ttest_pass"] = 1
        else: row["ttest_pass"] = 0

        row["pct_agree"] = numpy.average([1 if c[0]==c[1] else 0 for c in zip(code_preds, code_true)])

        if sum(code_preds) > 0 and sum(code_true) > 0:

            result_dict = {0: defaultdict(int), 1: defaultdict(int)}
            for pred, true in zip(code_preds, code_true):
                result_dict[pred][true] += 1
            kappa = cohens_kappa([
                [result_dict[0][0], result_dict[0][1]],
                [result_dict[1][0], result_dict[1][1]]
            ])
            row["kappa"] = kappa["kappa"]
            row["kappa_err"] = kappa["std_kappa"]

        return row


    def apply_model(self, documents, keep_cols=None, clear_temp_cache=True):

        if not keep_cols: keep_cols = ["pk"]
        documents = self.filter_documents(documents)
        if len(documents) > 0:
            return super(DocumentClassificationHandler, self).apply_model(documents, keep_cols=keep_cols, clear_temp_cache=clear_temp_cache)
        else:
            return pandas.DataFrame()


    @require_model
    @temp_cache_wrapper
    def apply_model_to_database(self, documents, num_cores=2, chunk_size=1000, clear_temp_cache=True):

        if not self.saved_model:

            print "Model is not currently saved in the database; please call 'save_to_database(name)' and pass a unique name"

        else:

            # documents = self._apply_document_filters(documents)

            # TODO: edited this to assume that it's a queryset, because it's MUCH more efficient to do a .count check
            # if you want to allow for lists of documents, do it in a way that checks whether it's a queryset BEFORE checking length
            # so you can avoid having to load/evaluate an unloaded queryset in full, just to check the length
            #if len(documents) > 0:
            if documents.count() > 0:

                try: document_ids = list(documents.values_list("pk", flat=True))
                except: document_ids = [getattr(d, "pk") for d in documents]

                print "Processing {} {}".format(len(document_ids), self.document_types)
                # for i, chunk in enumerate(chunker(document_ids, chunk_size)):
                #     codes = self.apply_model(get_model("Document").objects.filter(pk__in=chunk))
                #     print "Processing {} of {} ({}, {})".format((i+1)*chunk_size, len(document_ids), self.code_variable.name, self.document_types)
                #     for index, row in codes.iterrows():
                #         get_model("ClassifierDocumentCode").objects.create_or_update(
                #             {
                #                 "classifier": self.saved_model,
                #                 "document_id": row["pk"]
                #             },
                #             {"code_id": row[self.outcome_variable]},
                #             save_nulls=True,
                #             return_object=False
                #         )

                pool = Pool(processes=num_cores)
                for i, chunk in enumerate(chunker(document_ids, chunk_size)):
                    print "Creating chunk %i of %i" % (i+1, (i+1)*chunk_size)
                    pool.apply_async(_process_document_chunk, args=(self.saved_model.pk, chunk, i))
                    # _process_document_chunk(self.saved_model.pk, chunk, i)
                    # break
                pool.close()
                pool.join()

            else:

                print "All documents were filtered, nothing to do"


    @require_model
    def coded_documents(self):

        if self.saved_model:
            return self.saved_model.coded_documents.all()
        else:
            return []


def _process_document_chunk(model_id, chunk, i):

    try:

        import os, django, sys, traceback
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "{}.settings".format(settings.SITE_NAME))
        django.setup()
        from django.db import connection
        connection.close()

        from django_learning.models import DocumentClassificationModel, ClassifierDocumentCode

        model = DocumentClassificationModel.objects.get(pk=model_id)
        ClassifierDocumentCode.objects.filter(document_id__in=chunk, classifier=model).delete()
        handler = model.handler
        handler.load_model()
        codes = handler.apply_model(get_model("Document").objects.filter(pk__in=chunk))

        doc_codes = []
        for index, row in codes.iterrows():
            doc_codes.append(
                ClassifierDocumentCode(**{
                    "classifier_id": model_id,
                    "document_id": row["pk"],
                    "code_id": row[handler.outcome_variable]
                })
            )
        ClassifierDocumentCode.objects.bulk_create(doc_codes)

        print "Done processing chunk %i" % (int(i)+1)

    except Exception as e:

        print e
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print exc_type
        print exc_value
        print exc_traceback
        traceback.print_exc(exc_traceback)
        raise