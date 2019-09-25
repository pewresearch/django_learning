# ### ML Related UTILS
#
# import sklearn
#
# from sklearn.metrics import f1_score, precision_score, recall_score, make_scorer
# from sklearn.cross_validation import train_test_split
# from sklearn.grid_search import GridSearchCV
# from sklearn.metrics import classification_report
#
#
# def setup_learning_df(df,
#                       class_column,
#                       class_label_map,
#                       feature_cols = ['text'],
#                       test_percent = .2,
#                       seed = 10 ):
#
#     if class_label_map:
#         y = df[class_column].map(class_label_map)
#     else:
#         y = df[class_column]
#
#     if feature_cols:
#         X = df[feature_cols]
#     else:
#         X = df['text']
#
#
#     X_train, X_test, y_train, y_test, train_ids, test_ids = train_test_split(X, y, y.index, test_size= test_percent , random_state=seed)
#     print "Selected %i training cases and %i" % (
#         len(y_train),
#         len(y_test) )
#
#     return  X_train, X_test, y_train, y_test, train_ids, test_ids
#
#
# def make_scoring_function(scoring_paramemeter, unique_outcomes, smallest_code):
#     if unique_outcomes == 2:
#         if scoring_paramemeter == "f1":
#             scoring_function = make_scorer(f1_score, pos_label=smallest_code) # equivalent to f1_binary
#         elif scoring_paramemeter == "precision":
#             scoring_function = make_scorer(precision_score, pos_label=smallest_code)
#         else:
#             scoring_function = make_scorer(recall_score, pos_label=smallest_code)
#     else:
#         if scoring_paramemeter == "f1":
#             scoring_function = "f1_macro"
#             # scoring_function = "f1_micro"
#             # scoring_function = "f1_weighted"
#         elif scoring_paramemeter == "precision":
#             scoring_function = "precision"
#         else:
#             scoring_function = "recall"
#     return scoring_function
#
# def model_factory(df, pipeline
#                   , parameters = None
#                   , cv_folds = 5 # If none, then no cross-validation will occur
#                   , test_percent = .3
#                   , scoring_function = 'f1'
#                   , seed = 5
#                   , class_column = 'code_id' # column for the outcome variable
#                   , slh = None
#                   , class_label_map = None
#                   , feature_cols = None # List of columns used as features
#                   , weighted = True
#                  ):
#     X_train, X_test, y_train, y_test, train_ids, test_ids = setup_learning_df(
#                       df = df,
#                       class_column = class_column,
#                       class_label_map = class_label_map,
#                       feature_cols = feature_cols,
#                       test_percent = test_percent,
#                       seed = seed
#     )
#     print "Selected %i training cases and %i" % (
#         len(y_train),
#         len(y_test) )
#
#     scoring_function = make_scoring_function(
#                   scoring_paramemeter = scoring_function,
#                   unique_outcomes=len(y_train.unique()),
#                   smallest_code=y_train.value_counts(ascending=True).index[0]
#     )
#
#     grid_search_parameters = parameters or {}
#
#
#     if cv_folds:
#         print "Beginning grid search using %s and %i cores for" % (
#             str(scoring_function),
#             0,# self.num_cores,
#
#         )
#
#         clf = GridSearchCV(
#             pipeline,
#             grid_search_parameters,
#             cv=cv_folds,
#             scoring=scoring_function,
#            # n_jobs=self.num_cores,
#             verbose=1
#         )
#     else :
#         clf = pipeline
#
#     if weighted & ('weight' in df.columns):
#         print("weight the samples")
#         sample_weights = df[df.index.isin(X_train.index)].weight
#         clf.fit(X_train, y_train, sample_weights = sample_weights)
#     else:
#         clf.fit(X_train, y_train)
#
#     y_test, y_pred = y_test, clf.predict(X_test)
#
#     print(classification_report(y_test, y_pred))
#     #print(confusion_matrix(y_test, y_pred))
#
#     #if parameters:
#     #    print(clf.best_params_)
#
#     return  (clf, X_train, y_train, train_ids, X_test, y_test, test_ids)#, scoring_output
#

# def get_regex_patterns():
#
#     ML_regex_patterns = {
#         'waste_pattern':          r'waste*|ineffici*|frivolous|mismanage*|overruns',
#         'regulation_pattern':     r'regulati*',
#         'regulationbad_pattern':  r'burden*|excessive|hurt|overreach',
#         'regulationgood_pattern': r'protect|saf*|responsible',
#         'benefits_pattern':       r'welfare|benefits|handout',
#         'helppoor_pattern':       r'poverty|assist*',
#         'gov_pattern':            r'governm+|federal|U\.S\.|US',
#         'execu_pattern':          r'Obama|[Pp]resident|White House|executive',
#         'bureaucracy_pattern':    r'agency|department'
#     }
#
#     return ML_regex_patterns
#
#
# def match_class(df, pattern_dict, text_col='text'):
#
#     """
#     df= data frame to apply on
#     pattern_dict = dictionary of regex patterns to match
#     text_col= name of column to match on
#
#     returns the dataframe with a boolean column for each pattern
#     """
#
#     keys=pattern_dict.keys()
#     for k in keys:
#         df[k.split('_')[0]] = df[text_col].str.contains(pattern_dict[k])
#
#
# #def show_kwic(target, window, texts, maxres = 20):
# #    """
#     # adapted from http://conjugateprior.org/software/ca-in-python/
# #    target = target pattern to search for
# #    window = terms before and terms after the match to show
# #    texts = list of texts to show
# #    maxres = ?
# #    """
# #    res = 0
# #    for text in texts:
# #        tokens = text.split() # split on whitespace
# #        keyword = re.compile(target, re.IGNORECASE)
# #        for index in range( len(tokens) ):
# #            if keyword.match( tokens[index] ):
# #                res += 1
# #                if res >= maxres:
# #                    break
# #                start = max(0, index-window)
# #                finish = min(len(tokens), index+window+1)
# #                lhs = string.join( tokens[start:index] )
# #                rhs = string.join( tokens[index+1:finish] )
# #                print "%s [%s] %s" % (lhs, tokens[index], rhs)
#
#
# def show_kwic(target, window, texts, maxres=20, p=False, df='summary'):
#
#     """
#     # adapted from http://conjugateprior.org/software/ca-in-python/
#     target = target pattern to search for
#     window = terms before and terms after the match to show
#     texts = list of texts to show
#     maxres = ?
#
#     p = True means print text matches , false means don't.
#     df = "summary" returns a dataframe of with terms that frequently occur near your target term
#          otherwise it returns a dictionary of the text near the term you searched for.
#     """
#
#     res = 0
#     kwik = {}
#     for text in texts:
#         tokens = text.split() if type(text) in [str,unicode] else ''.split()# split on whitespace
#         keyword = re.compile(target, re.IGNORECASE)
#         for index in range( len(tokens) ):
#             if keyword.match( tokens[index] ):
#                 kwik[tokens[index]] = kwik.get(tokens[index]) or ''
#                 res += 1
#                 if res >= maxres:
#                     break
#                 start = max(0, index-window)
#                 finish = min(len(tokens), index+window+1)
#                 lhs = string.join( tokens[start:index] )
#                 rhs = string.join( tokens[index+1:finish] )
#                 if p:
#                     print "%s [%s] %s" % (lhs, tokens[index], rhs)
#                 kwik[tokens[index]] +="%s %s " % (lhs, rhs)
#     nonempty = { k:kwik[k] for k in kwik if kwik[k] != '' }
#     if df == 'summary':
#         if nonempty != {} :
#             wordcnt = {k: dict(Counter(kwik[k].split())) for k in nonempty}
#             ind = [{(innerKey, k) : [v] for k, v in values.iteritems()}  for innerKey, values in wordcnt.iteritems()]
#             df=pandas.concat([pandas.DataFrame(pandas.DataFrame(k).unstack()) for k in ind])
#             df.columns = ['freq']
#             df.index.names=['mainterm','closeterm','0']
#             df.reset_index(level=(0,1,2), inplace=True)
#             df =df[['mainterm','closeterm','freq']]
#         #df.groupby(['mainterm','closeterm']).sort('freq', ascending = False)
#             return df
#         else:
#             return pandas.DataFrame()
#     else:
#         return nonempty
#
#
# def show_most_informative_features(vectorizer, clf, n=20):
#
#     """
#     Shows the most informative features in a sklearn classifier.
#     http://stackoverflow.com/questions/11116697/how-to-get-most-informative-features-for-scikit-learn-classifiers
#     vectorizer = vectorizer function
#     clf = model (sklearn) function
#     n = top number of features
#     """
#
#     c_f = sorted(zip(clf.coef_[0], vectorizer.get_feature_names()))
#     top = zip(c_f[:n], c_f[:-(n+1):-1])
#     for (c1,f1),(c2,f2) in top:
#         print "\t%.4f\t%-15s\t\t%.4f\t%-15s" % (c1,f1,c2,f2)
#
#
# def regex_most_informative_features(vectorizer, clf, n=20):
#
#     """
#     Creates a regular expression from the most informative features.
#     Returns a dictionary of 'negative' + regex & 'positive' + regex
#     """
#
#     c_f = sorted(zip(clf.coef_[0], vectorizer.get_feature_names()))
#     negative = [f for c, f in c_f[:n]]
#     positive = [f for c, f in c_f[:-(n+1):-1]]
#     return {'positive': r'|'.join(positive),
#             'negative': r'|'.join(negative)}
#
#
# def make_new_labels(regex_labels, text_series):
#
#     """
#     Returns positive and negative class labels from dictionary of regex labels
#     """
#
#     return pandas.DataFrame({k : text_series.str.contains( regex_labels[k]) for k in regex_labels} )
#
#
# def view_misclass_text(df,pred_vec,kw_pred_vec):
#
#     """
#     Returns mis-classified entries from DF
#     pass in a df
#     pred_vec = vector of predicted labels
#     kw_pred_vec = vector of keywork-predicted labels
#     """
#
#     new=df.join(pandas.DataFrame({'pred':pred_vec}), how='inner').join(pandas.DataFrame({'kw_pred':kw_pred_vec}))
#     missclassified=new[['text','pred','kw_pred']][new.pred!=new.kw_pred]
#     print len(missclassified)
#     print len(df)
#     return missclassified
