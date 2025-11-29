from mysklearn.decision_tree import DecisionTreeClassifier

X_train_interview = [
    ["Senior", "Java", "no", "no"],
    ["Senior", "Java", "no", "yes"],
    ["Mid", "Python", "no", "no"],
    ["Junior", "Python", "no", "no"],
    ["Junior", "R", "yes", "no"],
    ["Junior", "R", "yes", "yes"],
    ["Mid", "R", "yes", "yes"],
    ["Senior", "Python", "no", "no"],
    ["Senior", "R", "yes", "no"],
    ["Junior", "Python", "yes", "no"],
    ["Senior", "Python", "yes", "yes"],
    ["Mid", "Python", "no", "yes"],
    ["Mid", "Java", "yes", "no"],
    ["Junior", "Python", "no", "yes"],
]

y_train_interview = [
    "False",
    "False",
    "True",
    "True",
    "True",
    "False",
    "True",
    "False",
    "True",
    "True",
    "True",
    "True",
    "True",
    "False",
]

tree_interview = [
    "Attribute",
    "att0",
    ["Value", "Junior", ["Attribute", "att3", ["Value", "no", ["Leaf", "True", 3, 5]], ["Value", "yes", ["Leaf", "False", 2, 5]]]],
    ["Value", "Mid", ["Leaf", "True", 4, 14]],
    ["Value", "Senior", ["Attribute", "att2", ["Value", "no", ["Leaf", "False", 3, 5]], ["Value", "yes", ["Leaf", "True", 2, 5]]]],
]

X_train_iphone = [
    [1, 3, "fair"],
    [1, 3, "excellent"],
    [2, 3, "fair"],
    [2, 2, "fair"],
    [2, 1, "fair"],
    [2, 1, "excellent"],
    [2, 1, "excellent"],
    [1, 2, "fair"],
    [1, 1, "fair"],
    [2, 2, "fair"],
    [1, 2, "excellent"],
    [2, 2, "excellent"],
    [2, 3, "fair"],
    [2, 2, "excellent"],
    [2, 3, "fair"],
]

y_train_iphone = [
    "no",
    "no",
    "yes",
    "yes",
    "yes",
    "no",
    "yes",
    "no",
    "yes",
    "yes",
    "yes",
    "yes",
    "yes",
    "no",
    "yes",
]

tree_iphone = [
    "Attribute",
    "att0",
    ["Value", 1, ["Attribute", "att1", ["Value", 1, ["Leaf", "yes", 1, 5]], ["Value", 2, ["Attribute", "att2", ["Value", "excellent", ["Leaf", "yes", 1, 2]], ["Value", "fair", ["Leaf", "no", 1, 2]]]], ["Value", 3, ["Leaf", "no", 2, 5]]]],
    ["Value", 2, ["Attribute", "att2", ["Value", "excellent", ["Attribute", "att1", ["Value", 1, ["Leaf", "no", 1, 4]], ["Value", 2, ["Leaf", "no", 1, 4]], ["Value", 3, ["Leaf", "no", 2, 4]]]], ["Value", "fair", ["Leaf", "yes", 6, 10]]]],
]


def test_decision_tree_classifier_fit():
    interview_clf = DecisionTreeClassifier()
    interview_clf.fit(X_train_interview, y_train_interview)
    assert interview_clf.tree == tree_interview

    iphone_clf = DecisionTreeClassifier()
    iphone_clf.fit(X_train_iphone, y_train_iphone)
    assert iphone_clf.tree == tree_iphone


def test_decision_tree_classifier_predict():
    interview_clf = DecisionTreeClassifier()
    interview_clf.fit(X_train_interview, y_train_interview)
    interview_test = [
        ["Junior", "Java", "yes", "no"],
        ["Senior", "Python", "no", "yes"],
    ]
    interview_expected = ["True", "False"]
    assert interview_clf.predict(interview_test) == interview_expected

    iphone_clf = DecisionTreeClassifier()
    iphone_clf.fit(X_train_iphone, y_train_iphone)
    iphone_test = [
        [1, 1, "excellent"],
        [2, 3, "excellent"],
    ]
    iphone_expected = ["yes", "no"]
    assert iphone_clf.predict(iphone_test) == iphone_expected