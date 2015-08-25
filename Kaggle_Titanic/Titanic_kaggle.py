import pandas
import numpy
import csv
import re
import operator
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import xgboost

# A function to get the title from a name.
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

# A function to get the id given a row
def get_family_id(row):
    # Find the last name by splitting on a comma
    last_name = row["Name"].split(",")[0]
    # Create the family id
    family_id = "{0}{1}".format(last_name, row["FamilySize"])
    # Look up the id in the mapping
    if family_id not in family_id_mapping:
        if len(family_id_mapping) == 0:
            current_id = 1
        else:
            # Get the maximum id from the mapping and add one to it if we don't have an id
            current_id = (max(family_id_mapping.items(), key=operator.itemgetter(1))[1] + 1)
        family_id_mapping[family_id] = current_id
    return family_id_mapping[family_id]    

# Main 
if __name__ == '__main__':
	# Chargement des donnees
	titanic = pandas.read_csv("train.csv")
	titanic_test = pandas.read_csv("test.csv")

	titanic["Cabin"] = titanic["Cabin"].fillna("N")
	titanic.loc[titanic["Cabin"].str.startswith("N", na=True), "Cabin"] = 0
	titanic.loc[titanic["Cabin"].str.startswith("A", na=False), "Cabin"] = 1
	titanic.loc[titanic["Cabin"].str.startswith("B", na=False), "Cabin"] = 2
	titanic.loc[titanic["Cabin"].str.startswith("C", na=False), "Cabin"] = 3
	titanic.loc[titanic["Cabin"].str.startswith("D", na=False), "Cabin"] = 4
	titanic.loc[titanic["Cabin"].str.startswith("E", na=False), "Cabin"] = 5
	titanic.loc[titanic["Cabin"].str.startswith("F", na=False), "Cabin"] = 6
	titanic.loc[titanic["Cabin"].str.startswith("G", na=False), "Cabin"] = 7
	titanic.loc[titanic["Cabin"].str.startswith("T", na=False), "Cabin"] = 8

	# Clean les donnees train
	titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

	titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())

	titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
	titanic.loc[titanic["Sex"] == "female", "Sex"] = 1

	titanic["Embarked"] = titanic["Embarked"].fillna("S")
	titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
	titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
	titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2

	# Clean les donnees test
	titanic_test["Age"] = titanic_test["Age"].fillna(titanic["Age"].median())

	titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())

	titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0 
	titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1

	titanic_test["Embarked"] = titanic_test["Embarked"].fillna("S")
	titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0
	titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1
	titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2


	titanic_test["Cabin"] = titanic_test["Cabin"].fillna("N")
	titanic_test.loc[titanic_test["Cabin"].str.startswith("N", na=True), "Cabin"] = 0
	titanic_test.loc[titanic_test["Cabin"].str.startswith("A", na=False), "Cabin"] = 1
	titanic_test.loc[titanic_test["Cabin"].str.startswith("B", na=False), "Cabin"] = 2
	titanic_test.loc[titanic_test["Cabin"].str.startswith("C", na=False), "Cabin"] = 3
	titanic_test.loc[titanic_test["Cabin"].str.startswith("D", na=False), "Cabin"] = 4
	titanic_test.loc[titanic_test["Cabin"].str.startswith("E", na=False), "Cabin"] = 5
	titanic_test.loc[titanic_test["Cabin"].str.startswith("F", na=False), "Cabin"] = 6
	titanic_test.loc[titanic_test["Cabin"].str.startswith("G", na=False), "Cabin"] = 7
	titanic_test.loc[titanic_test["Cabin"].str.startswith("T", na=False), "Cabin"] = 8

	# Creer des nouveaux attributs train
	titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]
	titanic["NameLength"] = titanic["Name"].apply(lambda x: len(x))
	titles = titanic["Name"].apply(get_title)
	title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}
	for k,v in title_mapping.items():
		titles[titles == k] = v
	titanic["Title"] = titles

	family_id_mapping = {}
	family_ids = titanic.apply(get_family_id, axis=1) # Get the family ids with the apply method
	family_ids[titanic["FamilySize"] < 3] = -1 # There are a lot of family ids, so we'll compress all of the families under 3 members into one code.
	titanic["FamilyId"] = family_ids

	# Creer des nouveaux attributs test
	titles = titanic_test["Name"].apply(get_title)
	# We're adding the Dona title to the mapping, because it's in the test set, but not the training set
	title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2, "Dona": 10}
	for k,v in title_mapping.items():
	    titles[titles == k] = v
	titanic_test["Title"] = titles
	# Check the counts of each unique title.
	#print(pandas.value_counts(titanic_test["Title"]))

	# Now, we add the family size column.
	titanic_test["FamilySize"] = titanic_test["SibSp"] + titanic_test["Parch"]

	# Now we can add family ids.
	family_ids = titanic_test.apply(get_family_id, axis=1)
	family_ids[titanic_test["FamilySize"] < 3] = -1
	titanic_test["FamilyId"] = family_ids
	titanic_test["NameLength"] = titanic_test["Name"].apply(lambda x: len(x))
	
	predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]
	algorithms = [
		[GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), predictors],
		[svm.SVC(random_state=1, probability=True), ["Pclass", "Sex", "Fare", "Title", "Embarked" ]],
		[RandomForestClassifier(random_state=1, n_estimators=100, min_samples_split=8, min_samples_leaf=4), predictors],
		[xgboost.XGBClassifier(max_depth=4,n_estimators=128), ["Pclass", "Sex", "Fare", "Title", "Embarked"]],
		[LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]
	]

	full_predictions = []
	for alg, predictors in algorithms:
		print("Train en cour")
		alg.fit(numpy.array(titanic[list(predictors)]), titanic["Survived"])
		predictions = alg.predict_proba(titanic_test[predictors].astype(float))[:,1]
		#predictions = alg.predict(numpy.array(titanic[list(predictors)]))
		full_predictions.append(predictions)

	print(len(titanic_test["PassengerId"]))
	
	print(predictions)

	# The gradient boosting classifier generates better predictions, so we weight it higher.
	predictions = (full_predictions[0] * 4 + full_predictions[1] * 2 + full_predictions[2] * 2 + full_predictions[3] * 1 + full_predictions[4] * 2) / 11
	print(predictions)
	predictions[predictions <= .5] = 0
	predictions[predictions > .5] = 1
	predictions = predictions.astype(int)
	csvInter = titanic_test["PassengerId"]
	predictions_file = open("prediction_v33.csv", "wb")
	open_file_object = csv.writer(predictions_file)
	open_file_object.writerow(["PassengerId","Survived"])
	open_file_object.writerows(zip(csvInter, predictions))
	predictions_file.close()
	#"""

