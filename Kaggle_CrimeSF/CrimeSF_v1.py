import pandas
import numpy
import csv
import re
import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.lda import LDA
from sklearn import svm
from sklearn import cross_validation
from ggplot import *

def get_month(date):
	month_search = re.search('-[0-9][0-9]-', date)
	if month_search:
		return month_search.group(0)
	return ""

def get_hours(date):
	hours_search = re.search(' [0-9][0-9]:', date)
	if hours_search:
		return hours_search.group(0)
	return ""

def get_typeStreet(street):
	street_search = re.search(' AV', street)
	if street_search:
		return street_search.group(0)
	street_search = re.search(' BL', street)
	if street_search:
		return street_search.group(0)
	street_search = re.search(' CR', street)
	if street_search:
		return street_search.group(0)
	street_search = re.search(' CT', street)
	if street_search:
		return street_search.group(0)
	street_search = re.search(' DR', street)
	if street_search:
		return street_search.group(0)
	street_search = re.search(' HY', street)
	if street_search:
		return street_search.group(0)
	street_search = re.search(' LN', street)
	if street_search:
		return street_search.group(0)
	street_search = re.search(' PL', street)
	if street_search:
		return street_search.group(0)
	street_search = re.search(' PZ', street)
	if street_search:
		return street_search.group(0)
	street_search = re.search(' RD', street)
	if street_search:
		return street_search.group(0)
	street_search = re.search(' ST', street)
	if street_search:
		return street_search.group(0)
	street_search = re.search(' TR', street)
	if street_search:
		return street_search.group(0)
	street_search = re.search(' WY', street)
	if street_search:
		return street_search.group(0)
	return " ST"

############ Main ############
if __name__ == '__main__':

	############ Chargement des donnees ############

	print("Chargement des donnes")
	crime_train = pandas.read_csv("train.csv")
	crime_test = pandas.read_csv("test.csv")

	############ Ajout de nouvelles colonnes Train ############

	print("Ajout colonne Month (Train)")
	month = crime_train["Dates"].apply(get_month)
	month_mapping = {"-01-": 1, "-02-": 2, "-03-": 3, "-04-": 4, "-05-": 5, "-06-": 6, "-07-": 7, "-08-": 8, "-09-": 9, "-10-": 10, "-11-": 11, "-12-": 12}
	for k,v in month_mapping.items():
		month[month == k] = v
	crime_train["Month"] = month
	#print(pandas.value_counts(crime_train["Month"]))
	#print ggplot(crime_train, aes(x='Month', fill='Category')) + geom_bar()

	print("Ajout colonne Season (Train)")
	season = crime_train["Month"].copy(deep=True)
	season_mapping = {1:"Winter", 2:"Winter", 3:"Winter", 4:"Spring", 5:"Spring", 6:"Spring", 7:"Summer", 8:"Summer", 9:"Summer", 10:"Fall", 11:"Fall", 12:"Fall"}
	for k,v in season_mapping.items():
		season[season == k] = v
	crime_train["Season"] = season
	#print(pandas.value_counts(crime_train["Season"]))
	#print ggplot(crime_train, aes(x='Season', fill='Category')) + geom_bar()

	print("Ajout colonne Hours (Train)")
	hours = crime_train["Dates"].apply(get_hours)
	#hours_mapping = { " 00:":"00h-06h", " 01:":"00h-06h", " 02:":"00h-06h", " 03:":"00h-06h", " 04:":"00h-06h", " 05:":"00h-06h", " 06:":"06h-09h", " 07:":"06h-09h", " 08:":"06h-09h", " 09:":"09h-12h", " 10:":"09h-12h", " 11:":"09h-12h", " 12:":"12h-18h", " 13:":"12h-18h", " 14:":"12h-18h", " 15:":"12h-18h", " 16:":"12h-18h", " 17:":"12h-18h", " 18:":"18h-21h", " 19:":"18h-21h", " 20:":"18h-21h", " 21:":"21h-00h", " 22:":"21h-00h", " 23:":"21h-00h"}
	hours_mapping = { " 00:":"00h", " 01:":"01h", " 02:":"02h", " 03:":"03h", " 04:":"04h", " 05:":"05h", " 06:":"06h", " 07:":"07h", " 08:":"08h", " 09:":"09h", " 10:":"10h", " 11:":"11h", " 12:":"12h", " 13:":"13h", " 14:":"14h", " 15:":"15h", " 16:":"16h", " 17:":"17h", " 18:":"18h", " 19:":"19h", " 20:":"20h", " 21:":"21h", " 22:":"22h", " 23:":"23h"}
	for k,v in hours_mapping.items():
		hours[hours == k] = v
	crime_train["Hours"] = hours
	#print(pandas.value_counts(crime_train["Hours"]))
	#print ggplot(crime_train, aes(x='Hours', fill='Category')) + geom_bar()

	print("Ajout colonne TypeStreet (Train)")
	type_street = crime_train["Address"].apply(get_typeStreet)
	type_mapping = { " ST":0, " AV":1, " BL":2, " DR":3, " HY":4, " WY":5, " RD":6, " CT":7, " PL":8, " PZ":9, " LN":10, " CR":11, " TR":12}
	for k,v in type_mapping.items():
		type_street[type_street == k] = v
	crime_train["TypeStreet"] = type_street
	#print(pandas.value_counts(crime_train["TypeStreet"]))
	#print ggplot(crime_train, aes(x='TypeStreet', fill='Category')) + geom_bar()

	############ Ajout de nouvelles colonnes Test ############

	print("Ajout colonne Month (Test)")
	month = crime_test["Dates"].apply(get_month)
	month_mapping = {"-01-": 1, "-02-": 2, "-03-": 3, "-04-": 4, "-05-": 5, "-06-": 6, "-07-": 7, "-08-": 8, "-09-": 9, "-10-": 10, "-11-": 11, "-12-": 12}
	for k,v in month_mapping.items():
		month[month == k] = v
	crime_test["Month"] = month

	print("Ajout colonne Season (Test)")
	season = crime_test["Month"].copy(deep=True)
	season_mapping = {1:"Winter", 2:"Winter", 3:"Winter", 4:"Spring", 5:"Spring", 6:"Spring", 7:"Summer", 8:"Summer", 9:"Summer", 10:"Fall", 11:"Fall", 12:"Fall"}
	for k,v in season_mapping.items():
		season[season == k] = v
	crime_test["Season"] = season

	print("Ajout colonne Hours (Test)")
	hours = crime_test["Dates"].apply(get_hours)
	#hours_mapping = { " 00:":"00h-06h", " 01:":"00h-06h", " 02:":"00h-06h", " 03:":"00h-06h", " 04:":"00h-06h", " 05:":"00h-06h", " 06:":"06h-09h", " 07:":"06h-09h", " 08:":"06h-09h", " 09:":"09h-12h", " 10:":"09h-12h", " 11:":"09h-12h", " 12:":"12h-18h", " 13:":"12h-18h", " 14:":"12h-18h", " 15:":"12h-18h", " 16:":"12h-18h", " 17:":"12h-18h", " 18:":"18h-21h", " 19:":"18h-21h", " 20:":"18h-21h", " 21:":"21h-00h", " 22:":"21h-00h", " 23:":"21h-00h"}
	hours_mapping = { " 00:":"00h", " 01:":"01h", " 02:":"02h", " 03:":"03h", " 04:":"04h", " 05:":"05h", " 06:":"06h", " 07:":"07h", " 08:":"08h", " 09:":"09h", " 10:":"10h", " 11:":"11h", " 12:":"12h", " 13:":"13h", " 14:":"14h", " 15:":"15h", " 16:":"16h", " 17:":"17h", " 18:":"18h", " 19:":"19h", " 20:":"20h", " 21:":"21h", " 22:":"22h", " 23:":"23h"}
	for k,v in hours_mapping.items():
		hours[hours == k] = v
	crime_test["Hours"] = hours

	print("Ajout colonne TypeStreet (Test)")
	type_street = crime_test["Address"].apply(get_typeStreet)
	type_mapping = { " ST":0, " AV":1, " BL":2, " DR":3, " HY":4, " WY":5, " RD":6, " CT":7, " PL":8, " PZ":9, " LN":10, " CR":11, " TR":12}
	for k,v in type_mapping.items():
		type_street[type_street == k] = v
	crime_test["TypeStreet"] = type_street

	############ Clean Data Train ############

	print("Clean Season (Train)")
	crime_train["Season"] = crime_train["Season"].fillna("Spring")
	crime_train.loc[crime_train["Season"] == "Winter", "Season"] = 0
	crime_train.loc[crime_train["Season"] == "Spring", "Season"] = 1
	crime_train.loc[crime_train["Season"] == "Summer", "Season"] = 2
	crime_train.loc[crime_train["Season"] == "Fall", "Season"] = 3

	print("Clean Hours (Train)")
	crime_train["Hours"] = crime_train["Hours"].fillna("12h")
	crime_train.loc[crime_train["Hours"] == "00h", "Hours"] = 0
	crime_train.loc[crime_train["Hours"] == "01h", "Hours"] = 1
	crime_train.loc[crime_train["Hours"] == "02h", "Hours"] = 2
	crime_train.loc[crime_train["Hours"] == "03h", "Hours"] = 3
	crime_train.loc[crime_train["Hours"] == "04h", "Hours"] = 4
	crime_train.loc[crime_train["Hours"] == "05h", "Hours"] = 5
	crime_train.loc[crime_train["Hours"] == "06h", "Hours"] = 6
	crime_train.loc[crime_train["Hours"] == "07h", "Hours"] = 7
	crime_train.loc[crime_train["Hours"] == "08h", "Hours"] = 8
	crime_train.loc[crime_train["Hours"] == "09h", "Hours"] = 9
	crime_train.loc[crime_train["Hours"] == "10h", "Hours"] = 10
	crime_train.loc[crime_train["Hours"] == "11h", "Hours"] = 11
	crime_train.loc[crime_train["Hours"] == "12h", "Hours"] = 12
	crime_train.loc[crime_train["Hours"] == "13h", "Hours"] = 13
	crime_train.loc[crime_train["Hours"] == "14h", "Hours"] = 14
	crime_train.loc[crime_train["Hours"] == "15h", "Hours"] = 15
	crime_train.loc[crime_train["Hours"] == "16h", "Hours"] = 16
	crime_train.loc[crime_train["Hours"] == "17h", "Hours"] = 17
	crime_train.loc[crime_train["Hours"] == "18h", "Hours"] = 18
	crime_train.loc[crime_train["Hours"] == "19h", "Hours"] = 19
	crime_train.loc[crime_train["Hours"] == "20h", "Hours"] = 20
	crime_train.loc[crime_train["Hours"] == "21h", "Hours"] = 21
	crime_train.loc[crime_train["Hours"] == "22h", "Hours"] = 22
	crime_train.loc[crime_train["Hours"] == "23h", "Hours"] = 23

	print("Clean PdDistrict (Train)")
	crime_train["PdDistrict"] = crime_train["PdDistrict"].fillna("SOUTHERN")
	crime_train.loc[crime_train["PdDistrict"] == "SOUTHERN", "PdDistrict"] = 0
	crime_train.loc[crime_train["PdDistrict"] == "MISSION", "PdDistrict"] = 1
	crime_train.loc[crime_train["PdDistrict"] == "NORTHERN", "PdDistrict"] = 2
	crime_train.loc[crime_train["PdDistrict"] == "BAYVIEW", "PdDistrict"] = 3
	crime_train.loc[crime_train["PdDistrict"] == "CENTRAL", "PdDistrict"] = 4
	crime_train.loc[crime_train["PdDistrict"] == "TENDERLOIN", "PdDistrict"] = 5
	crime_train.loc[crime_train["PdDistrict"] == "INGLESIDE", "PdDistrict"] = 6
	crime_train.loc[crime_train["PdDistrict"] == "TARAVAL", "PdDistrict"] = 7
	crime_train.loc[crime_train["PdDistrict"] == "PARK", "PdDistrict"] = 8
	crime_train.loc[crime_train["PdDistrict"] == "RICHMOND", "PdDistrict"] = 9

	print("Clean DayOfWeek (Train)")
	crime_train["DayOfWeek"] = crime_train["DayOfWeek"].fillna("Friday")
	crime_train.loc[crime_train["DayOfWeek"] == "Friday", "DayOfWeek"] = 0
	crime_train.loc[crime_train["DayOfWeek"] == "Monday", "DayOfWeek"] = 1
	crime_train.loc[crime_train["DayOfWeek"] == "Saturday", "DayOfWeek"] = 2
	crime_train.loc[crime_train["DayOfWeek"] == "Sunday", "DayOfWeek"] = 3
	crime_train.loc[crime_train["DayOfWeek"] == "Thursday", "DayOfWeek"] = 4
	crime_train.loc[crime_train["DayOfWeek"] == "Tuesday", "DayOfWeek"] = 5
	crime_train.loc[crime_train["DayOfWeek"] == "Wednesday", "DayOfWeek"] = 6


	print("Clean Address (Train)")
	crime_train['StreetNo'] = crime_train['Address'].apply(lambda x: x.split(' ', 1)[0] if x.split(' ', 1)[0].isdigit() else 0)
	crime_train['Address'] = crime_train['Address'].apply(lambda x: x.split(' ', 1)[1] if x.split(' ', 1)[0].isdigit() else x)
	streets = list(enumerate(numpy.unique(crime_train["Address"])))
	streets_dict = { name : i for i, name in streets }
	crime_train["Address"] = crime_train["Address"].map( lambda x: streets_dict[x]).astype(int)


	print("Clean Longitude and Latitude (Train)")
	zonesTemp = []

	for X, Y in crime_train[['X','Y']].values:
		if X > -122.52 and X < -122.50:
			if Y > 37.70 and Y < 37.725:
				zonesTemp.append(1)
			elif Y >= 37.725 and Y < 37.750:
				zonesTemp.append(2)
			elif Y >= 37.750 and Y < 37.775:
				zonesTemp.append(3)
			elif Y >= 37.775 and Y < 37.80:
				zonesTemp.append(4)
			else:
				zonesTemp.append(5)
		elif X >= -122.50 and X < -122.48:
			if Y > 37.70 and Y < 37.725:
				zonesTemp.append(6)
			elif Y >= 37.725 and Y < 37.750:
				zonesTemp.append(7)
			elif Y >= 37.750 and Y < 37.775:
				zonesTemp.append(8)
			elif Y >= 37.775 and Y < 37.80:
				zonesTemp.append(9)
			else:
				zonesTemp.append(10)
		elif X >= -122.48 and X < -122.46:
			if Y > 37.70 and Y < 37.725:
				zonesTemp.append(11)
			elif Y >= 37.725 and Y < 37.750:
				zonesTemp.append(12)
			elif Y >= 37.750 and Y < 37.775:
				zonesTemp.append(13)
			elif Y >= 37.775 and Y < 37.80:
				zonesTemp.append(14)
			else:
				zonesTemp.append(15)
		elif X >= -122.46 and X < -122.44:
			if Y > 37.70 and Y < 37.725:
				zonesTemp.append(16)
			elif Y >= 37.725 and Y < 37.750:
				zonesTemp.append(17)
			elif Y >= 37.750 and Y < 37.775:
				zonesTemp.append(18)
			elif Y >= 37.775 and Y < 37.80:
				zonesTemp.append(19)
			else:
				zonesTemp.append(20)
		elif X >= -122.44 and X < -122.42:
			if Y > 37.70 and Y < 37.725:
				zonesTemp.append(21)
			elif Y >= 37.725 and Y < 37.750:
				zonesTemp.append(22)
			elif Y >= 37.750 and Y < 37.775:
				zonesTemp.append(23)
			elif Y >= 37.775 and Y < 37.80:
				zonesTemp.append(24)
			else:
				zonesTemp.append(25)
		elif X >= -122.42 and X < -122.40:
			if Y > 37.70 and Y < 37.725:
				zonesTemp.append(26)
			elif Y >= 37.725 and Y < 37.750:
				zonesTemp.append(27)
			elif Y >= 37.750 and Y < 37.775:
				zonesTemp.append(28)
			elif Y >= 37.775 and Y < 37.80:
				zonesTemp.append(29)
			else:
				zonesTemp.append(30)
		elif X >= -122.40 and X < -122.38:
			if Y > 37.70 and Y < 37.725:
				zonesTemp.append(31)
			elif Y >= 37.725 and Y < 37.750:
				zonesTemp.append(32)
			elif Y >= 37.750 and Y < 37.775:
				zonesTemp.append(33)
			elif Y >= 37.775 and Y < 37.80:
				zonesTemp.append(34)
			else:
				zonesTemp.append(35)
		else:
			if Y > 37.70 and Y < 37.725:
				zonesTemp.append(36)
			elif Y >= 37.725 and Y < 37.750:
				zonesTemp.append(37)
			elif Y >= 37.750 and Y < 37.775:
				zonesTemp.append(38)
			elif Y >= 37.775 and Y < 37.80:
				zonesTemp.append(39)
			else:
				zonesTemp.append(40)
	zonesFinal = numpy.array(zonesTemp)
	crime_train['Zone'] = zonesFinal

	############ Clean Data Test ############

	print("Clean Season (Test)")
	crime_test["Season"] = crime_test["Season"].fillna("Spring")
	crime_test.loc[crime_test["Season"] == "Winter", "Season"] = 0
	crime_test.loc[crime_test["Season"] == "Spring", "Season"] = 1
	crime_test.loc[crime_test["Season"] == "Summer", "Season"] = 2
	crime_test.loc[crime_test["Season"] == "Fall", "Season"] = 3

	print("Clean Hours (Test)")
	crime_test["Hours"] = crime_test["Hours"].fillna("12h")
	crime_test.loc[crime_test["Hours"] == "00h", "Hours"] = 0
	crime_test.loc[crime_test["Hours"] == "01h", "Hours"] = 1
	crime_test.loc[crime_test["Hours"] == "02h", "Hours"] = 2
	crime_test.loc[crime_test["Hours"] == "03h", "Hours"] = 3
	crime_test.loc[crime_test["Hours"] == "04h", "Hours"] = 4
	crime_test.loc[crime_test["Hours"] == "05h", "Hours"] = 5
	crime_test.loc[crime_test["Hours"] == "06h", "Hours"] = 6
	crime_test.loc[crime_test["Hours"] == "07h", "Hours"] = 7
	crime_test.loc[crime_test["Hours"] == "08h", "Hours"] = 8
	crime_test.loc[crime_test["Hours"] == "09h", "Hours"] = 9
	crime_test.loc[crime_test["Hours"] == "10h", "Hours"] = 10
	crime_test.loc[crime_test["Hours"] == "11h", "Hours"] = 11
	crime_test.loc[crime_test["Hours"] == "12h", "Hours"] = 12
	crime_test.loc[crime_test["Hours"] == "13h", "Hours"] = 13
	crime_test.loc[crime_test["Hours"] == "14h", "Hours"] = 14
	crime_test.loc[crime_test["Hours"] == "15h", "Hours"] = 15
	crime_test.loc[crime_test["Hours"] == "16h", "Hours"] = 16
	crime_test.loc[crime_test["Hours"] == "17h", "Hours"] = 17
	crime_test.loc[crime_test["Hours"] == "18h", "Hours"] = 18
	crime_test.loc[crime_test["Hours"] == "19h", "Hours"] = 19
	crime_test.loc[crime_test["Hours"] == "20h", "Hours"] = 20
	crime_test.loc[crime_test["Hours"] == "21h", "Hours"] = 21
	crime_test.loc[crime_test["Hours"] == "22h", "Hours"] = 22
	crime_test.loc[crime_test["Hours"] == "23h", "Hours"] = 23

	print("Clean PdDistrict (Test)")
	crime_test["PdDistrict"] = crime_test["PdDistrict"].fillna("SOUTHERN")
	crime_test.loc[crime_test["PdDistrict"] == "SOUTHERN", "PdDistrict"] = 0
	crime_test.loc[crime_test["PdDistrict"] == "MISSION", "PdDistrict"] = 1
	crime_test.loc[crime_test["PdDistrict"] == "NORTHERN", "PdDistrict"] = 2
	crime_test.loc[crime_test["PdDistrict"] == "BAYVIEW", "PdDistrict"] = 3
	crime_test.loc[crime_test["PdDistrict"] == "CENTRAL", "PdDistrict"] = 4
	crime_test.loc[crime_test["PdDistrict"] == "TENDERLOIN", "PdDistrict"] = 5
	crime_test.loc[crime_test["PdDistrict"] == "INGLESIDE", "PdDistrict"] = 6
	crime_test.loc[crime_test["PdDistrict"] == "TARAVAL", "PdDistrict"] = 7
	crime_test.loc[crime_test["PdDistrict"] == "PARK", "PdDistrict"] = 8
	crime_test.loc[crime_test["PdDistrict"] == "RICHMOND", "PdDistrict"] = 9

	print("Clean DayOfWeek (Test)")
	crime_test["DayOfWeek"] = crime_test["DayOfWeek"].fillna("Friday")
	crime_test.loc[crime_test["DayOfWeek"] == "Friday", "DayOfWeek"] = 0
	crime_test.loc[crime_test["DayOfWeek"] == "Monday", "DayOfWeek"] = 1
	crime_test.loc[crime_test["DayOfWeek"] == "Saturday", "DayOfWeek"] = 2
	crime_test.loc[crime_test["DayOfWeek"] == "Sunday", "DayOfWeek"] = 3
	crime_test.loc[crime_test["DayOfWeek"] == "Thursday", "DayOfWeek"] = 4
	crime_test.loc[crime_test["DayOfWeek"] == "Tuesday", "DayOfWeek"] = 5
	crime_test.loc[crime_test["DayOfWeek"] == "Wednesday", "DayOfWeek"] = 6

	print("Clean Address (Test)")
	crime_test['StreetNo'] = crime_test['Address'].apply(lambda x: x.split(' ', 1)[0] if x.split(' ', 1)[0].isdigit() else 0)
	crime_test['Address'] = crime_test['Address'].apply(lambda x: x.split(' ', 1)[1] if x.split(' ', 1)[0].isdigit() else x)
	streets = list(enumerate(numpy.unique(crime_test["Address"])))
	streets_dict = { name : i for i, name in streets }
	crime_test["Address"] = crime_test["Address"].map( lambda x: streets_dict[x]).astype(int)

	print("Clean Longitude and Latitude (Test)")
	zonesTempTest = []

	for X, Y in crime_test[['X','Y']].values:
		if X > -122.52 and X < -122.50:
			if Y > 37.70 and Y < 37.725:
				zonesTempTest.append(1)
			elif Y >= 37.725 and Y < 37.750:
				zonesTempTest.append(2)
			elif Y >= 37.750 and Y < 37.775:
				zonesTempTest.append(3)
			elif Y >= 37.775 and Y < 37.80:
				zonesTempTest.append(4)
			else:
				zonesTempTest.append(5)
		elif X >= -122.50 and X < -122.48:
			if Y > 37.70 and Y < 37.725:
				zonesTempTest.append(6)
			elif Y >= 37.725 and Y < 37.750:
				zonesTempTest.append(7)
			elif Y >= 37.750 and Y < 37.775:
				zonesTempTest.append(8)
			elif Y >= 37.775 and Y < 37.80:
				zonesTempTest.append(9)
			else:
				zonesTempTest.append(10)
		elif X >= -122.48 and X < -122.46:
			if Y > 37.70 and Y < 37.725:
				zonesTempTest.append(11)
			elif Y >= 37.725 and Y < 37.750:
				zonesTempTest.append(12)
			elif Y >= 37.750 and Y < 37.775:
				zonesTempTest.append(13)
			elif Y >= 37.775 and Y < 37.80:
				zonesTempTest.append(14)
			else:
				zonesTempTest.append(15)
		elif X >= -122.46 and X < -122.44:
			if Y > 37.70 and Y < 37.725:
				zonesTempTest.append(16)
			elif Y >= 37.725 and Y < 37.750:
				zonesTempTest.append(17)
			elif Y >= 37.750 and Y < 37.775:
				zonesTempTest.append(18)
			elif Y >= 37.775 and Y < 37.80:
				zonesTempTest.append(19)
			else:
				zonesTempTest.append(20)
		elif X >= -122.44 and X < -122.42:
			if Y > 37.70 and Y < 37.725:
				zonesTempTest.append(21)
			elif Y >= 37.725 and Y < 37.750:
				zonesTempTest.append(22)
			elif Y >= 37.750 and Y < 37.775:
				zonesTempTest.append(23)
			elif Y >= 37.775 and Y < 37.80:
				zonesTempTest.append(24)
			else:
				zonesTempTest.append(25)
		elif X >= -122.42 and X < -122.40:
			if Y > 37.70 and Y < 37.725:
				zonesTempTest.append(26)
			elif Y >= 37.725 and Y < 37.750:
				zonesTempTest.append(27)
			elif Y >= 37.750 and Y < 37.775:
				zonesTempTest.append(28)
			elif Y >= 37.775 and Y < 37.80:
				zonesTempTest.append(29)
			else:
				zonesTempTest.append(30)
		elif X >= -122.40 and X < -122.38:
			if Y > 37.70 and Y < 37.725:
				zonesTempTest.append(31)
			elif Y >= 37.725 and Y < 37.750:
				zonesTempTest.append(32)
			elif Y >= 37.750 and Y < 37.775:
				zonesTempTest.append(33)
			elif Y >= 37.775 and Y < 37.80:
				zonesTempTest.append(34)
			else:
				zonesTempTest.append(35)
		else:
			if Y > 37.70 and Y < 37.725:
				zonesTempTest.append(36)
			elif Y >= 37.725 and Y < 37.750:
				zonesTempTest.append(37)
			elif Y >= 37.750 and Y < 37.775:
				zonesTempTest.append(38)
			elif Y >= 37.775 and Y < 37.80:
				zonesTempTest.append(39)
			else:
				zonesTempTest.append(40)
	zonesFinalTest = numpy.array(zonesTempTest)
	crime_test['Zone'] = zonesFinalTest
	
	############ TRAIN PREDICT ############

	predictors = ["PdDistrict","Zone"]
	algorithms = [
		[RandomForestClassifier(max_depth=16,n_estimators=128), ["PdDistrict", "Zone", "Hours", "DayOfWeek", "Address", "StreetNo", "TypeStreet"]],
    	[GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3),["PdDistrict", "Zone", "Hours", "DayOfWeek", "Address", "Month", "StreetNo", "TypeStreet"]],
    	[xgboost.XGBClassifier(max_depth=8,n_estimators=64), ["PdDistrict", "Zone", "Hours", "DayOfWeek", "Address", "Month", "StreetNo", "TypeStreet"]]
	 ]

	full_predictions = []
	for alg, predictors in algorithms:
		print("Train en cours ",alg.__class__.__name__ , " Model ")
		alg.fit(numpy.array(crime_train[list(predictors)]), crime_train["Category"].values.ravel())
		print("Predict en cours ",alg.__class__.__name__ , " Model ")
		predictionCrimeSF = alg.predict_proba(numpy.array(crime_test[list(predictors)]))
		full_predictions.append(predictionCrimeSF)
	print(full_predictions)
	predictionCrimeSF = numpy.around((full_predictions[0] * 2 + full_predictions[1] + full_predictions[2] * 4) / 7,decimals=4)
	print(predictionCrimeSF)

	############ Construction CSV ############
	
	print("Construction CSV")
	csvInter = crime_test["Id"]
	predictions_file = open("prediction_v33.csv", "wb")
	open_file_object = csv.writer(predictions_file)
	open_file_object.writerow(["Id","ARSON","ASSAULT","BAD CHECKS","BRIBERY","BURGLARY","DISORDERLY CONDUCT","DRIVING UNDER THE INFLUENCE","DRUG/NARCOTIC","DRUNKENNESS","EMBEZZLEMENT","EXTORTION","FAMILY OFFENSES","FORGERY/COUNTERFEITING","FRAUD","GAMBLING","KIDNAPPING","LARCENY/THEFT","LIQUOR LAWS","LOITERING","MISSING PERSON","NON-CRIMINAL","OTHER OFFENSES","PORNOGRAPHY/OBSCENE MAT","PROSTITUTION","RECOVERED VEHICLE","ROBBERY","RUNAWAY","SECONDARY CODES","SEX OFFENSES FORCIBLE","SEX OFFENSES NON FORCIBLE","STOLEN PROPERTY","SUICIDE","SUSPICIOUS OCC","TREA","TRESPASS","VANDALISM","VEHICLE THEFT","WARRANTS","WEAPON LAWS"])
	open_file_object.writerows(zip(csvInter, numpy.around(predictionCrimeSF[:,0],decimals=4),numpy.around(predictionCrimeSF[:,1],decimals=4), numpy.around(predictionCrimeSF[:,2],decimals=4), numpy.around(predictionCrimeSF[:,3],decimals=4), numpy.around(predictionCrimeSF[:,4],decimals=4), numpy.around(predictionCrimeSF[:,5],decimals=4), numpy.around(predictionCrimeSF[:,6],decimals=4), numpy.around(predictionCrimeSF[:,7],decimals=4), numpy.around(predictionCrimeSF[:,8],decimals=4), numpy.around(predictionCrimeSF[:,9],decimals=4), numpy.around(predictionCrimeSF[:,10],decimals=4), numpy.around(predictionCrimeSF[:,11],decimals=4), numpy.around(predictionCrimeSF[:,12],decimals=4), numpy.around(predictionCrimeSF[:,13],decimals=4), numpy.around(predictionCrimeSF[:,14],decimals=4), numpy.around(predictionCrimeSF[:,15],decimals=4), numpy.around(predictionCrimeSF[:,16],decimals=4), numpy.around(predictionCrimeSF[:,17],decimals=4), numpy.around(predictionCrimeSF[:,18],decimals=4), numpy.around(predictionCrimeSF[:,19],decimals=4), numpy.around(predictionCrimeSF[:,20],decimals=4), numpy.around(predictionCrimeSF[:,21],decimals=4), numpy.around(predictionCrimeSF[:,22],decimals=4), numpy.around(predictionCrimeSF[:,23],decimals=4), numpy.around(predictionCrimeSF[:,24],decimals=4), numpy.around(predictionCrimeSF[:,25],decimals=4), numpy.around(predictionCrimeSF[:,26],decimals=4), numpy.around(predictionCrimeSF[:,27],decimals=4), numpy.around(predictionCrimeSF[:,28],decimals=4), numpy.around(predictionCrimeSF[:,29],decimals=4), numpy.around(predictionCrimeSF[:,30],decimals=4), numpy.around(predictionCrimeSF[:,31],decimals=4), numpy.around(predictionCrimeSF[:,32],decimals=4), numpy.around(predictionCrimeSF[:,33],decimals=4), numpy.around(predictionCrimeSF[:,34],decimals=4), numpy.around(predictionCrimeSF[:,35],decimals=4), numpy.around(predictionCrimeSF[:,36],decimals=4), numpy.around(predictionCrimeSF[:,37],decimals=4), numpy.around(predictionCrimeSF[:,38],decimals=4)))
	predictions_file.close()
	#"""




