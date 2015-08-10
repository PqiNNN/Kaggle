import pandas
import numpy
import csv
from ggplot import *
import datetime

if __name__ == '__main__':

	print("Loading data...")
	stock_train = pandas.read_csv("./csv_files/Stock_Market_July_2015.csv")

	#Cleaning data
	print("Cleaning data...")
	stock_train.columns = ['Ticker','Date','Open','High','Low','Close','Volume']
	stock_train_spec = stock_train[stock_train.Ticker == 'ACE']
	stock_train_spec = stock_train_spec[['Date','Close','Open']]
	liste_dates = []
	for element in stock_train_spec['Date']:
		liste_dates.append(datetime.datetime.strptime(element, '%d-%b-%Y'))
	stock_train_spec['Date'] = liste_dates
	#liste_IDs = [x+1 for x in range(len(stock_train_spec))]

	#Bootstraping example
	# liste_modeles = []
	# for x in range(0, 100):
	# 	boot_sample = stock_train_spec.sample(n = len(stock_train_spec), replace = True)
	# 	#modele(boot_sample), predict(modele)
	# 	#liste_modeles.append(result_predictions)
	# for element in liste_modeles:
	# 	#average results




	#Ploting data
	x = ggplot(stock_train_spec, aes(x='Date',y='Close')) + geom_line() + scale_x_date(labels='%b %d')
	y = ggplot(stock_train_spec, aes(x='Date')) + geom_line(aes(y = 'Close', colour = "b")) + geom_line(aes(y = 'Open', colour = "r")) + scale_x_date(labels='%b %d')
	ggsave(x,'./Plots/CloseAnalysisJuly_ACE.png')
	ggsave(y,'./Plots/OpenCloseAnalysisJuly_ACE.png')