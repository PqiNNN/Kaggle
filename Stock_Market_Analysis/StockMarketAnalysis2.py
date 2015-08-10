import pandas
import numpy
import csv
from ggplot import *
import datetime

if __name__ == '__main__':

	print("Loading data...")
	stock_train = pandas.read_csv("Stock_Market_July_2015.csv")

	#Cleaning data
	print("Cleaning data...")
	stock_train.columns = ['Ticker','Date','Open','High','Low','Close','Volume']
	stock_train_MMM = stock_train[stock_train.Ticker == 'ACE']
	print(stock_train_MMM.head(5))
	stock_train_MMM = stock_train_MMM[['Date','Close']]
	print(stock_train_MMM.head(5))
	liste_dates = []
	for element in stock_train_MMM['Date']:
		liste_dates.append(datetime.datetime.strptime(element, '%d-%b-%Y'))
	stock_train_MMM['Date'] = liste_dates

	#Ploting data
	print ggplot(stock_train_MMM, aes(x='Date',y='Close')) + geom_line() + scale_x_date(labels='%b %d')