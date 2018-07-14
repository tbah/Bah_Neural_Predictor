#Thierno Bah
#Stock A! predictor
#


import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt 
import datetime
from pandas_datareader import data
import argparse


#plt.switch_backend('new_backend')

dates = []
prices = []

period = 0.5
option = 'close_price'
filename = None
data_source = 'robinhood'
nDaysTopredict = 5

def get_stoctData(sticker, nMonth,opt):
	ndays = 20 * nMonth
	ndays = int(ndays)
	f = data.DataReader(sticker, data_source)
	d = 0
	for i in f.tail(ndays)[opt]:	
		dates.append(d)
		prices.append(float(i))	
		d = d + 1
def get_option(op):
	if op[0] == 'O' or op[0] == 'o':
		return 'open_price'
	elif op[0] == 'c' or op[0] == 'C':
		return 'close_price'
	elif op[0] == 'v' or op[0] == 'V':
		return 'volume'

def parse_args():
    parser = argparse.ArgumentParser(
        description="Grab command line arguments sticker symbol, number of months, and open/close price or volume, or a csv file")
    parser.add_argument('-s', help="sticker symbol")
    parser.add_argument('-n', type=float, help="number of months or a fraction i.e 1.5 = 1 and 1/2 months")
    parser.add_argument('-o', help="options: v for volume, o for open price and c for close price")
    parser.add_argument('-f', help="csv file (exact path required)")
    args = parser.parse_args()
    global period, option, filename
    if args.s == None and args.f == None:
        print("No sticker symbol or file name given, exiting. use -h or --help for help")
        exit()
    if args.n == None:
        print("No time period given, default will be used.")
    else:
    	period = args.n
    if args.o == None:
        print("No option given, default will be used.")
    else:
    	option = get_option(args.o)
    if args.f != None:
    	filename = args.f

    return args.s#, args.n, args.o, args.f
def get_data(filename):
	with open(filename, 'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
		next(csvFileReader)
		i = 0
		for row in csvFileReader:
			darr = row[0].split('-')
			dates.append(i)
			prices.append(float(row[1]))
			i = i + 1
	return
def predict_prices(dates, prices, x, name, opt):
	global nDaysTopredict
	print('reshape...')
	dates = np.reshape(dates, (len(dates), 1))
	svr_lin = SVR(kernel= 'linear', C=1e3)
	#svr_poly = SVR(kernel= 'poly', C=1e3, degree = 2)
	svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
	print('linear fit...')
	svr_lin.fit(dates, prices)
	print('Polynomial fit...')
	#svr_poly.fit(dates, prices)
	print('rbf fit...')
	svr_rbf.fit(dates, prices)
	print('plot')
	plt.scatter(dates, prices, color='black', label='Data')
	plt.plot(dates, svr_rbf.predict(dates), color='red', label='RBF model')
	plt.plot(dates, svr_lin.predict(dates), color='green', label='Linear model')
	#plt.plot(dates, svr_poly.predict(dates), color='blue', label="Polynomial model")
	print('predict..')
	predW = [None]*nDaysTopredict
	pred = svr_rbf.predict(x)[0], svr_lin.predict(x)[0], #svr_poly.predict(x)[0]
	for i in range(1,nDaysTopredict+1):
		predW[i-1] = (svr_rbf.predict(x+i)[0], svr_lin.predict(x+i)[0])#, svr_poly.predict(x+i)[0])
	plt.xlabel('Date')

	plt.scatter(x, pred[0], color='red', label='RBF prediction')
	plt.scatter(x, pred[1], color='green', label='LM prediction')
	#plt.scatter(x, pred[2], color='blue', label='PM prediction')
	for i in range(1,nDaysTopredict+1):
		plt.scatter(x+i, predW[i-1][0], color='red')
		plt.scatter(x+i, predW[i-1][1], color='green')
		#plt.scatter(x+i, predW[i-1][2], color='blue')
	plt.ylabel('Price')
	plt.title('Support Vector Rgression '+name+'('+opt+')')
	plt.legend()
	plt.show()

	return pred

name = parse_args()
opti = "open_price"
if(name == None):
	name = filename.split('.')[0]
	get_data(filename)
else:
	opti = option
	if(period > 2):
		nDaysTopredict = int(nDaysTopredict + period)
	get_stoctData(name, period, option)

#fileN = input("Enter Sticker and number of months: ")
#name, num = fileN.split(' ')

predicted_price = predict_prices(dates, prices, len(dates), name, opti)
print(predicted_price)


