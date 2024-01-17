from pyArango.connection import Connection
import sys
import datetime
import time
import helper.telegramsend

class DATABASE():

    def connect(self):
        try:
            self.conn = Connection(arangoURL="http://127.0.0.1:8529", username="root", password="openSesame")
        except ZeroDivisionError:
            print("ZeroficisionError")
        except TypeError:
            errortext = "TypeError: Connection to Database failed\n " + str(sys.exc_info())
            print(errortext)

    def setup(self):
        if not self.conn.hasDatabase(name='Lending'):
            db = self.conn.createDatabase(name='Lending')
        db = self.conn['Lending']

        if not db.hasCollection(name='Ticker'):
            db.createCollection(name='Ticker')

        if not db.hasCollection(name='BTC'):
            db.createCollection(name='BTC')

        if not db.hasCollection(name='ActiveLoans'):
            db.createCollection(name='ActiveLoans')

        if not db.hasCollection(name='HistoryLoans'):
            db.createCollection(name='HistoryLoans')

    def insertTicker(self,data):
        db = self.conn['Lending']
        tickerCollection = db['Ticker']
        tickerDoc = tickerCollection.createDocument()
        for keys in data:
            tickerDoc[keys] = data[keys]
        tickerDoc['TIMESTAMP'] = datetime.datetime.strftime(datetime.datetime.now(),"%Y-%m-%d %H:%M:%S")
        tickerDoc.save()

    def insertBTC(self,BTC):
        db = self.conn['Lending']
        BTCCollection = db['BTC']
        BTCDoc = BTCCollection.createDocument()
        BTCDoc['value'] = BTC
        #BTCDoc['TIMESTAMP'] = datetime.datetime.strftime(datetime.datetime.now(),"%Y-%m-%d %H:%M:%S")
        BTCDoc['date'] = time.time()
        BTCDoc.save()

    def insertActiveLoan(self,data):
        db = self.conn['Lending']
        aql = "FOR x IN ActiveLoans FILTER x.id == " + str(data['id']) + " RETURN x"
        queryResult = db.AQLQuery(aql, rawResults=True)
        if len(queryResult) == 0:
            Text = "New Loan: \nID: " + str(data['id']) + "\nCurrency: " + data['currency'] + "\nRate: " + str(data['rate']) + "\nAmount: " + str(data['amount'])  + "\nDuration: " + str(data['duration'])
            helper.telegramsend.send(Text)
            ActiveLoanCollection = db['ActiveLoans']
            ActiveLoanDoc = ActiveLoanCollection.createDocument()
            for keys in data:
                ActiveLoanDoc[keys] = data[keys]
            ActiveLoanDoc.save()

    def insertHistoryLoan(self,data):
        db = self.conn['Lending']
        for datakey in data:
            aql = "FOR x IN HistoryLoans FILTER x.id == " + str(datakey['id']) + " RETURN x"
            queryResult = db.AQLQuery(aql, rawResults=True)
            if len(queryResult) == 0:
                Text = "New History Loan: \nID: " + str(datakey['id']) + "\nCurrency: " + datakey['currency'] + "\nRate: " + str(datakey['rate']) + "\nAmount: " + str(datakey['amount']) + "\nDuration: " + str(datakey['duration']) +  "\nEarned: " + str(datakey['earned'])
                helper.telegramsend.send(Text)
                HistoryLoanCollection = db['HistoryLoans']
                HistoryLoanDoc = HistoryLoanCollection.createDocument()
                for keys in datakey:
                    HistoryLoanDoc[keys] = datakey[keys]
                HistoryLoanDoc.save()
