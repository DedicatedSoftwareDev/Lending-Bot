import CLASS.POLONIEXAPI
import helper.config
import CLASS.DATABASE
from datetime import datetime,timedelta
import os
import time

Poloniexapi = CLASS.POLONIEXAPI.POLONIEXAPI()

Database = CLASS.DATABASE.DATABASE()
Database.connect()
Database.setup()

def cls():
    os.system('cls' if os.name=='nt' else 'clear')

def printscreen():
    Tickerdata = Poloniexapi.ticker
    AvailableBalancedata = Poloniexapi.availableAccountBalances
    ActiveLoans = Poloniexapi.activeLoans
    OpenLoans = Poloniexapi.OpenLoanOffers
    #CompleteBalance = Poloniexapi.completeBalances
    cls()
    print("LENDING BOT")
    print("USDT_BTC: " + Tickerdata["USDT_BTC"]["last"])
    print("")
    if 'lending' in AvailableBalancedata:
        for key in AvailableBalancedata["lending"]:
            if float(AvailableBalancedata["lending"][key]) > 0.0:
                print("You have " + AvailableBalancedata["lending"][key] + " " + str(key) + " available")
    else:
        print("You have nothing available")
    print("")
    if ActiveLoans != []:
        print("Active Loans")
        for key in ActiveLoans["provided"]:
            present = datetime.utcnow()
            future  = datetime.strptime(key["date"], '%Y-%m-%d %H:%M:%S') + timedelta(days = key['duration'])
            difference = future - present
            print("Currency: " + key["currency"] + " Rate: " + key["rate"] + "% Amount: " + key ["amount"] + " Time to End: " + str(difference))
    print("")
    if OpenLoans != []:
        print("Open Loans")
        for key in OpenLoans["BTC"]:
            present = datetime.utcnow()
            future  = datetime.strptime(key["date"], '%Y-%m-%d %H:%M:%S') + timedelta(days = key['duration'])
            difference = future - present
            print("Rate: " + key["rate"] + "% Amount: " + key ["amount"])
    print("")
    print(str(Poloniexapi.completeBalancesSum()) + " BTC")


def check():
    AvailableBalancedata = Poloniexapi.availableAccountBalances
    ActiveLoans = Poloniexapi.activeLoans
    HistoryLoans = Poloniexapi.LendingHistory
    if 'lending' in AvailableBalancedata:
        if float(AvailableBalancedata['lending']['BTC']) > 0.005:
            avg = Poloniexapi.avgLoanOrders('BTC')
            completeammount = AvailableBalancedata['lending']['BTC']
            print("Genug BTC zum Lending")
            print("{:.8f}".format(avg))
            print(completeammount)
            if conf['activeLending'] == "True":
                Poloniexapi.createLoanOffer(completeammount,avg)
    if conf['Database'] == "True":
        for key in ActiveLoans["provided"]:
            Database.insertActiveLoan(key)
        Database.insertHistoryLoan(HistoryLoans)
        Database.insertTicker(Poloniexapi.ticker)
        Database.insertBTC(Poloniexapi.ticker["USDT_BTC"]["last"])
    
    
    
while True:
    conf = helper.config.initconfig()
    Poloniexapi.request()
    printscreen()
    check()

    time.sleep(10)