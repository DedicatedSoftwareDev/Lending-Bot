import helper.config
from poloniex import Poloniex

import statistics

class POLONIEXAPI():

    def request(self):
        conf = helper.config.initconfig()

        self.polo = Poloniex()
        self.polo.key = conf["api_key"]
        self.polo.secret = conf["secret"]
        self.ticker = self.polo('returnTicker')
        self.completeBalances = self.polo.returnCompleteBalances()
        self.availableAccountBalances = self.polo.returnAvailableAccountBalances()
        self.activeLoans = self.polo.returnActiveLoans()
        self.LendingHistory = self.polo.returnLendingHistory() #TODO Test
        self.OpenLoanOffers = self.polo.returnOpenLoanOffers() #TODO Test



    def returnLoanOrders(self,currency):
        return self.polo.returnLoanOrders(currency=currency)

    def avgLoanOrders(self,currency):
        data = self.returnLoanOrders(currency)
        dicti = []
        for key in data['offers']:
            dicti.append(float(key['rate']))
        return statistics.mean(dicti)

    def createLoanOffer(self,amount, price):
        r = self.polo.createLoanOffer("BTC", amount, price, autoRenew=0)

    def completeBalancesSum(self):
        sum = 0.0
        for btcs in self.completeBalances:
            sum = sum + float(self.completeBalances[btcs]['btcValue'])
            self.completeBalances
        return sum
