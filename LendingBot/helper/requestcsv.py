import requests
import time
import csv
import os


def request(CurrencyPair,Period):
    now = time.time()
    _1month = 2678400
    now_1month = now - _1month
    for x in range(72):
        url = "https://poloniex.com/public?command=returnChartData&currencyPair="+ CurrencyPair + "&start=" + str(now_1month) + "&end=" + str(now) + "&period=" + Period + ""
        resp = requests.get(url=url)
        data = resp.json()

        csv_path = './Data/' + CurrencyPair + '_Poloniex_' + str(x) + '_' + Period + '.csv'
        with open(csv_path, 'w', newline='') as f:
            dict_writer = csv.DictWriter(f, data[0].keys())
            dict_writer.writeheader()
            dict_writer.writerows(data)
            print("CSV file saved at path {}".format(csv_path))
            f.close()

        now = now - _1month
        now_1month = now - _1month
    time.sleep(5)

    alle = []
    for y in range(72):
        csv_path = './Data/' + CurrencyPair + '_Poloniex_' + str(y) + '_' + Period + '.csv'
        with open(csv_path, newline='') as f:
            dict_reader = csv.DictReader(f)
            for row in dict_reader:
                alle.append(row)
        os.remove(csv_path)

    csv_path = './Data/' + CurrencyPair + '_Poloniex_all_' + Period + '.csv'
    with open(csv_path, 'w', newline='') as f:
            dict_writer = csv.DictWriter(f, alle[0].keys())
            dict_writer.writeheader()
            dict_writer.writerows(alle)
            print("CSV file saved at path {}".format(csv_path))