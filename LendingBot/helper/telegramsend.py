import telegram
import helper.config

conf = helper.config.initconfig()

bot = telegram.Bot(token=conf['telegramtoken'])

def send(text):
    bot.send_message(353575753, text=text)