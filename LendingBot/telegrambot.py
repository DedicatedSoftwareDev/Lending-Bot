#!/usr/bin/env python
# pylint: disable=C0116,W0613
# This program is dedicated to the public domain under the CC0 license.

"""
Simple Bot to reply to Telegram messages.

First, a few handler functions are defined. Then, those functions are passed to
the Dispatcher and registered at their respective places.
Then, the bot is started and runs until we press Ctrl-C on the command line.

Usage:
Basic Echobot example, repeats messages.
Press Ctrl-C on the command line or send a signal to the process to stop the
bot.
"""

import logging
import helper.config
import CLASS.POLONIEXAPI
from datetime import datetime,timedelta

from telegram import Update, ForceReply, ReplyKeyboardMarkup
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

conf = helper.config.initconfig()

Poloniexapi = CLASS.POLONIEXAPI.POLONIEXAPI()

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

logger = logging.getLogger(__name__)

reply_keyboard = [
    ['/status'],
    ['/help'],
    ['/Lend_on','/Lend_off']
]
#markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)
markup = ReplyKeyboardMarkup(reply_keyboard)

# Define a few command handlers. These usually take the two arguments update and
# context.
def start(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    update.message.reply_markdown_v2(
        fr'Hi {user.mention_markdown_v2()}\!',
        reply_markup=markup,
    )

def status_command(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /help is issued."""
    Poloniexapi.request()
    AvailableBalancedata = Poloniexapi.availableAccountBalances
    ActiveLoans = Poloniexapi.activeLoans
    OpenLoans = Poloniexapi.OpenLoanOffers
    Text = ""
    if 'lending' in AvailableBalancedata:
        for key in AvailableBalancedata["lending"]:
            if float(AvailableBalancedata["lending"][key]) > 0.0:
                Text = Text + ("You have " + AvailableBalancedata["lending"][key] + " " + str(key) + " available\n")
    else:
        Text = Text +("You have nothing available\n")
    if 'lending' in AvailableBalancedata:
        for key in AvailableBalancedata["lending"]:
            if float(AvailableBalancedata["lending"][key]) > 0.0:
                Text = Text + ("You have " + AvailableBalancedata["lending"][key] + " " + str(key) + " available")
    if ActiveLoans != []:
        Text = Text + "Active Loans:\n"
        for key in ActiveLoans["provided"]:
            present = datetime.utcnow()
            future  = datetime.strptime(key["date"], '%Y-%m-%d %H:%M:%S') + timedelta(days = key['duration'])
            difference = future - present
            Text = Text + ("Currency: " + key["currency"] + " Rate: " + key["rate"] + "% Amount: " + key ["amount"] + " Time to End: " + str(difference)) + "\n"
    if OpenLoans != []:
        Text = Text + "Open Loans:\n"
        for key in OpenLoans["BTC"]:
            present = datetime.utcnow()
            future  = datetime.strptime(key["date"], '%Y-%m-%d %H:%M:%S') + timedelta(days = key['duration'])
            difference = future - present
            Text = Text + ("Rate: " + key["rate"] + "% Amount: " + key ["amount"]) + "\n"

    Text = Text + (str(Poloniexapi.completeBalancesSum()) + " BTC")
    update.message.reply_text(Text,reply_markup=markup)

def lend_on_command(update: Update, context: CallbackContext) -> None:
    helper.config.setactiveLendingTrue()
    update.message.reply_text('Lending On',reply_markup=markup)

def lend_off_command(update: Update, context: CallbackContext) -> None:
    helper.config.setactiveLendingFalse()
    update.message.reply_text('Lending Off',reply_markup=markup)

def help_command(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /help is issued."""
    update.message.reply_text('/status\n/help',reply_markup=markup)


def echo(update: Update, context: CallbackContext) -> None:
    """Echo the user message."""
    #update.message.reply_text(update.message.text)
    update.message.reply_text("unknown command")


def main() -> None:
    """Start the bot."""
    # Create the Updater and pass it your bot's token.
    updater = Updater(conf["telegramtoken"])

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # on different commands - answer in Telegram
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_command))
    dispatcher.add_handler(CommandHandler("status", status_command))
    dispatcher.add_handler(CommandHandler("Lend_on", lend_on_command))
    dispatcher.add_handler(CommandHandler("Lend_off", lend_off_command))

    # on non command i.e message - echo the message on Telegram
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, help_command))

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    main()
