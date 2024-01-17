import configparser
import logging
import sys
import os

config = configparser.ConfigParser()
configFile = "config/config.ini"

def initconfig():
    if os.path.isfile(configFile):
        pass
    else:
        create()
    #Konfigdatei initialisieren
    try:
        #Config Datei auslesen
        config.read(configFile)
        conf = config['DEFAULT']
        return conf
    except:
        print("Error while loading the Config file:" + str(sys.exc_info()))
        logging.error("Error while loading the Config file" + str(sys.exc_info()))

def create():
    
    config['DEFAULT'] = {'API_KEY' : '',
                        'Secret' : '',
                        'Database' : True,
                        'telegramtoken' : '',
                        'activeLending' : True,
                        }

    with open(configFile, 'w') as configfile:
        config.write(configfile)

def setactiveLendingTrue():
    config.set('DEFAULT','activeLending','True')
    print("Lending On")
    with open('config/config.ini', 'w') as configfile:
        config.write(configfile)

def setactiveLendingFalse():
    config.set('DEFAULT','activeLending','False')
    print("Lending Off")
    with open('config/config.ini', 'w') as configfile:
        config.write(configfile)