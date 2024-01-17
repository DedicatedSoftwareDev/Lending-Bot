import mysql.connector
import sys

class MYSQL():
    def getDataBaseConnection(self):
        try:
            #Datenbankverbindung herstellen
            self.mydb = mysql.connector.connect(
                host="localhost",
                user="root",
                password="password",
                database="Lending",
                auth_plugin='mysql_native_password'
                )
        except:
            print("Unexpected error Datenbankverbindung functions.py:" + str(sys.exc_info()))

            