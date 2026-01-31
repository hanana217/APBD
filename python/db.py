import mysql.connector

def get_conn():
    return mysql.connector.connect(
        host="localhost",
        port=3308,
        user="apbd_user",
        password="apbd_pass",
        database="pos"
    )
