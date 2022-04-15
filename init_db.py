import sqlite3
import pandas as pd
import yfinance as yf

tickers = pd.read_csv("russell3000.csv")['Ticker'].tolist()

print("Finished reading tickers")

connection = sqlite3.connect('database.db')

with open('schema.sql') as f:
    connection.executescript(f.read())
    
cur = connection.cursor()
   
for tick in tickers:
    data = yf.download(tickers = tick, period = '10y', interval = '1mo', group_by = 'ticker')
    for row in data.itertuples(index=True, name='Pandas'):
        if(not pd.isna(row.Open)):
            cur.execute("INSERT INTO stocks ([Date], ticker, [Close], High, Low, [Open], Volume) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (row.Index.strftime('%Y-%m-%d'), tick, row.Close, row.High, row.Low, row.Open, row.Volume)
                        )
    print("Loaded " + tick)

connection.commit()
connection.close()
