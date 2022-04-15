import sqlite3
import numpy as np
import pandas as pd
import scipy.stats as stats
from datetime import datetime
from flask import Flask, render_template, request, url_for, flash, redirect
from werkzeug.exceptions import abort

def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

app = Flask(__name__)
app.config['SECRET_KEY'] = 'edwin'
    
def clean_data(stockdata, minmonths = 0):     
    # filter out assets with less observations than minmonths
    # data includes current date, remove to avoid skewed data
    recent = stockdata['Date'].iloc[-1] - pd.DateOffset(day = 1)
    stockdata = stockdata.loc[stockdata['Date'] <= recent].copy()
    return stockdata
        
def compute_weights(stockdata):  
    #start = kwargs.get('start', min(self.stockdata.index))
    #end = kwargs.get('end', max(self.stockdata.index))
    start = min(stockdata['Date'])
    end = max(stockdata['Date'])
    returns, mktreturn, labels = compute_returns(stockdata)
    sim = reg_params(returns, mktreturn, labels)

    sim_cutoff = cut(sim, mktreturn)
    z = (sim_cutoff['beta'] / sim_cutoff['eps']) * (sim_cutoff['excess'] - sim_cutoff['C'])
    weights = z.sort_values(ascending = False) / z.sum()
    return weights

def compute_returns(stockdata):
    #start = kwargs.get('start', min(self.stockdata.index))
    #end = kwargs.get('end', max(self.stockdata.index))
    start = min(stockdata['Date'])
    end = max(stockdata['Date'])
    df = stockdata.loc[(stockdata['Date'] >= start) & (stockdata['Date'] <= end)]

    # pivot and drop columns with all NaNs, pandas ignores partial NaNs
    pivoted = df.pivot(index = 'Date', columns = 'ticker', values = 'Open').dropna(axis=1, how='all')
    # extract and drop market from data
    spprices = pivoted['^GSPC'].copy().to_numpy()
    pivoted = pivoted.drop(columns = ['^GSPC'])

    # convert to numpy for easier computing
    prices = pivoted.to_numpy()

    # compute percent returns by subtracting previous day from current and dividing by previous
    returnarr = (prices[1:, :] - prices[:(prices.shape[0] - 1), :]) / prices[:(prices.shape[0] - 1), :]
    spreturns = (spprices[1:] - spprices[:(len(spprices) - 1)]) / spprices[:(len(spprices) - 1)]
    
    return(returnarr, spreturns, pivoted.columns.values)


def reg_params(returns, mktreturn, labels):
    # compute alphas and betas by regression a stock's mean return on the market mean return
    alphas = np.zeros(returns.shape[1])
    betas = np.zeros(returns.shape[1])
    unsyserr = np.zeros(returns.shape[1])
    for i in np.arange(returns.shape[1]):
        treturn = returns[:,i]
        tnonan = treturn[np.logical_not(np.isnan(treturn))]
        mktmatch = mktreturn[(len(mktreturn) - len(tnonan)):]
        betas[i], alphas[i], r, p, se = stats.linregress(mktmatch, tnonan)
        unsyserr[i] = np.sum((tnonan - alphas[i] - betas[i]*mktmatch)**2) / (len(mktmatch) - 2)

    params = pd.DataFrame(data = {'alpha': alphas, 'beta': betas, 'eps': unsyserr, 'rmean': returns.mean(axis=0)}, index = labels)
    return(params)


def cut(sim_params, mktreturn):

    sim_params['excess'] = sim_params['rmean'] / sim_params['beta']
    sim_params = sim_params.sort_values(by=['excess'], ascending = False)
    sim_params = sim_params.loc[(sim_params['excess'] > 0) & (sim_params['beta'] > 0)]

    # compute C values and cutoff
    num = sim_params['rmean'] * sim_params['beta'] / sim_params['eps']
    den = sim_params['beta']**2 / sim_params['eps']
    sim_params['C'] = mktreturn.var() * num.cumsum() / (1 + mktreturn.var() * den.cumsum())

    return(sim_params.loc[sim_params['C'] < sim_params['excess']])

@app.route('/')
def index():
    conn = get_db_connection()
    stocks = conn.execute('SELECT * FROM stocks ORDER BY Date, ticker, Volume LIMIT 1000').fetchall()
    conn.close()
    
    return render_template('index.html', stocks=stocks)        

@app.route('/create', methods=('GET', 'POST'))
def create():
    if request.method == 'POST':
        title = request.form['title']
        #content = request.form['content']

        if not title:
            flash('Title is required!')
        else:
            ticker_list_raw = title.split(',')
            ticker_list = [t.lstrip() for t in ticker_list_raw] + ['^GSPC']

            #conn = get_db_connection()
            #data = conn.execute('SELECT * FROM stocks WHERE ticker IN ({0}) ORDER BY Date'.format(', '.join     ('?' for _ in ticker_list)),
            #                    (ticker_list)
            #                    ).fetchall()
            #conn.commit()
            #conn.close()
            conn = sqlite3.connect('database.db')
            query = conn.execute('SELECT * FROM stocks WHERE Date >= DATE(\'now\', \'-5 years\') AND ticker IN ({0}) ORDER BY Date'.format(', '.join     ('?' for _ in ticker_list)),
                                 (ticker_list))
            #raw_data = query.fetchall()
            cols = [column[0] for column in query.description]
            data = pd.DataFrame.from_records(data = query.fetchall(), columns = cols)
            conn.commit()
            conn.close()
            
            data['Date'] = pd.to_datetime(data['Date'])
            stockdata = clean_data(data, minmonths = 36)

            weights = compute_weights(stockdata)

            output = []
            for i in range(len(weights)):
                output.append(weights.index[i] + ": " + str(weights[i]))

            
            #return redirect(url_for('index'), weights = weights, data = [title, data])
            return render_template('result.html', output = output, data = [title, data])

    return render_template('create.html')

