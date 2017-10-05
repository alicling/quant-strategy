## imort library
import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
from datetime import date

def get_stocklist(fdate):
    stocks=get_index_stocks('000300.XSHG',fdate)
    return stocks

#直接能从数据库获得的因子
##（1）估值：账面市值比（B/M)、盈利收益率（EPS）、动态市盈（PEG）
##（2）成长性：ROE、ROA、主营毛利率（GP/R)、净利率(P/R)
##（3）资本结构：资产负债（L/A)、固定资产比例（FAP）、流通市值（CMV
def get_easy_factors(fdate):
    ##直接能从数据库获得的因子
    stocks=get_stocklist(fdate)
    factors=['B/M','EPS','PEG','ROE','ROA','GP/R','P/R','L/A','FAP','CMV']
    q = query(
        valuation.code,
        balance.total_owner_equities/valuation.market_cap/100000000,
        income.basic_eps,
        valuation.pe_ratio,
        income.net_profit/balance.total_owner_equities,
        income.net_profit/balance.total_assets,
        income.total_profit/income.operating_revenue,
        income.net_profit/income.operating_revenue,
        balance.total_liability/balance.total_assets,
        balance.fixed_assets/balance.total_assets,
        valuation.circulating_market_cap
        ).filter(
        valuation.code.in_(stocks),
        valuation.circulating_market_cap
    )
    fdf = get_fundamentals(q, date=fdate)
    fdf.index = fdf['code']
    fdf.columns = ['code'] + factors
    return fdf.iloc[:,-10:]


## get altmanz
 def get_altmanz(fdate):
    stocks=get_stocklist(fdate)
    fdf=pd.DataFrame()
    df_altmanz=query(valuation.code,
             balance.total_current_assets,
             balance.total_current_liability,
             balance.total_assets,
             balance.retained_profit,
             income.operating_profit,
             valuation.circulating_market_cap,
             balance.total_liability,
             income.total_composite_income).filter(
        valuation.code.in_(stocks),
        valuation.circulating_market_cap
    )
    df_altmanz=get_fundamentals(df_altmanz,date=fdate)
    df_altmanz.index=df_altmanz['code']
    df_altmanz['X1']=(df_altmanz['total_current_assets']-df_altmanz['total_current_liability'])/df_altmanz['total_assets']
    df_altmanz['X2']=df_altmanz['retained_profit']/df_altmanz['total_assets']
    df_altmanz['X3']=df_altmanz['operating_profit']/df_altmanz['total_assets']
    df_altmanz['X4']=df_altmanz['circulating_market_cap']*100000000/df_altmanz['total_liability']
    df_altmanz['X5']=df_altmanz['total_composite_income']/df_altmanz['total_assets']
    df_altmanz['altmanz']=1.2*df_altmanz['X1'] + 1.4*df_altmanz['X2'] + 3.3*df_altmanz['X3'] + 0.6*df_altmanz['X4'] + 1.0*df_altmanz['X5']
    fdf['altmanz']=df_altmanz['altmanz']
    fdf.index=df_altmanz['code']
    return fdf.iloc[:,-1:]
    
## get_beta&indiocrysist risk
def get_beta(fdate):   
    stocks=get_stocklist(fdate)
    benchmark=get_price(security='000300.XSHG', count=252, end_date=fdate, frequency='daily', fields='close', skip_paused=False, fq='pre')
    stockprice=get_price(security=stocks, count=252, end_date=fdate, frequency='daily', fields='close', skip_paused=False, fq='pre')
    stockprice=stockprice[0,:,:]
    ## 计算大盘收益率：
    benchmark_ret=[]
    for i in range(0,251):
        benchmark_ret.append(log(benchmark.iloc[i+1]/benchmark.iloc[i]))
    
    ## stockprice return
    stockprice_ret={}
    for i in range(0,300):
        df=stockprice.iloc[:,i]
        newdf=[]
        for j in range(1,252):
            newdf.append(log(df[j]/df[j-1]))
        stockprice_ret[i]=newdf
    ## 计算beta，risk
    benchmark=pd.DataFrame(benchmark_ret)
    beta=[]
    ssr=[]
    for i in range(0,300):
        stock=pd.DataFrame(stockprice_ret[i])
        stock=stock.fillna(0)
        stock=pd.DataFrame(stock)
        df=pd.concat([benchmark,stock],axis=1)
        df=df.dropna(axis=0,how="all")
        df.columns=['benchmark','stock']
        results=sm.ols(formula="stock~benchmark",data=df).fit()
        beta.append(results.params[1])
        ssr.append(results.ssr)
    beta=pd.DataFrame(beta)
    ssr=pd.DataFrame(ssr)
    df=pd.concat([beta,ssr],axis=1)
    df.index=stocks
    df.columns=['beta','ssr']
    return df
    
## get Fscore:
def get_fscore(fdate):
    stocks=get_stocklist(fdate)
    stocks1=pd.DataFrame(stocks)
    stocks1.index=stocks
    ## 计算fscore——probability
    q = query(
        valuation.code,
        indicator.roa,
        cash_flow.net_operate_cash_flow/(balance.total_assets),
        (income.net_profit-cash_flow.net_operate_cash_flow)/(balance.total_assets),
        ).filter(
        valuation.code.in_(stocks),
        valuation.circulating_market_cap
    )
    prof=get_fundamentals(q,statDate=fdate.year)
    prof_pre=get_fundamentals(q,statDate=fdate.year-1)
    prof.columns=['code','roa','cfo','accrual']
    prof_pre.columns=['code1','roa1','cfo1','accrual1']
    prof.index=prof['code']
    prof_pre.index=prof_pre['code1']
    profdf=pd.DataFrame.join(prof,prof_pre)
    profdff=pd.DataFrame.join(stocks1,profdf)
    f_roa=[]
    for i in range(0,len(profdff)):
        if profdff['roa'][i]>=0:
            f_roa.append(1)
        else:
            f_roa.append(0)
    f_cfo=[]
    for i in range(0,len(profdff)):
        if profdff['cfo'][i]>=0:
            f_cfo.append(1)
        else:
            f_cfo.append(0)
    f_accrual=[]
    for i in range(0,len(profdff)):
        if profdff['accrual'][i]<0:
            f_accrual.append(1)
        else:
            f_accrual.append(0)
    f_d_roa=[]
    for i in range(0,len(profdff)):
        if profdff['roa'][i]-profdff['roa1'][i]>=0:
            f_d_roa.append(1)
        else:
            f_d_roa.append(0)
## Fscore-profitablity 计算
    f_profitablity=[]
    for i in range(0,len(profdff)):
        f_profitablity.append(f_roa[i]+f_cfo[i]+f_accrual[i]+f_d_roa[i])
#Financial performance signals: Changes in  nancial leverage/liquidity
    q = query(
        valuation.code,
        balance.longterm_loan/balance.total_assets,
        balance.total_current_assets/balance.total_current_liability,
        cash_flow.other_finance_act_cash,
        ).filter(
        valuation.code.in_(stocks),
        valuation.circulating_market_cap
)
    lev=get_fundamentals(q,statDate=fdate.year)
    lev.index=lev['code']
    lev.columns=['code','leverage','liquidity','issuance']
    lev_pre=get_fundamentals(q,statDate=fdate.year-1)
    lev_pre.index=lev_pre['code']
    lev_pre.columns=['code1','leverage1','liquidity1','issuance1']
    levdf=pd.DataFrame.join(lev,lev_pre)
    levdff=pd.DataFrame.join(stocks1,levdf)
    f_lev=[]
    for i in range(0,len(levdff)):
        if levdff['leverage'][i]-levdff['leverage1'][i]<0:
            f_lev.append(1)
        else:
            f_lev.append(0)
    f_liquid=[]
    for i in range(0,len(levdff)):
        if levdff['liquidity'][i]-levdff['liquidity1'][i]>=0:
            f_liquid.append(1)
        else:
            f_liquid.append(0)
    f_issuance=[]
    for i in range(0,len(levdff)):
        if levdff['issuance'][i]-levdff['issuance1'][i]<0:
            f_issuance.append(1)
        else:
            f_issuance.append(0)
    f_leverage=[]
    for i in range(0,len(levdff)):
        f_leverage.append(f_lev[i]+f_liquid[i]+f_issuance[i])
# A.3 Financial performance signals: Operating ef ciency
    q = query(
        valuation.code,
        indicator.operation_profit_to_total_revenue,
        income.operating_revenue/balance.total_assets,
        ).filter(
        valuation.code.in_(stocks),
        valuation.circulating_market_cap
)
    op=get_fundamentals(q,statDate=fdate.year)
    op.index=op['code']
    op.columns=['code','margin','turn']
    op_pre=get_fundamentals(q,statDate=fdate.year-1)
    op_pre.index=op_pre['code']
    op_pre.columns=['code1','margin1','turn1']
    opdf=pd.DataFrame.join(op,op_pre)
    opdff=pd.DataFrame.join(stocks1,opdf)
    f_margin=[]
    for i in range(0,len(opdff)):
        if opdff['margin'][i]-opdff['margin1'][i]>=0:
            f_margin.append(1)
        else:
            f_margin.append(0)
    f_turn=[]
    for i in range(0,len(opdff)):
        if opdff['turn'][i]-opdff['turn1'][i]>=0:
            f_turn.append(1)
        else:
            f_turn.append(0)
    f_operation=[]
    for i in range(0,len(opdff)):
        f_operation.append(f_margin[i]+f_turn[i])
    f_score=[]
    for i in range(0,len(stocks)):
        f_score.append(f_profitablity[i]+f_operation[i]+f_leverage[i])
    f_score=pd.DataFrame(f_score)
    f_score.index=stocks
    f_score.columns=['fscore']
    return f_score
