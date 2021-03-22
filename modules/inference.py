import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy.stats import norm

def plot_objective_hist(objective):

    fig = go.Figure(
        data=[go.Histogram(x=objective.objective)],
        layout={
                'title': 'NPV NOPAT - CAPEX',
                'xaxis':{
                        'title': 'MXN',
                        'showticklabels': True,
                        'side':'bottom',
                        'tickfont': {'size':10},
                        },
                'yaxis':{'title': '# of scenarios'},
            }
        )

    fig.show()

def plot_plant_production_plan(production, period_stats):

    production['date'] = production['T'].map(period_stats['date'].to_dict())

    data_dc = {}

    for plant in production.plant.unique():
        
        g = production.loc[production.plant==plant, ['date', 'production']]
        
        g = g.groupby(['date'], as_index=False).agg({'production': 'mean'}).to_dict('list')

        data_dc.update({plant: g})

    data = [
            go.Scatter(
                x=data_dc[plant]['date'],
                y=data_dc[plant]['production'],
                mode='lines',
                name=plant)
            for plant in data_dc.keys()
        ]

    layout = {
                'title': 'Production plan (avg over scenarios)',
                'xaxis':{
                        'title': 'Period',
                        'showticklabels': True,
                        'side':'bottom',
                        'tickfont': {'size':10},
                        },
                'yaxis':{'title': 'Litres'},
            }

    fig = go.Figure(data=data, layout = layout)
    fig.show()

def plot_plant_capex_plan(production, period_stats):

    production['date'] = production['T'].map(period_stats['date'].to_dict())

    def get_stat(c, avg=False):

        if avg:
            agg_fun = {c: 'mean'}
        else:
            agg_fun = {c: sum}

        g = production.groupby(['file', 'plant'], as_index=False).agg(agg_fun)
        
        g = g.groupby(['plant'], as_index=False).agg({c: 'mean'}).to_dict('list')

        return g

    data = [
            go.Bar(
                x=get_stat('capex', avg=False)['plant'],
                y=get_stat('capex', avg=False)['capex'],
                name='capex',
                yaxis='y'),
        ]

    layout = {
                'title': 'Plant CAPEX (avg over scenarios)',
                'barmode': 'group',
                'xaxis':{
                        'title': 'Refinery',
                        'showticklabels': True,
                        'side':'bottom',
                        'tickfont': {'size':10},
                        },
                'yaxis':{'title': 'MXN'},
            }

    fig = go.Figure(data=data, layout = layout)
    fig.show()

def plot_market_consumption(market, demand, period_stats):

    market['date'] = market['T'].map(period_stats['date'].to_dict())

    demand['date'] = demand['T'].map(period_stats['date'].to_dict())

    g = market.groupby(['file', 'loc'], as_index=False).agg({'sales': 'sum'})
    
    g = g.groupby(['loc'], as_index=False).agg({'sales': 'mean'}).sort_values(by=['sales'], ascending=[False]).to_dict('list')

    gd = demand.groupby(['file', 'loc'], as_index=False).agg({'demand': 'sum'})
    
    gd = gd.groupby(['loc'], as_index=False).agg({'demand': 'mean'}).to_dict('list')

    gp = demand.groupby(['loc'], as_index=False).agg({'price': 'mean'}).to_dict('list')

    data = [
            go.Bar(
                x=g['loc'],
                y=g['sales'],
                name='Sales',
                yaxis='y'),
            go.Bar(
                x=gd['loc'],
                y=gd['demand'],
                name='Demand',
                yaxis='y'),
            go.Scatter(
                x=gp['loc'],
                y=gp['price'],
                name='Price (right axis)',
                yaxis='y2',
                mode='markers'),
        ]

    layout = {
                'title': 'Consumption (avg over scenarios)',
                'barmode': 'group',
                'xaxis':{
                        'title': 'Location',
                        'showticklabels': True,
                        'side':'bottom',
                        'tickfont': {'size':10},
                        },
                'yaxis':{'title': 'Litres'},
                'yaxis2':{'title': 'MXN/L', 'overlaying': 'y', 'side': 'right'}
            }

    fig = go.Figure(data=data, layout = layout)
    fig.show()

def get_NPV_stats(production, period_stats, wacc):

    production = production.copy()
    
    d_wacc = (1 + wacc)**(1/365) - 1

    period_stats['D_WACC'] = d_wacc

    period_stats['accum_days'] = period_stats['days'].cumsum() - 365

    period_stats['DF'] = 1/((1+period_stats['D_WACC'])**period_stats['accum_days'])

    production['date'] = production['T'].map(period_stats['date'].to_dict())

    production['DF'] = production['T'].map(period_stats['DF'].to_dict())

    production['NPV_revenues'] =  production['revenues']*production['DF']

    production['NPV_costs'] =  production['costs']*production['DF']

    production = production.assign(NPV_ebitda=lambda r: r.DF*(r.revenues-r.costs))

    production = production.assign(NPV_fcf=lambda r: r.DF*((r.revenues-r.costs-r.depreciation)*0.65+r.depreciation-r.capex))

    production = production.groupby(['file', 'plant']).agg({c: sum for c in production.columns if ('NPV' in c)|(c in ['capex', 'depreciation'])})

    return production.groupby(['plant']).agg({c: ['mean', 'std'] for c in production.columns})

def plot_plant_performance(production, period_stats, wacc):

    stats = get_NPV_stats(production, period_stats, wacc).reset_index()

    data = [
            go.Bar(
                x=stats['plant'],
                y=stats['NPV_ebitda']['mean'],
                name='ebitda',
                yaxis='y'),
            go.Bar(
                x=stats['plant'],
                y=stats['NPV_fcf']['mean'],
                name='fcf_wo_nwc',
                yaxis='y'),
            go.Bar(
                x=stats['plant'],
                y=stats['capex']['mean'],
                name='capex',
                yaxis='y'),
        ]

    layout = {
                'title': 'Plant NPV (avg over scenarios)',
                'barmode': 'group',
                'xaxis':{
                        'title': 'Refinery',
                        'showticklabels': True,
                        'side':'bottom',
                        'tickfont': {'size':10},
                        },
                'yaxis':{'title': 'MXN'},
            }

    fig = go.Figure(data=data, layout = layout)
    fig.show()

def get_NPV_sensitivity(production, period_stats, wacc):

    production = production.copy()
    
    d_wacc = (1 + wacc)**(1/365) - 1

    period_stats['D_WACC'] = d_wacc

    period_stats['accum_days'] = period_stats['days'].cumsum() - 365

    period_stats['DF'] = 1/((1+period_stats['D_WACC'])**period_stats['accum_days'])

    production['date'] = production['T'].map(period_stats['date'].to_dict())

    production['DF'] = production['T'].map(period_stats['DF'].to_dict())

    production_base = production.copy()

    production = production.assign(NPV_fcf=lambda r: r.DF*((r.revenues-r.costs-r.depreciation)*0.65+r.depreciation-r.capex))

    production = production.assign(revenues=lambda r: r.revenues + abs(r.revenues*(0.5))).assign(NPV_revenues_up_50pct=lambda r: r.DF*((r.revenues-r.costs-r.depreciation)*0.65+r.depreciation-r.capex)).assign(revenues=lambda r: production_base.loc[r.index].revenues)

    production = production.assign(revenues=lambda r: r.revenues - abs(r.revenues*(0.5))).assign(NPV_revenues_down_50pct=lambda r: r.DF*((r.revenues-r.costs-r.depreciation)*0.65+r.depreciation-r.capex)).assign(revenues=lambda r: production_base.loc[r.index].revenues)

    production = production.assign(capex=lambda r: r.capex*(1.5)).assign(NPV_capex_up50pct=lambda r: r.DF*((r.revenues-r.costs-r.depreciation)*0.65+r.depreciation-r.capex)).assign(capex=lambda r: r.capex/(1.5))

    production = production.assign(capex=lambda r: r.capex*(1.2)).assign(NPV_capex_up20pct=lambda r: r.DF*((r.revenues-r.costs-r.depreciation)*0.65+r.depreciation-r.capex)).assign(capex=lambda r: r.capex/(1.2))

    d_wacc = (1 + wacc*1.2)**(1/365) - 1

    period_stats['D_WACC'] = d_wacc

    period_stats['accum_days'] = period_stats['days'].cumsum() - 365

    period_stats['DF'] = 1/((1+period_stats['D_WACC'])**period_stats['accum_days'])

    production['date'] = production['T'].map(period_stats['date'].to_dict())

    production['DF'] = production['T'].map(period_stats['DF'].to_dict())

    production = production.assign(NPV_WACC_up20pct=lambda r: r.DF*((r.revenues-r.costs-r.depreciation)*0.65+r.depreciation-r.capex))

    def get_stat(c):

        g = production.groupby(['file', 'plant'], as_index=False).agg({c: sum})
        
        g = g.groupby(['plant']).agg({c: ['mean', 'std', 'count']})

        return g

    gc = pd.concat(
        [
            get_stat('NPV_fcf'),
            get_stat('NPV_revenues_up_50pct'),
            get_stat('NPV_revenues_down_50pct'),
            get_stat('NPV_capex_up50pct'),
            get_stat('NPV_capex_up20pct'),
            get_stat('NPV_WACC_up20pct'),
        ], 
        axis=1
    )

    return gc

def sensitivity_diagram(production, period_stats, wacc):

    df = get_NPV_sensitivity(production, period_stats, wacc).T

    def get_bars(plant):

        ref_df = df.loc[(slice(None), 'mean'), [plant]].droplevel(1)

        ref_df = ref_df - ref_df.loc['NPV_fcf']

        ref_df = ref_df.reset_index()

        ref_df = ref_df.loc[ref_df['index']!='NPV_fcf']

        return ref_df.to_dict('list')

    data = [
            go.Bar(
                x=get_bars(plant)[plant],
                y=get_bars(plant)['index'],
                name=plant,
                yaxis='y',
                orientation='h'
                )
            for plant in df.columns
        ]

    layout = {
                'title': 'Sensitivity diagram',
                'barmode': "group",
                'autosize': True, 
                'showlegend': True,
                'xaxis':{
                        'title': 'Refinery',
                        'showticklabels': True,
                        'side':'bottom',
                        'tickfont': {'size':10},
                        },
                'yaxis':{'title': 'MXN'},
            }

    fig = go.Figure(data=data, layout = layout)
    fig.show()  

def mc_simulation(prod_stats, period_stats, WACC, n_samples=500, plot=True):

    performance = get_NPV_stats(prod_stats, period_stats, WACC)

    cpx_df_factor = (performance['depreciation']['mean']/performance['capex']['mean']).dropna().mean()
    
    samples = {col:
        {
            idx: np.random.normal(row['mean'], row['std'], size=n_samples) if (row['std'] != 0)&(col not in ['capex']) else np.random.normal(row['mean'], abs(row['mean']*0.2), size=n_samples) for idx, row in performance[col].iterrows()
        } for col in np.unique(performance.T.droplevel(1).index)
    }

    samples.update({'depreciation': {pln: v*cpx_df_factor for pln, v in samples['capex'].items()}})

    npvs = (pd.DataFrame(samples['NPV_revenues']) - pd.DataFrame(samples['NPV_costs']) - pd.DataFrame(samples['depreciation']))*0.65 + pd.DataFrame(samples['depreciation']) - pd.DataFrame(samples['capex'])

    fig = make_subplots(rows=int(len(npvs.columns)), cols=1)

    layout={
            'title': 'NPV NOPAT - CAPEX',
            'xaxis':{
                    'title': 'MXN',
                    'showticklabels': True,
                    'side':'bottom',
                    'tickfont': {'size':10},
                    },
            'yaxis':{'title': '# of scenarios'},
            'height':2000,
            'width':800
        }

    # fig = go.Figure(data=data, layout=layout)
    for n, plnt in enumerate(npvs.columns):
        fig.append_trace(go.Histogram(x=npvs[plnt], name=plnt), row=n+1, col=1)

    fig.update_layout(layout)
    fig.show()

    benefits = (pd.DataFrame(samples['NPV_revenues']) - pd.DataFrame(samples['NPV_costs']))*(1-0.35)

    costs = pd.DataFrame(samples['capex']) - pd.DataFrame(samples['depreciation'])*(0.35)

    assert ((benefits - costs) - npvs < 1).all().all()

    return benefits, costs

def real_opt_valuation(prod_stats, benefits, costs, capex, rf, T, plot=True):
    log_chg = (1+prod_stats.set_index(['T', 'file', 'plant']).ebitda.unstack(level=[0]).drop(columns=0).pct_change(axis=1)).apply(np.log).replace([np.inf, -np.inf], np.nan)
    mu_log_chg = log_chg.mean(axis=1)
    vol_samples = ((log_chg.T - mu_log_chg)**2).mean(axis=0).apply(np.sqrt)
    vol = vol_samples.reset_index().rename(columns={0: 'vol'}).groupby('plant').agg({'vol': 'mean'}).fillna(0)

    def d1(S, K, rf, vol, t):
        numer = np.log(S/K) + (rf + vol/2)*t
        denom = np.sqrt(vol*t)
        return numer/denom

    def d2(S, K, rf, vol, t):
        return d1(S, K, rf, vol, t) - np.sqrt(vol*t)

    def BS_model(S, K, rf, vol, t):
        N = norm.cdf
        n_d1 = d1(S, K, rf, vol, t)
        n_d2 = d2(S, K, rf, vol, t)
        BS = S*N(n_d1) - K*np.exp(-rf*t)*N(n_d2)
        return BS

    K = capex.groupby(['loc']).agg({'capex': sum})['capex']

    npv = benefits - costs

    call = pd.concat([benefits.mean(axis=0), K, vol**2], axis=1).apply(lambda r: BS_model(r[0], r['capex'], rf, r['vol'], T), axis=1).fillna(0)

    def step_fn(x):
        if x>0:
            return 1
        else:
            return 0

    def delta_fn(x):
        if abs(x)>1e-5:
            return 1
        else:
            return 0

    if plot:
        valuation = pd.concat([benefits.mean(axis=0)-K*benefits.mean(axis=0).apply(delta_fn), npv.mean(axis=0), call], axis=1)
        valuation.columns = ['DCF', 'MC', 'BS']
        valuation.index.names = ['plant']

        data = [
                go.Bar(
                    x=valuation.reset_index().plant,
                    y=valuation.DCF,
                    name='DCF',
                    yaxis='y'),
                go.Bar(
                    x=valuation.reset_index().plant,
                    y=valuation.MC,
                    name='Monte Carlo',
                    yaxis='y'),
            ]

        layout = {
                    'title': 'DCF vs Monte Carlo',
                    'barmode': 'group',
                    'xaxis':{
                            'title': 'Refinery',
                            'showticklabels': True,
                            'side':'bottom',
                            'tickfont': {'size':10},
                            },
                    'yaxis':{'title': 'MXN', 'range': [valuation.min().min()*1.5, valuation.max().max()*1.5]},
                }

        fig = go.Figure(data=data, layout = layout)
        fig.show()

    return npv, call

# END