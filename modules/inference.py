import plotly.graph_objects as go
import pandas as pd
import numpy as np

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

def plot_plant_performance(production, period_stats):

    production['date'] = production['T'].map(period_stats['date'].to_dict())

    def get_stat(c):

        g = production.groupby(['file', 'plant'], as_index=False).agg({c: sum})
        
        g = g.groupby(['plant'], as_index=False).agg({c: 'mean'}).to_dict('list')

        return g

    data = [
            go.Bar(
                x=get_stat('capex')['plant'],
                y=get_stat('capex')['capex'],
                name='capex',
                yaxis='y'),
            go.Bar(
                x=get_stat('ebitda')['plant'],
                y=get_stat('ebitda')['ebitda'],
                name='ebitda',
                yaxis='y'),
            go.Bar(
                x=get_stat('fcf_wo_nwc')['plant'],
                y=get_stat('fcf_wo_nwc')['fcf_wo_nwc'],
                name='fcf_wo_nwc',
                yaxis='y'),
        ]

    layout = {
                'title': 'Plant performance (avg over scenarios)',
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

# def tornado_diagram():

# def mc_simulation():




# END