import numpy as np
import pandas as pd
import pyomo.environ as pme
import pyomo.gdp as pmg
import pyomo
from pyomo.core.util import sum_product
import os
import sys
import re
from modules import timeseries
from math import radians, cos, sin, asin, sqrt, isnan
from scipy.spatial.distance import cdist
from tqdm import tqdm

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def monthly_data(version, window, init_date, final_date, plot=True, p_pbar=None):
    
    FILTERS = 32 # number of filters for the convolution layers
    KERNEL_SIZE = 15 # size of the convolution window (kernel)
    STRIDES = 2 # number of timesteps to skip between samples
    H_UNITS = 2**8 # size of the hidden neurons for prediction
    LATENT_DIMS = 2**4 # size of the embedding
    
    # vae random blocks
    # price
    price_ts, price_encoder, price_decoder, price_vae = timeseries.get_models(batch_size=window.batch, input_width=window.input_width-1, input_dims=len(window.label_columns), latent_dims=LATENT_DIMS, filters=FILTERS, kernel_size=KERNEL_SIZE, strides=STRIDES, h_units=H_UNITS, random=True, amplitude=20)

    # litres
    litres_ts, litres_encoder, litres_decoder, litres_vae = timeseries.get_models(batch_size=window.batch, input_width=window.input_width-1, input_dims=len(window.label_columns), latent_dims=LATENT_DIMS, filters=FILTERS, kernel_size=KERNEL_SIZE, strides=STRIDES, h_units=H_UNITS, random=True, amplitude=100)
    
    timeseries.load_weights(version, price_encoder, price_decoder, price_vae, litres_encoder, litres_decoder, litres_vae)
    
    price_ts_rand, litres_ts_rand = timeseries.forecast(price_vae, price_ts, litres_vae, litres_ts, window=window, init_date=init_date, final_date=final_date, plot=False, p_pbar=p_pbar, saturate=True)
    
    df = pd.merge(
        left = price_ts_rand,
        right = litres_ts_rand,
        on=['date', 'state', 'loc']).assign(revenue=lambda r: r.price*r.litres).assign(date=lambda r: r.date.dt.to_period('M')).groupby(['date', 'state', 'loc']).agg({'litres': sum, 'revenue': sum})\
    .assign(price=lambda r: r.revenue.div(r.litres)).drop(columns='revenue')

    month_stats = pd.DataFrame(litres_ts_rand.date.drop_duplicates())

    month_stats = month_stats.assign(days=1).groupby([month_stats.date.dt.to_period('M')]).agg({'days': sum})

    month_stats['T'] = 1

    month_stats['T'] = month_stats['T'].cumsum().values - 1

    dates_map = {dt: n for n, dt in enumerate(df.reset_index().date.unique())}

    df['T'] = df.reset_index().date.map(dates_map).values

    df = df.set_index(['T'], append=True)
    
    price_ts_rand = df.price.reset_index()
    
    litres_ts_rand = df.litres.reset_index()
    
    return price_ts_rand, litres_ts_rand, month_stats

def annual_data(version, window, init_date, final_date, plot=True, p_pbar=None):
    
    FILTERS = 32 # number of filters for the convolution layers
    KERNEL_SIZE = 15 # size of the convolution window (kernel)
    STRIDES = 2 # number of timesteps to skip between samples
    H_UNITS = 2**8 # size of the hidden neurons for prediction
    LATENT_DIMS = 2**4 # size of the embedding
    
    # vae random blocks
    # price
    price_ts, price_encoder, price_decoder, price_vae = timeseries.get_models(batch_size=window.batch, input_width=window.input_width-1, input_dims=len(window.label_columns), latent_dims=LATENT_DIMS, filters=FILTERS, kernel_size=KERNEL_SIZE, strides=STRIDES, h_units=H_UNITS, random=True, amplitude=20)

    # litres
    litres_ts, litres_encoder, litres_decoder, litres_vae = timeseries.get_models(batch_size=window.batch, input_width=window.input_width-1, input_dims=len(window.label_columns), latent_dims=LATENT_DIMS, filters=FILTERS, kernel_size=KERNEL_SIZE, strides=STRIDES, h_units=H_UNITS, random=True, amplitude=100)
    
    timeseries.load_weights(version, price_encoder, price_decoder, price_vae, litres_encoder, litres_decoder, litres_vae)
    
    price_ts_rand, litres_ts_rand = timeseries.forecast(price_vae, price_ts, litres_vae, litres_ts, window=window, init_date=init_date, final_date=final_date, plot=False, p_pbar=p_pbar, saturate=True)
    
    df = pd.merge(
        left = price_ts_rand,
        right = litres_ts_rand,
        on=['date', 'state', 'loc']).assign(revenue=lambda r: r.price*r.litres).assign(date=lambda r: r.date.dt.to_period('Y')).groupby(['date', 'state', 'loc']).agg({'litres': sum, 'revenue': sum})\
    .assign(price=lambda r: r.revenue.div(r.litres)).drop(columns='revenue')

    year_stats = pd.DataFrame(litres_ts_rand.date.drop_duplicates())

    year_stats = year_stats.assign(days=1).groupby([year_stats.date.dt.to_period('Y')]).agg({'days': sum})

    year_stats['T'] = 1

    year_stats['T'] = year_stats['T'].cumsum().values - 1

    dates_map = {dt: n for n, dt in enumerate(df.reset_index().date.unique())}

    df['T'] = df.reset_index().date.map(dates_map).values

    df = df.set_index(['T'], append=True)
    
    price_ts_rand = df.price.reset_index()
    
    litres_ts_rand = df.litres.reset_index()
    
    return price_ts_rand, litres_ts_rand, year_stats

def get_refinery_data():
    return pd.read_csv('capacity.csv', header=0).rename(columns={'site': 'loc'})

def get_distances(window):
    coords = pd.concat([window.coords, get_refinery_data()], axis=0).drop(columns=['daily_capacity'])

    distances = cdist(coords.set_index(['state', 'loc']).values, coords.set_index(['state', 'loc']).values, metric = lambda XA, XB: haversine(XA[1], XA[0], XB[1], XB[0]))

    distances = pd.DataFrame(distances, index = coords['loc'].values, columns = coords['loc'].values).unstack()

    distances.index.names = ['source', 'destination']

    distances = distances.reset_index().rename(columns={0: 'kms'})

    return distances

def fix_freight(window, freight, overall_mxnxkms_L, specific_state_connection_rates, specific_states_rates):
    coords = pd.concat([window.coords, get_refinery_data()], axis=0).drop(columns=['daily_capacity'])

    if not not specific_state_connection_rates:
        print('Modifying rates for specific state connections')
        specific_modifications = [] # a list to save the modified locations so they are not modified in the general case
        specific_states = list(np.unique([k[0] for k in specific_state_connection_rates.keys()]+[k[1] for k in specific_state_connection_rates.keys()]))
        specific_states = coords.loc[coords.state.isin(specific_states), ['state', 'loc']].drop_duplicates()
        for k, _ in specific_state_connection_rates.items():
            sources = specific_states.loc[specific_states.state == k[0], 'loc'].values
            destinations = specific_states.loc[specific_states.state == k[1], 'loc'].values
            for s in sources:
                for d in destinations:
                    source_state = specific_states.set_index('loc').loc[s].values[0]
                    dest_state = specific_states.set_index('loc').loc[d].values[0]
                    rate = specific_state_connection_rates[(source_state, dest_state)]
                    freight.loc[(freight.source==s)&(freight.destination==d), 'freightxkms']*=rate
                    specific_modifications.append([s, d])
        
    if not not specific_states_rates:
        print('Modifying rates for states')
        state_locs = coords.loc[coords.state.isin(specific_states_rates.keys()), ['state', 'loc']].drop_duplicates()
        for _, row in state_locs.iterrows():
            selection = (freight.source==row['loc'])|(freight.destination==row['loc'])
            selection = selection&(freight.freightxkms==overall_mxnxkms_L) # select only the unmodified rates
            freight.loc[selection,'freightxkms']*=specific_states_rates[row.state]

    freight['freight'] = freight['kms']*freight['freightxkms']

    return freight
            
def production_timeseries(opt_periods, period_stats, fixed_base, fixed_ratio, variable_base, variable_ratio, storage_base, storage_ratio, safety_inventory_base, safety_inventory_ratio):
    refineries = ['Cadereyta Refinery',
        'Madero Refinery',
        'Tula Refinery',
        'Salamanca Refinery',
        'Minatitlan Refinery',
        'Salina Cruz Refinery',
        'Cangrejera Refinery',
        'Dos Bocas Refinery']

    refineries_profle = {
        'Cadereyta Refinery': 0.95,
        'Madero Refinery': 0.92,
        'Tula Refinery': 1.0,
        'Salamanca Refinery': 1.0,
        'Minatitlan Refinery': 1.15,
        'Salina Cruz Refinery': 1.07,
        'Cangrejera Refinery': 2.0,
        'Dos Bocas Refinery': 0.9
        }

    production = get_refinery_data()
    
    fixed = fixed_base
    fixed_costs_dict = {'fixed': {r: fixed_ratio*refineries_profle[r] for r in refineries}}

    production = pd.merge(
        left=production,
        right=pd.DataFrame(fixed_costs_dict),
        left_on = 'loc',
        right_index=True
    ).assign(fixed=lambda r: r.daily_capacity*r.fixed*30+fixed)

    variable = variable_base
    variable_costs_dict = {'variable': {r: variable_ratio*refineries_profle[r] for r in refineries}}

    production = pd.merge(
        left=production,
        right=pd.DataFrame(variable_costs_dict),
        left_on = 'loc',
        right_index=True
    ).assign(variable=lambda r: r.daily_capacity*r.variable+variable)

    storage = storage_base
    storage_dict = {'storage': {r: storage_ratio for r in refineries}}

    production = pd.merge(
        left=production,
        right=pd.DataFrame(storage_dict),
        left_on = 'loc',
        right_index=True
    ).assign(storage=lambda r: r.daily_capacity*r.storage+storage)
    
    safety_inventory = safety_inventory_base
    safety_inventory_dict = {'safety_inventory': {r: safety_inventory_ratio for r in refineries}}

    production = pd.merge(
        left=production,
        right=pd.DataFrame(safety_inventory_dict),
        left_on = 'loc',
        right_index=True
    ).assign(safety_inventory=lambda r: r.daily_capacity*r.safety_inventory+safety_inventory)

    t = 0
    production['T'] = t
    production['days'] = period_stats.loc[period_stats['T']==t].days.values[0]
    production_base = production.copy()
    for t in range(1,opt_periods):
        days = period_stats.loc[period_stats['T']==t].days.values[0]
        production = pd.concat(
            [production, production_base.assign(
                T=t,
                days=days,
                fixed=lambda r: r.fixed*(1.05)**t,
                variable=lambda r: r.variable*(1.09)**t,
            )], axis = 0
        )
    production.reset_index(drop=True, inplace=True)
    production['capacity'] = production['daily_capacity']*production['days']

    return production

def build_freight_costs(w, overall_mxnxton_kms):
    distances = get_distances(window=w)

    # overall_mxnxton_kms = 0.2 # MXN/(Ton-Km)
    pipe_size = 20000 # (L/trip) 
    fuel_weight = (0.8508+0.7489)/2/1000 # (Ton/L) = (Kg/L)/(1 Ton/1000 Kg) 0.8508 is for fuel, 0.7489 is for gasoline

    ### trip = 20,000 litres ###

    tonsxpipe = pipe_size*fuel_weight # (Ton/trip) = (L/trip)*(Ton/L)
    overall_mxnxkms = overall_mxnxton_kms*tonsxpipe # MXN/(Km-trip) = MXN/(Ton-Km)*(Ton/trip)
    overall_mxnxkms_L = overall_mxnxkms/pipe_size # MXN/(Km-L) = MXN/(Km-trip)/(L/trip)

    freight = distances.copy()
    freight['freightxkms'] = overall_mxnxkms_L
            
    freight['freight'] = freight['kms']*freight['freightxkms']

    specific_states_rates = {
        'Baja California': 1.2, # Peninsula
        'Baja California Sur': 1.5, # Peninsula
        'Yucatán': 1.3, # Peninsula
        'Oaxaca': 1.2, # Mountain range
        'San Luis Potosí': 0.85 # Logistics hub
    }

    specific_state_connection_rates = {
        ('Baja California', 'Baja California Sur'): 1.3, # Mountain range
        ('Oaxaca', 'Baja California Sur'): 0.2, # Vessel
        ('Tamaulipas', 'Yucatán'): 0.2, # Vessel
        ('Veracruz de Ignacio de la Llave', 'Yucatán'): 0.2, # Vessel
    }

    freight = fix_freight(w, freight, overall_mxnxkms_L, specific_state_connection_rates, specific_states_rates)

    return freight

def market_block_rule(b, t, market, DEMAND, PRICE, FREIGHT, freight_inflation = 0.12, step=None, pbar=None):
    
    if pbar is not None:

        template = 'Step: {step}; Building market {market} at time {t}'

        pbar.set_description(template.format(step=step, market=market, t = t))

    fuel_opt=b.parent_block()

    dict_demand = {None: DEMAND.loc[(DEMAND['loc']==market)&(DEMAND['T']==t)].litres.values[0]}

    dict_price = {None: PRICE.loc[(PRICE['loc']==market)&(PRICE['T']==t)].price.values[0]}

    dict_freight = FREIGHT.assign(freight=lambda r: r.freight*(1+freight_inflation)**t).set_index(['source']).loc[FREIGHT.set_index(['source'])['destination']==market].freight.to_dict()

    dict_freight = {k: it for k, it in dict_freight.items() if k[0] in fuel_opt.REFINERIES}

    dict_freight = {k: 30+10*FREIGHT.freight.max() if np.isnan(it) else it for k, it in dict_freight.items()}

    b.DEMAND = pme.Param(initialize=dict_demand, default=0, name=f'{market} market demand', doc=f'{market} market demand')

    b.PRICE = pme.Param(initialize=dict_price, default=0, name=f'{market} market price', doc=f'{market} market price')    

    b.FREIGHT = pme.Param(fuel_opt.REFINERIES,
                    initialize=dict_freight,
                    default=FREIGHT.freight.max()*2,
                    name=f'Sourcing cost for each refinery to {market}',
                    doc=f'Sourcing cost for each refinery to {market}')

    # define variables
    def L_Bounds_rule(model, plant):
        return (0, model.DEMAND) # This plant will sell at most the demand at the market
    b.L = pme.Var(fuel_opt.REFINERIES, bounds=L_Bounds_rule)

    # define expressions
    def sourcing_cost_rule(model, plant):
        return model.FREIGHT[plant]*model.L[plant]
    b.FREIGHT_COST = pme.Expression(fuel_opt.REFINERIES, rule=sourcing_cost_rule)

    def revenue_rule(model, plant):
        return model.PRICE*model.L[plant]
    b.REVENUE = pme.Expression(fuel_opt.REFINERIES, rule=revenue_rule)

    def EBITDA_rule(model, plant):
        return (model.PRICE - model.FREIGHT[plant])*model.L[plant]
    b.EBITDA = pme.Expression(fuel_opt.REFINERIES, rule=EBITDA_rule)

    # define constraints
    def demand_fullfilment_rule(model):
        return sum(model.L[plant] for plant in fuel_opt.REFINERIES) <= model.DEMAND
    b.DEMAND_Fullfilment_Constraint = pme.Constraint(rule=demand_fullfilment_rule)

    sum([pme.value(x) for x in b.component_data_objects(ctype=pme.Param)])

def plant_block_rule(b, t, plant, FREIGHT, PRODUCTION, PROJECTS, depreciation_years, minimum_production_months, freight_inflation=0.09, step=None, pbar=None, annual=False):

    if pbar is not None:

        template = 'Step: {step}; Building plant {plant} at time {t}'

        pbar.set_description(template.format(step=step, plant=plant, t = t))
    
    fuel_opt=b.parent_block()

    b.INTER_PLANTS = pme.Set(initialize=[p for p in fuel_opt.REFINERIES if p not in [plant, 'Slack']], name='Inter facilities connection', doc='Inter facilities connection')

    dict_freight = FREIGHT.assign(freight=lambda r: r.freight*(1+freight_inflation)**t).set_index(['source']).loc[FREIGHT.set_index(['source'])['destination']==plant].freight.to_dict()

    dict_freight = {k: it for k, it in dict_freight.items() if k[0] in b.INTER_PLANTS}

    dict_freight = {k: 30+10*FREIGHT.freight.max() if np.isnan(it) else it for k, it in dict_freight.items()}

    refinery_df = PRODUCTION.loc[(PRODUCTION['loc']==plant)&(PRODUCTION['T']==t)]

    b.FREIGHT = pme.Param(b.INTER_PLANTS,
                    initialize=dict_freight,
                    default=FREIGHT.freight.max()*2,
                    name=f'Sourcing cost for each refinery to {plant}',
                    doc=f'Sourcing cost for each refinery to {plant}')

    b.VARIABLE = pme.Param(initialize={None: refinery_df.variable.values[0]},
                            default=PRODUCTION.variable.max()*2,
                            name=f'{plant} production costs',
                            doc=f'{plant} production costs')

    b.FIXED = pme.Param(initialize={None: refinery_df.fixed.values[0]},
                            default=PRODUCTION.fixed.max()*1000,
                            name=f'{plant} fixed costs',
                            doc=f'{plant} fixed costs')

    b.CAPACITY = pme.Param(initialize={None: refinery_df.capacity.values[0]},
                            default=0,
                            name=f'{plant} production capacity',
                            doc=f'{plant} production capacity',
                            mutable=True)

    b.STORAGE = pme.Param(initialize={None: refinery_df.storage.values[0]},
                            default=0,
                            name=f'{plant} storage capacity',
                            doc=f'{plant} storage capacity')

    b.SAFETY_INVENTORY = pme.Param(initialize={None: refinery_df.safety_inventory.values[0]},
                            default=0,
                            name=f'{plant} safety inventory',
                            doc=f'{plant} safety inventory')

    b.SWITCH_COSTS = pme.Param(initialize={None: refinery_df.safety_inventory.values[0]},
                            default=0,
                            name=f'{plant} turn on costs',
                            doc=f'{plant} turn on costs')

    plant_projects = PROJECTS.loc[(PROJECTS['loc']==plant)&(PROJECTS.type=='INITIAL')]

    b.INITIAL_CAPEX_COSTS = pme.Param(
                    initialize={None: plant_projects.capex.values[0]},
                    default=0,
                    name=f'{plant} Inital CAPEX',
                    doc=f'{plant} Initial CAPEX')

    inc_plant_projects = PROJECTS.loc[(PROJECTS['loc']==plant)&(PROJECTS.type=='STRATEGIC')]

    b.STRATEGIC_CAPEX_COSTS = pme.Param(
                    initialize={None: inc_plant_projects.capex.values[0]},
                    default=0,
                    name=f'{plant} Capacity expansion CAPEX',
                    doc=f'{plant} Capacity expansion CAPEX')

    b.STRATEGIC_CAPEX_SIZE = pme.Param(
                    initialize={None: inc_plant_projects.increment.values[0]},
                    default=0,
                    name=f'{plant} Capacity expansion',
                    doc=f'{plant} Capacity expansion')

    b.EXPANSION_CAPACITY = pme.Param(initialize={None: refinery_df.capacity.values[0]*max(0,(inc_plant_projects.increment.values[0]-1))},
                            default=0,
                            name=f'{plant} production expanded capacity',
                            doc=f'{plant} production expanded capacity')

    # define variables
    def L_Bounds_rule(model):
        return (0, model.CAPACITY+model.EXPANSION_CAPACITY) # This plant will sell at most its capacity
    b.L = pme.Var(bounds=L_Bounds_rule)

    def OUT_INTER_L_Bounds_rule(model, inter_plant):
        return (0, model.CAPACITY+model.EXPANSION_CAPACITY)
    b.OUT_INTER_L = pme.Var(b.INTER_PLANTS, bounds=OUT_INTER_L_Bounds_rule)

    def IN_INTER_L_Bounds_rule(model, inter_plant):
        return (0, model.CAPACITY+model.EXPANSION_CAPACITY)
    b.IN_INTER_L = pme.Var(b.INTER_PLANTS, bounds=IN_INTER_L_Bounds_rule)

    b.STATES = pme.Set(initialize=[0,1])
    
    b.INITIAL_CAPEX = pme.Var(domain=pme.Binary)

    b.STRATEGIC_CAPEX = pme.Var(domain=pme.Binary)

    b.SWITCH = pme.Var(domain=pme.Binary)

    b.LAYOFF = pme.Var(domain=pme.Binary)

    b.INITIAL_STATE = pme.Var(domain=pme.Binary, initialize=0)

    b.STATE = pme.Var(domain=pme.Binary)

    b.INITIAL_AVAILABLE = pme.Var(domain=pme.Binary, initialize=0)

    b.AVAILABLE = pme.Var(domain=pme.Binary)

    b.INITIAL_EXPANSION = pme.Var(domain=pme.Binary, initialize=0)

    b.EXPANSION = pme.Var(domain=pme.Binary)

    def L_INITIAL_INVENTORY_Bounds_rule(model):
        return (0, model.STORAGE) #
    b.L_INITIAL_INVENTORY = pme.Var(bounds=L_INITIAL_INVENTORY_Bounds_rule)

    # INVENTORY as Expression
    def end_product_inventories_and_production_rule(model):
        fuel_opt=model.parent_block()
        return model.L_INITIAL_INVENTORY + model.L - fuel_opt.DEMAND[t, plant] + sum(model.IN_INTER_L[ip] - model.OUT_INTER_L[ip] for ip in b.INTER_PLANTS)        
    b.L_INVENTORY = pme.Expression(rule=end_product_inventories_and_production_rule)

    # define constraints
    def l_inventories_bottom_constraint_rule(model):
        return 0 <= model.L_INVENTORY
    b.L_INVENTORY_LIMITS_Constraint = pme.Constraint(rule=l_inventories_bottom_constraint_rule)

    def inventories_capacity_constraint_rule(model):
        return model.STATE*model.SAFETY_INVENTORY <= model.L_INVENTORY # ALREADY BOUNDED FROM TOP WITH INITIAL_INVENTORY BOUNDS
    b.INVENTORY_LIMITS_Constraint = pme.Constraint(rule=inventories_capacity_constraint_rule)

    def SWITCH_Constraint_rule(disjunct, a0, a1, s0, s1, c0, c1):

        b=disjunct.parent_block()
        fuel_opt=b.parent_block()

        disjunct.SWITCHConstraint = pme.Constraint()
        disjunct.LAYOFFConstraint = pme.Constraint()
        disjunct.INITIAL_CAPEXConstraint = pme.Constraint()
        disjunct.STRATEGIC_CAPEXConstraint = pme.Constraint()

        if (c0==0)&(c1==1):
            disjunct.STRATEGIC_CAPEXConstraint = b.STRATEGIC_CAPEX == 1
        else:
            disjunct.STRATEGIC_CAPEXConstraint = b.STRATEGIC_CAPEX == 0

        if c1==0:
            def production_state_rule(model):
                return b.L <= b.STATE*b.CAPACITY
        else:
            def production_state_rule(model):
                return b.L <= b.STATE*(b.CAPACITY+b.EXPANSION_CAPACITY)
        disjunct.PRODUCTION_CAPACITY_Constraint = pme.Constraint(rule=production_state_rule)

        if (a0==0)&(a1==1):
            disjunct.INITIAL_CAPEXConstraint = b.INITIAL_CAPEX == 1
        else:
            disjunct.INITIAL_CAPEXConstraint = b.INITIAL_CAPEX == 0

        if s0==s1:
            disjunct.SWITCHConstraint = b.SWITCH == 0
            disjunct.LAYOFFConstraint = b.LAYOFF == 0

        elif (s0==0)&(s1==1):
            disjunct.SWITCHConstraint = b.SWITCH == 1
            disjunct.LAYOFFConstraint = b.LAYOFF == 0

        elif (s0==1)&(s1==0):
            disjunct.SWITCHConstraint = b.SWITCH == 0
            disjunct.LAYOFFConstraint = b.LAYOFF == 1

        def initial_available_coherence_constraint_rules(model):
            return b.INITIAL_AVAILABLE == a0
        disjunct.INITIAL_AVAILABLE_COHERENCE_Constraint = pme.Constraint(rule=initial_available_coherence_constraint_rules)

        def available_coherence_constraint_rules(model):
            return b.AVAILABLE == a1
        disjunct.AVAILABLE_COHERENCE_Constraint = pme.Constraint(rule=available_coherence_constraint_rules)

        def initial_strategic_coherence_constraint_rules(model):
            return b.INITIAL_EXPANSION == c0
        disjunct.INITIAL_OVERHAUL_COHERENCE_Constraint = pme.Constraint(rule=initial_strategic_coherence_constraint_rules)

        def strategic_coherence_constraint_rules(model):
            return b.EXPANSION == c1
        disjunct.OVERHAUL_COHERENCE_Constraint = pme.Constraint(rule=strategic_coherence_constraint_rules)

        # initial_state coherence constraint
        def initial_state_coherence_constraint_rules(model):
            return b.INITIAL_STATE == s0
        disjunct.INITIAL_STATE_COHERENCE_Constraint = pme.Constraint(rule=initial_state_coherence_constraint_rules)

        # state coherence constraint
        def state_coherence_constraint_rules(model):
            if t == fuel_opt.T.last():
                return b.STATE == s0
            else:
                return b.STATE == s1
        disjunct.STATE_COHERENCE_Constraint = pme.Constraint(rule=state_coherence_constraint_rules)
    b.SWITCH_GDP = pmg.Disjunct(b.STATES, b.STATES, b.STATES, b.STATES, b.STATES, b.STATES, rule=SWITCH_Constraint_rule)

    possible_states = [
        [0,0,0,0,0,0],

        [0,1,0,0,0,0],
        [0,1,0,0,0,1],

        [1,1,0,0,0,0],
        [1,1,0,1,0,0],
        [1,1,1,0,0,0],
        [1,1,1,1,0,0],

        [1,1,0,0,0,1],
        [1,1,1,0,0,1],

        [1,1,0,0,1,1],
        [1,1,0,1,1,1],
        [1,1,1,0,1,1],
        [1,1,1,1,1,1],
    ]

    def BindSTATES_rule(model):
        return [model.SWITCH_GDP[s[0], s[1], s[2], s[3], s[4], s[5]] for s in possible_states]
    b.BindSTATES = pmg.Disjunction(rule=BindSTATES_rule)

    def SWITCH_COSTS_rules(model):
        return model.SWITCH_COSTS*model.SWITCH
    b.SWITCH_COSTS_Expression = pme.Expression(rule=SWITCH_COSTS_rules)

    if annual:
        def LAYOFF_COSTS_rules(model):
            return model.FIXED*model.LAYOFF*minimum_production_months/12
        b.LAYOFF_COSTS_Expression = pme.Expression(rule=LAYOFF_COSTS_rules)
    else:
        def LAYOFF_COSTS_rules(model):
            return model.FIXED*model.LAYOFF*minimum_production_months
        b.LAYOFF_COSTS_Expression = pme.Expression(rule=LAYOFF_COSTS_rules)

    def CAPEX_COSTS_rules(model):
        return model.INITIAL_CAPEX_COSTS*model.INITIAL_CAPEX + model.STRATEGIC_CAPEX_COSTS*model.STRATEGIC_CAPEX
    b.CAPEX_COSTS_Expression = pme.Expression(rule=CAPEX_COSTS_rules)

    def production_costs_rules(model):
        return\
                model.SWITCH_COSTS_Expression + model.LAYOFF_COSTS_Expression\
                + model.STATE*model.FIXED\
                + model.L*model.VARIABLE\
                + sum(model.IN_INTER_L[rp]*model.FREIGHT[rp] for rp in model.INTER_PLANTS)
    b.PRODUCTION_COSTS_Expression = pme.Expression(rule=production_costs_rules)

    def production_revenue_rules(model):
        fuel_opt=model.parent_block()
        return sum(fuel_opt.MARKETS[t, mkt].REVENUE[plant] for mkt in fuel_opt.LOCATIONS)
    b.PRODUCTION_REVENUES_Expression = pme.Expression(rule=production_revenue_rules)

    def production_EBITDA_rules(model):
        fuel_opt=model.parent_block()
        return model.PRODUCTION_REVENUES_Expression\
            - model.PRODUCTION_COSTS_Expression\
            - sum(fuel_opt.MARKETS[t, mkt].FREIGHT_COST[plant] for mkt in fuel_opt.LOCATIONS)
    b.PRODUCTION_EBITDA_Expression = pme.Expression(rule=production_EBITDA_rules)

    if annual:
        def production_DEP_rules(model):
            return (model.INITIAL_AVAILABLE*model.INITIAL_CAPEX_COSTS+model.INITIAL_EXPANSION*model.STRATEGIC_CAPEX_COSTS)/depreciation_years
        b.PRODUCTION_DEP_Expression = pme.Expression(rule=production_DEP_rules)
    else:
        def production_DEP_rules(model):
            return (model.INITIAL_AVAILABLE*model.INITIAL_CAPEX_COSTS+model.INITIAL_EXPANSION*model.STRATEGIC_CAPEX_COSTS)/depreciation_years/12
        b.PRODUCTION_DEP_Expression = pme.Expression(rule=production_DEP_rules)

    def production_EBIT_rules(model):
        return model.PRODUCTION_EBITDA_Expression - model.PRODUCTION_DEP_Expression
    b.PRODUCTION_EBIT_Expression = pme.Expression(rule=production_EBIT_rules)

    tax_charge = 0.35

    def production_NOPAT_rules(model):
        return model.PRODUCTION_EBIT_Expression*(1-tax_charge)
    b.PRODUCTION_NOPAT_Expression = pme.Expression(rule=production_NOPAT_rules)

    def production_FCF_rules(model):
        return model.PRODUCTION_NOPAT_Expression + model.PRODUCTION_DEP_Expression - model.CAPEX_COSTS_Expression
    b.PRODUCTION_FCF_Expression = pme.Expression(rule=production_FCF_rules)

    def production_TOTAL_COSTS_rules(model):
        return model.PRODUCTION_COSTS_Expression + model.CAPEX_COSTS_Expression
    b.PRODUCTION_TOTAL_COSTS_Expression = pme.Expression(rule=production_TOTAL_COSTS_rules)

    sum([pme.value(x) for x in b.component_data_objects(ctype=pme.Param)])

def get_fuel_opt_model(opt_periods, REFINERIES, LOCATIONS, DEMAND, PRICE, FREIGHT, PRODUCTION, PROJECTS, WACC, depreciation_years, minimum_production_months, step=None, pbar=None, annual=False, baseline=False):

    if pbar is not None:

        template = 'Step: {step}; Building optimization model'

        pbar.set_description(template.format(step=step))

    # Parent optimiztion object that will hold the whole model
    fuel_opt = pme.ConcreteModel()

    # Optimzation periods
    fuel_opt.T = pme.RangeSet(0,opt_periods-1)

    fuel_opt.REFINERIES = pme.Set(initialize=REFINERIES, name='Refineries', doc='Refineries')

    fuel_opt.LOCATIONS = pme.Set(initialize=LOCATIONS, name='Locations', doc='Locations')

    fuel_opt.MARKETS = pme.Block(fuel_opt.T, fuel_opt.LOCATIONS,
                        rule=lambda self, t, mkt: market_block_rule(self, t, mkt, DEMAND, PRICE, FREIGHT, step=step, pbar=pbar))

    def DEMAND_Expression_rule(model, t, plant):
        return sum(fuel_opt.MARKETS[t, mkt].L[plant] for mkt in fuel_opt.LOCATIONS)
    fuel_opt.DEMAND = pme.Expression(fuel_opt.T, fuel_opt.REFINERIES, rule=DEMAND_Expression_rule)

    fuel_opt.PLANTS = pme.Block(fuel_opt.T, fuel_opt.REFINERIES,
                        rule=lambda self, t, plant: plant_block_rule(self, t, plant, FREIGHT, PRODUCTION, PROJECTS, depreciation_years, minimum_production_months, step=step, pbar=pbar, annual=annual))

    if baseline:
        for plant in fuel_opt.REFINERIES:
            for t in fuel_opt.T:
                fuel_opt.PLANTS[t, plant].EXPANSION.fix(0)
                fuel_opt.PLANTS[t, plant].INITIAL_EXPANSION.fix(0)
                if PROJECTS.loc[(PROJECTS['loc']==plant)&(PROJECTS.type=='INITIAL')].capex.values[0] != 0:
                    fuel_opt.PLANTS[t, plant].AVAILABLE.fix(0)
                    fuel_opt.PLANTS[t, plant].INITIAL_AVAILABLE.fix(0)

    if pbar is not None:

        template = 'Step: {step}; Setting linking constraints'

        pbar.set_description(template.format(step=step))

    # inter-facilities linking constraints
    def L_INTER_PLANT_rule(model, t, source_plant, receiving_plant):
        if (source_plant == receiving_plant):
            return pme.Constraint.Skip
        elif 'Slack' in [source_plant]:
            return model.PLANTS[t, source_plant].OUT_INTER_L[receiving_plant] == 0
        elif 'Slack' in [receiving_plant]:
            return model.PLANTS[t, receiving_plant].IN_INTER_L[source_plant] == 0
        else:
            return model.PLANTS[t, source_plant].OUT_INTER_L[receiving_plant] == model.PLANTS[t, receiving_plant].IN_INTER_L[source_plant]
    fuel_opt.L_INTER_PLANT_linking = pme.Constraint(fuel_opt.T, fuel_opt.REFINERIES, fuel_opt.REFINERIES, rule=L_INTER_PLANT_rule)

    # inter-period linking constraints
    def L_INVENTORY_inter_period_rule(model, t, plant):
        if t == model.T.first():
            return model.PLANTS[t, plant].L_INITIAL_INVENTORY == 0
        else:
            return model.PLANTS[t, plant].L_INITIAL_INVENTORY == model.PLANTS[t-1, plant].L_INVENTORY
    fuel_opt.L_INVENTORY_linking = pme.Constraint(fuel_opt.T, fuel_opt.REFINERIES, rule=L_INVENTORY_inter_period_rule)    

    def STATE_inter_period_rule(model, t, plant):
        if t == model.T.first():
            model.PLANTS[t, plant].INITIAL_STATE.fix(0)
            return model.PLANTS[t, plant].INITIAL_STATE == 0
        else:
            return model.PLANTS[t, plant].INITIAL_STATE == model.PLANTS[t-1, plant].STATE
    fuel_opt.STATE_linking = pme.Constraint(fuel_opt.T, fuel_opt.REFINERIES, rule=STATE_inter_period_rule)

    def AVAILABLE_inter_period_rule(model, t, plant):
        if t == model.T.first():
            model.PLANTS[t, plant].INITIAL_AVAILABLE.fix(0)
            return model.PLANTS[t, plant].INITIAL_AVAILABLE == 0
        else:
            return model.PLANTS[t, plant].INITIAL_AVAILABLE == model.PLANTS[t-1, plant].AVAILABLE
    fuel_opt.AVAILABLE_linking = pme.Constraint(fuel_opt.T, fuel_opt.REFINERIES, rule=AVAILABLE_inter_period_rule)

    def EXPANSION_inter_period_rule(model, t, plant):
        if t == model.T.first():
            model.PLANTS[t, plant].INITIAL_EXPANSION.fix(0)
            return model.PLANTS[t, plant].INITIAL_EXPANSION == 0
        else:
            return model.PLANTS[t, plant].INITIAL_EXPANSION == model.PLANTS[t-1, plant].EXPANSION
    fuel_opt.EXPANSION_linking = pme.Constraint(fuel_opt.T, fuel_opt.REFINERIES, rule=EXPANSION_inter_period_rule)

    if pbar is not None:

        template = 'Step: {step}; Building objective'

        pbar.set_description(template.format(step=step))

    if annual:
        def MAX_FCF_optimization_rule(model):
            return sum(model.PLANTS[t, p].PRODUCTION_FCF_Expression/((1+WACC)**(t)) for t in model.T for p in model.REFINERIES)
        fuel_opt.OBJECTIVE = pme.Objective(rule=MAX_FCF_optimization_rule, sense=pme.maximize)
    else:
        def MAX_FCF_optimization_rule(model):
            return sum(model.PLANTS[t, p].PRODUCTION_FCF_Expression/((1+WACC/12)**(t)) for t in model.T for p in model.REFINERIES)
        fuel_opt.OBJECTIVE = pme.Objective(rule=MAX_FCF_optimization_rule, sense=pme.maximize)

    sum([pme.value(x) for x in fuel_opt.component_data_objects(ctype=pme.Param)])

    if pbar is not None:

        template = 'Step: {step}; Model built'

        pbar.set_description(template.format(step=step))

    return fuel_opt

def optimize_MIP_model(model, opt_timelimit, mip_tolerance, save_path, opt_version, step, pbar=None, verbose=True):

    xfrm_gdp = pme.TransformationFactory('gdp.hull')

    xfrm_gdp.apply_to(model)

    opt = pme.SolverFactory('glpk', executable='/usr/bin/glpsol')
    opt.options['tmlim'] = opt_timelimit # glpk setMaximumSeconds
    opt.options["mipgap"] = mip_tolerance

    solver_manager = pme.SolverManagerFactory("serial")
    if solver_manager is None:
        print("Failed to create solver manager.")
        sys.exit(1)
        
    if not os.path.exists(save_path+opt_version+'/LOGS/'):
        os.makedirs(save_path+opt_version+'/LOGS/')

    if pbar is not None:
    
        template = 'Step: {step}; Optimizing model'

        pbar.set_description(template.format(step=step))

    results = solver_manager.solve(model, opt=opt, tee=verbose, keepfiles=True, logfile=save_path+opt_version+'/LOGS/'+'_'.join(opt_version.split(' '))+'_'+str(step)+".log")

    log_file = open(save_path+opt_version+'/LOGS/'+'_'.join(opt_version.split(' '))+'_'+str(step)+'.log', "r")

    res = re.findall(r'\d\.\d*%', log_file.read())

    log_file.close()

    if pbar is not None:

        template = 'Step: {step}; Found optimal solution (mip gap={mip})'

        pbar.set_description(template.format(step=step, mip=res))

    try:

        last_mip_gap = float(res[-1][:-1])/100

    except(IndexError):

        last_mip_gap = -1

    return results, last_mip_gap

def retrieve_opt_info(model):

    market_stats = pd.concat(
        [
            # pd.DataFrame({'demand': {(t, mkt, 'ALL'): pme.value(model.MARKETS[t, mkt].DEMAND) for t in model.T for mkt in model.LOCATIONS}}),
            # pd.DataFrame({'price': {(t, mkt, 'ALL'): pme.value(model.MARKETS[t, mkt].PRICE) for t in model.T for mkt in model.LOCATIONS}}),
            pd.DataFrame({'sales': {(t, mkt, plant): pme.value(model.MARKETS[t, mkt].L[plant]) for t in model.T for mkt in model.LOCATIONS for plant in model.REFINERIES}}),
            pd.DataFrame({'freight': {(t, mkt, plant): pme.value(model.MARKETS[t, mkt].FREIGHT_COST[plant]) for t in model.T for mkt in model.LOCATIONS for plant in model.REFINERIES}}),
            pd.DataFrame({'revenue': {(t, mkt, plant): pme.value(model.MARKETS[t, mkt].REVENUE[plant]) for t in model.T for mkt in model.LOCATIONS for plant in model.REFINERIES}}),
            pd.DataFrame({'ebitda': {(t, mkt, plant): pme.value(model.MARKETS[t, mkt].EBITDA[plant]) for t in model.T for mkt in model.LOCATIONS for plant in model.REFINERIES}}),
        ], axis=1
        )

    demand = pd.concat(
        [
            pd.DataFrame({'demand': {(t, mkt, ): pme.value(model.MARKETS[t, mkt].DEMAND) for t in model.T for mkt in model.LOCATIONS}}),
            pd.DataFrame({'price': {(t, mkt): pme.value(model.MARKETS[t, mkt].PRICE) for t in model.T for mkt in model.LOCATIONS}}),
        ], axis=1
        )

    prod_stats = pd.concat(
        [
            pd.DataFrame({'state': {(t, plant): pme.value(model.PLANTS[t, plant].STATE) for t in model.T for plant in model.REFINERIES}}),
            pd.DataFrame({'available': {(t, plant): pme.value(model.PLANTS[t, plant].INITIAL_CAPEX) for t in model.T for plant in model.REFINERIES}}),
            pd.DataFrame({'expansion': {(t, plant): pme.value(model.PLANTS[t, plant].STRATEGIC_CAPEX) for t in model.T for plant in model.REFINERIES}}),
            pd.DataFrame({'capacity': {(t, plant): pme.value(model.PLANTS[t, plant].CAPACITY) for t in model.T for plant in model.REFINERIES}}),
            pd.DataFrame({'production': {(t, plant): pme.value(model.PLANTS[t, plant].L) for t in model.T for plant in model.REFINERIES}}),
            pd.DataFrame({'init_inventory': {(t, plant): pme.value(model.PLANTS[t, plant].L_INITIAL_INVENTORY) for t in model.T for plant in model.REFINERIES}}),
            pd.DataFrame({'inventory': {(t, plant): pme.value(model.PLANTS[t, plant].L_INVENTORY) for t in model.T for plant in model.REFINERIES}}),
            pd.DataFrame({'switch_cost': {(t, plant): pme.value(model.PLANTS[t, plant].SWITCH_COSTS_Expression) for t in model.T for plant in model.REFINERIES}}),
            pd.DataFrame({'layoff_cost': {(t, plant): pme.value(model.PLANTS[t, plant].LAYOFF_COSTS_Expression) for t in model.T for plant in model.REFINERIES}}),
            pd.DataFrame({'capex': {(t, plant): pme.value(model.PLANTS[t, plant].CAPEX_COSTS_Expression) for t in model.T for plant in model.REFINERIES}}),
            pd.DataFrame({'costs': {(t, plant): pme.value(model.PLANTS[t, plant].PRODUCTION_COSTS_Expression) for t in model.T for plant in model.REFINERIES}}),
            pd.DataFrame({'revenues': {(t, plant): pme.value(model.PLANTS[t, plant].PRODUCTION_REVENUES_Expression) for t in model.T for plant in model.REFINERIES}}),
            pd.DataFrame({'ebitda': {(t, plant): pme.value(model.PLANTS[t, plant].PRODUCTION_EBITDA_Expression) for t in model.T for plant in model.REFINERIES}}),
            pd.DataFrame({'depreciation': {(t, plant): pme.value(model.PLANTS[t, plant].PRODUCTION_DEP_Expression) for t in model.T for plant in model.REFINERIES}}),
            pd.DataFrame({'nopat': {(t, plant): pme.value(model.PLANTS[t, plant].PRODUCTION_NOPAT_Expression) for t in model.T for plant in model.REFINERIES}}),
            pd.DataFrame({'fcf_wo_nwc': {(t, plant): pme.value(model.PLANTS[t, plant].PRODUCTION_FCF_Expression) for t in model.T for plant in model.REFINERIES}}),
        ], axis=1
    ).assign(utilization=lambda r: r.production.div(r.capacity))

    inter_facilites = pd.concat(
        [
            pd.DataFrame({'receiving': {(t, plant, out_plant): pme.value(model.PLANTS[t, plant].IN_INTER_L[out_plant]) for t in model.T for plant in model.REFINERIES for out_plant in model.PLANTS[t, plant].INTER_PLANTS}}),
            pd.DataFrame({'freight_cost': {(t, plant, out_plant): pme.value(model.PLANTS[t, plant].IN_INTER_L[out_plant]*model.PLANTS[t, plant].FREIGHT[out_plant]) for t in model.T for plant in model.REFINERIES for out_plant in model.PLANTS[t, plant].INTER_PLANTS}}),
        ], axis=1
        )
    
    objective = pd.DataFrame({'objective': {0: pme.value(model.OBJECTIVE)}})

    return demand, market_stats, prod_stats, inter_facilites, objective

def load_opt_info(save_path, opt_version):

    objective_ls = []

    for s in os.listdir(os.path.join(save_path, opt_version, 'objective')):

        objective_ls.append(pd.read_csv(save_path+opt_version+f"/objective/{s}").assign(file=s))

    market_stats_ls = []

    for s in os.listdir(os.path.join(save_path, opt_version, 'market')):

        market_stats_ls.append(pd.read_csv(save_path+opt_version+f"/market/{s}").assign(file=s))
    
    demand_ls = []

    for s in os.listdir(os.path.join(save_path, opt_version, 'demand')):
        
        demand_ls.append(pd.read_csv(save_path+opt_version+f"/demand/{s}").assign(file=s))

    prod_stats_ls =  []
    
    for s in os.listdir(os.path.join(save_path, opt_version, 'production')):

        prod_stats_ls.append(pd.read_csv(save_path+opt_version+f"/production/{s}").assign(file=s))

    market_stats = pd.concat(market_stats_ls)

    demand = pd.concat(demand_ls)

    prod_stats = pd.concat(prod_stats_ls)

    objective = pd.concat(objective_ls)

    period_stats = pd.read_csv(save_path+opt_version+f"/ts_specs/stats.csv")

    return market_stats, demand, prod_stats, objective, period_stats

# END