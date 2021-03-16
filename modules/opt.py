import numpy as np
import pandas as pd
import pyomo.environ as pme
import pyomo.gdp as pmg
import pyomo
from pyomo.core.util import sum_product
import sys
import re

def market_block_rule(b, t, market, DEMAND, PRICE, FREIGHT):
    fuel_opt=b.parent_block()

    dict_demand = {None: DEMAND.loc[DEMAND['loc']==market].litres.values[0]}

    dict_price = {None: PRICE.loc[PRICE['loc']==market].price.values[0]}

    dict_freight = FREIGHT.set_index(['refinery']).loc[FREIGHT['loc']==market].freight.to_dict()

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

def plant_block_rule(b, t, plant, FREIGHT, PRODUCTION, PROJECTS, depreciation_years, minimum_production_period):
    fuel_opt=b.parent_block()

    b.INTER_PLANTS = pme.Set(initialize=[p for p in fuel_opt.REFINERIES if p not in [plant, 'Slack']], name='Inter facilities connection', doc='Inter facilities connection')

    dict_freight = FREIGHT.set_index(['refinery']).loc[FREIGHT['loc']==plant].freight.to_dict()

    dict_freight = {k: it for k, it in dict_freight.items() if k[0] in b.INTER_PLANTS}

    dict_freight = {k: 30+10*FREIGHT.freight.max() if np.isnan(it) else it for k, it in dict_freight.items()}

    refinery_df = PRODUCTION.loc[(PRODUCTION.refinery==plant)&(PRODUCTION.T==t)]

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
                            doc=f'{plant} production capacity')

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

    plant_projects = PROJECTS.loc[(PROJECTS.refinery==plant)&(PROJECTS.type=='INITIAL')]

    b.INITIAL_CAPEX_COSTS = pme.Param(
                    initialize={None: plant_projects.capex.values[0]},
                    default=0,
                    name=f'{plant} Inital CAPEX',
                    doc=f'{plant} Initial CAPEX')

    b.EXPANSION_CAPEX_COSTS = pme.Param(
                    initialize={None: plant_projects.capex.values[0]},
                    default=0,
                    name=f'{plant} Capacity expansion CAPEX',
                    doc=f'{plant} Capacity expansion CAPEX')

    b.EXPANSION_CAPEX_SIZE = pme.Param(
                    initialize={None: plant_projects.increment.values[0]},
                    default=0,
                    name=f'{plant} Capacity expansion',
                    doc=f'{plant} Capacity expansion')

    # define variables
    def L_Bounds_rule(model):
        return (0, model.CAPACITY) # This plant will sell at most its capacity
    b.L = pme.Var(bounds=L_Bounds_rule)

    def OUT_INTER_L_Bounds_rule(model, inter_plant):
        return (0, model.CAPACITY)
    b.OUT_INTER_L = pme.Var(b.INTER_PLANTS, bounds=OUT_INTER_L_Bounds_rule)

    def IN_INTER_L_Bounds_rule(model, inter_plant):
        return (0, model.CAPACITY)
    b.IN_INTER_L = pme.Var(b.INTER_PLANTS, bounds=IN_INTER_L_Bounds_rule)

    b.STATES = pme.Set(initialize=[0,1])
    
    b.INITIAL_CAPEX = pme.Var(domain=pme.Binary)

    b.EXPANSION_CAPEX = pme.Var(domain=pme.Binary)

    b.SWITCH = pme.Var(domain=pme.Binary)

    b.LAYOFF = pme.Var(domain=pme.Binary)

    b.INITIAL_STATE = pme.Var(domain=pme.Binary)

    b.STATE = pme.Var(domain=pme.Binary)

    b.INITIAL_AVAILABLE = pme.Var(domain=pme.Binary)

    b.AVAILABLE = pme.Var(domain=pme.Binary)

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
        return 0 <= model.CLK_INVENTORY
    b.L_INVENTORY_LIMITS_Constraint = pme.Constraint(rule=l_inventories_bottom_constraint_rule)

    def inventories_capacity_constraint_rule(model):
        return model.STATE*model.SAFETY_INVENTORY <= model.L_INVENTORY # ALREADY BOUNDED FROM TOP WITH INITIAL_INVENTORY BOUNDS
    b.INVENTORY_LIMITS_Constraint = pme.Constraint(fuel_opt.PROCESSES, rule=inventories_capacity_constraint_rule)

    def production_state_rule(model):
        return model.L <= model.STATE*model.CAPACITY
    b.PRODUCTION_CAPACITY_Constraint = pme.Constraint(rule=production_state_rule)

    def SWITCH_Constraint_rule(disjunct, a0, a1, s0, s1):

        b=disjunct.parent_block()
        fuel_opt=b.parent_block()

        disjunct.SWITCHConstraint = pme.Constraint()
        disjunct.LAYOFFConstraint = pme.Constraint()
        disjunct.INITIAL_CAPEXConstraint = pme.Constraint()

        if (a0==0)&(a1==0)&(s0==0)&(s1==0):
            disjunct.SWITCHConstraint = b.SWITCH == 0
            disjunct.LAYOFFConstraint = b.LAYOFF == 0
            disjunct.INITIAL_CAPEXConstraint = b.INITIAL_CAPEX == 0

        elif (a0==0)&(a1==1)&(s0==0)&(s1==1):
            disjunct.SWITCHConstraint = b.SWITCH == 1
            disjunct.LAYOFFConstraint = b.LAYOFF == 0
            disjunct.INITIAL_CAPEXConstraint = b.INITIAL_CAPEX == 1

        elif (a0==1)&(a1==1)&(s0==0)&(s1==0):
            disjunct.SWITCHConstraint = b.SWITCH == 0
            disjunct.LAYOFFConstraint = b.LAYOFF == 0
            disjunct.INITIAL_CAPEXConstraint = b.INITIAL_CAPEX == 0

        elif (a0==1)&(a1==1)&(s0==0)&(s1==1):
            disjunct.SWITCHConstraint = b.SWITCH == 1
            disjunct.LAYOFFConstraint = b.LAYOFF == 0
            disjunct.INITIAL_CAPEXConstraint = b.INITIAL_CAPEX == 0

        elif (a0==1)&(a1==1)&(s0==1)&(s1==0):
            disjunct.SWITCHConstraint = b.SWITCH == 0
            disjunct.LAYOFFConstraint = b.LAYOFF == 1
            disjunct.INITIAL_CAPEXConstraint = b.INITIAL_CAPEX == 0

        elif (a0==1)&(a1==1)&(s0==1)&(s1==1):
            disjunct.SWITCHConstraint = b.SWITCH == 0
            disjunct.LAYOFFConstraint = b.LAYOFF == 0
            disjunct.INITIAL_CAPEXConstraint = b.INITIAL_CAPEX == 0

        else:
            disjunct.SWITCHConstraint = b.SWITCH == 0
            disjunct.LAYOFFConstraint = b.LAYOFF == 0
            disjunct.INITIAL_CAPEXConstraint = b.INITIAL_CAPEX == 0

        def initial_available_coherence_constraint_rules(model):
            return b.INITIAL_AVAILABLE == a0
        disjunct.INITIAL_AVAILABLE_COHERENCE_Constraint = pme.Constraint(rule=initial_available_coherence_constraint_rules)

        def available_coherence_constraint_rules(model):
            return b.AVAILABLE == a1
        disjunct.AVAILABLE_COHERENCE_Constraint = pme.Constraint(rule=available_coherence_constraint_rules)

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
    b.SWITCH_GDP = pmg.Disjunct(b.STATES, b.STATES, b.STATES, b.STATES, rule=SWITCH_Constraint_rule)

    possible_states = [
        [0,0,0,0],
        [0,1,0,1],
        [1,1,0,0],
        [1,1,0,1],
        [1,1,1,0],
        [1,1,1,1]
    ]

    def BindSTATES_rule(model):
        return [model.SWITCH_GDP[s[0], s[1], s[2], s[3]] for s in possible_states]
    b.BindSTATES = pmg.Disjunction(rule=BindSTATES_rule)

    def SWITCH_COSTS_rules(model):
        return model.SWITCH_COSTS*model.SWITCH
    b.SWITCH_COSTS_Expression = pme.Expression(rule=SWITCH_COSTS_rules)

    def LAYOFF_COSTS_rules(model):
        return model.FIXED_COSTS*minimum_production_period*model.LAYOFF
    b.LAYOFF_COSTS_Expression = pme.Expression(rule=LAYOFF_COSTS_rules)

    def CAPEX_COSTS_rules(model):
        return model.INITIAL_CAPEX_COSTS*model.INITIAL_CAPEX
    b.CAPEX_COSTS_Expression = pme.Expression(rule=CAPEX_COSTS_rules)

    def production_costs_rules(model):
        return\
                model.SWITCH_COSTS_Expression + model.LAYOFF_COSTS_Expression\
                + model.STATE*model.FIXED_COSTS\
                + model.L*model.VARIABLE\
                + sum(model.IN_INTER_CLK[rp]*model.FREIGHT[rp] for rp in model.INTER_PLANTS)
    b.PRODUCTION_COSTS_Expression = pme.Expression(rule=production_costs_rules)

    def production_revenue_rules(model):
        fuel_opt=model.parent_block()
        return sum(fuel_opt.MARKETS[t, mkt].REVENUE[plant] for mkt in fuel_opt.LOCATIONS)
    b.PRODUCTION_REVENUES_Expression = pme.Expression(rule=production_revenue_rules)

    def production_EBITDA_rules(model):
        fuel_opt=model.parent_block()
        return model.PRODUCTION_REVENUES_Expression\
            - model.PRODUCTION_COSTS_Expression\
            - sum(fuel_opt.MARKETS[t, mkt].FREIGHT_COST[plant] for mkt in fuel_opt.DESTINATION_SET)
    b.PRODUCTION_EBITDA_Expression = pme.Expression(rule=production_EBITDA_rules)

    def production_DEP_rules(model):
        return model.INITIAL_CAPEX_COSTS/depreciation_years/12*model.INITIAL_AVAILABLE
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

def get_fuel_opt_model(opt_periods, REFINERIES, LOCATIONS, DEMAND, PRICE, FREIGHT, PRODUCTION, PROJECTS, WACC, depreciation_years, minimum_production_period):
    global pbar

    # Parent optimiztion object that will hold the whole model
    fuel_opt = pme.ConcreteModel()

    # Optimzation periods
    fuel_opt.T = pme.RangeSet(opt_periods)

    fuel_opt.REFINERIES = pme.Set(initialize=REFINERIES+['Slack'], name='Refineries', doc='Refineries')

    fuel_opt.LOCATIONS = pme.Set(initialize=LOCATIONS, name='Locations', doc='Locations')

    fuel_opt.MARKETS = pme.Block(fuel_opt.T, fuel_opt.LOCATIONS,
                        rule=lambda self, t, mkt: market_block_rule(self, t, mkt, DEMAND, PRICE, FREIGHT))

    def DEMAND_Expression_rule(model, t, plant):
        return sum(fuel_opt.MARKETS[t, mkt].L[plant] for mkt in fuel_opt.LOCATIONS)
    fuel_opt.DEMAND = pme.Expression(fuel_opt.T, fuel_opt.REFINERIES, rule=DEMAND_Expression_rule)

    fuel_opt.PLANTS = pme.Block(fuel_opt.T, fuel_opt.REFINERIES,
                        rule=lambda self, t, plant: plant_block_rule(self, t, plant, FREIGHT, PRODUCTION, PROJECTS, depreciation_years, minimum_production_period))

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
    fuel_opt.L_INTER_PLANT_linking = pme.Constraint(fuel_opt.T, fuel_opt.LOCATIONS, fuel_opt.LOCATIONS, rule=L_INTER_PLANT_rule)

    # inter-period linking constraints
    def L_INVENTORY_inter_period_rule(model, t, plant):
        if t == model.T.first():
            return model.PLANTS[t, plant].L_INITIAL_INVENTORY == 0
        else:
            return model.PLANTS[t, plant].L_INITIAL_INVENTORY == model.PLANTS[t-1, plant].L_INVENTORY
    fuel_opt.L_INVENTORY_linking = pme.Constraint(fuel_opt.T, fuel_opt.PLANTS_SET, rule=L_INVENTORY_inter_period_rule)    

    def STATE_inter_period_rule(model, t, plant):
        if t == model.T.first():
            model.PLANTS[t, plant].INITIAL_STATE.fix(0)
            return model.PLANTS[t, plant].INITIAL_STATE == 0
        else:
            return model.PLANTS[t, plant].INITIAL_STATE == model.PLANTS[t-1, plant].STATE
    fuel_opt.STATE_linking = pme.Constraint(fuel_opt.T, fuel_opt.LOCATIONS, rule=STATE_inter_period_rule)

    def AVAILABLE_inter_period_rule(model, t, plant):
        if t == model.T.first():
            model.PLANTS[t, plant].INITIAL_AVAILABLE.fix(0)
            return model.PLANTS[t, plant].INITIAL_AVAILABLE == 0
        else:
            return model.PLANTS[t, plant].INITIAL_AVAILABLE == model.PLANTS[t-1, plant].AVAILABLE
    fuel_opt.AVAILABLE_linking = pme.Constraint(fuel_opt.T, fuel_opt.LOCATIONS, rule=AVAILABLE_inter_period_rule)
    
    def MAX_FCF_optimization_rule(model):
        return sum(model.PLANTS[t, p].PRODUCTION_FCF_Expression/((1+WACC/12)**(t)) for t in model.T for p in model.PLANTS_SET)
    fuel_opt.OBJECTIVE = pme.Objective(rule=MAX_FCF_optimization_rule, sense=pme.maximize)

    return fuel_opt

def optimize_MIP_model(model, opt_timelimit, mip_tolerance, save_path, opt_version, step):

    xfrm_gdp = pme.TransformationFactory('gdp.hull')

    xfrm_gdp.apply_to(model)

    opt = pme.SolverFactory("glpk")
    opt.options['tmlim'] = opt_timelimit # glpk setMaximumSeconds
    opt.options["mipgap"] = mip_tolerance

    solver_manager = pme.SolverManagerFactory("serial")
    if solver_manager is None:
        print("Failed to create solver manager.")
        sys.exit(1)

    results = solver_manager.solve(model, opt=opt, tee=True, keepfiles=True, logfile=save_path+opt_version+'\\LOGS\\'+'_'.join(opt_version.split(' '))+'_'+str(step)+".log")

    log_file = open(save_path+opt_version+'\\LOGS\\'+'_'.join(opt_version.split(' '))+'_'+str(step)+'.log', "r")

    res = re.findall(r'\d\.\d*%', log_file.read())

    log_file.close()

    try:

        last_mip_gap = float(res[-1][:-1])/100

    except(IndexError):

        last_mip_gap = -1

    return results, last_mip_gap
