from modules import data
from modules import timeseries
from modules import opt

class Session1(object):
    def __init__(self):
        self.session = '1'
        
        self.get_coords = data.get_coords

        self.get_prices = data.get_prices

class Session2(object):
    def __init__(self):
        self.session = '2'
        
        self.get_coords = data.get_coords

        self.get_prices = data.get_prices

        self.get_pop = data.get_pop

        self.get_demand = data.get_demand

        self.calc_fpc = data.calc_fpc

        self.retrieve_daily_avg_prices = data.retrieve_daily_avg_prices

        self.get_price_dataframe = data.get_price_dataframe

        self.calc_consumption_data = data.calc_consumption_data

        self.create_fuel_dataframe = data.create_fuel_dataframe

        self.plot_monthly_data = data.plot_monthly_data

class Session3(object):
    def __init__(self):
        self.session = '3'
        
        self.get_coords = data.get_coords

        self.get_prices = data.get_prices

        self.get_pop = data.get_pop

        self.get_demand = data.get_demand

        self.calc_fpc = data.calc_fpc

        self.retrieve_daily_avg_prices = data.retrieve_daily_avg_prices

        self.get_price_dataframe = data.get_price_dataframe

        self.calc_consumption_data = data.calc_consumption_data

        self.create_fuel_dataframe = data.create_fuel_dataframe

        self.plot_monthly_data = data.plot_monthly_data

        self.WindowGenerator = timeseries.WindowGenerator

        self.get_timeseries = timeseries.get_timeseries

        self.EncoderModel = timeseries.EncoderModel

        self.DecoderModel = timeseries.DecoderModel

        self.vae_model = timeseries.vae_model

        self.get_models = timeseries.get_models

        self.train = timeseries.train

        self.run_backtest = timeseries.run_backtest

        self.save_weights = timeseries.save_weights

        self.load_weights = timeseries.load_weights

        self.test_df = timeseries.test_df

        self.forecast = timeseries.forecast

class Session4(object):
    def __init__(self):
        self.session = '4'
        
        self.get_coords = data.get_coords

        self.get_prices = data.get_prices

        self.get_pop = data.get_pop

        self.get_demand = data.get_demand

        self.calc_fpc = data.calc_fpc

        self.retrieve_daily_avg_prices = data.retrieve_daily_avg_prices

        self.get_price_dataframe = data.get_price_dataframe

        self.calc_consumption_data = data.calc_consumption_data

        self.create_fuel_dataframe = data.create_fuel_dataframe

        self.plot_monthly_data = data.plot_monthly_data

        self.WindowGenerator = timeseries.WindowGenerator

        self.get_timeseries = timeseries.get_timeseries

        self.EncoderModel = timeseries.EncoderModel

        self.DecoderModel = timeseries.DecoderModel

        self.vae_model = timeseries.vae_model

        self.get_models = timeseries.get_models

        self.train = timeseries.train

        self.run_backtest = timeseries.run_backtest

        self.save_weights = timeseries.save_weights

        self.load_weights = timeseries.load_weights

        self.test_df = timeseries.test_df

        self.forecast = timeseries.forecast

        self.monthly_data = opt.monthly_data

        self.get_refinery_data = opt.get_refinery_data

        self.get_distances = opt.get_distances

        self.fix_freight = opt.fix_freight

        self.market_block_rule = opt.market_block_rule

        self.plant_block_rule = opt.plant_block_rule

        self.get_fuel_opt_model = opt.get_fuel_opt_model

        self.optimize_MIP_model = opt.optimize_MIP_model

class Session5(object):
    def __init__(self):
        self.session = '5'
        
        self.get_coords = data.get_coords

        self.get_prices = data.get_prices

        self.get_pop = data.get_pop

        self.get_demand = data.get_demand

        self.calc_fpc = data.calc_fpc

        self.retrieve_daily_avg_prices = data.retrieve_daily_avg_prices

        self.get_price_dataframe = data.get_price_dataframe

        self.calc_consumption_data = data.calc_consumption_data

        self.create_fuel_dataframe = data.create_fuel_dataframe

        self.plot_monthly_data = data.plot_monthly_data

        self.WindowGenerator = timeseries.WindowGenerator

        self.get_timeseries = timeseries.get_timeseries

        self.EncoderModel = timeseries.EncoderModel

        self.DecoderModel = timeseries.DecoderModel

        self.vae_model = timeseries.vae_model

        self.get_models = timeseries.get_models

        self.train = timeseries.train

        self.run_backtest = timeseries.run_backtest

        self.save_weights = timeseries.save_weights

        self.load_weights = timeseries.load_weights

        self.test_df = timeseries.test_df

        self.forecast = timeseries.forecast

        self.monthly_data = opt.monthly_data

        self.get_refinery_data = opt.get_refinery_data

        self.get_distances = opt.get_distances

        self.fix_freight = opt.fix_freight

        self.market_block_rule = opt.market_block_rule

        self.plant_block_rule = opt.plant_block_rule

        self.get_fuel_opt_model = opt.get_fuel_opt_model

        self.optimize_MIP_model = opt.optimize_MIP_model

class ProjectSessions(object):
    def __init__(self):
        self.session = 'A'
        
        self.get_coords = data.get_coords

        self.get_prices = data.get_prices

        self.get_pop = data.get_pop

        self.get_demand = data.get_demand

        self.calc_fpc = data.calc_fpc

        self.retrieve_daily_avg_prices = data.retrieve_daily_avg_prices

        self.get_price_dataframe = data.get_price_dataframe

        self.calc_consumption_data = data.calc_consumption_data

        self.create_fuel_dataframe = data.create_fuel_dataframe

        self.plot_monthly_data = data.plot_monthly_data

        self.WindowGenerator = timeseries.WindowGenerator

        self.get_timeseries = timeseries.get_timeseries

        self.EncoderModel = timeseries.EncoderModel

        self.DecoderModel = timeseries.DecoderModel

        self.vae_model = timeseries.vae_model

        self.get_models = timeseries.get_models

        self.train = timeseries.train

        self.run_backtest = timeseries.run_backtest

        self.save_weights = timeseries.save_weights

        self.load_weights = timeseries.load_weights

        self.test_df = timeseries.test_df

        self.forecast = timeseries.forecast

        self.monthly_data = opt.monthly_data

        self.get_refinery_data = opt.get_refinery_data

        self.get_distances = opt.get_distances

        self.fix_freight = opt.fix_freight

        self.market_block_rule = opt.market_block_rule

        self.plant_block_rule = opt.plant_block_rule

        self.get_fuel_opt_model = opt.get_fuel_opt_model

        self.optimize_MIP_model = opt.optimize_MIP_model

        self.retrieve_opt_info = opt.retrieve_opt_info

        self.production_timeseries = opt.production_timeseries

        


# END #