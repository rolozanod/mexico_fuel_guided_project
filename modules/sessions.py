from modules import data

class Session1(object):
    def __init__(self):
        self.session = '1'
        
        self.get_coords = data.get_coords

        self.get_prices = data.get_prices

class Session2(object):
    def __init__(self):
        self.session = '1'
        
        self.get_coords = data.get_coords

        self.get_prices = data.get_prices

        self.get_pop = data.get_pop

        self.get_demand = data.get_demand

        self.calc_fpc = data.calc_fpc

        self.retrieve_daily_avg_prices = data.retrieve_daily_avg_prices

        self.get_fuel_dataframe = data.get_fuel_dataframe

        self.calc_consumption_data = data.calc_consumption_data

        self.create_fuel_dataframe = data.create_fuel_dataframe

        self.plot_monthly_data = data.plot_monthly_data



# END #