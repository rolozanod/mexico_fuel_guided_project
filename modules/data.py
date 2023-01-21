import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression

from statsmodels.formula.api import ols

import matplotlib.pyplot as plt

from modules import census, businesses

month_map = {
    'ENE': 1,
    'FEB': 2,
    'MAR': 3,
    'ABR': 4,
    'MAY': 5,
    'JUN': 6,
    'JUL': 7,
    'AGO': 8,
    'SEP': 9,
    'OCT': 10,
    'NOV': 11,
    'DIC': 12
}

def download_project_data():

    census.download_INEGI_files(
        url=r"https://www.inegi.org.mx/contenidos/programas/ccpv/2020/microdatos/ageb_manzana/RESAGEBURB_{}_2020_csv.zip",
        zipped_file="RESAGEBURB_{}CSV20.csv",
        period="2020"
    )

    businesses.download_zip_files()

    census.download_censo_economico_INEGI_files()

    census.get_coords()

def get_prices():
    # SITE
    # 'http://www.cre.gob.mx/da/PreciosPromedioMensuales.csv'

    # URL
    prices_url = 'http://transparenciacre.westcentralus.cloudapp.azure.com/PNT/73/III/E/PL/Precios_promedio_diarios_y_mensuales_en_estaciones_de_servicio.xlsx'

    # DATA
    daily_prices = pd.read_excel(prices_url, sheet_name='Cuadro 1.1', skiprows=3, header=0, usecols="A:D", engine='openpyxl').rename(columns={"Fecha": "date"})
    regular_state_prices = pd.read_excel(prices_url, sheet_name='Cuadro 1.2', skiprows=3, header=[0,1], nrows=33, engine='openpyxl')
    premium_state_prices = pd.read_excel(prices_url, sheet_name='Cuadro 1.3', skiprows=3, header=[0,1], nrows=33, engine='openpyxl')
    diesel_state_prices = pd.read_excel(prices_url, sheet_name='Cuadro 1.4', skiprows=3, header=[0,1], nrows=33, engine='openpyxl')

    return daily_prices, regular_state_prices, premium_state_prices, diesel_state_prices

def get_pop():
    return census.read_INEGI_data().rename(columns={"POBTOT": "pop", "TVIVPARHAB": "hab_houses", "VPH_AUTOM": "cars_per_house"}).reset_index()

def get_demand():
    return pd.read_csv('demand.csv').rename(columns={'Estado': 'state'}).set_index('state').T.unstack().reset_index().rename(columns={'level_1': 'year', 0: 'MBBL'}).assign(year=lambda r: r['year'].astype(int), BBL=lambda r: r.MBBL*1e3)

def get_coords():

    coords = census.dask_read_coords()
    coords = coords.assign(state_key=lambda r: r.state_key.astype(int).astype(str), mun_key=lambda r: r.mun_key.astype(int).astype(str))
    coords = coords.groupby(['state', 'state_key', 'mun', 'mun_key']).agg({'lat': 'mean', 'lon': 'mean'}).compute()

    return coords

def calc_fpc(pop, demand, regress = True):

    state_data = pd.merge(
        left=demand.loc[(demand.year==2020)].drop(columns=['year']),
        right=pop.groupby('state', as_index=False).agg({'pop': sum, 'hab_houses': sum, 'cars_per_house': sum}),
        on='state',
    )

    if regress:

        inputs = ["pop", 'cars_per_house', 'hab_houses']
        response = ["BBL"]

        def build_features(df):
            X = df[inputs].values
            # X = np.concatenate((X, X**2, X**3), axis=1)
            return X

        X = build_features(state_data.dropna())
        y = state_data.dropna()[response].values

        reg = LinearRegression().fit(X, y)

        print(f'Score: {reg.score(X, y)}')

        X_p = build_features(state_data.loc[state_data.isnull().any(axis=1)])
        
        state_data.loc[state_data.isnull().any(axis=1), response] = reg.predict(X_p)

    state_data['consumption'] = state_data['BBL'].div(state_data['pop'])

    pop['MUN_FUEL'] = pop.apply(lambda r: state_data.set_index('state').consumption.to_dict()[r['state']]*r['pop'], axis=1)

    return pop

def retrieve_daily_avg_prices(df, daily_data, ftype):
    # Get the monthly state price for the fuel type (ftype)
    df0 = df.loc[df[('ENTIDAD', 'Unnamed: 0_level_1')]=='Nacional'].drop(columns=[('ENTIDAD', 'Unnamed: 0_level_1')]).T.reset_index().rename(columns={'level_0': 'year', 'level_1': 'month_str', 0: 'price'})

    # translate the month name
    df0['month'] = df0['month_str'].map(month_map)

    # get the daily national price for the fuel type
    daily = daily_data.set_index('date')[ftype]
    daily = daily_data[['date', ftype]]
    daily.loc[:, 'date'] = pd.to_datetime(daily.date, format='%d/%m/%Y').values
    
    # merge both dataframes to calculate the daily price for each state
    # we have the average month price for each state and the daily national price 
    # the national price set the curve of how the price of each state will vary over the month
    df0 = pd.merge(
        left=df0,
        right=daily,
        left_on=['year', 'month'],
        right_on=[daily.date.dt.year, daily.date.dt.month]
    )
    df0 = df0[['date', 'price', ftype]]
    df0['curve'] = df0['price'].div(df0[ftype])


    df1 = df.loc[df[('ENTIDAD', 'Unnamed: 0_level_1')]!='Nacional'].set_index(('ENTIDAD', 'Unnamed: 0_level_1')).T.reset_index().rename(columns={'level_0': 'year', 'level_1': 'month_str', 0: 'price'})
    df1['month'] = df1['month_str'].map(month_map)

    # merge the curve dataframe with the monthly state dataframe
    df0 = pd.merge(
        left=df0[['date', 'curve']],
        right=df1,
        left_on=[df0.date.dt.year, df0.date.dt.month],
        right_on=['year', 'month']
    ).set_index('date').drop(columns=['year', 'month_str', 'month'])

    # map the daily curve for each state
    for c in df0.columns:
        if c not in ['curve']:
            df0[c] = df0[c]*df0['curve']

    # map of fuel types
    translate_fuel = {'Gasolina Regular': 'regular_price', 'Gasolina Premium': 'premium_price', 'Diésel': 'diesel_price'}

    # reshape the dataframe and remove undisired columns
    df0 = df0.drop(columns=['curve']).T.unstack().reset_index().rename(columns={'level_1': 'state', 0: translate_fuel[ftype]}).set_index(['date', 'state'])

    return df0

def get_price_dataframe():
    daily_prices, regular_state_prices, premium_state_prices, diesel_state_prices = get_prices()

    regular = retrieve_daily_avg_prices(regular_state_prices, daily_prices, 'Gasolina Regular')
    premium = retrieve_daily_avg_prices(premium_state_prices, daily_prices, 'Gasolina Premium')
    diesel = retrieve_daily_avg_prices(diesel_state_prices, daily_prices, 'Diésel')

    fuel = pd.concat(
            (
                regular,
                premium,
                diesel
                ),
                axis=1
        ).mean(axis=1)

    fuel = fuel.reset_index().rename(columns={0: 'price'})

    return fuel

def calc_consumption_data(return_model=True, aggregated=True, min_share=0.1, keep_above=0.9, local=True):

    pop = get_pop()
    demand = get_demand()

    consumption_data = calc_fpc(pop, demand, regress = True)
    state_data_stats = consumption_data.groupby('state').MUN_FUEL.sum().div(pop.MUN_FUEL.sum())

    fuel_price = get_price_dataframe()

    fuel_monthly_state_price = fuel_price.groupby([fuel_price.date.dt.year, 'state']).agg({'price': 'mean'}).reset_index().rename(columns={'date': 'year'})

    fuel_monthly_state_consumption = pd.read_csv('demand.csv').rename(columns={'Estado': 'state'}).set_index('state').T

    synthetic_states = []

    # iterate over each state
    for stt in fuel_monthly_state_consumption.columns:
    # estimate if the state is missing
        if (fuel_monthly_state_consumption[stt].isnull().any()):
            # iterate each entry of the missing state
            for idx, _ in fuel_monthly_state_consumption[stt].iteritems():
                # if is nan fill it wiht the estimate
                if np.isnan(fuel_monthly_state_consumption.loc[idx, stt]):
                    # add the state to the synthetic_states record
                    if stt not in synthetic_states:
                        synthetic_states.append(stt)
                    # The state_data_stats has the percentage of consumption of each state
                    # We will fill the missing states data with this statistic using the known consumption for all the states
                    fuel_monthly_state_consumption.loc[idx, stt] = fuel_monthly_state_consumption.loc[idx,:].sum()*state_data_stats[stt]

    fuel_monthly_state_consumption = fuel_monthly_state_consumption.unstack().reset_index().rename(columns={'level_1': 'year', 0: 'MBBL'}).assign(BBL=lambda r: r['MBBL']*1e3).assign(year=lambda r: r['year'].astype(int))

    elasticity = pd.merge(
        left=fuel_monthly_state_consumption,
        right=fuel_monthly_state_price,
        on=['year', 'state']
    ).dropna()

    model = ols('BBL ~ C(state) * price', elasticity).fit()

    # Calculate the revenue for each location and estimate its participation on each state
    consumption_data = consumption_data.set_index(["state", "mun"])
    consumption_data["state_share"] = consumption_data.groupby(["state", "mun"]).agg({"MUN_FUEL": sum}).div(consumption_data.assign(MUN_FUEL=lambda r: r['MUN_FUEL']).groupby(["state"]).agg({"MUN_FUEL": sum})).MUN_FUEL

    # Calculate the treshhold of relevant locations
    # This is done using the Box and Whisker methodology -> Q(50) + 1.5 x (Q(75) - Q(25))
    # This is cliped over the 10% and 90% range so no locations are found under 10% revenue and locations avbove 90% are not grouped
    consumption_data = pd.merge(
        left=consumption_data,
        right=consumption_data.groupby(["state"]).agg({"state_share": lambda x: max(min([np.percentile(x, 50) + 1.5*(np.percentile(x, 75) - np.percentile(x, 25)), keep_above]), min_share)}).rename(columns={"state_share": 'cutoff'}),
        left_index=True,
        right_index=True
    )

    # create the aggregate binary variable, if the aggregate variable is one that location will be grouped into a single clusteres location
    consumption_data['aggregate'] = (consumption_data['state_share'] >= consumption_data['cutoff'])*1

    relevant_loc_map = (consumption_data.loc[consumption_data['aggregate']==1].groupby(["state"]).cumcount(ascending=True)+1)

    def robust_map(idx):
        if idx in relevant_loc_map.to_dict().keys():
            return relevant_loc_map.to_dict()[idx]
        else:
            return 0

    if aggregated:
        consumption_data['aggregate'] = consumption_data.index.map(robust_map)
        def group_wa(s):
            dropped = s.dropna()
            if dropped.empty:
                return np.nan
            else:
                return np.average(dropped, weights = consumption_data.reset_index().loc[dropped.index].MUN_FUEL)

        def loc_name(x):
            if len(x)!=1:
                return f'{consumption_data.reset_index().loc[x.index].state.unique()[0]}_aggregate({len(x)})'
            else:
                return x

        consumption_data = consumption_data.reset_index().groupby(['state', 'aggregate'], as_index=False).agg({'lat': group_wa, 'lon': group_wa, 'MUN_FUEL': sum, 'mun': loc_name}).drop(columns=['aggregate'])

    return consumption_data, model

def create_fuel_dataframe(min_share=0.1, keep_above=0.9, local=True):

    consumption_data, model = calc_consumption_data(return_model=True, aggregated=True, min_share=min_share, keep_above=keep_above, local=local)

    data = get_price_dataframe()

    # Add consumption estimate
    data['state_fuel_predict'] = model.predict(data)

    # Esteimate the fuel consumption average to calculate a curve of consumption over the month
    fuel_demand_avg = data.groupby([data.date.dt.year, 'state']).agg({'state_fuel_predict': 'mean'}).state_fuel_predict

    # Calculate the curve over the average of the month
    data['state_fuel_predict'] = data.set_index([data.date.dt.year, 'state']).state_fuel_predict.div(fuel_demand_avg).values

    data = pd.merge(
        left=consumption_data,
        right=data,
        on=['state'],
    ).drop_duplicates()

    data['MUN_FUEL'] *= data['state_fuel_predict']

    data = data.drop(columns=['state_fuel_predict']).rename(columns={'MUN_FUEL': 'bbl'})[['state', 'mun', 'lat', 'lon', 'date', 'bbl', 'price']]

    data['litres'] = data['bbl']*158.98730272810

    data = data.drop(columns=['bbl'])

    return data

def plot_monthly_data(data):
    
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(18, 12))

    m = data.groupby(data.date.dt.to_period("M")).agg({'price': 'mean'}).price
    # m.index = m.index.to_period('M')
    axes[0].set_title('Price')
    m.plot(ax=axes[0])

    m = data.groupby(data.date.dt.to_period("M")).agg({'litres': 'sum'}).litres
    # m.index = m.index.to_period('M')
    axes[1].set_title('Litres')
    m.plot(ax=axes[1])

    fig.show()

# END