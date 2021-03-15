import os
import requests
import zipfile, io
from tqdm import tqdm
import geopandas as gpd
import re

import pandas as pd
import numpy as np

# https://www.inegi.org.mx/app/biblioteca/ficha.html?upc=889463807469
# https://www.inegi.org.mx/temas/mg/#Descargas

def process_inegi_shp_request(r, head):
    shp_file_path = "conjunto_de_datos/"
    shp_files_headers = [head]
    shp_resources = ['.shp', '.shx', '.cpg', '.dbf', '.prj']
    zf = zipfile.ZipFile(io.BytesIO(r.content))

    # Delete current files in path because gpd reads the whole folder
    for root, dirs, files in os.walk(shp_file_path):
        for file in files:
            os.remove(os.path.join(root, file))

    # Extract files
    for shp_file in shp_files_headers:
        for shp_ext in shp_resources:
            zf.extract(f'{shp_file_path}{shp_file}{shp_ext}', path='/content/')

    # Read path
    coords = gpd.read_file(shp_file_path)

    # Delete current files in path because gpd reads the whole folder
    for root, dirs, files in os.walk(shp_file_path):
        for file in files:
            os.remove(os.path.join(root, file))

    coords = coords.to_crs("wgs84")

    coords['centroid'] = coords.geometry.centroid

    def get_latlon(s):
        centroid_re = r"POINT \(([+-]\d+.\d+) (\d+.\d+)\)"

        lat = float(re.search(centroid_re, s).group(1))
        lon = float(re.search(centroid_re, s).group(2))

        return lat, lon

    coords['lat'] = coords.centroid.apply(lambda r: r.x)
    coords['lon'] = coords.centroid.apply(lambda r: r.y)

    coords = coords.drop(columns=['geometry', 'centroid'])

    return coords

def get_coords():
    try:
        coords = process_inegi_shp_request(r, '00mun')
        state_coords = process_inegi_shp_request(r, '00ent')
    except:
        geo_zip_url = "https://www.inegi.org.mx/contenidos/productos/prod_serv/contenidos/espanol/bvinegi/productos/geografia/marcogeo/889463807469/mg_2020_integrado.zip"
        r = requests.get(geo_zip_url)
        coords = process_inegi_shp_request(r, '00mun')
        state_coords = process_inegi_shp_request(r, '00ent')

    coords = pd.merge(
        left=state_coords[["CVE_ENT", "NOMGEO"]].rename(columns={"NOMGEO": "STATE"}),
        right=coords.rename(columns={"NOMGEO": "MUN"}),
        on=["CVE_ENT"]
    )

    return coords

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

# END