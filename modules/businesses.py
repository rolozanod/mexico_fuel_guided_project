from pandas import DataFrame, concat, read_csv, read_excel
from tqdm import tqdm
import re
from os import listdir, path
from requests import get
from zipfile import ZipFile
from io import BytesIO
from pathlib import Path
from dask import dataframe as dd

LOCAL_PATH = "local/"

DENUE_ACT = {
    "Construcción":"23",
    "Actividades legislativas, gubernamentales, de impartición de justicia y de organismos internacionales y extraterritoriales": "93",
    "Agricultura, cría y explotación de animales, aprovechamiento forestal, pesca y caza": "11",
    "Comercio al por mayor": "43",
    "Comercio al por menor (1 de 4)": "46111",
    "Comercio al por menor (2 de 4)": "46112-46311",
    "Comercio al por menor (3 de 4)": "46321-46531",
    "Comercio al por menor (4 de 4)": "46591-46911",
    "Corporativos": "55",
    "Información en medios masivos": "51",
    "Minería": "21",
    "Industrias manufactureras": "31-33",
    "Generación, transmisión y distribución de energía eléctrica, suministro de agua y de gas por ductos al consumidor final": "22",
    "Otros servicios excepto actividades gubernamentales (1 de 2)": "81_1",
    "Otros servicios excepto actividades gubernamentales (2 de 2)": "81_2",
    "Servicios de alojamiento temporal y de preparación de alimentos y bebidas (1 de 2)": "72_1",
    "Servicios de alojamiento temporal y de preparación de alimentos y bebidas (2 de 2)": "72_2",
    "Servicios de apoyo a los negocios y manejo de desechos y servicios de remediación": "56",
    "Servicios de esparcimiento culturales y deportivos, y otros servicios recreativos": "71",
    "Servicios de salud y de asistencia social": "62",
    "Servicios educativos": "61",
    "Servicios financieros y de seguros": "52",
    "Servicios inmobiliarios y de alquiler de bienes muebles e intangibles": "53",
    "Servicios profesionales, científicos y técnicos": "54",
    "Transportes, correos y almacenamiento": "48-49"
}

files_head = 'denue_inegi_'
def download_zip_files(sector:list = None, period:str = None):
    """
    sector: str
    period: str - {yyyy}_{mm}
    """

    url_tail = "_csv.zip"
    if period is None:
        url_header = "https://www.inegi.org.mx/contenidos/masiva/denue/denue_00_"
    else:
        url_header = f"https://www.inegi.org.mx/contenidos/masiva/denue/{period}/denue_00_"

    if sector is None:
        # download all sectors
        queue = [v for _, v in DENUE_ACT.items()]
    else:
        # download specific sector
        queue = sector

    pbar = tqdm(queue)
    for act in pbar:
        
        res = get(url = url_header + act + url_tail)

        zf = ZipFile(BytesIO(res.content))

        csv_files = list(filter(lambda f: f.startswith("conjunto_de_datos"), zf.namelist()))
        for fname in csv_files:
            df = read_csv(zf.open(fname), encoding='latin-1')

            if period is None:
                filepath = f"{LOCAL_PATH}DENUE/LATEST"
            else:
                filepath = f"{LOCAL_PATH}DENUE/{period}"

            csv_name = fname.split("conjunto_de_datos/")[1]

            pbar.set_description(desc=f"saving: {filepath}/{csv_name}")
            Path(filepath).mkdir(parents=True, exist_ok=True)
            df.to_csv(f"{filepath}/{csv_name}")


def get_scian(level: str = 'CLASE'):
    """
    level: str = [SECTOR, SUBSECTOR, RAMA, SUBRAMA, CLASE]
    """
    # scian = read_excel(path.join(db_path, 'scian_2018_categorias_y_productos.xlsx'), sheet_name=level, skiprows=1, index_col=[0,1,2], dtype={'Código': 'str'})
    # scian = scian.groupby(['Código', 'Título', 'Descripción']).agg({v: lambda r: ', '.join(list(set(r.dropna().values))) for v in scian.columns}).reset_index()
    # scian['Código'] = scian['Código'].astype(str)

    # def remove_T(s):
    #     if s[-1] == 'T':
    #         return s[:-1]
    #     else:
    #         return s

    # scian['Título'] = scian['Título'].str.strip().map(remove_T)

    return None


def get_denue(period:str = None):
    if period is None:
        filepath = f"{LOCAL_PATH}DENUE/LATEST"
    else:
        filepath = f"{LOCAL_PATH}DENUE/{period}"
    
    denue_db = [f for f in listdir(filepath) if f.startswith(files_head)]
    data = DataFrame()
    pbar = tqdm(denue_db)
    for f in pbar:
        pbar.set_description(f'denue file: {f}')
        d = read_csv(path.join(filepath, f), encoding='latin-1', dtype={'codigo_act': str, 'id': int}, low_memory=False)
        
        data = concat(
            [data, d], axis=0
        )
    
    for k, v in {'SECTOR':2, 'SUBSECTOR':3, 'RAMA':4, 'SUBRAMA':5, 'CLASE':6}.items():
        data[k] = data.codigo_act.str[:v].map(get_scian(k).set_index(['Código'])['Título'])

    tamaños = {p:max([int(x) for x in re.findall("\d+", p.replace('más', '500'))]) for p in data.per_ocu.unique()}

    data['personal'] = data.per_ocu.map(tamaños)

    return data


def dask_read_denue(period=None):
    if period is None:
        filepath = f"{LOCAL_PATH}DENUE/LATEST"
    else:
        filepath = f"{LOCAL_PATH}DENUE/{period}"

    data = dd.read_csv(
        f'{filepath}/denue_inegi_*.csv',
        encoding='latin-1',
        dtype={
            'codigo_act': str,
            'id': int,
            'ageb': str,
            'edificio': str,
            'edificio_e': str,
            'letra_int': str,
            'nom_CenCom': str,
            'num_local': str,
            'tipoCenCom': str
            }
        )

    for k, v in {'SECTOR':2, 'SUBSECTOR':3, 'RAMA':4, 'SUBRAMA':5, 'CLASE':6}.items():
        data[k] = data.codigo_act.str[:v].map(get_scian(k).set_index(['Código'])['Título'])

    tamaños = {p:max([int(x) for x in re.findall("\d+", p.replace('más', '500'))]) for p in data.per_ocu.unique()}

    data['personal'] = data.per_ocu.map(tamaños)

    return data


# END
