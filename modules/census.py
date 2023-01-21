from dask import dataframe as dd
from pandas import read_csv as pdread_csv, read_excel as pdread_excel, DataFrame, concat, merge
from numpy import nan, inf
from requests import get
from zipfile import ZipFile
from io import BytesIO
from tqdm import tqdm
from pathlib import Path
from unidecode import unidecode
import os
import requests
import zipfile, io
from pathlib import Path
import shutil
import geopandas as gpd

CENSUS_PATH = "local/"

### Censo poblacional

# Descripción 2020: https://www.inegi.org.mx/contenidos/programas/ccpv/2020/doc/fd_agebmza_urbana_cpv2020.pdf
# Descripción 2010: https://www.inegi.org.mx/contenidos/programas/ccpv/2010/doc/fd_agebmza_urbana.pdf

estados_inegi_2010 = {
    "01": "aguascalientes",
    "02": "baja_california",
    "03": "baja_california_sur",
    "04": "campeche",
    "05": "coahuila",
    "06": "colima",
    "07": "chiapas",
    "08": "chihuahua",
    "09": "distrito_federal",
    "10": "durango",
    "11": "guanajuato",
    "12": "guerrero",
    "13": "hidalgo",
    "14": "jalisco",
    "15": "mexico",
    "16": "michoacan",
    "17": "morelos",
    "18": "nayarit",
    "19": "nuevo_leon",
    "20": "oaxaca",
    "21": "puebla",
    "22": "queretaro",
    "23": "quintana_roo",
    "24": "san_luis_potosi",
    "25": "sinaola",
    "26": "sonora",
    "27": "tabasco",
    "28": "tamaulipas",
    "29": "tlaxcala",
    "30": "veracruz",
    "31": "yucatan",
    "32": "zacatecas",
}

files = {
    2020: "https://www.inegi.org.mx/contenidos/programas/ccpv/2020/microdatos/ageb_manzana/RESAGEBURB_{}_2020_csv.zip",
    2010: "https://www.inegi.org.mx/contenidos/programas/ccpv/2010/microdatos/iter/ageb_manzana/{}_{}_2010_ageb_manzana_urbana_xls.zip",
    2000: "https://www.inegi.org.mx/contenidos/programas/ccpv/2000/microdatos/iter/{}_{}_2000_iter_xls.zip",
}

zipped = {
    2020: "RESAGEBURB_{}CSV20.csv",
    2010: "RESAGEBURB_{}XLS10.xls",
    2000: "ITER_{}XLS00",
    # 2020: "iter_{}_cpv2020/conjunto_de_datos/conjunto_de_datos_iter_{}CSV20.csv"
    
}

def download_INEGI_files(url, zipped_file, period:str = None):
    # Censo poblacional 2020
    # URL = r"https://www.inegi.org.mx/contenidos/programas/ccpv/2020/microdatos/ageb_manzana/RESAGEBURB_{}_2020_csv.zip"
    # ZIP_FILENAME = "RESAGEBURB_{}CSV20.csv"

    # Censo econmomico 2019:
    # URL = r"https://www.inegi.org.mx/contenidos/programas/ce/2019/Datosabiertos/ce2019_{}_csv.zip"
    # ZIP_FILENAME = "ce2019_{}.csv"

    # DATA
    # pop = DataFrame()

    for state in tqdm(range(1, 33)):
        if len(str(state)) == 1:
            str_state = "0" + str(state)
        else:
            str_state = str(state)

        try:
            r = get(url.format(str_state))
        except (IndexError):
            r = get(url.format(str_state, estados_inegi_2010[str_state]))

        zf = ZipFile(BytesIO(r.content))
        # print(zf.filelist)
        if "csv" in url:
            try:
                pop = pdread_csv(zf.open(zipped_file.format(str_state)))
            except (IndexError):
                pop = pdread_csv(
                        zf.open(
                            zipped_file.format(str_state, estados_inegi_2010[str_state])
                        )
                    )
        elif "xls" in url:
            try:
                pop = pdread_excel(zf.open(zipped_file.format(str_state)))
            except (IndexError):
                pop = pdread_excel(
                        zf.open(
                            zipped_file.format(str_state, estados_inegi_2010[str_state])
                        )
                    )
        else:
            raise NotImplementedError
        del r, zf

        for c in group_func_2020.keys():
            pop[c] = pop[c].replace(["*", "N/D", "N/A"], 0).fillna(0).astype(float)

        for c in group_cols_2020:
            pop[c] = pop[c].fillna("N/A").replace(["*", "N/D"], "N/A")


        str_path = f"{CENSUS_PATH}INEGI/{period}"
        Path(str_path).mkdir(parents=True, exist_ok=True)
        pop.to_csv(f"{str_path}/censo{period}_{str_state}.csv")

group_cols_2020 = [
    "ENTIDAD",
    "NOM_ENT",
    "MUN",
    "NOM_MUN",
    "LOC",
    "NOM_LOC",
    "AGEB",
    "MZA",
]

special_calc_cols = {
    "REL_H_M": "mean",
    "PROM_HNV": "mean",
    "GRAPROES": "mean",
    "GRAPROES_F": "mean",
    "GRAPROES_M": "mean",
    "PROM_OCUP": "mean",
    "PRO_OCUP_C": "mean",
}

pop_2020_columns = [
    "POBTOT",
    "POBFEM",
    "POBMAS",
    "AGEB",
    "P_0A2",
    "P_0A2_F",
    "P_0A2_M",
    "P_3YMAS",
    "P_3YMAS_F",
    "P_3YMAS_M",
    "P_5YMAS",
    "P_5YMAS_F",
    "P_5YMAS_M",
    "P_12YMAS",
    "P_12YMAS_F",
    "P_12YMAS_M",
    "P_15YMAS",
    "P_15YMAS_F",
    "P_15YMAS_M",
    "P_18YMAS",
    "P_18YMAS_F",
    "P_18YMAS_M",
    "P_3A5",
    "P_3A5_F",
    "P_3A5_M",
    "P_6A11",
    "P_6A11_F",
    "P_6A11_M",
    "P_8A14",
    "P_8A14_F",
    "P_8A14_M",
    "P_12A14",
    "P_12A14_F",
    "P_12A14_M",
    "P_15A17",
    "P_15A17_F",
    "P_15A17_M",
    "P_18A24",
    "P_18A24_F",
    "P_18A24_M",
    "P_15A49_F",
    "P_60YMAS",
    "P_60YMAS_F",
    "P_60YMAS_M",
    "REL_H_M",
    "POB0_14",
    "POB15_64",
    "POB65_MAS",
    "PROM_HNV",
    "PNACENT",
    "PNACENT_F",
    "PNACENT_M",
    "PNACOE",
    "PNACOE_F",
    "PNACOE_M",
    "PRES2015",
    "PRES2015_F",
    "PRES2015_M",
    "PRESOE15",
    "PRESOE15_F",
    "PRESOE15_M",
    "P3YM_HLI",
    "P3YM_HLI_F",
    "P3YM_HLI_M",
    "P3HLINHE",
    "P3HLINHE_F",
    "P3HLINHE_M",
    "P3HLI_HE",
    "P3HLI_HE_F",
    "P3HLI_HE_M",
    "P5_HLI",
    "P5_HLI_NHE",
    "P5_HLI_HE",
    "PHOG_IND",
    "POB_AFRO",
    "POB_AFRO_F",
    "POB_AFRO_M",
    "PCON_DISC",
    "PCDISC_MOT",
    "PCDISC_VIS",
    "PCDISC_LENG",
    "PCDISC_AUD",
    "PCDISC_MOT2",
    "PCDISC_MEN",
    "PCON_LIMI",
    "PCLIM_CSB",
    "PCLIM_VIS",
    "PCLIM_HACO",
    "PCLIM_OAUD",
    "PCLIM_MOT2",
    "PCLIM_RE_CO",
    "PCLIM_PMEN",
    "PSIND_LIM",
    "P3A5_NOA",
    "P3A5_NOA_F",
    "P3A5_NOA_M",
    "P6A11_NOA",
    "P6A11_NOAF",
    "P6A11_NOAM",
    "P12A14NOA",
    "P12A14NOAF",
    "P12A14NOAM",
    "P15A17A",
    "P15A17A_F",
    "P15A17A_M",
    "P18A24A",
    "P18A24A_F",
    "P18A24A_M",
    "P8A14AN",
    "P8A14AN_F",
    "P8A14AN_M",
    "P15YM_AN",
    "P15YM_AN_F",
    "P15YM_AN_M",
    "P15YM_SE",
    "P15YM_SE_F",
    "P15YM_SE_M",
    "P15PRI_IN",
    "P15PRI_INF",
    "P15PRI_INM",
    "P15PRI_CO",
    "P15PRI_COF",
    "P15PRI_COM",
    "P15SEC_IN",
    "P15SEC_INF",
    "P15SEC_INM",
    "P15SEC_CO",
    "P15SEC_COF",
    "P15SEC_COM",
    "P18YM_PB",
    "P18YM_PB_F",
    "P18YM_PB_M",
    "GRAPROES",
    "GRAPROES_F",
    "GRAPROES_M",
    "PEA",
    "PEA_F",
    "PEA_M",
    "PE_INAC",
    "PE_INAC_F",
    "PE_INAC_M",
    "POCUPADA",
    "POCUPADA_F",
    "POCUPADA_M",
    "PDESOCUP",
    "PDESOCUP_F",
    "PDESOCUP_M",
    "PSINDER",
    "PDER_SS",
    "PDER_IMSS",
    "PDER_ISTE",
    "PDER_ISTEE",
    "PAFIL_PDOM",
    "PDER_SEGP",
    "PDER_IMSSB",
    "PAFIL_IPRIV",
    "PAFIL_OTRAI",
    "P12YM_SOLT",
    "P12YM_CASA",
    "P12YM_SEPA",
    "PCATOLICA",
    "PRO_CRIEVA",
    "POTRAS_REL",
    "PSIN_RELIG",
    "TOTHOG",
    "HOGJEF_F",
    "HOGJEF_M",
    "POBHOG",
    "PHOGJEF_F",
    "PHOGJEF_M",
    "VIVTOT",
    "TVIVHAB",
    "TVIVPAR",
    "VIVPAR_HAB",
    "VIVPARH_CV",
    "TVIVPARHAB",
    "VIVPAR_DES",
    "VIVPAR_UT",
    "OCUPVIVPAR",
    "PROM_OCUP",
    "PRO_OCUP_C",
    "VPH_PISODT",
    "VPH_PISOTI",
    "VPH_1DOR",
    "VPH_2YMASD",
    "VPH_1CUART",
    "VPH_2CUART",
    "VPH_3YMASC",
    "VPH_C_ELEC",
    "VPH_S_ELEC",
    "VPH_AGUADV",
    "VPH_AEASP",
    "VPH_AGUAFV",
    "VPH_TINACO",
    "VPH_CISTER",
    "VPH_EXCSA",
    "VPH_LETR",
    "VPH_DRENAJ",
    "VPH_NODREN",
    "VPH_C_SERV",
    "VPH_NDEAED",
    "VPH_DSADMA",
    "VPH_NDACMM",
    "VPH_SNBIEN",
    "VPH_REFRI",
    "VPH_LAVAD",
    "VPH_HMICRO",
    "VPH_AUTOM",
    "VPH_MOTO",
    "VPH_BICI",
    "VPH_RADIO",
    "VPH_TV",
    "VPH_PC",
    "VPH_TELEF",
    "VPH_CEL",
    "VPH_INTER",
    "VPH_STVP",
    "VPH_SPMVPI",
    "VPH_CVJ",
    "VPH_SINRTV",
    "VPH_SINLTC",
    "VPH_SINCINT",
    "VPH_SINTIC",
]

group_func_2020 = {c: sum for c in pop_2020_columns if c not in group_cols_2020}

group_func_2020.update(special_calc_cols)


def dask_read_INEGI_census(period, short=True):
    
    if short:
        data_index_cols = ['NOM_ENT', 'ENTIDAD', 'NOM_MUN', 'MUN', 'NOM_LOC', 'LOC', 'AGEB', 'MZA']
        short_cols = [
            "POB15_64",
            "POB65_MAS",
            "GRAPROES",
            "PROM_OCUP",
            "VIVPARH_CV",
            'POBTOT', 'P_18YMAS', 'P_60YMAS', 'PDER_SS',
            'TVIVPARHAB', 'VPH_C_SERV', 'VPH_S_ELEC', 'VPH_AGUAFV', 'VPH_NODREN', 'VPH_INTER', 'VPH_PISOTI', 'VPH_AUTOM'
            ]
        pop = dd.read_csv(
            f"{CENSUS_PATH}INEGI/{period}/censo{period}*.csv",
            encoding="utf-8",
            dtype={x: 'object' for x in pop_2020_columns},
            usecols=data_index_cols+short_cols,
            # blocksize=25e6,
        )
    else:
        pop = dd.read_csv(
            f"{CENSUS_PATH}INEGI/{period}/censo{period}*.csv",
            encoding="utf-8",
            dtype={x: 'object' for x in pop_2020_columns},
            # blocksize=25e6,
        )

    pop = pop.drop(columns=[c for c in pop.columns if 'Unnamed' in c])

    for c in group_func_2020.keys():
        if c in pop.columns:
            pop[c] = pop[c].replace(["*", "N/D", "N/A"], 0).astype(float)

    for c in group_cols_2020:
        if c in pop.columns:
            pop[c] = pop[c].fillna("N/A").replace(["*", "N/D"], "N/A")

    return pop


def clean_INEGI_census_2020(pop):

    for c in group_func_2020.keys():
        pop[c] = pop[c].replace(["*", "N/D", "N/A"], 0).astype(float)

    for c in group_cols_2020:
        pop[c] = pop[c].fillna("N/A").replace(["*", "N/D"], "N/A")

    return pop


def read_INEGI_census(file_name, clean_pop=True):
    pop = pdread_csv(
        f"{CENSUS_PATH}/INEGI/{file_name}.csv", encoding="utf-8", index_col=0, header=0
    ).reset_index(drop=True)
    if clean_pop:
        pop = clean_INEGI_census_2020(pop=pop)
    return pop


# Agregate state data
def aggregate_2020_state_data(edo_i, pop=None, clean_pop=False):

    if pop is None:
        pop = read_INEGI_census(file_name="censo2020")
        clean_pop = False

    pop_shrt = pop.loc[(pop["ENTIDAD"] == edo_i)]

    if clean_pop:
        pop_shrt = clean_INEGI_census_2020(pop=pop_shrt)

    # print(pop_shrt.loc[pop_shrt.NOM_LOC=='Total de la entidad'].POBTOT.sum())

    # The summarized AGEB must equal sum(MZA)
    agg_var = (
        pop_shrt.loc[
            (pop["MZA"] != 0)
            & (pop_shrt["AGEB"] != "0000")
            & (pop_shrt["LOC"] != 0)
            & (pop_shrt["MUN"] != 0)
            & (pop_shrt["ENTIDAD"] == edo_i)
        ]
        .groupby(["ENTIDAD", "MUN", "LOC", "AGEB"])
        .agg(group_func_2020)
    )

    det_var = (
        pop_shrt.loc[
            (pop_shrt.NOM_LOC == "Total AGEB urbana") & (pop_shrt["ENTIDAD"] == edo_i)
        ]
        .groupby(["ENTIDAD", "MUN", "LOC", "AGEB"])
        .agg(group_func_2020)
    )

    norm_pop_shrt = (
        pop_shrt.loc[
            (pop_shrt["MZA"] != 0)
            & (pop_shrt["AGEB"] != "0000")
            & (pop_shrt["LOC"] != 0)
            & (pop_shrt["MUN"] != 0)
            & (pop_shrt["ENTIDAD"] == edo_i)
        ]
        .set_index(["ENTIDAD", "MUN", "LOC", "AGEB", "MZA"])
        .drop(columns=["NOM_ENT", "NOM_MUN", "NOM_LOC"])
    )

    if not agg_var.empty:
        norm_pop_shrt = (
            norm_pop_shrt * (agg_var / det_var).fillna(1)
        ).reset_index()  # .groupby(['ENTIDAD', 'MUN', 'LOC']).agg(group_func_2020)
    else:
        norm_pop_shrt = (
            norm_pop_shrt.reset_index()
        )  # .groupby(['ENTIDAD', 'MUN', 'LOC']).agg(group_func_2020)

    delta = (
        norm_pop_shrt.groupby(["ENTIDAD", "MUN", "LOC", "AGEB"]).agg(group_func_2020)
        - det_var
    )
    n = norm_pop_shrt.groupby(["ENTIDAD", "MUN", "LOC", "AGEB"])["POBTOT"].count()
    for col in delta.columns:
        delta[col] /= n.rename({"POBTOT": col})

    norm_pop_shrt.set_index(["ENTIDAD", "MUN", "LOC", "AGEB", "MZA"], inplace=True)

    norm_pop_shrt -= delta

    norm_pop_shrt.reset_index(inplace=True)

    # print(norm_pop_shrt.POBTOT.sum())

    # The summarized LOC must equal sum(AGEB)
    agg_var = (
        norm_pop_shrt.loc[
            (norm_pop_shrt["MZA"] != 0)
            & (norm_pop_shrt["AGEB"] != "0000")
            & (norm_pop_shrt["LOC"] != 0)
            & (norm_pop_shrt["MUN"] != 0)
            & (norm_pop_shrt["ENTIDAD"] == edo_i)
        ]
        .groupby(["ENTIDAD", "MUN", "LOC"])
        .agg(group_func_2020)
    )

    det_var = (
        pop_shrt.loc[
            (pop_shrt.NOM_LOC == "Total de la localidad urbana")
            & (pop_shrt["ENTIDAD"] == edo_i)
        ]
        .groupby(["ENTIDAD", "MUN", "LOC"])
        .agg(group_func_2020)
    )

    norm_pop_shrt = (
        pop_shrt.loc[
            (pop_shrt["MZA"] != 0)
            & (pop_shrt["AGEB"] != "0000")
            & (pop_shrt["LOC"] != 0)
            & (pop_shrt["MUN"] != 0)
            & (pop_shrt["ENTIDAD"] == edo_i)
        ]
        .set_index(["ENTIDAD", "MUN", "LOC", "AGEB", "MZA"])
        .drop(columns=["NOM_ENT", "NOM_MUN", "NOM_LOC"])
    )

    if not agg_var.empty:
        norm_pop_shrt = (
            norm_pop_shrt * (agg_var / det_var).fillna(1)
        ).reset_index()  # .groupby(['ENTIDAD', 'MUN', 'LOC']).agg(group_func_2020)
    else:
        norm_pop_shrt = (
            norm_pop_shrt.reset_index()
        )  # .groupby(['ENTIDAD', 'MUN', 'LOC']).agg(group_func_2020)

    delta = (
        norm_pop_shrt.groupby(["ENTIDAD", "MUN", "LOC"]).agg(group_func_2020) - det_var
    )
    n = norm_pop_shrt.groupby(["ENTIDAD", "MUN", "LOC"])["POBTOT"].count()
    for col in delta.columns:
        delta[col] /= n.rename({"POBTOT": col})

    norm_pop_shrt.set_index(["ENTIDAD", "MUN", "LOC", "AGEB", "MZA"], inplace=True)

    norm_pop_shrt -= delta

    norm_pop_shrt.reset_index(inplace=True)

    # print(norm_pop_shrt.POBTOT.sum())

    # The summarized MUN must equal sum(LOC)
    agg_var = (
        norm_pop_shrt.loc[
            (norm_pop_shrt["MZA"] != 0)
            & (norm_pop_shrt["AGEB"] != "0000")
            & (norm_pop_shrt["LOC"] != 0)
            & (norm_pop_shrt["MUN"] != 0)
            & (norm_pop_shrt["ENTIDAD"] == edo_i)
        ]
        .groupby(["ENTIDAD", "MUN"])
        .agg(group_func_2020)
    )

    det_var = (
        pop_shrt.loc[
            (pop_shrt.NOM_LOC == "Total del municipio") & (pop_shrt["ENTIDAD"] == edo_i)
        ]
        .groupby(["ENTIDAD", "MUN"])
        .agg(group_func_2020)
    )

    norm_pop_shrt = (
        pop_shrt.loc[
            (pop_shrt["MZA"] != 0)
            & (pop_shrt["AGEB"] != "0000")
            & (pop_shrt["LOC"] != 0)
            & (pop_shrt["MUN"] != 0)
            & (pop_shrt["ENTIDAD"] == edo_i)
        ]
        .set_index(["ENTIDAD", "MUN", "LOC", "AGEB", "MZA"])
        .drop(columns=["NOM_ENT", "NOM_MUN", "NOM_LOC"])
    )

    if not agg_var.empty:
        norm_pop_shrt = (
            norm_pop_shrt * (agg_var / det_var).fillna(1)
        ).reset_index()  # .groupby(['ENTIDAD', 'MUN', 'LOC']).agg(group_func_2020)
    else:
        norm_pop_shrt = (
            norm_pop_shrt.reset_index()
        )  # .groupby(['ENTIDAD', 'MUN', 'LOC']).agg(group_func_2020)

    delta = norm_pop_shrt.groupby(["ENTIDAD", "MUN"]).agg(group_func_2020) - det_var
    n = norm_pop_shrt.groupby(["ENTIDAD", "MUN"])["POBTOT"].count()
    for col in delta.columns:
        delta[col] /= n.rename({"POBTOT": col})

    norm_pop_shrt.set_index(["ENTIDAD", "MUN", "LOC", "AGEB", "MZA"], inplace=True)

    norm_pop_shrt -= delta

    norm_pop_shrt.reset_index(inplace=True)

    # print(norm_pop_shrt.POBTOT.sum())

    # The summarized ENT must equal sum(MUN)
    agg_var = (
        norm_pop_shrt.loc[
            (norm_pop_shrt["MZA"] != 0)
            & (norm_pop_shrt["AGEB"] != "0000")
            & (norm_pop_shrt["LOC"] != 0)
            & (norm_pop_shrt["MUN"] != 0)
            & (norm_pop_shrt["ENTIDAD"] == edo_i)
        ]
        .groupby(["ENTIDAD"])
        .agg(group_func_2020)
    )

    det_var = (
        pop_shrt.loc[
            (pop_shrt.NOM_LOC == "Total de la entidad") & (pop_shrt["ENTIDAD"] == edo_i)
        ]
        .groupby(["ENTIDAD"])
        .agg(group_func_2020)
    )

    norm_pop_shrt = (
        pop_shrt.loc[
            (pop_shrt["MZA"] != 0)
            & (pop_shrt["AGEB"] != "0000")
            & (pop_shrt["LOC"] != 0)
            & (pop_shrt["MUN"] != 0)
            & (pop_shrt["ENTIDAD"] == edo_i)
        ]
        .set_index(["ENTIDAD", "MUN", "LOC", "AGEB", "MZA"])
        .drop(columns=["NOM_ENT", "NOM_MUN", "NOM_LOC"])
    )

    if not agg_var.empty:
        norm_pop_shrt = (
            norm_pop_shrt * (agg_var / det_var).fillna(1)
        ).reset_index()  # .groupby(['ENTIDAD', 'MUN', 'LOC']).agg(group_func_2020)
    else:
        norm_pop_shrt = (
            norm_pop_shrt.reset_index()
        )  # .groupby(['ENTIDAD', 'MUN', 'LOC']).agg(group_func_2020)

    delta = norm_pop_shrt.groupby(["ENTIDAD"]).agg(group_func_2020) - det_var
    n = norm_pop_shrt.groupby(["ENTIDAD"])["POBTOT"].count()
    for col in delta.columns:
        delta[col] /= n.rename({"POBTOT": col})

    norm_pop_shrt.set_index(["ENTIDAD", "MUN", "LOC", "AGEB", "MZA"], inplace=True)

    norm_pop_shrt -= delta

    norm_pop_shrt.reset_index(inplace=True)

    # print(norm_pop_shrt.POBTOT.sum())

    edo_nom = list(pop_shrt.NOM_ENT.unique())[0]
    norm_pop_shrt["NOM_ENT"] = edo_nom
    norm_pop_shrt["NOM_MUN"] = (
        norm_pop_shrt["MUN"]
        .map(pop_shrt.loc[:, ["MUN", "NOM_MUN"]].set_index("MUN")["NOM_MUN"].to_dict())
        .fillna("N/A")
    )
    norm_pop_shrt["NOM_LOC"] = (
        norm_pop_shrt.assign(LOC=lambda r: r.MUN.astype(str) + "-" + r.LOC.astype(str))[
            "LOC"
        ]
        .map(
            pop_shrt.loc[:, ["MUN", "LOC", "NOM_LOC"]]
            .assign(LOC=lambda r: r.MUN.astype(str) + "-" + r.LOC.astype(str))
            .set_index("LOC")["NOM_LOC"]
            .to_dict()
        )
        .fillna("N/A")
    )

    # norm_pop_shrt.groupby(['ENTIDAD']).agg(group_func_2020)
    assert (
        abs(
            norm_pop_shrt["POBTOT"].sum()
            / pop_shrt.loc[pop_shrt.NOM_LOC == "Total de la entidad"].POBTOT.sum()
            - 1
        )
        < 1e-2
    )

    return norm_pop_shrt


def aggregate_2020_states(file_name="censo2020", pop=None):

    if pop is None:
        pop = read_INEGI_census(file_name=file_name)

    for i in tqdm(range(1, 33)):
        s = aggregate_2020_state_data(i, pop=pop, clean_pop=False)
        s.to_csv(f"{CENSUS_PATH}/DET/{file_name}_{str(i)}.csv", encoding="utf-8")
        del s


# DEPRECATED
def construct_detailed_census(file_name, pop=None):

    if pop is None:
        pop = read_INEGI_census(file_name)

    group_cols = [
        "ENTIDAD",
        "NOM_ENT",
        "MUN",
        "NOM_MUN",
        "LOC",
        "NOM_LOC",
        "AGEB",
        "MZA",
    ]

    group_func = {c: sum for c in pop.columns if c not in group_cols}

    for c in group_func.keys():
        pop[c] = pop[c].replace(["*", "N/D", "N/A"], 0).astype(float)

    for c in group_cols:
        pop[c] = pop[c].fillna("N/A").replace(["*", "N/D"], "N/A")

    def aggregate_data(df, group_cols, agg_var, ref_var, det_var):
        pop_shrt = (
            df.loc[
                ~df[det_var].str.contains("Total") | df[ref_var].str.contains("Total")
            ]
            .groupby(group_cols, as_index=False)
            .agg(group_func)
        )

        pop_shrt["idx"] = 1
        pop_shrt.loc[pop_shrt[ref_var].str.contains("Total"), "idx"] = 0

        norm = pop_shrt.groupby([agg_var, "idx"]).agg(group_func).unstack()
        for c in group_func.keys():
            norm[(c, "R")] = norm[(c, 0)].div(norm[(c, 1)])
        norm = (
            norm[[(c, "R") for c in group_func.keys()]]
            .stack()
            .reset_index(["idx"])
            .drop(columns=["idx"])
        )

        for c in group_func.keys():
            pop_shrt[c + "_o"] = pop_shrt[c]
            pop_shrt[c] = pop_shrt.apply(lambda r: r[c] * norm[c][r.NOM_MUN], axis=1)
            pop_shrt.loc[pop_shrt.idx == 0, c] = pop_shrt.loc[
                pop_shrt.idx == 0, c + "_o"
            ]
            pop_shrt = pop_shrt.drop(columns=[c + "_o"])

        norm = pop_shrt.groupby([agg_var, "idx"]).agg(group_func).unstack()
        for c in group_func.keys():
            norm[(c, "R")] = norm[(c, 0)].div(norm[(c, 1)])

        pop_shrt = pop_shrt.drop(columns=["idx"])
        pop_shrt = pop_shrt.loc[~pop_shrt[ref_var].str.contains("Total")]

        return pop_shrt

    # Normalize LOC to MUN
    group_cols = ["ENTIDAD", "NOM_ENT", "MUN", "NOM_MUN", "LOC", "NOM_LOC"]
    agg_var = "NOM_MUN"
    ref_var = "NOM_LOC"
    det_var = "AGEB"

    pop_shrt = aggregate_data(pop, group_cols, agg_var, ref_var, det_var)

    # Normalize MUN to ENT
    group_cols = ["ENTIDAD", "NOM_ENT", "MUN", "NOM_MUN"]
    agg_var = "NOM_ENT"
    ref_var = "NOM_MUN"
    det_var = "NOM_LOC"

    pop_shrt = aggregate_data(pop_shrt, group_cols, agg_var, ref_var, det_var)

    pop_shrt.to_csv(f"{CENSUS_PATH}/MUN/{file_name}.csv")


def read_census(file_name):
    pop = DataFrame()
    for entry in tqdm(os.listdir(f"{CENSUS_PATH}/DET/")):
        if entry.endswith(".csv") & (file_name in entry):
            s = pdread_csv(
                f"{CENSUS_PATH}/DET/{entry}", encoding="utf-8", index_col=0, header=0
            )
            pop = pop.append(s)
    return pop.reset_index(drop=True)


### Localidades reportadas
### Change str to int fxn
def get_int(s):
    try:
        return int(s)
    except (ValueError, AttributeError):
        return nan


def read_localities():

    locs = pdread_excel("D:/DBs/LOCALITIES/LOCALIDADES_202106.xlsx")
    muns = pdread_excel("D:/DBs/LOCALITIES/MUNICIPIOS_20210701.xlsx")

    locs = locs.rename(
        columns={"CVE_ENT": "ENTIDAD", "CVE_MUN": "MUN", "CVE_LOC": "LOC"}
    ).drop(columns=["Estatus"])
    muns = muns.rename(
        columns={"CATALOG_KEY": "MUN", "MUNICIPIO": "NOM_MUN", "EFE_KEY": "ENTIDAD"}
    ).drop(columns=["ESTATUS"])

    ### Ent str to int
    locs["ENTIDAD"] = locs.ENTIDAD.apply(get_int)
    locs.dropna(subset=["ENTIDAD"], inplace=True)
    locs["ENTIDAD"] = locs["ENTIDAD"].astype(int)
    ### Mun str to int
    locs["MUN"] = locs.MUN.apply(get_int)
    locs.dropna(subset=["MUN"], inplace=True)
    locs["MUN"] = locs["MUN"].astype(int)
    ### Loc str to int
    locs["LOC"] = locs.LOC.apply(get_int)
    locs.dropna(subset=["LOC"], inplace=True)
    locs["LOC"] = locs["LOC"].astype(int)

    ### Ent str to int
    muns["ENTIDAD"] = muns.ENTIDAD.apply(get_int)
    muns.dropna(subset=["ENTIDAD"], inplace=True)
    muns["ENTIDAD"] = muns["ENTIDAD"].astype(int)
    ### Mun str to int
    muns["MUN"] = muns.MUN.apply(get_int)
    muns.dropna(subset=["MUN"], inplace=True)
    muns["MUN"] = muns["MUN"].astype(int)
    ### Loc str to int

    return muns, locs


### Censo economico


def download_censo_economico_INEGI_files():

    INEGI_states_map = {
        "01": "ags",
        "02": "bc",
        "03": "bcs",
        "04": "camp",
        "05": "coah",
        "06": "col",
        "07": "chis",
        "08": "chih",
        "09": "cdmx",
        "10": "dgo",
        "11": "gto",
        "12": "gro",
        "13": "hgo",
        "14": "jal",
        "15": "mex",
        "16": "mich",
        "17": "mor",
        "18": "nay",
        "19": "nl",
        "20": "oax",
        "21": "pue",
        "22": "qro",
        "23": "qroo",
        "24": "slp",
        "25": "sin",
        "26": "son",
        "27": "tab",
        "28": "tamps",
        "29": "tlax",
        "30": "ver",
        "31": "yuc",
        "32": "zac",
    }

    # Censo econmomico 2019:
    url = r"https://www.inegi.org.mx/contenidos/programas/ce/2019/Datosabiertos/ce2019_{}_csv.zip"
    zipped_file = "conjunto_de_datos/ce2019_{}.csv"
    file_name = "CENSO_ECONOMICO"

    # DATA
    pop = DataFrame()
    for state in tqdm(range(1, 33)):
        if len(str(state)) == 1:
            str_state = "0" + str(state)
        else:
            str_state = str(state)

        r = get(url.format(INEGI_states_map[str_state]))

        zf = ZipFile(BytesIO(r.content))
        db = pdread_csv(
            zf.open(zipped_file.format(INEGI_states_map[str_state])),
            index_col=False,
            dtype={"CODIGO": str},
        )

        db["ID_ESTRATO"] = db["ID_ESTRATO"].fillna(0).replace(" ", 0)
        db = db.loc[(db.ID_ESTRATO == 99) | (db.ID_ESTRATO == 0)]
        columns = ["ENTIDAD", "MUNICIPIO", "CODIGO"]
        db = db.groupby(columns, as_index=False).agg(
            {
                c: max
                for c in db.columns
                if (c not in columns) & (c not in ["ID_ESTRATO"])
            }
        )

        cat_Act = pdread_csv(
            zf.open("catalogos/tc_codigo_actividad.csv"),
            index_col=False,
            dtype={"CODIGO": str},
        )
        db = merge(
            left=db, right=cat_Act, how="left", left_on=["CODIGO"], right_on=["CODIGO"]
        )

        Path(f"{CENSUS_PATH}/ECON/2019").mkdir(parents=True, exist_ok=True)
        db.to_csv(f"{CENSUS_PATH}/ECON/2019/{file_name}_{str_state}.csv", encoding="utf-8")

        del r, zf


def read_economy():
    file_name = "CENSO_ECONOMICO"
    ce = pdread_csv(
        f"{CENSUS_PATH}/ECON/2019/{file_name}.csv", encoding="utf-8", index_col=0, header=0
    )
    return ce


def dask_read_INEGI_economy():

    file_name = "CENSO_ECONOMICO"

    df = dd.read_csv(
        f"{CENSUS_PATH}/ECON/2019/{file_name}*.csv",
        encoding="utf-8",
        # blocksize=25e6,
        low_memory=False,
        dtype={'CODIGO': 'object',
            'MUNICIPIO': 'object'}
    )
    df = df.drop(columns=[c for c in df.columns if 'Unnamed' in c])

    return df

### COORDS

# https://www.inegi.org.mx/app/biblioteca/ficha.html?upc=889463807469
# https://www.inegi.org.mx/temas/mg/#Descargas

def process_inegi_shp_request(r, head):
    shp_file_path = "conjunto_de_datos/"
    shp_files_headers = [head]
    shp_resources = ['.shp', '.shx', '.cpg', '.dbf', '.prj']
    if r is not None:
        zf = zipfile.ZipFile(io.BytesIO(r.content))

        # Delete current files in path because gpd reads the whole folder
        for root, _, files in os.walk(shp_file_path):
            for file in files:
                os.remove(os.path.join(root, file))

        GIS_HEAD = '/content/mexico_fuel_guided_project/local/GIS/'
        Path(f'{GIS_HEAD}{shp_file_path}/{head}').mkdir(parents=True, exist_ok=True)
        # Extract files
        for shp_file in shp_files_headers:
            for shp_ext in shp_resources:
                zf.extract(f'{shp_file_path}{shp_file}{shp_ext}', path=GIS_HEAD)
                shutil.move(f"{GIS_HEAD}{shp_file_path}{shp_file}{shp_ext}", f"{GIS_HEAD}{shp_file_path}{shp_file}/{shp_file}{shp_ext}")

    # Read path
    coords = gpd.read_file(f'/content/mexico_fuel_guided_project/local/GIS/{shp_file_path}{head}')

    coords = coords.to_crs("wgs84")

    coords['centroid'] = coords.geometry.centroid

    coords['lat'] = coords.centroid.apply(lambda r: r.x)
    coords['lon'] = coords.centroid.apply(lambda r: r.y)

    coords = coords.drop(columns=['geometry', 'centroid'])

    return coords


def download_coords(save_local=True):
    if os.path.exists('/content/mexico_fuel_guided_project/local/COORDS/')&os.path.isfile('/content/mexico_fuel_guided_project/local/COORDS/geoinfo.csv'):
        coords = pdread_csv('/content/mexico_fuel_guided_project/local/COORDS/geoinfo.csv', sep='|', encoding='cp1252')
    else:
        try:
            r = None
            coords = process_inegi_shp_request(r, '00a')
            loc_coords = process_inegi_shp_request(r, '00l')
            rur_coords = process_inegi_shp_request(r, '00lrp')
            mun_coords = process_inegi_shp_request(r, '00mun')
            state_coords = process_inegi_shp_request(r, '00ent')
        except:
            geo_zip_url = "https://www.inegi.org.mx/contenidos/productos/prod_serv/contenidos/espanol/bvinegi/productos/geografia/marcogeo/889463807469/mg_2020_integrado.zip"
            r = requests.get(geo_zip_url)
            coords = process_inegi_shp_request(r, '00a')
            loc_coords = process_inegi_shp_request(r, '00l')
            rur_coords = process_inegi_shp_request(r, '00lrp')
            mun_coords = process_inegi_shp_request(r, '00mun')
            state_coords = process_inegi_shp_request(r, '00ent')

        coords = concat(
            [coords, rur_coords.drop(columns=["NOMGEO", "PLANO", "CVE_MZA"]).assign(Ambito='Rural_puntual')], axis=0
        )

        coords = merge(
            left=loc_coords.rename(columns={"NOMGEO": "LOC"}).drop(columns=["CVEGEO"]),
            right=coords.rename(columns={"Ambito": "AMBITO"}),
            on=["CVE_ENT", "CVE_MUN", "CVE_LOC"],
            how="right", suffixes=["_LOC", ""]
        )

        coords = merge(
            left=mun_coords.rename(columns={"NOMGEO": "MUN"}).drop(columns=["CVEGEO"]),
            right=coords,
            on=["CVE_ENT", "CVE_MUN"],
            how="right", suffixes=["_MUN", ""]
        )

        coords = merge(
            left=state_coords.rename(columns={"NOMGEO": "STATE"}).drop(columns=["CVEGEO"]),
            right=coords,
            on=["CVE_ENT"],
            how="right", suffixes=["_STATE", ""]
        )

        coords = coords.rename(columns={"CVE_ENT": "state_key", "STATE": "state", "CVE_MUN": "mun_key", "MUN": "mun", "CVE_LOC": "loc_key", "LOC": "loc", "AMBITO_LOC": "loc_type", "CVEGEO": 'block', "CVE_AGEB": "block_key", "AMBITO": "block_type"})

        if save_local:
            Path("/content/mexico_fuel_guided_project/local/COORDS").mkdir(parents=True, exist_ok=True)
            coords.to_csv('/content/mexico_fuel_guided_project/local/COORDS/geoinfo.csv', index=False, sep='|', encoding='cp1252')


def dask_read_coords():

    df = dd.read_csv(
        '/content/mexico_fuel_guided_project/local/COORDS/geoinfo.csv',
        sep='|',
        encoding='cp1252',
    )
    df = df.drop(columns=[c for c in df.columns if 'Unnamed' in c])

    return df


def read_INEGI_data():
    
    ### DEMOGRAPHICS
    data_index_cols = ['NOM_ENT', 'ENTIDAD', 'NOM_MUN', 'MUN']
    cols2use = [
            "POB15_64",
            "POB65_MAS",
            "GRAPROES",
            "PROM_OCUP",
            "VIVPARH_CV",
            'POBTOT', 'P_18YMAS', 'P_60YMAS', 'PDER_SS',
            'TVIVPARHAB', 'VPH_C_SERV', 'VPH_S_ELEC', 'VPH_AGUAFV', 'VPH_NODREN', 'VPH_INTER', 'VPH_PISOTI', 'VPH_AUTOM'
            ]

    data = dask_read_INEGI_census(period='2020')

    data = data.where(lambda r: (r.MUN>0)&(r.LOC>0)&(r.MZA>0)&(r.AGEB != "0000")).dropna()
    # print(data.POBTOT.sum().compute())
    data = data.rename(columns={'NOM_ENT': 'state', 'ENTIDAD': 'state_key', 'NOM_MUN': 'mun', 'MUN': 'mun_key'})

    data = data.assign(
            POBESC=lambda r: r.POB15_64+r.POB65_MAS,
            GRAPROESxPOBESC=lambda r: r.GRAPROES*r.POBESC,
            PROM_OCUPxVIVPARH_CV = lambda r: r.PROM_OCUP*r.VIVPARH_CV,
            )

    data = data.assign(state_key=lambda r: r.state_key.astype(int).astype(str), mun_key=lambda r: r.mun_key.astype(int).astype(str))
    data = data.groupby(['state', 'state_key', 'mun', 'mun_key']).agg({
        'POBTOT': sum, 'P_18YMAS': sum, 'P_60YMAS': sum, 'POB15_64': sum, 'POB65_MAS': sum, 'PDER_SS': sum,
        'TVIVPARHAB': sum, 'VPH_C_SERV': sum, 'VPH_S_ELEC': sum, 'VPH_AGUAFV': sum, 'VPH_NODREN': sum, 'VPH_INTER': sum, 'VPH_PISOTI': sum, 'VPH_AUTOM': sum,
        'PROM_OCUPxVIVPARH_CV': sum, 'VIVPARH_CV': sum,
        'GRAPROESxPOBESC': sum, 'POBESC': sum,
        'MZA': 'count'
        })

    coords = dask_read_coords()
    coords = coords.assign(state_key=lambda r: r.state_key.astype(int).astype(str), mun_key=lambda r: r.mun_key.astype(int).astype(str))
    coords = coords.groupby(['state', 'state_key', 'mun', 'mun_key']).agg({'lat': 'mean', 'lon': 'mean'}).compute()
    #.set_index(['NOM_ENT', 'ENTIDAD', 'NOM_MUN', 'MUN', 'NOM_LOC', 'LOC', 'AGEB'])
    data = merge(
        left=data.compute(),
        right=coords,
        left_index=True,
        right_index=True,
        how='left'
    )
    # print(data.POBTOT.sum())

    data['GRAPROES']=data['GRAPROESxPOBESC']/data['POBESC']
    data['GRAPROES']=data['GRAPROES'].replace([-inf, inf], nan).fillna(0)

    data['PROM_OCUP']=data['PROM_OCUPxVIVPARH_CV']/data['VIVPARH_CV']
    data['PROM_OCUP']=data['PROM_OCUP'].replace([-inf, inf], nan).fillna(0)

    data = data.assign(
        PEA = lambda r: r['P_18YMAS']-r['P_60YMAS'], # POB >18,<60
        PNEA = lambda r: r['POBTOT']-r['PEA'], # POB <18,>60
        POBEST = lambda r: r['POB15_64']+r['POB65_MAS'], # POB QUE DEBE DE TENER ESTUDIOS
        CES = lambda r: r['PNEA']/r['PEA'], # CARGA ECONÓMICA SOCIAL
        SST = lambda r: r['PDER_SS']/r['POBTOT'],  # SEGURIDAD SOCIAL TOTAL
        # SSP = lambda r: (r['PDER_SS']-r['PAFIL_IPRIV'])/(r['POBTOT']-r['PAFIL_IPRIV']), # SEGURIDAD SOCIAL PÚBLICA
        VPH_S_SERV = lambda r: r['TVIVPARHAB']-r['VPH_C_SERV'], # VIV CON SERVICIOS DE ELECTRICIDAD, AGUA y DRENAJE
        VSB = lambda r: r['VPH_S_SERV']/r['TVIVPARHAB'], # VIV CON SERVICIOS DE ELECTRICIDAD, AGUA y DRENAJE
        VSE = lambda r: r['VPH_S_ELEC']/r['TVIVPARHAB'], # VIV SIN ELECTRICIDAD
        VSA = lambda r: r['VPH_AGUAFV']/r['TVIVPARHAB'], # VIV SIN AGUA
        VSD = lambda r: r['VPH_NODREN']/r['TVIVPARHAB'], # VIV SIN DRENAJE
        VPH_S_INTER = lambda r: r['TVIVPARHAB']-r['VPH_INTER'], # VIV SIN INTERNET
        VSI = lambda r: r['VPH_S_INTER']/r['TVIVPARHAB'], # VIV CON INTERNET
        PDT = lambda r: r['VPH_PISOTI']/r['TVIVPARHAB'], # VIV CON PISO DE TIERRA
        GES = lambda r: r['GRAPROES'], # PCT DE PERSONAS SIN ESTUDIO (MAX ESTUDIO = 19)
        DENS_HAB = lambda r: r['PROM_OCUP'], # IDX DENSIDAD CASA HABITACIÓN
        )
    
    ### ECONOMY
    act_eco = [
        'A111A',
        'A121A',
        'A131A',
        'A211A',
        'A221A',
        'A511A',
        'A700A',
        'A800A',
        'H000A',
        'H000B',
        'H000C',
        'H000D',
    ]

    name_eco = [
        'Producción bruta total',
        'Consumo intermedio',
        'Valor agregado censal bruto',
        'Inversión total',
        'Formación bruta de capital fijo',
        'Margen por reventa de mercancías',
        'Total de gastos',
        'Total de ingresos',
        'Personal dependiente de la razón social total',
        'Personal dependiente de la razón social, hombres',
        'Personal dependiente de la razón social, mujeres',
        'Horas trabajadas por personal dependiente de la razón social'
    ]
    name_eco = [unidecode(s.upper().replace(' ', '_')) for s in name_eco]

    act_eco_map = dict(zip(act_eco, name_eco))

    eco = dask_read_INEGI_economy()

    eco = eco.rename(columns=act_eco_map)
    eco = eco.rename(columns={'NOM_ENT': 'state', 'ENTIDAD': 'state_key', 'NOM_MUN': 'mun', 'MUNICIPIO': 'mun_key'})

    eco['ACTIVIDAD'] = eco['DESC_CODIGO'].str.upper()
    for c in name_eco:
        eco[c] = eco[c].astype(float)

    eco = eco.where((eco.mun_key!=' ')&(eco.CLASIFICADOR_CODIGO.str.contains('Rama|Subrama'))).dropna()
    eco = eco.assign(state_key=lambda r: r.state_key.astype(int).astype(str), mun_key=lambda r: r.mun_key.astype(int).astype(str))
    
    eco = eco.groupby(['state_key', 'mun_key', 'CLASIFICADOR_CODIGO', 'ACTIVIDAD']).agg({x: sum for x in name_eco})

    eco = eco.reset_index().compute()

    eco.loc[eco.CLASIFICADOR_CODIGO=='Rama', 'ACTIVIDAD'] = 'RAMA - ' + eco.loc[eco.CLASIFICADOR_CODIGO=='Rama', 'ACTIVIDAD']
    eco.loc[eco.CLASIFICADOR_CODIGO=='Subrama', 'ACTIVIDAD'] = 'SUBRAMA - ' + eco.loc[eco.CLASIFICADOR_CODIGO=='Subrama', 'ACTIVIDAD']

    eco = eco.drop(columns=['CLASIFICADOR_CODIGO'])
    
    eco = eco.set_index(['state_key', 'mun_key', 'ACTIVIDAD']).unstack(['ACTIVIDAD']).fillna(0)

    for cc in name_eco:
        eco[(cc, None)] = eco[[(cc, c) for c in eco[cc].columns if c.startswith('RAMA -')]].sum(axis=1)

    eco.columns = [' - '.join([x for x in c if not x is None]) for c in eco.columns]

    data = merge(
        left=data,
        right=eco,
        left_index=True,
        right_index=True,
        how='left'
    ).fillna(0)
    # print(data.POBTOT.sum())

    return data

# END
