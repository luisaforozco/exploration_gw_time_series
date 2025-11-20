import re
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from io import StringIO


def get_bro_data(bro_GLD_id: str) -> pd.DataFrame:
    headers = {"accept": "text/plain"}
    response = requests.get(f"https://publiek.broservices.nl/gm/gld/v1/seriesAsCsv/{bro_GLD_id}", headers=headers)
    # Check error code from the request
    if response.status_code == 400:
        error_data = response.json()
        description = error_data.get('description', 'Unknown error')
        raise ValueError(f"Bad request seriesAsCsv for BRO ID {bro_GLD_id}: {description}")
    # Check if response body is empty or whitespace, TODO: decide if return an empty df or raise an error
    if not response.text.strip(): 
        raise ValueError(f"Er is geen data beschikbaar voor dit dossier {bro_GLD_id}. The response body is empty.")
    df = pd.read_csv(StringIO(response.text))
    return clean_csv(df)


def clean_csv(df: pd.DataFrame) -> pd.DataFrame:
    # remove columns whose name contains "Opmerking" or "Controle" (case-insensitive)
    mask = df.columns.str.contains(r'Opmerking|Controle', case=False, na=False)
    # remove columns containing NaNs
    df_clean = df.loc[:, ~mask].dropna(axis=1, how='all')
    # remove rows containing NaNs
    df_clean.dropna(inplace=True)
    return df_clean


def get_gmw_of_gld(bro_GLD_id: str) -> str:
    headers = {"accept": "text/plain"}
    response = requests.get(f"https://publiek.broservices.nl/gm/gld/v1/objectsAsCsv/{bro_GLD_id}?rapportagetype=volledig", headers=headers)
    if response.status_code == 400:
        error_data = response.json()
        description = error_data.get('description', 'Unknown error')
        raise ValueError(f"Bad request objectsAsCsv for BRO ID {bro_GLD_id}: {description}")
    gmw_matches = re.search(r'\b(GMW[0-9]+)\b', response.text)
    if gmw_matches: return gmw_matches.group(1)
    else: raise ValueError(f"No GMW found for {bro_GLD_id}")
    

def get_coordinates_gmw(bro_GMW_id:str) -> tuple:
    response = requests.get(f"https://publiek.broservices.nl/gm/gmw/v1/objects/{bro_GMW_id}?fullHistory=", headers= {"accept": "application/xml"})
    xml = response.text
    root = ET.fromstring(xml)
    if response.status_code == 400:
        try:
            desc = root.find(".//brocom:rejectionReason", ns)
            description = desc.text if desc is not None else xml
        except Exception:
            description = xml
        raise ValueError(f"Bad request for {bro_GMW_id}: {description}")
    ns = {
        "gml": "http://www.opengis.net/gml/3.2",
        "gmwcommon": "http://www.broservices.nl/xsd/gmwcommon/1.1",
        "brocom": "http://www.broservices.nl/xsd/brocommon/3.0",
        "dsgmw": "http://www.broservices.nl/xsd/dsgmw/1.1",
    }
    pos = root.find(".//gmwcommon:location/gml:pos", ns)
    x, y = map(float, pos.text.split())
    if x is not None and y is not None:
        return x, y
    else:
        raise ValueError(f"Coordinates not found for well {bro_GMW_id}")