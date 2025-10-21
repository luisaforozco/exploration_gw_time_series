import requests
import pandas as pd
from io import StringIO

def get_bro_data(bro_GLD_id: str) -> pd.DataFrame:
    headers = {"accept": "text/plain"}
    response = requests.get(f"https://publiek.broservices.nl/gm/gld/v1/seriesAsCsv/{bro_GLD_id}", headers=headers)
    # Check error code from the request
    if response.status_code == 400:
        error_data = response.json()
        description = error_data.get('description', 'Unknown error')
        raise ValueError(f"Bad request for BRO ID {bro_GLD_id}: {description}")
    # Check if response body is empty or whitespace, TODO: decide if return an empty df or raise an error
    if not response.text.strip(): 
        raise ValueError(f"Er is geen data beschikbaar voor dit dossier {bro_GLD_id}. The response body is empty.")
    df = pd.read_csv(StringIO(response.text))
    return clean_csv(df)


def clean_csv(df: pd.DataFrame) -> pd.DataFrame:
    # remove columns whose name contains "Opmerking" (case-insensitive)
    mask = df.columns.str.contains(r'Opmerking', case=False, na=False)
    # remove columns containing NaNs
    df_clean = df.loc[:, ~mask].dropna(axis=1, how='all')
    # remove rows containing NaNs
    df_clean.dropna(inplace=True)
    return df_clean
