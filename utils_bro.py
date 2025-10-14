import requests
import pandas as pd
from io import StringIO

def get_bro_data(bro_GLD_id: str) -> pd.DataFrame:
    headers = {"accept": "text/plain"}
    response = requests.get(f"https://publiek.broservices.nl/gm/gld/v1/seriesAsCsv/{bro_GLD_id}", headers=headers)
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
