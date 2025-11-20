import os
import re
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
from sqlalchemy import create_engine

def init_connection_to_dino():
    """Initialize and return a connection to the DINO Oracle database."""
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    load_dotenv(PROJECT_ROOT / "env" / "env.sh")
    USR_DINO = os.getenv("USR_DINO")
    PWD_DINO = os.getenv("PWD_DINO")
    return create_engine(f"oracle+oracledb://{USR_DINO}:{PWD_DINO}@gdnoradb01.gdnnet.lan:1522/?service_name=dinoprd03")


def get_migrated_data_by_bro_id(bro_GLD_id: str, engine=None) -> pd.DataFrame:
    """
    Fetch successfully migrated data from DINO to BRO by BRO ID (GLDxxxx).
    Args:
        bro_GLD_id: BRO GLD identifier as a string.
        engine: SQLAlchemy engine instance. If None, a new connection will be initialized.
    Returns:
        pd.DataFrame: DataFrame containing the migrated data with 'monitor_date' converted to Unix time.
    """
    if engine is None:
        engine = init_connection_to_dino()
    sql_query = f"""
        WITH
        cte_migration_event as (
        SELECT
        GWS_W.NITG_NR
        , GWS_PZM.PIEZOMETER_DBK
        , BRO_M_E.EVENT_DBK
        , BRO_M_E.BRO_ID
        , BRO_M_E.BRO_DETAIL_ID
        , LOC.X_RD_CRD
        , LOC.Y_RD_CRD
        FROM
            DINO_DBA.GWS_WELL GWS_W
            INNER JOIN DINO_DBA.GWS_PIEZOMETER GWS_PZM
                ON GWS_W.WELL_DBK = GWS_PZM.WELL_DBK
            INNER JOIN DINO_DBA.BRO_MIGRATION_EVENT BRO_M_E
                ON GWS_PZM.PIEZOMETER_DBK = BRO_M_E.EVENT_RECORD_DBK
            INNER JOIN DINO_DBA.LOC_SURFACE_LOCATION LOC 
                ON LOC.SURFACE_LOCATION_DBK = GWS_W.SURFACE_LOCATION_DBK
        WHERE
            BRO_M_E.BRO_ID = '{bro_GLD_id}'
            AND BRO_M_E.RO_TYPE_CD = 'GLD'
            AND BRO_M_E.EVENT_TYPE_CD = 'ADDITION'
            AND BRO_M_E.TABLE_NM_DBK = (SELECT TABLE_NM_DBK FROM DINO_DBA.REF_BRO_MIGRATION_TABLE_NM WHERE TABLE_NM = 'GWS_PIEZOMETER')
        )
        SELECT
        cte_m_e.NITG_NR
        , cte_m_e.PIEZOMETER_DBK
        , cte_m_e.BRO_ID
        , cte_m_e.BRO_DETAIL_ID
        , cte_m_e.X_RD_CRD as X
        , cte_m_e.Y_RD_CRD as Y
        , GWS_MSM_H.MONITOR_DATE
        , GWS_MSM_H.VALUE
        , (
            SELECT gph.MSM_NAP_HEIGHT
            FROM DINO_DBA.GWS_PIE_HISTORY gph
            WHERE gph.PIEZOMETER_DBK = cte_m_e.PIEZOMETER_DBK
            AND gph.START_DATE = (
                    SELECT MAX(gph2.START_DATE)
                    FROM DINO_DBA.GWS_PIE_HISTORY gph2
                    WHERE gph2.PIEZOMETER_DBK = cte_m_e.PIEZOMETER_DBK AND gph2.START_DATE <= GWS_MSM_H.MONITOR_DATE
            )
          ) AS MSM_NAP_HEIGHT
        FROM cte_migration_event cte_m_e
        INNER JOIN DINO_DBA.BRO_MIGRATION_RECORD BRO_M_R
            ON cte_m_e.EVENT_DBK = BRO_M_R.EVENT_DBK
        INNER JOIN DINO_DBA.GWS_MSM_HEAD GWS_MSM_H
            ON BRO_M_R.MIGRATED_RECORD_DBK = GWS_MSM_H.MSM_HEAD_DBK
        WHERE
            BRO_M_R.TABLE_NM_DBK = (SELECT TABLE_NM_DBK FROM DINO_DBA.REF_BRO_MIGRATION_TABLE_NM WHERE TABLE_NM = 'GWS_MSM_HEAD')
            AND BRO_M_R.MIGRATION_STATUS_CD in ('REWORK', 'SUCCESS')
        ORDER BY GWS_MSM_H.MONITOR_DATE
    """
    df = pd.read_sql(sql_query, engine)
    if df.empty: 
        raise ValueError(f"No data found for BRO_ID: {bro_GLD_id}")
    try:
        df['monitor_date'] = df['monitor_date'].dt.tz_localize('Europe/Amsterdam')
    except Exception as e:
        df['monitor_date'] = df['monitor_date'].dt.tz_localize('Europe/Amsterdam', ambiguous=[True]*len(df), nonexistent='shift_forward')
    df['monitor_date'] = parse_date_to_unix(df['monitor_date'])
    # convert values to meters above NAP
    df['value'] = (df['msm_nap_height'] - df['value'])/100
    # delete msm_nap_height column and remove rows with NaNs
    df.drop(columns=['msm_nap_height'], inplace=True)
    df.dropna(inplace=True)
    return df


def parse_date_to_unix(timestamps, default_tz='UTC'):
    """
    Convert ISO 8601 or date-only strings to Unix time (milliseconds since epoch).
    For date-only values, set time to midnight.
    Args:
        timestamps: list of strings (ISO or date-only)
        default_tz: timezone to assume if none provided, univ time is in UTC format
        dayfirst: interpret day-first for date-only formats like '28-08-1973' as given by BRO because we are in Europe.
    
    Returns:
        list of floats (Unix time in milliseconds)
    """
    first_ts = timestamps[0]
    if isinstance(first_ts, pd.Timestamp):
        unix_times = [ts.timestamp() * 1000 for ts in timestamps]
    elif isinstance(first_ts, str):
        _YFIRST = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}")
        unix_times = []
        for ts in timestamps:
            ts = ts.strip()
            if not ts:
                unix_times.append(None)
                continue
            try:
                if _YFIRST.match(ts) or "T" in ts: # ISO & Year-first
                    dt = pd.to_datetime(ts, utc=True)  
                else:  # Day-first branch
                    dt = pd.to_datetime(ts, dayfirst=True)
                
                if dt.tzinfo is None:
                    dt = dt.tz_localize(default_tz)
                
                unix_times.append(dt.timestamp() * 1000)  # seconds since epoch
            
            except Exception:
                unix_times.append(None)
    else:
        raise ValueError(f"Cannot convert timestamps of type {type(ts)}")
    return unix_times


def get_DINO_data_by_piezometer(piezometer_dbk, engine=None) -> pd.DataFrame:
    """
    Fetch data from DINO by piezometer_dbk.
    Args:
        piezometer_dbk: Piezometer DBK identifier as a string.
        engine: SQLAlchemy engine instance. If None, a new connection will be initialized.
    Returns:
        pd.DataFrame: DataFrame containing the data with 'monitor_date' converted to Unix time and values adjusted to meters above NAP.
    """
    sql = f"""
        SELECT
            w.NITG_NR,
            l.X_RD_CRD AS X,
            l.Y_RD_CRD AS Y,
            p.PIEZOMETER_NR,
            h.MONITOR_DATE,
            h.VALUE,
            (   SELECT g.MSM_NAP_HEIGHT
                FROM DINO_DBA.GWS_PIE_HISTORY g
                WHERE g.PIEZOMETER_DBK = h.PIEZOMETER_DBK AND g.START_DATE <= h.MONITOR_DATE
                ORDER BY g.START_DATE DESC
                FETCH FIRST 1 ROW ONLY
            ) AS MSM_NAP_HEIGHT
        FROM DINO_DBA.GWS_MSM_HEAD h
        INNER JOIN DINO_DBA.GWS_PIEZOMETER p
        ON h.PIEZOMETER_DBK = p.PIEZOMETER_DBK
        INNER JOIN DINO_DBA.GWS_WELL w
        ON p.WELL_DBK = w.WELL_DBK
        INNER JOIN DINO_DBA.LOC_SURFACE_LOCATION l
        ON l.SURFACE_LOCATION_DBK = w.SURFACE_LOCATION_DBK
        WHERE h.PIEZOMETER_DBK  = '{piezometer_dbk}'
        ORDER BY h.MONITOR_DATE
    """
    if engine is None:
        engine = init_connection_to_dino()

    df = pd.read_sql(sql, engine)
    if df.empty: 
        raise ValueError(f"No data found for PIEZOMETER_DBK: {piezometer_dbk}")
    try:
        df['monitor_date'] = df['monitor_date'].dt.tz_localize('Europe/Amsterdam')
    except Exception as e:
        df['monitor_date'] = df['monitor_date'].dt.tz_localize('Europe/Amsterdam', ambiguous=[True]*len(df), nonexistent='shift_forward')
    df['monitor_date'] = parse_date_to_unix(df['monitor_date'])
    # convert values to meters above NAP
    df['value'] = (df['msm_nap_height'] - df['value'])/100
    # delete msm_nap_height column and remove rows with NaNs
    df.drop(columns=['msm_nap_height'], inplace=True)
    df.dropna(inplace=True)
    #df.sort_values('monitor_date', inplace=True) # redundant sorting (already in sql query), but leaving it in case we might find time-jumps in the data down the road
    return df
    