import os
import re
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

def init_connection_to_dino():
    """Initialize and return a connection to the DINO Oracle database."""
    load_dotenv("../env/env.sh")
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
    sql_query_migrated_GLD = f"""
    WITH
    cte_migration_event as (
    SELECT
        GWS_W.WELL_DBK
    , GWS_W.NITG_NR
    , GWS_PZM.PIEZOMETER_DBK
    , GWS_PZM.PIEZOMETER_NR
    , BRO_M_E.EVENT_DBK
    , BRO_M_E.BRO_ID
    , BRO_M_E.BRO_DETAIL_ID AS observationId
    , LOC.X_RD_CRD
    , LOC.Y_RD_CRD
    FROM
        DINO_DBA.GWS_WELL GWS_W
        INNER JOIN DINO_DBA.GWS_PIEZOMETER GWS_PZM
        on GWS_W.WELL_DBK = GWS_PZM.WELL_DBK
        INNER JOIN DINO_DBA.BRO_MIGRATION_EVENT BRO_M_E
        on GWS_PZM.PIEZOMETER_DBK = BRO_M_E.EVENT_RECORD_DBK
        INNER JOIN DINO_DBA.LOC_SURFACE_LOCATION LOC 
        on LOC.SURFACE_LOCATION_DBK = GWS_W.SURFACE_LOCATION_DBK
    WHERE
        BRO_M_E.BRO_ID = '{bro_GLD_id}'
        and BRO_M_E.RO_TYPE_CD = 'GLD'
        and BRO_M_E.EVENT_TYPE_CD = 'ADDITION'
        and BRO_M_E.TABLE_NM_DBK = (SELECT TABLE_NM_DBK FROM DINO_DBA.REF_BRO_MIGRATION_TABLE_NM WHERE TABLE_NM = 'GWS_PIEZOMETER')
    )
    SELECT
    cte_m_e.NITG_NR
    , cte_m_e.PIEZOMETER_NR
    , cte_m_e.BRO_ID
    , cte_m_e.observationId
    , cte_m_e.X_RD_CRD
    , cte_m_e.Y_RD_CRD
    , GWS_MSM_H.MONITOR_DATE
    , GWS_MSM_H.VALUE
    FROM
    cte_migration_event cte_m_e
    INNER JOIN DINO_DBA.BRO_MIGRATION_RECORD BRO_M_R
        on cte_m_e.EVENT_DBK = BRO_M_R.EVENT_DBK
    INNER JOIN DINO_DBA.GWS_MSM_HEAD GWS_MSM_H
        on BRO_M_R.MIGRATED_RECORD_DBK = GWS_MSM_H.MSM_HEAD_DBK
    WHERE
    BRO_M_R.TABLE_NM_DBK = (SELECT TABLE_NM_DBK FROM DINO_DBA.REF_BRO_MIGRATION_TABLE_NM WHERE TABLE_NM = 'GWS_MSM_HEAD')
    and BRO_M_R.MIGRATION_STATUS_CD in ('REWORK', 'SUCCESS')
    ORDER BY
    cte_m_e.WELL_DBK
    , cte_m_e.PIEZOMETER_DBK
    , GWS_MSM_H.MONITOR_DATE
    """
    df = pd.read_sql(sql_query_migrated_GLD, engine)
    df['monitor_date'] = df['monitor_date'].dt.tz_localize('Europe/Amsterdam')
    df['monitor_date'] = parse_date_to_unix(df['monitor_date'])
    #TODO: correct values to get . t.ov. NAP
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
                if _YFIRST.match(ts) or "T" in ts: # ISO & Y-first
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