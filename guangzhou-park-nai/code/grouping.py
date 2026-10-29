"""Fixed station classifications; no concentration-derived classification."""
import pandas as pd
from package_paths import get_paths

def assign_fixed_site_groups(data):
    group_file=get_paths().station_groups
    groups=pd.read_csv(group_file,encoding='utf-8-sig',dtype=str,keep_default_na=False)
    if list(groups.columns)!=['area','group']:
        raise ValueError('Invalid station group columns: '+str(group_file))
    if groups.empty or groups[['area','group']].eq('').any().any():
        raise ValueError('Empty station name or group')
    if groups.area.duplicated().any():
        raise ValueError('Duplicate station names in grouping table')
    if not groups.group.isin(['A_urban','B_mountain']).all():
        raise ValueError('Invalid station group label')
    if data.area.isna().any():
        raise ValueError('Missing station name in input')
    labels=data.area.map(groups.set_index('area').group)
    if labels.isna().any():
        unknown=sorted(data.loc[labels.isna(),'area'].astype(str).unique())
        raise ValueError('Stations absent from fixed grouping table: '+repr(unknown))
    return labels
