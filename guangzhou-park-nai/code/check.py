"""Check inputs and recalculate published validation metrics without fitting."""
import importlib.metadata
import json
import numpy as np
import pandas as pd
from grouping import assign_fixed_site_groups
from package_paths import get_paths

def metrics(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float)
    return {'n':len(y),'R2':float(1-((y-p)**2).sum()/((y-y.mean())**2).sum()),'RMSE':float(np.sqrt(np.mean((y-p)**2))),'MAE':float(np.mean(abs(y-p)))}

def check_inputs():
    p=get_paths()
    required=[p.master_table,p.station_groups]
    missing=[str(x) for x in required if not x.is_file()]
    if missing: raise FileNotFoundError('Missing inputs: '+repr(missing))
    d=pd.read_csv(p.master_table)
    needed=['row_id','area','date','average_value','doy_sin','doy_cos','dem_elevation_m','dw_baseline2023_trees_prob','hls_model_evi2','hls_model_ndmi','gpm_precip_uncal_d0_mm','gpm_precip_uncal_prev7d_mm']
    absent=set(needed)-set(d.columns)
    if absent: raise ValueError('Missing model columns: '+repr(sorted(absent)))
    d['date']=pd.to_datetime(d.date)
    labels=assign_fixed_site_groups(d)
    assert len(d)==5998 and d.area.nunique()==14
    assert d.date.nunique()==449 and not d.duplicated(['area','date']).any()
    assert labels.value_counts().to_dict()=={'A_urban':3782,'B_mountain':2216}
    assert d[needed[4:]].notna().all().all()
    assert assign_fixed_site_groups(d[['area']]).equals(labels)
    try: assign_fixed_site_groups(pd.DataFrame({'area':['unknown station']}))
    except ValueError: pass
    else: raise AssertionError('Unknown station did not raise')
    out={'rows':len(d),'sites':d.area.nunique(),'dates':d.date.nunique(),'group_rows':labels.value_counts().to_dict(),'target_free_grouping':True,'models_fitted':False,'published_metrics':{},'environment':{}}
    for package in ['numpy','pandas','scikit-learn','lightgbm','xgboost','joblib']:
        try:out['environment'][package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:out['environment'][package]='not installed'
    source=p.published_validation/'oof_m4_timeblocked_predictions.csv'
    if source.is_file():
        a=pd.read_csv(source);y=a.average_value.to_numpy(float);pred=a.oof_pred_m4.to_numpy(float)
        out['published_metrics']={'n':len(a),'R2':float(1-((y-pred)**2).sum()/((y-y.mean())**2).sum()),'RMSE':float(np.sqrt(np.mean((y-pred)**2))),'MAE':float(np.mean(abs(y-pred)))}
        assert len(a)==5998
    return out

def check_results():
    paths=get_paths();v=paths.published_validation
    needed=[v/'oof_m4_timeblocked_predictions.csv',v/'grouped_loso_M4_results.csv',v/'published_model_comparison.csv']
    missing=[str(x) for x in needed if not x.is_file()]
    if missing:raise FileNotFoundError('Missing frozen validation inputs: '+repr(missing))
    d=pd.read_csv(needed[0]);d['date']=pd.to_datetime(d.date)
    assert len(d)==5998 and d.date.nunique()==449
    master=pd.read_csv(paths.master_table);master['date']=pd.to_datetime(master.date)
    check=d.merge(master[['area','date','average_value']],on=['area','date'],validate='one_to_one',suffixes=('','_master'))
    assert len(check)==5998 and np.array_equal(check.average_value,check.average_value_master)
    fold_by_date={day:i for i,block in enumerate(np.array_split(np.sort(d.date.unique()),5)) for day in block}
    d['fold']=d.date.map(fold_by_date);d['baseline']=np.nan
    for f in range(5):
        tr=d.fold.ne(f);te=~tr
        d.loc[te,'baseline']=d.loc[te,'area'].map(d.loc[tr].groupby('area').average_value.mean())
    assert d.baseline.notna().all()
    d['ym']=d.date.dt.to_period('M').astype(str)
    d['season']=d.date.dt.month.map({12:'DJF',1:'DJF',2:'DJF',3:'MAM',4:'MAM',5:'MAM',6:'JJA',7:'JJA',8:'JJA',9:'SON',10:'SON',11:'SON'})
    d['season_year']=d.date.dt.year+d.date.dt.month.eq(12).astype(int)
    rows=[];baselines=[];counts={}
    for scale,keys in [('Daily',None),('Monthly',['area','group','ym']),('Contiguous seasonal',['area','group','season_year','season'])]:
        a=d if keys is None else d.groupby(keys).agg(average_value=('average_value','mean'),oof_pred_m4=('oof_pred_m4','mean'),baseline=('baseline','mean')).reset_index()
        counts[scale]=len(a)
        for group in ['All stations','A_urban','B_mountain']:
            g=a if group=='All stations' else a[a.group.eq(group)]
            rows.append(dict(scale=scale,group=group,**metrics(g.average_value,g.oof_pred_m4)))
            baselines.append(dict(scale=scale,group=group,**metrics(g.average_value,g.baseline)))
    assert counts=={'Daily':5998,'Monthly':214,'Contiguous seasonal':80}
    formal=pd.read_csv(needed[1]);comp=pd.read_csv(needed[2])
    m4=comp[comp.model.eq('M4_TwoPart')&comp.protocol.str.startswith('C_LOSO')]
    m4=m4.merge(formal[~formal.test_site.str.startswith('__')],left_on='seed',right_on='test_site',validate='one_to_one',suffixes=('_comparison','_formal'))
    assert len(m4)==14
    assert np.allclose(m4.MAE_comparison,m4.MAE_formal,rtol=0,atol=1e-8)
    assert np.allclose(m4.RMSE_comparison,m4.RMSE_formal,rtol=0,atol=1e-8)
    out={'verified':True,'models_fitted':False,'frozen_outputs_modified':False,'table3_nine_rows':rows,'same_fold_station_mean_baselines':baselines,'aggregation_units':counts,'published_M4_C_matches_formal_smearing_results':True,'M4_C_station_mean_MAE':{g:float(x.MAE_formal.mean()) for g,x in m4.groupby('group')}}
    return out

def main():
    print(json.dumps({'inputs':check_inputs(),'results':check_results()},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
