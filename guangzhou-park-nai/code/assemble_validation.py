"""Assemble current 7-feature comparison; formal smearing M4 C wins explicitly."""
import json
import numpy as np
import pandas as pd
from package_paths import get_paths

def main():
    p=get_paths()
    ab=pd.read_csv(p.reproduced/'comparison/model_comparison_ab.csv')
    c=pd.read_csv(p.reproduced/'comparison/loso_comparators.csv')
    formal=pd.read_csv(p.reproduced/'m4_final/grouped_loso_M4_results.csv')
    raw=pd.read_csv(p.master_table)
    a_b=ab[ab.protocol.isin(['A_random_stratified','B_time_forward'])].copy()
    # Comparator C contains M1-M3 only; current M4 C is read from its formal results.
    cc=c[c.model.isin(['M1_LGBM','M2_XGB','M3_Ensemble'])].rename(columns={'site':'seed'}).copy()
    cc['n_test']=cc.seed.map(raw.groupby('area').size())
    cc['Bias']=np.nan
    m4=[]
    for _,r in formal[~formal.test_site.str.startswith('__')].iterrows():
        obs=raw.loc[raw.area.eq(r.test_site),'average_value'].to_numpy(float)
        r2=1-float(r.RMSE**2)/float(np.mean((obs-obs.mean())**2))
        m4.append({'protocol':'C_LOSO_'+r.group,'seed':r.test_site,'model':'M4_TwoPart','n_test':int(r.n),'R2':r2,'RMSE':r.RMSE,'MAE':r.MAE,'Bias':r.Bias})
    cols=['protocol','seed','model','n_test','R2','RMSE','MAE','Bias']
    combined=pd.concat([a_b[cols],cc[cols],pd.DataFrame(m4)[cols]],ignore_index=True)
    assert len(combined)==80
    output=p.output('comparison')
    combined.to_csv(output/'model_comparison_current.csv',index=False,encoding='utf-8-sig')
    report={'n_rows':len(combined),'M4_protocol_C_source':'reproduced/m4_final/grouped_loso_M4_results.csv','legacy_M4_C_excluded':True,'frozen_published_outputs_modified':False}
    frozen=p.published_validation/'published_model_comparison.csv'
    if frozen.is_file():
        pub=pd.read_csv(frozen)
        keys=['protocol','seed','model']
        for x in [combined,pub]:
            for k in keys:x[k]=x[k].astype(str)
        pair=combined.merge(pub,on=keys,suffixes=('_reproduced','_published'),validate='one_to_one')
        report['matched_published_rows']=len(pair)
        report['maximum_absolute_metric_difference']={k:float((pair[k+'_reproduced']-pair[k+'_published']).abs().max()) for k in ['R2','RMSE','MAE']}
    (output/'assembly_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
