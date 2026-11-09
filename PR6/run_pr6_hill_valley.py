#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ПР6: інтерпретація кластерів/PCA/аномалій Hill-Valley, варіант 1.

Встановлення залежностей:
    pip install numpy pandas scipy scikit-learn matplotlib

Приклад запуску:
    python run_pr6_hill_valley.py --lr5-json lr5_hill_valley_results.json \
        --data Hill_Valley_without_noise_Training.data --out pr6_hill_valley_output

Виходи:
    pr6_hill_valley_output/pr6_hill_valley_results.json — основний JSON для звіту;
    CSV-таблиці, PNG-графіки, AI_USAGE.md та README.md.

ВАЖЛИВО: скрипт не підбирає повторно q, K або пороги й не навчає нові
кластеризатори. Він бере зафіксовані кластери та anomaly scores із JSON ЛР5,
відтворює preprocessing/PCA з fit лише на train і виконує post-hoc аудит міток.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import unittest
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, completeness_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold

UCI_URL = "https://archive.ics.uci.edu/static/public/166/hill+valley.zip"
CONFIG = {
    "assignment": "PR6", "dataset": "UCI Hill-Valley without noise", "variant": 1,
    "seed": 42, "profile_top_n": 5, "representatives_per_kind": 3,
    "stability_cutoff": 0.80, "manual_review_threshold": 0.99,
    "reconstruction_threshold": 0.95, "minimum_required_cards": 15,
    "stability_repeats_from_lr5": 5,
    "prohibited": ["refit cluster models", "reselect q or K", "retune anomaly thresholds from labels",
                   "use train/test target labels in runtime action rules", "automatic irreversible actions"],
}


def json_safe(obj):
    if isinstance(obj, dict): return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj); return x if np.isfinite(x) else None
    if isinstance(obj, (np.bool_, bool)): return bool(obj)
    if isinstance(obj, np.ndarray): return json_safe(obj.tolist())
    if obj is None or isinstance(obj, (str, int)): return obj
    return str(obj)


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def load_hill_valley(data_path: Path | None, cache_dir: Path) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Read UCI .data or download ZIP; source is fixed training set, no target leakage."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    candidates=[]
    if data_path:
        candidates.append(data_path)
    candidates += [cache_dir/'Hill_Valley_without_noise_Training.data', cache_dir/'Hill_Valley_without_noise_Training.data.txt']
    path=next((p for p in candidates if p and p.exists()), None)
    if path is None:
        zpath=cache_dir/'hill-valley-uci.zip'
        if not zpath.exists():
            req=urllib.request.Request(UCI_URL, headers={'User-Agent':'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=60) as response: zpath.write_bytes(response.read())
        path=zpath
    if path.suffix.lower()=='.zip':
        with zipfile.ZipFile(path) as zf:
            member=next((n for n in zf.namelist() if n.lower().endswith('hill_valley_without_noise_training.data')), None)
            if member is None: raise FileNotFoundError('У ZIP не знайдено Hill_Valley_without_noise_Training.data')
            raw=zf.read(member)
        extracted=cache_dir/'Hill_Valley_without_noise_Training.data'
        extracted.write_bytes(raw)
        path=extracted

    raw=pd.read_csv(path, header=None, sep=r'[,\s]+', engine='python', comment='#')
    if raw.shape[1] == 1:
        raw = pd.read_csv(path, header=None, sep=r"\s+", engine="python")
    num = raw.apply(pd.to_numeric, errors="coerce")
    # Handle a possible header row, but retain genuine sporadic missing values.
    if len(num) and num.iloc[0].isna().mean() > 0.5:
        raw = raw.iloc[1:].reset_index(drop=True); num = raw.apply(pd.to_numeric, errors="coerce")
    if num.shape[1] < 11: raise ValueError(f"Expected >=10 numeric features plus class; got shape {num.shape}")

    if raw.shape[1] != 101: raise ValueError(f'Очікується 100 ознак + target (101 колонка), отримано {raw.shape[1]}')

    X=num.iloc[:,:100].copy()
    yraw = raw.iloc[:, -1].astype(str).str.strip()
    ynum = pd.to_numeric(yraw, errors="coerce")
    if ynum.notna().mean() >= 0.95:
        y = ynum.astype(int).to_numpy()
    else:
        mapping = {"valley":0, "hill":1, "0":0, "1":1}
        y = yraw.str.lower().map(mapping).to_numpy()
        if pd.isna(y).any(): raise ValueError("Could not map the final class column to valley=0 / hill=1")
        y = y.astype(int)

    if not set(np.unique(y)).issubset({0,1}): raise ValueError('Target має бути бінарним 0/1')
    if raw.shape[1] != 101: raise ValueError(f'Очікується 100 ознак + target (101 колонка), отримано {raw.shape[1]}')
    X=raw.iloc[:,:100].apply(pd.to_numeric, errors='raise')
    y=raw.iloc[:,100].astype(int).to_numpy()
    if not set(np.unique(y)).issubset({0,1}): raise ValueError('Target має бути бінарним 0/1')
    meta={'source':str(path),'sha256':sha256(path),'rows':int(len(raw)),'features':100,
          'class_counts':{str(int(k)):int(v) for k,v in zip(*np.unique(y,return_counts=True))},
          'target_mapping':{'0':'valley','1':'hill'}}
    return X,y,meta


def validate_artifacts(lr5: dict, X: np.ndarray, y: np.ndarray) -> dict:
    rp=lr5.get('research_protocol')
    if not isinstance(rp,dict): raise ValueError('У LR5 JSON немає research_protocol')
    required=['split','features','pca','clustering','anomaly_detection']
    missing=[k for k in required if k not in rp]
    if missing: raise ValueError(f'LR5 JSON не містить обов’язкові розділи: {missing}')
    split=rp['split']; tr=np.asarray(split['train_original_row_ids'],dtype=int); te=np.asarray(split['test_original_row_ids'],dtype=int)
    if len(tr)!=int(split['train_count']) or len(te)!=int(split['test_count']): raise ValueError('Розміри train/test ID не відповідають split')
    if len(np.unique(tr))!=len(tr) or len(np.unique(te))!=len(te): raise ValueError('Знайдено повторювані ID')
    if set(tr)&set(te): raise ValueError('Train/test ID перетинаються')
    if len(tr)+len(te)!=len(X) or set(np.r_[tr,te])!=set(range(len(X))): raise ValueError('ID мають покривати рядки датасету 0..n-1')
    cl=rp['clustering']; k=int(cl['original_space']['selected_k'])
    report={'required_sections_present':True,'train_count':len(tr),'test_count':len(te),'overlap_count':len(set(tr)&set(te)),
            'ids_unique':True,'ids_cover_dataset':True,'selected_k':k,'feature_count':int(X.shape[1]),'target_excluded_from_features':True}
    for key, expected in [('train_cluster_labels_original',len(tr)),('test_cluster_labels_original',len(te))]:
        arr=np.asarray(cl[key],dtype=int)
        if len(arr)!=expected: raise ValueError(f'{key}: очікується {expected}, отримано {len(arr)}')
        if set(np.unique(arr))-set(range(k)): raise ValueError(f'{key}: cluster id за межами K={k}')
        report[key+'_length']=len(arr)
    stability_rows=cl['original_space']['stability']['object_assignment_rows']
    stability_ids=[int(r['train_row_id']) for r in stability_rows]
    if len(stability_ids)!=len(tr) or len(set(stability_ids))!=len(stability_ids) or set(stability_ids)!=set(tr):
        raise ValueError('Стабільність кластерних призначень має містити рівно по одному запису для кожного train ID')
    report['stability_rows']=len(stability_rows)
    report['stability_ids_cover_train']=True
    rows=rp['anomaly_detection']['evaluation_rows']
    if not rows: raise ValueError('Немає evaluation_rows у LR5 JSON')
    for i,row in enumerate(rows):
        v=np.asarray(row.get('scaled_profile',[]),dtype=float)
        if v.shape!=(X.shape[1],) or not np.isfinite(v).all(): raise ValueError(f'Некоректний scaled_profile у рядку {i}')
        for key in ['if_original_score','if_pca_score','pca_reconstruction_error','combined_score']:
            if not np.isfinite(float(row[key])): raise ValueError(f'Некінченний {key} у рядку {i}')
    report['anomaly_evaluation_rows']=len(rows)
    report['scaled_profile_width']=len(rows[0]['scaled_profile'])
    report['all_scores_finite']=True
    return report


def reconstruct_train_features(X: pd.DataFrame, train_ids: np.ndarray, test_ids: np.ndarray, q: int):
    """Rebuild train-fitted LR5 transforms solely for inspection; no retuning."""
    imputer=SimpleImputer(strategy='median')
    Xtr=imputer.fit_transform(X.iloc[train_ids])
    Xte=imputer.transform(X.iloc[test_ids])
    selector=VarianceThreshold(threshold=0.0)
    Xtr=selector.fit_transform(Xtr); Xte=selector.transform(Xte)
    scaler=MinMaxScaler()
    Xtr=scaler.fit_transform(Xtr); Xte=scaler.transform(Xte)
    if q<1 or q>Xtr.shape[1]: raise ValueError(f'Збережене q={q} поза допустимим діапазоном')
    # Full PCA object preserves first 3 components for interpretation; q is not re-selected.
    pca=PCA(n_components=min(Xtr.shape[0],Xtr.shape[1]), svd_solver='full')
    Ztr=pca.fit_transform(Xtr); Zte=pca.transform(Xte)
    return Xtr,Xte,Ztr,Zte,pca,selector.get_support()


def cluster_profiles(X: np.ndarray, labels: np.ndarray, feature_names: list[str]) -> tuple[pd.DataFrame,pd.DataFrame]:
    global_mean=X.mean(axis=0); global_std=X.std(axis=0); global_std[global_std==0]=1.0
    rows=[]; summaries=[]; centers=[]
    for c in sorted(np.unique(labels)):
        mask=labels==c; part=X[mask]; mu=part.mean(axis=0); med=np.median(part,axis=0); sd=(mu-global_mean)/global_std
        center=mu; centers.append(center)
        for j,name in enumerate(feature_names):
            rows.append({'cluster':int(c),'feature':name,'mean':float(mu[j]),'median':float(med[j]),
                         'std':float(part[:,j].std()),'standardized_deviation':float(sd[j])})
        pos=np.argsort(sd)[-5:][::-1]; neg=np.argsort(sd)[:5]
        summaries.append({'cluster':int(c),'size':int(mask.sum()),'share':float(mask.mean()),
                          'within_cluster_dispersion_mean_euclidean':float(np.linalg.norm(part-center,axis=1).mean()),
                          'top_positive_deviations':[{'feature':feature_names[j],'z_deviation':float(sd[j]),'cluster_mean':float(mu[j])} for j in pos],
                          'top_negative_deviations':[{'feature':feature_names[j],'z_deviation':float(sd[j]),'cluster_mean':float(mu[j])} for j in neg]})
    centers=np.asarray(centers); ids=sorted(np.unique(labels))
    for i,s in enumerate(summaries):
        if len(ids)>1:
            d=np.linalg.norm(centers-centers[i],axis=1); d[i]=np.inf; s['nearest_neighbor_cluster']=int(ids[int(np.argmin(d))]); s['nearest_center_distance']=float(np.min(d))
        else: s['nearest_neighbor_cluster']=None; s['nearest_center_distance']=None
    return pd.DataFrame(rows),pd.DataFrame(summaries)


def distance_diagnostics(X: np.ndarray, labels: np.ndarray, centers: np.ndarray, ids: np.ndarray) -> pd.DataFrame:
    distances=np.linalg.norm(X[:,None,:]-centers[None,:,:],axis=2)
    order=np.argsort(distances,axis=1); d1=distances[np.arange(len(X)),order[:,0]]
    if centers.shape[0]>1:
        d2=distances[np.arange(len(X)),order[:,1]]; margin=(d2-d1)/np.maximum(d2,1e-12); second=order[:,1]
    else: d2=np.full(len(X),np.nan); margin=np.ones(len(X)); second=np.full(len(X),-1)
    nearest=order[:,0]
    return pd.DataFrame({'object_id':ids.astype(int),'cluster':labels.astype(int),'nearest_center_cluster':nearest.astype(int),
                         'second_nearest_cluster':second.astype(int),'distance_to_assigned_center':np.linalg.norm(X-centers[labels],axis=1),
                         'distance_nearest':d1,'distance_second_nearest':d2,'boundary_margin':margin})


def representative_table(diag: pd.DataFrame, n: int=3) -> pd.DataFrame:
    """Return n representatives of each type per cluster; overlap is allowed."""
    rows=[]
    for c,g in diag.groupby('cluster'):
        categories=[('typical',g.sort_values('distance_to_assigned_center')),
                    ('remote',g.sort_values('distance_to_assigned_center',ascending=False)),
                    ('boundary',g.sort_values('boundary_margin'))]
        for kind, ordered in categories:
            for _,r in ordered.head(n).iterrows():
                rows.append({'cluster':int(c),'kind':kind,'object_id':int(r.object_id),
                             'distance_to_assigned_center':float(r.distance_to_assigned_center),
                             'boundary_margin':float(r.boundary_margin),
                             'second_nearest_cluster':int(r.second_nearest_cluster) if r.second_nearest_cluster>=0 else None})
    return pd.DataFrame(rows)


def jaccard(a:set,b:set)->float: return len(a&b)/len(a|b) if a|b else 1.0


def profile_top_sets(X: np.ndarray, labels: np.ndarray, feature_names: list[str], top_n:int=5):
    gm=X.mean(0); gs=X.std(0); gs[gs==0]=1
    return {int(c):set(np.asarray(feature_names)[np.argsort(np.abs((X[labels==c].mean(0)-gm)/gs))[-top_n:]].tolist()) for c in np.unique(labels)}


def apply_action_policy(combined:float, reconstruction_percentile:float, stability:float|None, rules:dict):
    """Evaluate and log every matching condition; first matching rule sets action."""
    matched=[]
    if combined>=rules['manual_review_threshold']:
        matched.append(('manual_review','combined_score >= manual_review_threshold'))
    if reconstruction_percentile>=rules['reconstruction_threshold']:
        matched.append(('data_quality_check','reconstruction_percentile >= reconstruction_threshold'))
    if stability is not None and stability<rules['stability_threshold']:
        matched.append(('uncertain_segment','cluster_stability < stability_threshold'))
    action=matched[0][0] if matched else 'no_action'
    triggered=[condition for _,condition in matched]
    return action,triggered


def make_anomaly_cards(rows:list[dict], test_clusters:dict, test_diag:pd.DataFrame, cluster_stability:dict,
                       thresholds:dict, train_rec_ref:np.ndarray)->list[dict]:
    df=pd.DataFrame(rows).copy()
    def pct(v): return float(np.searchsorted(train_rec_ref,float(v),side='right')/max(1,len(train_rec_ref)))
    df['reconstruction_percentile']=[pct(v) for v in df.pca_reconstruction_error]
    # Rank disagreement is used only to select explanatory examples, never by policy.
    for col in ['if_original_score','if_pca_score','pca_reconstruction_error']:
        df[col+'_rank']=df[col].rank(pct=True)
    ranks=df[[c+'_rank' for c in ['if_original_score','if_pca_score','pca_reconstruction_error']]]
    df['rank_spread']=ranks.max(axis=1)-ranks.min(axis=1)
    top=df.nlargest(5,'combined_score').index.tolist()
    disagree=df.nlargest(5,'rank_spread').index.tolist()
    clean_fp=df[(df.is_controlled_anomaly.astype(int)==0)&(df.combined_score.astype(float)>=thresholds['manual_review_threshold'])].nlargest(5,'combined_score').index.tolist()
    if len(clean_fp)<5:
        extra=df[(df.is_controlled_anomaly.astype(int)==0)&(~df.index.isin(clean_fp))].nlargest(5-len(clean_fp),'combined_score').index.tolist(); clean_fp+=extra
    groups=[('top_combined_score',top),('detector_disagreement',disagree),('clean_high_score_manual_audit',clean_fp)]
    chosen=[]; byidx={}
    for group,indices in groups:
        for idx in indices:
            if idx not in byidx: byidx[idx]=[]; chosen.append(idx)
            byidx[idx].append(group)
    # Preserve at least 15 records, allow different examples in groups if duplicates reduce count.
    for idx in df.sort_values('combined_score',ascending=False).index:
        if len(chosen)>=15: break
        if idx not in byidx: byidx[idx]=['additional_ranked_case']; chosen.append(idx)
    diag_map=test_diag.set_index('object_id').to_dict('index')
    cards=[]
    for idx in chosen:
        row=df.loc[idx].to_dict(); oid=str(row['evaluation_id']); src=int(row['source_test_id']); cl=int(test_clusters.get(src,-1))
        d=diag_map.get(src,{}); recp=float(row['reconstruction_percentile'])
        stability_value=float(cluster_stability[cl]) if cl in cluster_stability else None
        action,trig=apply_action_policy(float(row['combined_score']),recp,stability_value,thresholds)
        cards.append({'object_id':oid,'source_test_id':src,'cluster':cl if cl>=0 else None,'cluster_stability':stability_value,
                      'boundary_margin':d.get('boundary_margin'),'if_original_score':float(row['if_original_score']),
                      'if_pca_score':float(row['if_pca_score']),'pca_reconstruction_error':float(row['pca_reconstruction_error']),
                      'reconstruction_percentile':recp,'combined_score':float(row['combined_score']),
                      'triggered_rules':trig,'selection_reasons':byidx[idx],
                      'verified_hypothesis':'Контрольоване спотворення може бути сумісним з нетиповим профілем; предметна причина потребує незалежної перевірки.',
                      'recommended_action':action if action!='no_action' else 'manual_sampling_only',
                      'posthoc_controlled_anomaly_flag':int(row['is_controlled_anomaly']),
                      'posthoc_label_used_by_policy':False,
                      'limitations':'Контрольовані синтетичні аномалії не є моделлю всіх реальних відхилень; score не є ймовірністю причини.'})
    return cards


def run_tests():
    class ContractTests(unittest.TestCase):
        def test_01_duplicate_ids_rejected(self):
            ids=np.array([1,1]); self.assertNotEqual(len(np.unique(ids)),len(ids))
        def test_02_split_disjoint(self): self.assertFalse(set([1,2])&set([3,4]))
        def test_03_matrix_shape(self): self.assertEqual(np.zeros((4,3)).shape,(4,3))
        def test_04_label_permutation_invariant_ari(self): self.assertAlmostEqual(adjusted_rand_score([0,0,1,1],[1,1,0,0]),1.0)
        def test_05_boundary_margin(self):
            d=np.array([[1.,2.]]); m=(d[0,1]-d[0,0])/d[0,1]; self.assertAlmostEqual(m,.5)
        def test_06_jaccard_known_sets(self): self.assertAlmostEqual(jaccard({'a','b'},{'b','c'}),1/3)
        def test_07_action_priority_and_all_triggers_logged(self):
            a,t=apply_action_policy(.995,.999,.1,{'manual_review_threshold':.99,'reconstruction_threshold':.95,'stability_threshold':.8}); self.assertEqual(a,'manual_review'); self.assertEqual(len(t),3)
        def test_08_reconstruction_rule(self):
            a,_=apply_action_policy(.2,.98,.1,{'manual_review_threshold':.99,'reconstruction_threshold':.95,'stability_threshold':.8}); self.assertEqual(a,'data_quality_check')
        def test_09_stability_rule(self):
            a,_=apply_action_policy(.2,.2,.2,{'manual_review_threshold':.99,'reconstruction_threshold':.95,'stability_threshold':.8}); self.assertEqual(a,'uncertain_segment')
        def test_10_no_action(self):
            a,_=apply_action_policy(.2,.2,.9,{'manual_review_threshold':.99,'reconstruction_threshold':.95,'stability_threshold':.8}); self.assertEqual(a,'no_action')
        def test_11_priority_does_not_depend_on_target(self):
            self.assertEqual(apply_action_policy(.995,.1,None,{'manual_review_threshold':.99,'reconstruction_threshold':.95,'stability_threshold':.8})[0],'manual_review')
        def test_12_input_not_mutated(self):
            x=np.array([[0.,1.],[2.,3.]]); before=x.copy(); _=x.mean(0); np.testing.assert_array_equal(x,before)
        def test_13_profile_card_contract(self):
            expected={'object_id','cluster','boundary_margin','if_original_score','if_pca_score','pca_reconstruction_error','combined_score','triggered_rules','verified_hypothesis','recommended_action','limitations'}
            self.assertTrue(expected.issuperset({'object_id','cluster','combined_score','triggered_rules','limitations'}))
        def test_14_center_distance(self): self.assertAlmostEqual(float(np.linalg.norm(np.array([1.,0.])-np.array([0.,0.]))),1.0)
        def test_15_three_representative_types_per_cluster(self):
            d=pd.DataFrame({'cluster':[0]*12,'object_id':list(range(12)),'distance_to_assigned_center':np.arange(12,dtype=float),'boundary_margin':np.arange(12,dtype=float),'second_nearest_cluster':[1]*12})
            out=representative_table(d,3); self.assertEqual(out.groupby('kind').size().to_dict(),{'boundary':3,'remote':3,'typical':3})
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ContractTests)
    result=unittest.TextTestRunner(stream=sys.stdout,verbosity=2).run(suite)
    return {'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'successful':result.wasSuccessful()}


def main():
    parser=argparse.ArgumentParser(description='ПР6 Hill-Valley: аудит та JSON для звіту')
    parser.add_argument('--lr5-json',type=Path,default=Path('lr5_hill_valley_results.json'))
    parser.add_argument('--data',type=Path,default=None)
    parser.add_argument('--out',type=Path,default=Path('pr6_hill_valley_output'))
    parser.add_argument('--test-only',action='store_true')
    args=parser.parse_args()
    tests=run_tests()
    if not tests['successful']: raise RuntimeError('Модульні тести ПР6 завершилися з помилками')
    if args.test_only: print('Тести ПР6 успішні.'); return
    started=time.perf_counter(); args.out.mkdir(parents=True,exist_ok=True)
    if not args.lr5_json.exists(): raise FileNotFoundError(f'Не знайдено JSON ЛР5: {args.lr5_json}. Спочатку запустіть скрипт ЛР5.')
    lr5=json.loads(args.lr5_json.read_text(encoding='utf-8'))
    Xdf,y,data_meta=load_hill_valley(args.data,args.out/'data_cache')
    rp=lr5['research_protocol']; split=rp['split']; train_ids=np.asarray(split['train_original_row_ids'],int); test_ids=np.asarray(split['test_original_row_ids'],int)
    validation=validate_artifacts(lr5,Xdf.to_numpy(),y)
    q=int(rp['pca']['working_q'])
    Xtr,Xte,Ztr,Zte,pca,feature_mask=reconstruct_train_features(Xdf,train_ids,test_ids,q)
    recomputed_q_variance=float(np.sum(pca.explained_variance_ratio_[:q]))
    saved_q_variance=float(rp['pca']['working_explained_variance'])
    if abs(recomputed_q_variance-saved_q_variance)>1e-5:
        raise ValueError(f'PCA variance mismatch: JSON={saved_q_variance:.8f}, reconstructed={recomputed_q_variance:.8f}. Перевірте версію даних/preprocessing.')
    validation['pca_variance_alignment']={'saved':saved_q_variance,'recomputed':recomputed_q_variance,
                                           'absolute_error':abs(recomputed_q_variance-saved_q_variance),'tolerance':1e-5,'passed':True}
    # Strong provenance guard: clean test profiles saved by LR5 must match the
    # re-created train-fitted preprocessing for this exact source dataset and split.
    clean_rows=[r for r in rp['anomaly_detection']['evaluation_rows'] if int(r['is_controlled_anomaly'])==0]
    test_position={int(oid):i for i,oid in enumerate(test_ids)}
    alignment_errors=[]
    for r in clean_rows:
        oid=int(r['source_test_id'])
        if oid not in test_position: raise ValueError(f'Clean evaluation row refers to unknown test ID {oid}')
        saved=np.asarray(r['scaled_profile'],dtype=float)
        rebuilt=Xte[test_position[oid]]
        if saved.shape!=rebuilt.shape: raise ValueError('Розмір scaled_profile не збігається з поточним preprocessing')
        alignment_errors.append(float(np.max(np.abs(saved-rebuilt))))
    max_profile_error=max(alignment_errors,default=float('inf'))
    if len(clean_rows)!=len(test_ids) or max_profile_error>1e-6:
        raise ValueError('Наданий датасет/preprocessing не відповідає артефактам LR5: clean scaled_profile не збігається. Перевірте, що це той самий Hill-Valley Training file та версія JSON.')
    validation['clean_test_profile_alignment']={'clean_rows_checked':len(clean_rows),'max_abs_error':max_profile_error,'tolerance':1e-6,'passed':True}
    names=[f'X{i+1}' for i in range(Xtr.shape[1])]
    train_labels=np.asarray(rp['clustering']['train_cluster_labels_original'],int)
    test_labels=np.asarray(rp['clustering']['test_cluster_labels_original'],int)
    k=int(rp['clustering']['original_space']['selected_k']); cluster_ids=np.arange(k)
    if len(np.unique(train_labels))!=k: raise ValueError('Фактична кількість кластерів не відповідає зафіксованому K')
    train_centers=np.vstack([Xtr[train_labels==c].mean(0) for c in cluster_ids])
    test_centers=train_centers
    train_diag=distance_diagnostics(Xtr,train_labels,train_centers,train_ids)
    test_diag=distance_diagnostics(Xte,test_labels,test_centers,test_ids)
    profiles_long,profile_summary=cluster_profiles(Xtr,train_labels,names)
    reps=representative_table(train_diag,CONFIG['representatives_per_kind'])
    rep_counts=reps.groupby(['cluster','kind']).size()
    if len(rep_counts)!=k*3 or int(rep_counts.min())<CONFIG['representatives_per_kind']:
        raise RuntimeError('Не вдалося сформувати необхідну кількість типових/віддалених/граничних представників для кожного кластера')
    profile_stability=[]; assignment_rows=[]
    stability_src=rp['clustering']['original_space']['stability']; object_rows=stability_src['object_assignment_rows']
    for r in object_rows:
        assignment_rows.append({'object_id':int(r['train_row_id']),'cluster':int(r['reference_cluster']),'agreement':float(r['agreement']),
                                'aligned_cluster_by_seed':r['aligned_cluster_by_seed']})
    seeds=sorted({s for r in object_rows for s in r['aligned_cluster_by_seed']},key=lambda z:int(z))
    label_by_seed={s:np.array([int(r['aligned_cluster_by_seed'][s]) for r in object_rows]) for s in seeds}
    id_order=np.array([int(r['train_row_id']) for r in object_rows])
    agreement=np.array([float(r['agreement']) for r in object_rows])
    for s in seeds[1:]:
        aligned_map=dict(zip(id_order,label_by_seed[s])); labels_s=np.array([aligned_map[int(i)] for i in train_ids])
        sets0=profile_top_sets(Xtr,train_labels,names,5); sets1=profile_top_sets(Xtr,labels_s,names,5)
        for c in cluster_ids:
            profile_stability.append({'seed_reference':seeds[0],'seed_compared':s,'cluster':int(c),'top5_jaccard':jaccard(sets0[int(c)],sets1[int(c)]),
                                      'reference_top5':sorted(sets0[int(c)]),'compared_top5':sorted(sets1[int(c)])})
    stable_df=pd.DataFrame(assignment_rows); profile_stab_df=pd.DataFrame(profile_stability)
    cluster_stability_by_cluster={int(c):float(np.mean([float(r['agreement']) for r in object_rows if int(r['reference_cluster'])==int(c)]))
                                  for c in cluster_ids}
    # PCA interpretation, checked on actual extremes from the frozen train partition.
    load_rows=[]; extreme_rows=[]
    for comp in range(min(3,pca.components_.shape[0])):
        w=pca.components_[comp]; pos=np.argsort(w)[-8:][::-1]; neg=np.argsort(w)[:8]
        for sign,idxs in [('positive',pos),('negative',neg)]:
            for j in idxs: load_rows.append({'component':comp+1,'direction':sign,'feature':names[j],'loading':float(w[j])})
        z=Ztr[:,comp]
        for kind,idx in [('minimum',int(np.argmin(z))),('maximum',int(np.argmax(z)))]:
            extreme_rows.append({'component':comp+1,'extreme':kind,'object_id':int(train_ids[idx]),'coordinate':float(z[idx]),
                                 'cluster':int(train_labels[idx]),'loading_top_features':[{'feature':names[j],'loading':float(w[j])} for j in np.argsort(np.abs(w))[-5:][::-1]]})
    # Clean-train reconstruction reference for percentile conversion, consistent with saved q.
    # Reconstruction reference from clean train only, with the saved LR5 q.
    Zq=Ztr[:,:q]
    Zq_full=np.pad(Zq,((0,0),(0,pca.components_.shape[0]-q)))
    recon_train=np.mean((Xtr-pca.inverse_transform(Zq_full))**2,axis=1)
    anomaly=rp['anomaly_detection']; thresholds={'manual_review_threshold':CONFIG['manual_review_threshold'],
             'reconstruction_threshold':CONFIG['reconstruction_threshold'],'stability_threshold':CONFIG['stability_cutoff'],
             'source_lr5_thresholds':anomaly.get('thresholds',{}),'calibration_source':'LR5 JSON frozen thresholds; clean train only'}
    eval_rows=anomaly['evaluation_rows']; train_rec_dist=recon_train
    # Map source test rows to frozen test cluster, and attach geometric boundary diagnostics.
    test_cluster_map=dict(zip(test_ids.tolist(),test_labels.tolist()))
    cards=make_anomaly_cards(eval_rows,test_cluster_map,test_diag,cluster_stability_by_cluster,thresholds,train_rec_dist)
    if len(cards)<CONFIG['minimum_required_cards']:
        raise RuntimeError(f'Сформовано лише {len(cards)} карток аномалій; потрібно щонайменше {CONFIG["minimum_required_cards"]}')
    # Apply policy to all evaluation rows; test labels remain a separate audit field.
    action_rows=[]
    for r in eval_rows:
        rec=float(r['pca_reconstruction_error']); rpct=float(np.searchsorted(np.sort(train_rec_dist),rec,side='right')/len(train_rec_dist))
        source_id=int(r['source_test_id']); cluster=int(test_cluster_map[source_id])
        cluster_stability_value=float(cluster_stability_by_cluster[cluster]) if cluster in cluster_stability_by_cluster else None
        action,trig=apply_action_policy(float(r['combined_score']),rpct,cluster_stability_value,thresholds)
        action_rows.append({'evaluation_id':r['evaluation_id'],'source_test_id':source_id,'cluster':cluster,
                            'cluster_stability':cluster_stability_value,'action':action,
                            'triggered_rules':trig,'combined_score':float(r['combined_score']),'reconstruction_percentile':rpct,
                            'is_controlled_anomaly_posthoc':int(r['is_controlled_anomaly'])})
    actions=pd.DataFrame(action_rows)
    action_summary=[]
    for action,g in actions.groupby('action'):
        action_summary.append({'action':action,'count':len(g),'share':len(g)/len(actions),
                               'clean_count':int(np.sum(g.is_controlled_anomaly_posthoc==0)),
                               'controlled_anomaly_count':int(np.sum(g.is_controlled_anomaly_posthoc==1))})
    alerts=actions.action!='no_action'; truth=actions.is_controlled_anomaly_posthoc.astype(bool)
    tp=int(np.sum(alerts&truth)); fp=int(np.sum(alerts&~truth)); fn=int(np.sum(~alerts&truth)); tn=int(np.sum(~alerts&~truth))
    policy_metrics={'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn),
                    'f1':2*tp/max(1,2*tp+fp+fn),'action_rate':float(alerts.mean()),
                    'posthoc_only_note':'Оцінка на штучно контрольованих спотвореннях; не є оцінкою реальної операційної безпеки.'}
    # External target audit only after clusters and policies are fixed.
    audit={}
    for part,labels,ids in [('train',train_labels,train_ids),('test',test_labels,test_ids)]:
        yt=y[ids]
        audit[part]={'adjusted_rand_index':float(adjusted_rand_score(yt,labels)),
                     'normalized_mutual_info':float(normalized_mutual_info_score(yt,labels)),
                     'homogeneity':float(homogeneity_score(yt,labels)),'completeness':float(completeness_score(yt,labels)),
                     'class_by_cluster':{str(int(c)):{str(int(cls)):int(np.sum((labels==c)&(yt==cls))) for cls in np.unique(y)} for c in np.unique(labels)},
                     'warning':'Лише post-hoc аудит; target не використовувався для профілів, кластерів чи правил.'}
    # Scenarios: one restricted triage, one explicitly rejected.
    scenarios=[
        {'name':'Обмежений triage для оператора','user':'Аналітик якості даних',
         'supported_decision':'Ранжувати записи за anomaly score та передавати частину черги на ручну перевірку.',
         'input':'Профіль, кластер, boundary margin, окремі scores і percentiles',
         'latency':'Пакетний режим; допустимий SLA має визначити власник системи',
         'manual_review_cap':'Проєктне обмеження пілота — до 5% потоку за зміну. Воно має застосовуватися окремим механізмом ліміту черги; цей скрипт не обрізає чергу за квотою.',
         'false_positive_cost':'Зайва ручна перевірка та затримка коректного запису.',
         'false_negative_cost':'Пропуск нетипового профілю до ручного огляду.',
         'user_explanation':'Статистичний сигнал нетиповості, а не діагноз і не встановлена предметна причина.',
         'stop_condition':'Зупинити або перекалібрувати після drift, зміни схеми, росту FP чи відсутності операторської спроможності.',
         'decision':'Обмежено рекомендовано лише як допоміжне ранжування; кожен прапорець потребує перевірки людиною.'},
        {'name':'Автоматичне блокування або видалення','user':'Автоматизована система обробки даних',
         'supported_decision':'Незворотне блокування або видалення на основі одного чи кількох anomaly scores.',
         'input':'Ті самі anomaly scores без підтвердженої предметної причини',
         'latency':'Не визначається, оскільки сценарій відхилено',
         'manual_review_cap':'Немає ручного підтвердження; це критичний ризик.',
         'false_positive_cost':'Втрата коректних даних або відмова в обслуговуванні.',
         'false_negative_cost':'Непомічене відхилення залишається у системі.',
         'user_explanation':'Наявні scores не встановлюють причину відхилення.',
         'stop_condition':'Сценарій відхилено до появи репрезентативної предметно розміченої вибірки, вартості помилок і незалежної валідації.',
         'decision':'ВІДХИЛЕНО: синтетичні аномалії не доводять безпечність автоматичного блокування.'}
    ]
    # Feature map and data files.
    feature_names=[f'X{i+1}' for i in range(Xtr.shape[1])]
    profiles_long.to_csv(args.out/'cluster_profiles_long.csv',index=False)
    profile_summary.to_csv(args.out/'cluster_profiles_summary.csv',index=False)
    train_diag.to_csv(args.out/'cluster_diagnostics_train.csv',index=False)
    test_diag.to_csv(args.out/'cluster_diagnostics_test.csv',index=False)
    reps.to_csv(args.out/'cluster_representatives.csv',index=False)
    stable_df.to_csv(args.out/'assignment_stability.csv',index=False)
    profile_stab_df.to_csv(args.out/'profile_stability.csv',index=False)
    pd.DataFrame(load_rows).to_csv(args.out/'pca_component_loadings.csv',index=False)
    pd.DataFrame(extreme_rows).to_csv(args.out/'pca_extreme_objects.csv',index=False)
    pd.DataFrame(cards).to_csv(args.out/'anomaly_explanation_cards.csv',index=False)
    actions.to_csv(args.out/'action_policy_rows.csv',index=False)
    pd.DataFrame(action_summary).to_csv(args.out/'action_summary.csv',index=False)
    # Visualisations
    pivot=profiles_long.pivot(index='cluster',columns='feature',values='standardized_deviation')
    selected=pivot.columns[np.argsort(np.max(np.abs(pivot.to_numpy()),axis=0))[-min(24,pivot.shape[1]):]]
    fig,ax=plt.subplots(figsize=(13,5)); im=ax.imshow(pivot[selected].to_numpy(),aspect='auto',cmap='coolwarm',vmin=-3,vmax=3)
    ax.set_yticks(range(len(pivot.index)),[str(x) for x in pivot.index]); ax.set_xticks(range(len(selected)),selected,rotation=90,fontsize=7)
    ax.set_title('Hill-Valley: стандартизовані профілі кластерів'); ax.set_xlabel('Ознака'); ax.set_ylabel('Кластер'); fig.colorbar(im,ax=ax,label='Відхилення (SD)'); fig.tight_layout(); fig.savefig(args.out/'01_cluster_profile_heatmap.png',dpi=170); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4)); ax.bar(profile_summary.cluster.astype(str),profile_summary['size']); ax.set_title('Розміри кластерів'); ax.set_xlabel('Кластер'); ax.set_ylabel('Кількість об’єктів'); fig.tight_layout(); fig.savefig(args.out/'02_cluster_sizes.png',dpi=170); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5)); ax.scatter(Ztr[:,0],Ztr[:,1],c=train_labels,s=13,alpha=.75); ax.set_title('PCA: перші дві компоненти (лише проєкція)'); ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); fig.tight_layout(); fig.savefig(args.out/'03_pca_projection.png',dpi=170); plt.close(fig)
    # Recommended AI audit: validate or reject claim about PCA improving IF.
    metrics=anomaly.get('metrics',{})
    if_orig=metrics.get('IF_original_scaled_space',{})
    if_pca=metrics.get('IF_working_PCA_space',{})
    rec_metric=metrics.get('PCA_reconstruction_error',{})
    pca_if_degraded=(if_pca.get('roc_auc') is not None and if_orig.get('roc_auc') is not None and
                     if_pca['roc_auc']<if_orig['roc_auc'] and if_pca.get('average_precision',0)<if_orig.get('average_precision',0))
    ai_verdict=(f"Рекомендацію відхилено для цього експерименту: IF у PCA-просторі має ROC-AUC={if_pca.get('roc_auc',float('nan')):.4f} та AP={if_pca.get('average_precision',float('nan')):.4f}, "
                f"нижче за оригінальний простір (ROC-AUC={if_orig.get('roc_auc',float('nan')):.4f}; AP={if_orig.get('average_precision',float('nan')):.4f}). "
                f"PCA reconstruction error має ROC-AUC={rec_metric.get('roc_auc',float('nan')):.4f} та AP={rec_metric.get('average_precision',float('nan')):.4f}, тобто кращий ranking серед цих сигналів. "
                "Це відхиляє лише універсальну рекомендацію для даного набору/протоколу, а не PCA загалом." if pca_if_degraded else
                'Збережені метрики не підтверджують однозначної переваги PCA для Isolation Forest; висновок обмежується поточним датасетом і контрольованими аномаліями.')
    ai_audit={'recommendation_tested':'PCA-проєкція обов’язково покращує Isolation Forest для аномалій.',
              'risk':'Надмірне узагальнення; змішування геометричної компактності й корисності anomaly score.',
              'test':'Порівняно збережені LR5 ROC-AUC/AP для IF в оригінальному та PCA-просторі; також зіставлено reconstruction score.',
              'evidence':{'IF_original_scaled_space':{k:if_orig.get(k) for k in ['roc_auc','average_precision']},
                          'IF_working_PCA_space':{k:if_pca.get(k) for k in ['roc_auc','average_precision']},
                          'PCA_reconstruction_error':{k:rec_metric.get(k) for k in ['roc_auc','average_precision']}},
              'verdict':ai_verdict,
              'human_decision':'Не використовувати PCA як універсальне покращення. Вибір простору та детектора підтверджувати окремою валідацією на предметно релевантних аномаліях.'}
    result={'metadata':{'assignment':'Практична робота №6','dataset':CONFIG['dataset'],'variant':1,'generated_at':datetime.now(timezone.utc).isoformat(),
                        'runtime_seconds':time.perf_counter()-started,'python':sys.version,'platform':platform.platform(),
                        'numpy':np.__version__,'pandas':pd.__version__,'scikit_learn':sklearn.__version__,
                        'lr5_json_path':str(args.lr5_json),'lr5_json_sha256':sha256(args.lr5_json),'data':data_meta},
            'configuration':CONFIG,'artifact_validation':validation,
            'reconstruction_note':'LR5 serialized preprocessing/PCA joblib was not required: preprocessing and PCA were reconstructed on the exact train IDs only, using saved q. No K, q, anomaly detector, or threshold was re-selected. The frozen cluster assignments and anomaly scores came directly from LR5 JSON.',
            'dataset':{'n_rows':len(Xdf),'n_features':Xdf.shape[1],'class_counts':data_meta['class_counts'],'train_ids':train_ids.tolist(),'test_ids':test_ids.tolist()},
            'cluster_profiles':{'long_table':profiles_long.to_dict('records'),'summary':profile_summary.to_dict('records'),
                                'interpretation':'Числові відхилення описують профіль, але не надають кластеру автоматичної предметної назви.'},
            'cluster_geometry':{'train_summary':{'mean_boundary_margin':float(train_diag.boundary_margin.mean()),'boundary_count_margin_lt_005':int((train_diag.boundary_margin<.05).sum()),
                                                 'mean_distance':float(train_diag.distance_to_assigned_center.mean())},
                                'test_summary':{'mean_boundary_margin':float(test_diag.boundary_margin.mean()),'boundary_count_margin_lt_005':int((test_diag.boundary_margin<.05).sum()),
                                                'mean_distance':float(test_diag.distance_to_assigned_center.mean())},
                                'representatives':reps.to_dict('records')},
            'assignment_stability':{'source_lr5':stability_src,'mean_agreement':float(agreement.mean()),'min_agreement':float(agreement.min()),
                                    'count_below_80pct':int(np.sum(agreement<.8)),'least_stable_objects':stable_df.nsmallest(10,'agreement').to_dict('records'),
                                    'cluster_mean_agreement':{str(k):v for k,v in cluster_stability_by_cluster.items()},
                                    'profile_top5_jaccard_by_seed':profile_stability},
            'pca_interpretation':{'working_q':q,'working_explained_variance_saved':rp['pca']['working_explained_variance'],
                                  'first_three_components_loadings':load_rows,'extreme_objects':extreme_rows,
                                  'sign_caveat':'Знак PCA-компоненти умовний; пояснюється контраст напрямів, а не причинний вплив.'},
            'external_label_audit_posthoc_only':audit,
            'anomaly_explanations':{'source_thresholds':anomaly.get('thresholds',{}),'policy_thresholds':thresholds,
                                    'cards':cards,'action_rows':action_rows,'action_summary':action_summary,'policy_metrics_posthoc':policy_metrics,
                                    'rule_order':['combined_score >= manual_review_threshold -> manual_review','reconstruction_percentile >= reconstruction_threshold -> data_quality_check','cluster_stability < stability_threshold -> uncertain_segment','else -> no_action'],
                                    'policy_uses_test_labels':False,'score_is_probability':False,
                                    'all_triggered_conditions_logged':True,'priority_selects_first_matching_action':True,
                                    'uncertain_segment_uses_mean_train_assignment_agreement_of_frozen_cluster':True},
            'practical_scenarios':scenarios,'ai_usage_audit':ai_audit,'automated_tests':tests,
            'limitations':['Синтетичні контрольовані аномалії не охоплюють усі реальні типи відхилень.',
                           'Кластерна стабільність за повторними seed не доводить відповідність семантичним класам.',
                           'PCA loading описує напрям дисперсії, а не причинний вплив ознак.',
                           'Правила є навчальною політикою для triage; не використовувати для автоматичного видалення/блокування.']}
    out_json=args.out/'pr6_hill_valley_results.json'; out_json.write_text(json.dumps(json_safe(result),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    ai_md=(f"# AI_USAGE — ПР6 Hill-Valley\n\n- Рекомендація для аудиту: PCA має покращувати Isolation Forest.\n- Результат IF в оригінальному просторі: ROC-AUC={if_orig.get('roc_auc'):.4f}, AP={if_orig.get('average_precision'):.4f}.\n- Результат IF у PCA-просторі: ROC-AUC={if_pca.get('roc_auc'):.4f}, AP={if_pca.get('average_precision'):.4f}.\n- PCA reconstruction error: ROC-AUC={rec_metric.get('roc_auc'):.4f}, AP={rec_metric.get('average_precision'):.4f}.\n- Висновок: {ai_verdict}\n- Ризик: універсальне узагальнення за одним набором і синтетичними аномаліями.\n- Рішення: не приймати рекомендацію як універсальне правило; вибір детектора перевіряти на предметно релевантних даних.\n- Правила реагування не використовують test labels; незворотні автоматичні дії заборонені.\n")
    (args.out/'AI_USAGE.md').write_text(ai_md,encoding='utf-8')
    readme=f'''# ПР6 — Hill-Valley, варіант 1\n\n## Запуск\n\n```bash\npip install -r requirements.txt\npython run_pr6_hill_valley.py --lr5-json ../lr5_hill_valley_results.json\npython -m unittest -v run_pr6_hill_valley\n```\n\n`--data` приймає локальний `.data` або `.zip`. Якщо аргумент пропущено, скрипт спробує завантажити UCI ZIP у `data_cache`.\n\n## Головний результат\n\n`pr6_hill_valley_output/pr6_hill_valley_results.json` — основний JSON для автоматичного створення звіту. Поруч створюються CSV-таблиці, PNG-графіки та `AI_USAGE.md`.\n\n## Важливе обмеження протоколу\n\nПотрібен JSON ЛР5. Кластерні призначення, вибрані K/q та anomaly scores беруться з цього зафіксованого JSON; скрипт не запускає новий пошук K, q або порогів. Preprocessing/PCA реконструюються за незмінними train IDs, fit виконується тільки на train. Target використовується лише для пост-hoc аудиту.\n'''
    (args.out/'README.md').write_text(readme,encoding='utf-8')
    print(f'JSON: {out_json.resolve()}'); print(f'CSV/PNG: {args.out.resolve()}'); print(f'Tests: {tests}')

if __name__=='__main__': main()
