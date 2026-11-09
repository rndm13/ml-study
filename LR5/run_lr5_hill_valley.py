#!/usr/bin/env python3
"""LR5: PCA, clustering and anomaly detection for Hill-Valley.

The class column is excluded from all unsupervised fitting, PCA dimension selection,
clustering selection and anomaly-threshold calibration. It is used only for the
post-hoc external audit and controlled-anomaly evaluation.
"""
from __future__ import annotations
import argparse, io, json, math, os, platform, sys, time, urllib.request, zipfile, unittest
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (adjusted_rand_score, average_precision_score,
    calinski_harabasz_score, completeness_score, confusion_matrix, davies_bouldin_score,
    f1_score, homogeneity_score, normalized_mutual_info_score, precision_score,
    recall_score, roc_auc_score, silhouette_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold

UCI_URL = "https://archive.ics.uci.edu/static/public/166/hill+valley.zip"
DATA_MEMBER = "Hill_Valley_without_noise_Training.data"
DEFAULT_CONFIG = {
    "seed": 42, "test_size": 0.20, "pca_thresholds": [0.85, 0.90, 0.95],
    "working_pca_threshold": 0.90, "k_values": [2,3,4,5,6,7,8],
    "cluster_seeds": [11,23,42,57,81], "kmeans_n_init": 10,
    "isolation_forest": {"n_estimators": 300, "max_samples": "auto", "max_features": 1.0},
    "anomaly_quantile": 0.95, "controlled_anomaly_fraction": 1.0,
    "controlled_anomaly_kinds": ["gaussian_noise", "local_spike", "segment_reverse"],
    "top_k_fraction": 0.05
}


def json_safe(obj):
    if isinstance(obj, dict): return {str(k): json_safe(v) for k,v in obj.items()}
    if isinstance(obj, (list, tuple)): return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray): return json_safe(obj.tolist())
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, (np.bool_,)): return bool(obj)
    if isinstance(obj, Path): return str(obj)
    return obj


def read_hill_valley(data_path: str | None = None):
    """Read UCI Hill-Valley data (.data/.csv/.txt or the original ZIP archive)."""
    source = None
    if data_path:
        p = Path(data_path).expanduser().resolve()
        if not p.exists(): raise FileNotFoundError(f"Data file not found: {p}")
        if zipfile.is_zipfile(p):
            with zipfile.ZipFile(p) as zf:
                names = zf.namelist()
                match = next((n for n in names if n.endswith(DATA_MEMBER)), None)
                if match is None: raise ValueError(f"{DATA_MEMBER} not found in ZIP")
                raw_text = zf.read(match).decode("utf-8", errors="replace"); source = f"{p}::{match}"
        else:
            raw_text = p.read_text(encoding="utf-8", errors="replace"); source = str(p)
    else:
        try:
            req = urllib.request.Request(UCI_URL, headers={"User-Agent":"Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=45) as resp: blob = resp.read()
            with zipfile.ZipFile(io.BytesIO(blob)) as zf:
                match = next((n for n in zf.namelist() if n.endswith(DATA_MEMBER)), None)
                if match is None: raise ValueError(f"{DATA_MEMBER} not found in downloaded UCI archive")
                raw_text = zf.read(match).decode("utf-8", errors="replace"); source = f"UCI:{UCI_URL}::{match}"
        except Exception as e:
            raise RuntimeError("Could not download UCI archive. Pass --data PATH to the local Hill-Valley .data file or ZIP.") from e
    # UCI files are comma-delimited and normally do not have a header.
    try: raw = pd.read_csv(io.StringIO(raw_text), header=None, sep=r"\s*,\s*", engine="python")
    except Exception: raw = pd.read_csv(io.StringIO(raw_text), header=None)
    if raw.shape[1] == 1:
        raw = pd.read_csv(io.StringIO(raw_text), header=None, sep=r"\s+", engine="python")
    num = raw.apply(pd.to_numeric, errors="coerce")
    # Handle a possible header row, but retain genuine sporadic missing values.
    if len(num) and num.iloc[0].isna().mean() > 0.5:
        raw = raw.iloc[1:].reset_index(drop=True); num = raw.apply(pd.to_numeric, errors="coerce")
    if num.shape[1] < 11: raise ValueError(f"Expected >=10 numeric features plus class; got shape {num.shape}")
    X = num.iloc[:, :-1].copy(); yraw = raw.iloc[:, -1].astype(str).str.strip()
    ynum = pd.to_numeric(yraw, errors="coerce")
    if ynum.notna().mean() >= 0.95:
        y = ynum.astype(int).to_numpy()
    else:
        mapping = {"valley":0, "hill":1, "0":0, "1":1}
        y = yraw.str.lower().map(mapping).to_numpy()
        if pd.isna(y).any(): raise ValueError("Could not map the final class column to valley=0 / hill=1")
        y = y.astype(int)
    X.columns = [f"X{i+1}" for i in range(X.shape[1])]
    # Keep missing numeric cells for fold-safe median imputation.
    if set(np.unique(y)) != {0,1}: raise ValueError(f"Expected binary classes 0/1; got {np.unique(y)}")
    return X, y, source


def choose_n_components(cumulative_variance, threshold):
    if not 0 < float(threshold) <= 1: raise ValueError("PCA threshold must be in (0, 1]")
    c = np.asarray(cumulative_variance, dtype=float)
    if c.ndim != 1 or len(c)==0 or not np.isfinite(c).all(): raise ValueError("Invalid cumulative variance vector")
    pos = np.flatnonzero(c >= threshold)
    return int(pos[0]+1) if len(pos) else int(len(c))


def align_cluster_labels(reference, candidate):
    ref = np.asarray(reference); cand = np.asarray(candidate)
    if ref.shape != cand.shape or ref.ndim != 1: raise ValueError("Cluster label arrays must be same 1D shape")
    rvals, cvals = np.unique(ref), np.unique(cand)
    mat = np.zeros((len(cvals),len(rvals)),dtype=int)
    for i,c in enumerate(cvals):
        for j,r in enumerate(rvals): mat[i,j] = int(np.sum((cand==c)&(ref==r)))
    rows, cols = linear_sum_assignment(-mat)
    mapping = {cvals[i]:rvals[j] for i,j in zip(rows,cols)}
    return np.array([mapping.get(v, v) for v in cand])


def mean_pairwise_ari_nmi(label_runs):
    if len(label_runs)<2: return {"mean_pairwise_ari":None,"mean_pairwise_nmi":None}
    ari=[]; nmi=[]
    for i in range(len(label_runs)):
        for j in range(i+1,len(label_runs)):
            ari.append(adjusted_rand_score(label_runs[i],label_runs[j]))
            nmi.append(normalized_mutual_info_score(label_runs[i],label_runs[j]))
    return {"mean_pairwise_ari":float(np.mean(ari)),"mean_pairwise_nmi":float(np.mean(nmi)),"pair_count":len(ari)}


def object_assignment_stability(label_runs):
    aligned=[np.asarray(label_runs[0])]
    for run in label_runs[1:]: aligned.append(align_cluster_labels(label_runs[0],run))
    mat=np.vstack(aligned).T
    agreement=np.array([Counter(row.tolist()).most_common(1)[0][1]/mat.shape[1] for row in mat])
    return agreement, aligned


def top_k_jaccard(scores_a, scores_b, k):
    a=np.asarray(scores_a); b=np.asarray(scores_b)
    if a.shape!=b.shape or a.ndim!=1: raise ValueError("score arrays must have same 1D shape")
    if k<1 or k>len(a): raise ValueError("k must be within score array")
    ia=set(np.argsort(a)[-k:].tolist()); ib=set(np.argsort(b)[-k:].tolist())
    return len(ia&ib)/len(ia|ib) if ia|ib else 1.0


def empirical_percentile(reference, values):
    ref=np.sort(np.asarray(reference,dtype=float)); val=np.asarray(values,dtype=float)
    return np.searchsorted(ref,val,side="right")/max(1,len(ref))


def anomaly_metrics(y_true, scores, threshold):
    y=np.asarray(y_true,dtype=int); s=np.asarray(scores,dtype=float)
    if y.shape!=s.shape or y.ndim!=1 or not np.isfinite(s).all(): raise ValueError("Invalid anomaly labels/scores")
    pred=(s>threshold).astype(int); cm=confusion_matrix(y,pred,labels=[0,1])
    pos=int(y.sum()); k=max(1,pos)
    top=np.argsort(s)[-k:]
    return {"roc_auc":float(roc_auc_score(y,s)),"average_precision":float(average_precision_score(y,s)),
        "precision":float(precision_score(y,pred,zero_division=0)),"recall":float(recall_score(y,pred,zero_division=0)),
        "f1":float(f1_score(y,pred,zero_division=0)),"precision_at_k":float(y[top].mean()),"k":int(k),
        "threshold":float(threshold),"flagged_count":int(pred.sum()),"flagged_fraction":float(pred.mean()),
        "confusion_matrix_tn_fp_fn_tp":cm.tolist(),"predictions":pred.tolist()}


def inject_controlled_anomalies(X, object_ids, fraction=1.0, seed=42, kinds=None):
    X=np.asarray(X,dtype=float); ids=np.asarray(object_ids)
    if X.ndim!=2 or len(ids)!=len(X): raise ValueError("X and object_ids shapes do not match")
    if not 0 < fraction <= 1: raise ValueError("fraction must be in (0,1]")
    kinds=kinds or ["gaussian_noise","local_spike","segment_reverse"]
    allowed={"gaussian_noise","local_spike","segment_reverse"}
    if not kinds or any(k not in allowed for k in kinds): raise ValueError("Unknown controlled anomaly kind")
    rng=np.random.default_rng(seed); n=max(1,int(round(len(X)*fraction)))
    chosen=np.arange(n) if n<=len(X) else np.resize(np.arange(len(X)),n)
    out=[]; meta=[]
    d=X.shape[1]
    for j,idx in enumerate(chosen):
        row=X[idx].copy(); kind=kinds[j%len(kinds)]; params={}
        if kind=="gaussian_noise":
            count=max(1,int(round(0.20*d))); cols=np.sort(rng.choice(d,size=count,replace=False))
            noise=rng.normal(0,0.22,size=count); row[cols]+=noise
            params={"changed_coordinates":cols.tolist(),"noise_sd_scaled_units":0.22}
        elif kind=="local_spike":
            width=min(max(5,d//12),d); start=int(rng.integers(0,max(1,d-width+1))); sign=int(rng.choice([-1,1]))
            row[start:start+width]+=sign*0.45
            params={"start_coordinate_1based":start+1,"width":width,"shift_scaled_units":sign*0.45}
        elif kind=="segment_reverse":
            width=max(5,d//3); start=int(rng.integers(0,max(1,d-width+1)))
            row[start:start+width]=row[start:start+width][::-1]
            params={"start_coordinate_1based":start+1,"width":width}
        out.append(row); meta.append({"source_object_id":int(ids[idx]),"anomaly_kind":kind,"parameters":params})
    return np.asarray(out), meta


def cluster_run_grid(X, k_values, seeds, n_init):
    rows=[]; labels_by_key={}
    for k in k_values:
        for seed in seeds:
            t0=time.perf_counter(); model=KMeans(n_clusters=int(k),n_init=int(n_init),random_state=int(seed))
            labels=model.fit_predict(X); elapsed=time.perf_counter()-t0
            nlab=len(np.unique(labels))
            row={"k":int(k),"seed":int(seed),"inertia":float(model.inertia_),"fit_seconds":float(elapsed),
                 "n_clusters_observed":int(nlab),"silhouette":None,"davies_bouldin":None,"calinski_harabasz":None}
            if 1<nlab<len(X):
                row["silhouette"]=float(silhouette_score(X,labels))
                row["davies_bouldin"]=float(davies_bouldin_score(X,labels))
                row["calinski_harabasz"]=float(calinski_harabasz_score(X,labels))
            rows.append(row); labels_by_key[(int(k),int(seed))]=labels
    return rows,labels_by_key


def select_k(rows):
    grouped={}
    for r in rows: grouped.setdefault(r["k"],[]).append(r)
    summaries=[]
    for k,items in grouped.items():
        def mean(key):
            vals=[x[key] for x in items if x[key] is not None]
            return float(np.mean(vals)) if vals else None
        summaries.append({"k":k,"silhouette_mean":mean("silhouette"),"davies_bouldin_mean":mean("davies_bouldin"),
                          "calinski_harabasz_mean":mean("calinski_harabasz"),"inertia_mean":mean("inertia"),
                          "fit_seconds_mean":mean("fit_seconds"),"runs":len(items)})
    valid=[x for x in summaries if x["silhouette_mean"] is not None]
    if not valid: raise RuntimeError("No valid KMeans candidate produced internal metrics")
    valid.sort(key=lambda x:(-x["silhouette_mean"], x["davies_bouldin_mean"] if x["davies_bouldin_mean"] is not None else float("inf"),
                             -(x["calinski_harabasz_mean"] or -float("inf")),x["k"]))
    return int(valid[0]["k"]),summaries


def external_audit(y, labels):
    y=np.asarray(y,dtype=int); labels=np.asarray(labels)
    return {"ari":float(adjusted_rand_score(y,labels)),"nmi":float(normalized_mutual_info_score(y,labels)),
            "homogeneity":float(homogeneity_score(y,labels)),"completeness":float(completeness_score(y,labels)),
            "cluster_class_counts":pd.crosstab(pd.Series(labels,name="cluster"),pd.Series(y,name="class")).to_dict()}


def fit_preprocess(X_train, X_test):
    imputer=SimpleImputer(strategy="median"); A=imputer.fit_transform(X_train); B=imputer.transform(X_test)
    variance=VarianceThreshold(threshold=0.0); A2=variance.fit_transform(A); B2=variance.transform(B)
    if A2.shape[1] < 2: raise ValueError("Fewer than two non-constant features after preprocessing")
    scaler=MinMaxScaler(); A3=scaler.fit_transform(A2); B3=scaler.transform(B2)
    return imputer,variance,scaler,A3,B3


def run_tests():
    class Tests(unittest.TestCase):
        def test_q_at_threshold(self): self.assertEqual(choose_n_components([.4,.8,.95],.9),3)
        def test_q_invalid(self):
            with self.assertRaises(ValueError): choose_n_components([.5,.8],0)
        def test_q_upper_endpoint(self): self.assertEqual(choose_n_components([.3,.6,1.],1.),3)
        def test_alignment_permutation(self):
            ref=np.array([0,0,1,1,2,2]); cand=np.array([2,2,0,0,1,1]); self.assertEqual(adjusted_rand_score(ref,cand),1.0)
            self.assertTrue(np.array_equal(ref,align_cluster_labels(ref,cand)))
        def test_pairwise_ari(self):
            r=mean_pairwise_ari_nmi([[0,0,1,1],[1,1,0,0]]); self.assertAlmostEqual(r["mean_pairwise_ari"],1.0)
        def test_topk_jaccard(self): self.assertEqual(top_k_jaccard([1,2,3],[4,2,0],1),0.0)
        def test_topk_invalid(self):
            with self.assertRaises(ValueError): top_k_jaccard([1,2],[2,1],3)
        def test_anomaly_metrics_range(self):
            m=anomaly_metrics([0,0,1,1],[.1,.2,.8,.9],.5); self.assertEqual(m["confusion_matrix_tn_fp_fn_tp"],[[2,0],[0,2]])
        def test_anomaly_score_shape(self):
            with self.assertRaises(ValueError): anomaly_metrics([0,1],[.1],.5)
        def test_injection_deterministic(self):
            x=np.tile(np.linspace(0,1,100),(6,1)); a,m=inject_controlled_anomalies(x,np.arange(6),1.,7)
            b,n=inject_controlled_anomalies(x,np.arange(6),1.,7); self.assertTrue(np.allclose(a,b)); self.assertEqual(m,n)
        def test_injection_bad_fraction(self):
            with self.assertRaises(ValueError): inject_controlled_anomalies(np.ones((2,10)),[1,2],0,1)
        def test_split_no_overlap(self):
            tr,te=train_test_split(np.arange(30),test_size=.2,random_state=42); self.assertFalse(set(tr)&set(te))
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful(): raise SystemExit("LR5 unit tests failed")
    return {"tests_run":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),"successful":result.wasSuccessful()}


def run_experiment(data_path, out_dir, config_path=None):
    cfg=DEFAULT_CONFIG.copy()
    if config_path:
        user_cfg=json.loads(Path(config_path).read_text(encoding="utf-8")); cfg.update(user_cfg)
    out=Path(out_dir); figdir=out/"figures"; tables=out/"tables"; artifacts=out/"artifacts"
    for p in (out,figdir,tables,artifacts): p.mkdir(parents=True,exist_ok=True)
    Xdf,y,source=read_hill_valley(data_path); row_ids=np.arange(len(Xdf)); X_train_df,X_test_df,y_train,y_test,ids_train,ids_test=train_test_split(
        Xdf,y,row_ids,test_size=float(cfg["test_size"]),stratify=y,random_state=int(cfg["seed"]))
    if set(ids_train)&set(ids_test): raise AssertionError("Train/test overlap")
    Xtr_raw=X_train_df.to_numpy(dtype=float); Xte_raw=X_test_df.to_numpy(dtype=float)
    imputer,variance,scaler,Xtr,Xte=fit_preprocess(Xtr_raw,Xte_raw)
    train_constant=[str(c) for c in X_train_df.columns[X_train_df.nunique(dropna=False)<=1]]
    data_audit={"rows":int(len(Xdf)),"features":int(Xdf.shape[1]),
        "missing_values_per_feature":Xdf.isna().sum().to_dict(),
        "rows_with_any_missing":int(Xdf.isna().any(axis=1).sum()),
        "constant_features_on_train":train_constant,
        "n_constant_features_dropped":int(Xtr_raw.shape[1]-Xtr.shape[1]),
        "duplicate_feature_rows":int(Xdf.duplicated().sum()),
        "target_not_in_features":True,"train_test_overlap_count":int(len(set(ids_train)&set(ids_test)))}
    pca_full=PCA(n_components=min(Xtr.shape[0],Xtr.shape[1]),svd_solver="full")
    t0=time.perf_counter(); Ztr_full=pca_full.fit_transform(Xtr); pca_fit_seconds=time.perf_counter()-t0
    Zte_full=pca_full.transform(Xte); ratios=pca_full.explained_variance_ratio_; cumulative=np.cumsum(ratios)
    threshold_rows=[]
    for threshold in cfg["pca_thresholds"]:
        q=choose_n_components(cumulative,float(threshold)); Z=Ztr_full[:,:q]
        recon=Z@pca_full.components_[:q]+pca_full.mean_
        rmse=float(np.sqrt(np.mean((Xtr-recon)**2)))
        # Same K and seed for a fair q comparison; K itself is selected in the working space below.
        threshold_rows.append({"threshold":float(threshold),"q":int(q),"explained_variance":float(cumulative[q-1]),
                               "train_reconstruction_rmse":rmse,"pca_fit_seconds_full":float(pca_fit_seconds)})
    working_q=choose_n_components(cumulative,float(cfg["working_pca_threshold"]))
    Ztr=Ztr_full[:,:working_q]; Zte=Zte_full[:,:working_q]
    # Cluster grids use train only; class labels do not enter the algorithms or K selection.
    raw_rows,raw_labels=cluster_run_grid(Xtr,cfg["k_values"],cfg["cluster_seeds"],cfg["kmeans_n_init"])
    pca_rows,pca_labels=cluster_run_grid(Ztr,cfg["k_values"],cfg["cluster_seeds"],cfg["kmeans_n_init"])
    k_raw,raw_k_summary=select_k(raw_rows); k_pca,pca_k_summary=select_k(pca_rows)
    seed_list=[int(s) for s in cfg["cluster_seeds"]]
    raw_selected_runs=[raw_labels[(k_raw,s)] for s in seed_list]
    pca_selected_runs=[pca_labels[(k_pca,s)] for s in seed_list]
    raw_stab,raw_aligned=object_assignment_stability(raw_selected_runs)
    pca_stab,pca_aligned=object_assignment_stability(pca_selected_runs)
    raw_assignment_rows=[{"train_row_id":int(ids_train[i]),"agreement":float(raw_stab[i]),"reference_cluster":int(raw_aligned[0][i]),
        "aligned_cluster_by_seed":{str(seed_list[j]):int(raw_aligned[j][i]) for j in range(len(seed_list))}} for i in range(len(ids_train))]
    pca_assignment_rows=[{"train_row_id":int(ids_train[i]),"agreement":float(pca_stab[i]),"reference_cluster":int(pca_aligned[0][i]),
        "aligned_cluster_by_seed":{str(seed_list[j]):int(pca_aligned[j][i]) for j in range(len(seed_list))}} for i in range(len(ids_train))]
    raw_stability={**mean_pairwise_ari_nmi(raw_selected_runs),"mean_object_agreement":float(raw_stab.mean()),
        "unstable_object_count_below_75pct":int(np.sum(raw_stab<.75)),"unstable_top10":[{"train_row_id":int(ids_train[i]),"agreement":float(raw_stab[i])} for i in np.argsort(raw_stab)[:10]],
        "object_assignment_rows":raw_assignment_rows}
    pca_stability={**mean_pairwise_ari_nmi(pca_selected_runs),"mean_object_agreement":float(pca_stab.mean()),
        "unstable_object_count_below_75pct":int(np.sum(pca_stab<.75)),"unstable_top10":[{"train_row_id":int(ids_train[i]),"agreement":float(pca_stab[i])} for i in np.argsort(pca_stab)[:10]],
        "object_assignment_rows":pca_assignment_rows}
    # Freeze working q and K before opening labels for external audit.
    km_raw=KMeans(n_clusters=k_raw,n_init=int(cfg["kmeans_n_init"]),random_state=int(cfg["seed"])).fit(Xtr)
    km_pca=KMeans(n_clusters=k_pca,n_init=int(cfg["kmeans_n_init"]),random_state=int(cfg["seed"])).fit(Ztr)
    raw_train_labels=km_raw.labels_; pca_train_labels=km_pca.labels_
    raw_test_labels=km_raw.predict(Xte); pca_test_labels=km_pca.predict(Zte)
    audit={"raw_space_train":external_audit(y_train,raw_train_labels),"raw_space_test":external_audit(y_test,raw_test_labels),
           "pca_space_train":external_audit(y_train,pca_train_labels),"pca_space_test":external_audit(y_test,pca_test_labels),
           "label_use_note":"Labels were not used in preprocessing, PCA, K selection, clustering, or threshold calibration; they were accessed only for post-hoc external audit."}
    # Compare the three PCA variance thresholds using selected PCA K as a fixed comparator.
    for item in threshold_rows:
        q=item["q"]
        t_proj=time.perf_counter(); Zq=pca_full.transform(Xtr)[:,:q]; item["projection_seconds"] = float(time.perf_counter()-t_proj)
        t_km=time.perf_counter(); km=KMeans(n_clusters=k_pca,n_init=int(cfg["kmeans_n_init"]),random_state=int(cfg["seed"])).fit(Zq)
        item["fixed_k_kmeans_fit_seconds"] = float(time.perf_counter()-t_km)
        item["fixed_k_silhouette"] = float(silhouette_score(Zq,km.labels_))
        item["fixed_k_davies_bouldin"] = float(davies_bouldin_score(Zq,km.labels_))
    # Anomaly model training and all thresholds are derived from clean train only.
    if_cfg=cfg["isolation_forest"]
    if_raw=IsolationForest(n_estimators=int(if_cfg["n_estimators"]),max_samples=if_cfg["max_samples"],max_features=float(if_cfg["max_features"]),contamination="auto",random_state=int(cfg["seed"]),n_jobs=-1).fit(Xtr)
    if_pca=IsolationForest(n_estimators=int(if_cfg["n_estimators"]),max_samples=if_cfg["max_samples"],max_features=float(if_cfg["max_features"]),contamination="auto",random_state=int(cfg["seed"]),n_jobs=-1).fit(Ztr)
    train_if_raw=-if_raw.score_samples(Xtr); train_if_pca=-if_pca.score_samples(Ztr)
    def reconstruction(A):
        ZA=pca_full.transform(A)[:,:working_q]; rec=ZA@pca_full.components_[:working_q]+pca_full.mean_
        return np.mean((A-rec)**2,axis=1)
    train_rec=reconstruction(Xtr)
    tr_if_pct=empirical_percentile(train_if_pca,train_if_pca); tr_rec_pct=empirical_percentile(train_rec,train_rec)
    train_comb=np.maximum(tr_if_pct,tr_rec_pct)
    def distribution_stats(values):
        a=np.asarray(values,dtype=float)
        return {"count":int(len(a)),"mean":float(np.mean(a)),"std":float(np.std(a,ddof=1)) if len(a)>1 else 0.0,
            "min":float(np.min(a)),"q50":float(np.quantile(a,.50)),"q90":float(np.quantile(a,.90)),
            "q95":float(np.quantile(a,.95)),"q99":float(np.quantile(a,.99)),"max":float(np.max(a))}
    q=float(cfg["anomaly_quantile"])
    thresholds={"if_raw":float(np.quantile(train_if_raw,q)),"if_pca":float(np.quantile(train_if_pca,q)),
        "pca_reconstruction":float(np.quantile(train_rec,q)),"combined_percentile":float(np.quantile(train_comb,q)),
        "quantile":q,"calibration_source":"clean training partition only"}
    anomaly_X,anomaly_meta=inject_controlled_anomalies(Xte,ids_test,cfg["controlled_anomaly_fraction"],int(cfg["seed"])+101,cfg["controlled_anomaly_kinds"])
    Xeval=np.vstack([Xte,anomaly_X]); Z_eval=pca_full.transform(Xeval)[:,:working_q]
    y_anom=np.r_[np.zeros(len(Xte),dtype=int),np.ones(len(anomaly_X),dtype=int)]
    eval_ids=[f"{int(i)}:clean" for i in ids_test]+[f"{m['source_object_id']}:anomaly:{m['anomaly_kind']}" for m in anomaly_meta]
    score_raw=-if_raw.score_samples(Xeval); score_pca=-if_pca.score_samples(Z_eval); score_rec=reconstruction(Xeval)
    score_comb=np.maximum(empirical_percentile(train_if_pca,score_pca),empirical_percentile(train_rec,score_rec))
    anomaly_eval={"IF_original_scaled_space":anomaly_metrics(y_anom,score_raw,thresholds["if_raw"]),
        "IF_working_PCA_space":anomaly_metrics(y_anom,score_pca,thresholds["if_pca"]),
        "PCA_reconstruction_error":anomaly_metrics(y_anom,score_rec,thresholds["pca_reconstruction"]),
        "combined_max_percentile":anomaly_metrics(y_anom,score_comb,thresholds["combined_percentile"])}
    error_examples={}
    for detector_name,detector_result in anomaly_eval.items():
        preds=np.asarray(detector_result["predictions"],dtype=int)
        score_vec={"IF_original_scaled_space":score_raw,"IF_working_PCA_space":score_pca,
                   "PCA_reconstruction_error":score_rec,"combined_max_percentile":score_comb}[detector_name]
        fp=np.flatnonzero((y_anom==0)&(preds==1)); fn=np.flatnonzero((y_anom==1)&(preds==0))
        error_examples[detector_name]={
            "false_positive_clean_examples":[{"evaluation_id":eval_ids[i],"score":float(score_vec[i]),"threshold":float(detector_result["threshold"])} for i in fp[:10]],
            "false_negative_controlled_anomaly_examples":[{"evaluation_id":eval_ids[i],"anomaly_kind":("clean" if i<len(ids_test) else anomaly_meta[i-len(ids_test)]["anomaly_kind"]),"score":float(score_vec[i]),"threshold":float(detector_result["threshold"])} for i in fn[:10]]}
    topk=max(1,int(math.ceil(float(cfg["top_k_fraction"])*len(score_raw))))
    topk_j=top_k_jaccard(score_raw,score_pca,topk)
    top_raw=set(np.argsort(score_raw)[-topk:]); top_pca=set(np.argsort(score_pca)[-topk:])
    disagreements=[]
    pred_raw=(score_raw>thresholds["if_raw"]).astype(int); pred_pca=(score_pca>thresholds["if_pca"]).astype(int)
    for i in np.flatnonzero(pred_raw!=pred_pca)[:100]:
        disagreements.append({"evaluation_id":eval_ids[i],"is_controlled_anomaly":int(y_anom[i]),"if_original_flag":int(pred_raw[i]),"if_pca_flag":int(pred_pca[i]),
            "if_original_score":float(score_raw[i]),"if_pca_score":float(score_pca[i]),"pca_reconstruction_error":float(score_rec[i])})
    anomaly_score_rows=[]
    for i in range(len(Xeval)):
        anomaly_score_rows.append({"evaluation_id":eval_ids[i],"source_test_id":int(ids_test[i] if i<len(ids_test) else anomaly_meta[i-len(ids_test)]["source_object_id"]),
            "is_controlled_anomaly":int(y_anom[i]),"anomaly_kind":"clean" if i<len(ids_test) else anomaly_meta[i-len(ids_test)]["anomaly_kind"],
            "if_original_score":float(score_raw[i]),"if_pca_score":float(score_pca[i]),"pca_reconstruction_error":float(score_rec[i]),"combined_score":float(score_comb[i]),
            "if_original_flag":int(pred_raw[i]),"if_pca_flag":int(pred_pca[i]),"reconstruction_flag":int(score_rec[i]>thresholds["pca_reconstruction"]),"combined_flag":int(score_comb[i]>thresholds["combined_percentile"]),
            "scaled_profile":Xeval[i].tolist()})
    pd.DataFrame(raw_rows).to_csv(tables/"kmeans_original_runs.csv",index=False)
    pd.DataFrame(pca_rows).to_csv(tables/"kmeans_pca_runs.csv",index=False)
    pd.DataFrame(anomaly_score_rows).to_csv(tables/"anomaly_scores.csv",index=False)
    pd.DataFrame(anomaly_meta).to_json(tables/"controlled_anomaly_manifest.json",orient="records",indent=2)
    # Figures and plot-ready numerical arrays.
    comp_df=pd.DataFrame({"component":np.arange(1,len(ratios)+1),"explained_variance_ratio":ratios,"cumulative_variance":cumulative})
    comp_df.to_csv(tables/"pca_variance.csv",index=False)
    fig,ax=plt.subplots(figsize=(8,4)); ax.plot(np.arange(1,len(ratios)+1),ratios,marker="."); ax.set(xlabel="Компонента",ylabel="Частка поясненої дисперсії",title="PCA scree plot"); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(figdir/"01_pca_scree.png",dpi=150); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4)); ax.plot(np.arange(1,len(cumulative)+1),cumulative); ax.axhline(cfg["working_pca_threshold"],ls="--"); ax.axvline(working_q,ls="--"); ax.set(xlabel="Кількість компонент",ylabel="Кумулятивна пояснена дисперсія",title="Вибір робочої розмірності PCA"); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(figdir/"02_pca_cumulative.png",dpi=150); plt.close(fig)
    Z2=Ztr_full[:,:2] if Ztr_full.shape[1]>=2 else np.c_[Ztr_full[:,0],np.zeros(len(Ztr_full))]
    fig,ax=plt.subplots(figsize=(7,5)); sc=ax.scatter(Z2[:,0],Z2[:,1],c=pca_train_labels,s=18,cmap="tab10",alpha=.8); ax.set(xlabel="PC1",ylabel="PC2",title=f"Кластеризація у PCA-просторі (q={working_q}, K={k_pca})"); fig.colorbar(sc,ax=ax,label="Кластер (умовний ID)"); fig.tight_layout(); fig.savefig(figdir/"03_pca_clusters.png",dpi=150); plt.close(fig)
    for rows,space,name in [(raw_rows,"Початковий простір","04_kmeans_original.png"),(pca_rows,"PCA-простір","05_kmeans_pca.png")]:
        summaries=[]
        for k in cfg["k_values"]:
            vals=[r["silhouette"] for r in rows if r["k"]==k and r["silhouette"] is not None]; summaries.append(float(np.mean(vals)) if vals else np.nan)
        fig,ax=plt.subplots(figsize=(7,4)); ax.plot(cfg["k_values"],summaries,marker="o"); ax.set(xlabel="K",ylabel="Середній silhouette",title=f"Вибір K: {space}"); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(figdir/name,dpi=150); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4)); ax.hist(score_raw,bins=30,alpha=.55,label="IF original"); ax.hist(score_pca,bins=30,alpha=.55,label="IF PCA"); ax.axvline(thresholds["if_raw"],ls="--"); ax.set(title="Розподіли anomaly score на clean+controlled test",xlabel="Anomaly score",ylabel="Кількість"); ax.legend(); fig.tight_layout(); fig.savefig(figdir/"06_anomaly_score_distributions.png",dpi=150); plt.close(fig)
    # Threshold comparison now gets corresponding fit+validation metrics after q selection.
    summary={"metadata":{"script":"run_lr5_hill_valley.py","lab":"Лабораторна робота №5 — кластеризація, PCA та аномалії","generated_at_utc":datetime.now(timezone.utc).isoformat(),
        "dataset_source":source,"python_version":sys.version,"platform":platform.platform(),"processor":platform.processor(),"numpy_version":np.__version__,"pandas_version":pd.__version__,"scikit_learn_version":sklearn.__version__,"elapsed_seconds":None},
        "research_protocol":{"research_question":"Чи зберігає PCA-редукований простір структуру Hill-Valley для кластеризації та чи допомагає він виявляти контрольовані аномалії?",
        "variant":"Hill-Valley, selected dataset variant 1 from prior classification work","target":"class (0=valley, 1=hill); excluded from unsupervised fitting",
        "split":{"train_count":len(Xtr),"test_count":len(Xte),"test_fraction":float(cfg["test_size"]),"seed":int(cfg["seed"]),"stratified_for_composition_only":True,
            "train_original_row_ids":ids_train.tolist(),"test_original_row_ids":ids_test.tolist(),"overlap_count":len(set(ids_train)&set(ids_test))},
        "features":{"original_numeric_features":int(Xdf.shape[1]),"nonconstant_features_after_train_fit":int(Xtr.shape[1]),"n_rows":int(len(Xdf)),"class_counts":pd.Series(y).value_counts().sort_index().to_dict()},
        "data_audit":data_audit,
        "preprocessing":{"imputer":"SimpleImputer(strategy=median), fit only on train","constant_features":"VarianceThreshold(0), fit only on train","scaler":"MinMaxScaler, fit only on train","leakage_guard":"test transformed only; no fit/fit_transform on test"},
        "pca":{"full_fit_seconds":pca_fit_seconds,"explained_variance_threshold_comparison":threshold_rows,"working_threshold":float(cfg["working_pca_threshold"]),"working_q":working_q,
            "working_explained_variance":float(cumulative[working_q-1]),"variance_table":comp_df.to_dict(orient="records"),"first_two_components_variance":float(np.sum(ratios[:2])),
            "first_two_component_loadings":pca_full.components_[:2].tolist()},
        "clustering":{"k_grid":cfg["k_values"],"seeds":seed_list,"n_init":int(cfg["kmeans_n_init"]),
            "original_space":{"selected_k":k_raw,"selection_rule":"max mean silhouette; tie-break by lower Davies-Bouldin, then higher Calinski-Harabasz, then smaller K","k_summaries":raw_k_summary,"all_runs":raw_rows,"stability":raw_stability},
            "pca_space":{"selected_k":k_pca,"selection_rule":"same rule as original space","k_summaries":pca_k_summary,"all_runs":pca_rows,"stability":pca_stability},
            "external_audit_after_selection":audit,
            "train_cluster_labels_original":raw_train_labels.tolist(),"test_cluster_labels_original":raw_test_labels.tolist(),
            "train_cluster_labels_pca":pca_train_labels.tolist(),"test_cluster_labels_pca":pca_test_labels.tolist(),
            "train_pca_2d_coordinates":Z2.tolist(),"train_object_ids":ids_train.tolist()},
        "anomaly_detection":{"isolation_forest_config":if_cfg,"thresholds":thresholds,"controlled_anomaly_kinds":cfg["controlled_anomaly_kinds"],
            "controlled_anomaly_count":len(anomaly_X),"controlled_anomaly_manifest":anomaly_meta,
            "train_clean_score_distributions":{"if_original":distribution_stats(train_if_raw),"if_pca":distribution_stats(train_if_pca),
                "pca_reconstruction_error":distribution_stats(train_rec),"combined_percentile":distribution_stats(train_comb)},
            "test_clean_score_distributions":{"if_original":distribution_stats(score_raw[:len(Xte)]),"if_pca":distribution_stats(score_pca[:len(Xte)]),
                "pca_reconstruction_error":distribution_stats(score_rec[:len(Xte)]),"combined_percentile":distribution_stats(score_comb[:len(Xte)])},
            "controlled_anomaly_score_distributions_by_kind":{kind:{"if_original":distribution_stats([score_raw[len(Xte)+i] for i,m in enumerate(anomaly_meta) if m["anomaly_kind"]==kind]),
                "if_pca":distribution_stats([score_pca[len(Xte)+i] for i,m in enumerate(anomaly_meta) if m["anomaly_kind"]==kind]),
                "pca_reconstruction_error":distribution_stats([score_rec[len(Xte)+i] for i,m in enumerate(anomaly_meta) if m["anomaly_kind"]==kind]),
                "combined_percentile":distribution_stats([score_comb[len(Xte)+i] for i,m in enumerate(anomaly_meta) if m["anomaly_kind"]==kind])}
                for kind in cfg["controlled_anomaly_kinds"]},
            "evaluation_protocol":"clean test rows are negatives; deterministic modified copies are positives; threshold calibrated only from clean train scores",
            "metrics":anomaly_eval,"false_positive_false_negative_examples":error_examples,
            "top_k_jaccard_IF_original_vs_PCA":float(topk_j),"top_k":int(topk),
            "top_k_overlap_count":len(top_raw&top_pca),"top_k_union_count":len(top_raw|top_pca),
            "score_disagreement_count":int(np.sum(pred_raw!=pred_pca)),"score_disagreement_examples":disagreements,
            "evaluation_rows":anomaly_score_rows},
        "artifacts":{"joblib":"artifacts/lr5_hill_valley_pipeline.joblib","pca_variance_csv":"tables/pca_variance.csv","original_kmeans_csv":"tables/kmeans_original_runs.csv",
            "pca_kmeans_csv":"tables/kmeans_pca_runs.csv","anomaly_scores_csv":"tables/anomaly_scores.csv","controlled_anomaly_manifest":"tables/controlled_anomaly_manifest.json",
            "figures":[f"figures/{p.name}" for p in sorted(figdir.glob("*.png"))]},
        "limitations":["Controlled anomalies are artificial profile perturbations; performance on them does not prove detection quality for naturally occurring anomalies.",
            "Hill-Valley labels describe hill/valley shape, not anomaly status; class labels are used only for external cluster audit.",
            "PCA and K-Means results depend on scaling, q, K, and initialization; the 2D projection is visual only."]}}
    run_test_result=run_tests()
    summary["automated_tests"]=run_test_result
    model_bundle={"imputer":imputer,"variance_filter":variance,"scaler":scaler,"pca_full":pca_full,"working_q":working_q,
        "kmeans_original":km_raw,"kmeans_pca":km_pca,"isolation_forest_original":if_raw,"isolation_forest_pca":if_pca,
        "anomaly_thresholds":thresholds,"config":cfg,"feature_names":list(Xdf.columns)}
    joblib.dump(model_bundle,artifacts/"lr5_hill_valley_pipeline.joblib",compress=3)
    summary["metadata"]["elapsed_seconds"]=float(time.perf_counter()-T_START)
    (out/"lr5_hill_valley_results.json").write_text(json.dumps(json_safe(summary),ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    (out/"experiment_config.json").write_text(json.dumps(json_safe(cfg),ensure_ascii=False,indent=2),encoding="utf-8")
    return summary

T_START=time.perf_counter()

def main():
    ap=argparse.ArgumentParser(description="Run LR5 Hill-Valley unsupervised learning experiment")
    ap.add_argument("--data",default=None,help="Local .data/.csv/.txt or UCI ZIP; if omitted, download UCI archive")
    ap.add_argument("--out",default="lr5_hill_valley_output")
    ap.add_argument("--config",default=None,help="Optional JSON configuration overriding defaults")
    ap.add_argument("--test-only",action="store_true",help="Run embedded unit tests and exit")
    args=ap.parse_args()
    if args.test_only: run_tests(); return
    res=run_experiment(args.data,args.out,args.config)
    print(f"LR5 completed: {Path(args.out)/'lr5_hill_valley_results.json'}")
    print(f"Rows={res['research_protocol']['features']['n_rows']}; q={res['research_protocol']['pca']['working_q']}; K(raw/PCA)={res['research_protocol']['clustering']['original_space']['selected_k']}/{res['research_protocol']['clustering']['pca_space']['selected_k']}")

if __name__=="__main__": main()
