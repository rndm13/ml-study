#!/usr/bin/env python3
"""PR5: Multi-criteria engineering selection of Hill-Valley classifiers.

Fixed binary task: class 1 = hill (positive), class 0 = valley.
All candidates use the same stratified CV folds and the same fixed probability
threshold (0.45). The test split is not used for ranking, constraints, weights,
or sensitivity analysis; it is evaluated once after selections are frozen.
"""
from __future__ import annotations
import argparse, io, json, math, os, platform, sys, tempfile, time, urllib.request, zipfile, unittest
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

UCI_URL = "https://archive.ics.uci.edu/static/public/166/hill+valley.zip"
DATA_MEMBER = "Hill_Valley_without_noise_Training.data"
FIXED_THRESHOLD = 0.45
DEFAULT_CONFIG = {
  "protocol": {"test_size":0.20,"seed":42,"cv_splits":5,"cv_seed":42,"fixed_threshold":FIXED_THRESHOLD,
               "batch_size":256,"warmup_calls":7,"timed_calls":40,"fit_repeats":3,"primary_scenario":"realtime_cpu_service",
               "positive_class":1,"positive_class_name":"hill","negative_class_name":"valley"},
  "scenarios": [
    {"name":"realtime_cpu_service","description":"Навчальний сценарій синхронного CPU-сервісу з пріоритетом Recall та малих ресурсних витрат.",
     "constraints":{"cv_recall_mean":{"min":0.35},"latency_p95_ms":{"max":20.0},"model_size_mb":{"max":2.0}},
     "criteria":{"cv_f1_mean":{"direction":"maximize","weight":0.25,"worst":0.30,"best":0.75},
       "cv_recall_mean":{"direction":"maximize","weight":0.30,"worst":0.30,"best":0.85},
       "latency_p95_ms":{"direction":"minimize","weight":0.20,"worst":20.0,"best":0.5},
       "model_size_mb":{"direction":"minimize","weight":0.15,"worst":2.0,"best":0.005},
       "cv_f1_std":{"direction":"minimize","weight":0.10,"worst":0.20,"best":0.01}}},
    {"name":"batch_quality_priority","description":"Навчальний пакетний сценарій із вищою вагою ROC-AUC та якості прогнозу.",
     "constraints":{"cv_f1_mean":{"min":0.30},"cv_recall_mean":{"min":0.30},"latency_p95_ms":{"max":100.0},"model_size_mb":{"max":10.0}},
     "criteria":{"cv_roc_auc_mean":{"direction":"maximize","weight":0.45,"worst":0.50,"best":0.95},
       "cv_f1_mean":{"direction":"maximize","weight":0.30,"worst":0.30,"best":0.75},
       "cv_recall_mean":{"direction":"maximize","weight":0.10,"worst":0.30,"best":0.85},
       "fit_seconds":{"direction":"minimize","weight":0.05,"worst":15.0,"best":0.01},
       "cv_f1_std":{"direction":"minimize","weight":0.10,"worst":0.20,"best":0.01}}}
  ],
  "sensitivity":{"scenario":"batch_quality_priority","criterion":"cv_roc_auc_mean","weights": [0.10,0.125,0.15,0.175,0.20,0.225,0.25,0.275,0.30,0.325,0.35,0.375,0.40,0.425,0.45,0.475,0.50,0.525,0.55,0.575,0.60,0.625,0.65,0.675,0.70]}
}


def json_safe(obj):
    if isinstance(obj,dict): return {str(k):json_safe(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)): return [json_safe(v) for v in obj]
    if isinstance(obj,np.ndarray): return json_safe(obj.tolist())
    if isinstance(obj,(np.integer,)): return int(obj)
    if isinstance(obj,(np.floating,float)): return float(obj) if np.isfinite(obj) else None
    if isinstance(obj,(np.bool_,)): return bool(obj)
    if isinstance(obj,Path): return str(obj)
    return obj


def read_hill_valley(data_path: str | None = None):
    source=None
    if data_path:
        p=Path(data_path).expanduser().resolve()
        if not p.exists(): raise FileNotFoundError(f"Data file not found: {p}")
        if zipfile.is_zipfile(p):
            with zipfile.ZipFile(p) as zf:
                match=next((n for n in zf.namelist() if n.endswith(DATA_MEMBER)),None)
                if match is None: raise ValueError(f"{DATA_MEMBER} not found in ZIP")
                txt=zf.read(match).decode("utf-8",errors="replace"); source=f"{p}::{match}"
        else: txt=p.read_text(encoding="utf-8",errors="replace"); source=str(p)
    else:
        try:
            req=urllib.request.Request(UCI_URL,headers={"User-Agent":"Mozilla/5.0"})
            with urllib.request.urlopen(req,timeout=45) as resp: blob=resp.read()
            with zipfile.ZipFile(io.BytesIO(blob)) as zf:
                match=next((n for n in zf.namelist() if n.endswith(DATA_MEMBER)),None)
                if match is None: raise ValueError(f"{DATA_MEMBER} not found in UCI archive")
                txt=zf.read(match).decode("utf-8",errors="replace"); source=f"UCI:{UCI_URL}::{match}"
        except Exception as e: raise RuntimeError("Could not download UCI archive; pass --data PATH to the local .data or ZIP file") from e
    try: df=pd.read_csv(io.StringIO(txt),header=None,sep=r"\s*,\s*",engine="python")
    except Exception: df=pd.read_csv(io.StringIO(txt),header=None)
    if df.shape[1]==1: df=pd.read_csv(io.StringIO(txt),header=None,sep=r"\s+",engine="python")
    numeric=df.apply(pd.to_numeric,errors="coerce")
    if len(numeric) and numeric.iloc[0].isna().mean()>.5:
        df=df.iloc[1:].reset_index(drop=True); numeric=df.apply(pd.to_numeric,errors="coerce")
    if numeric.shape[1]<11: raise ValueError(f"Expected at least 10 features and one target; got {numeric.shape}")
    X=numeric.iloc[:,:-1].copy(); yraw=df.iloc[:,-1].astype(str).str.strip()
    ynum=pd.to_numeric(yraw,errors="coerce")
    if ynum.notna().mean()>=.95: y=ynum.astype(int).to_numpy()
    else:
        y=yraw.str.lower().map({"valley":0,"hill":1,"0":0,"1":1}).to_numpy()
        if pd.isna(y).any(): raise ValueError("Cannot map target to 0=valley and 1=hill")
        y=y.astype(int)
    if set(np.unique(y))!={0,1}: raise ValueError(f"Expected binary target 0/1; got {np.unique(y)}")
    X.columns=[f"X{i+1}" for i in range(X.shape[1])]
    return X,y,source


def build_candidates(seed=42):
    return {
      "logistic_regression": Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler()),
          ("classifier",LogisticRegression(C=1.0,max_iter=3000,solver="lbfgs",random_state=seed))]),
      "linear_svm_calibrated_C100": Pipeline([("imputer",SimpleImputer(strategy="median")),
          ("classifier",CalibratedClassifierCV(estimator=LinearSVC(C=100.0,random_state=seed),method="sigmoid",cv=3))]),
      "controlled_decision_tree": Pipeline([("imputer",SimpleImputer(strategy="median")),
          ("classifier",DecisionTreeClassifier(max_depth=8,min_samples_leaf=4,criterion="gini",random_state=seed))]),
      "random_forest": Pipeline([("imputer",SimpleImputer(strategy="median")),
          ("classifier",RandomForestClassifier(n_estimators=200,max_features="sqrt",min_samples_leaf=2,
              class_weight="balanced",random_state=seed,n_jobs=1))]),
      "hist_gradient_boosting": Pipeline([("imputer",SimpleImputer(strategy="median")),
          ("classifier",HistGradientBoostingClassifier(max_iter=150,learning_rate=.06,max_leaf_nodes=15,
              l2_regularization=.5,random_state=seed))])
    }


def metric_pack(y_true, probabilities, threshold=FIXED_THRESHOLD):
    y=np.asarray(y_true,dtype=int); p=np.asarray(probabilities,dtype=float)
    if y.shape!=p.shape or y.ndim!=1 or not np.isfinite(p).all(): raise ValueError("Invalid y/probability arrays")
    pred=(p>=float(threshold)).astype(int); cm=confusion_matrix(y,pred,labels=[0,1])
    return {"f1":float(f1_score(y,pred,zero_division=0)),"precision":float(precision_score(y,pred,zero_division=0)),
        "recall":float(recall_score(y,pred,zero_division=0)),"balanced_accuracy":float(balanced_accuracy_score(y,pred)),
        "roc_auc":float(roc_auc_score(y,p)),"accuracy":float(accuracy_score(y,pred)),
        "brier_score":float(brier_score_loss(y,p)),"positive_action_rate":float(pred.mean()),
        "confusion_matrix_tn_fp_fn_tp":cm.tolist(),"predictions":pred.tolist()}


def evaluate_cv(candidates, X, y, n_splits, seed, threshold):
    cv=StratifiedKFold(n_splits=int(n_splits),shuffle=True,random_state=int(seed))
    splits=list(cv.split(X,y)); results={}; oof_by_model={}
    for name,estimator in candidates.items():
        fold_rows=[]; oof=np.full(len(y),np.nan,dtype=float)
        for fold,(tr,va) in enumerate(splits,1):
            model=clone(estimator); t0=time.perf_counter(); model.fit(X.iloc[tr],y[tr]); fit_s=time.perf_counter()-t0
            p_val=model.predict_proba(X.iloc[va])[:,1]; p_train=model.predict_proba(X.iloc[tr])[:,1]
            val=metric_pack(y[va],p_val,threshold); train=metric_pack(y[tr],p_train,threshold)
            oof[va]=p_val
            fold_rows.append({"fold":fold,"train_indices":tr.tolist(),"validation_indices":va.tolist(),"fit_seconds":float(fit_s),
                "train_metrics":{k:v for k,v in train.items() if k!="predictions"},"validation_metrics":{k:v for k,v in val.items() if k!="predictions"},
                "validation_predictions":val["predictions"],"validation_probabilities":p_val.tolist()})
        if np.isnan(oof).any(): raise AssertionError(f"OOF coverage incomplete for {name}")
        metric_names=["f1","precision","recall","balanced_accuracy","roc_auc","accuracy","brier_score","positive_action_rate"]
        summary={}
        for key in metric_names:
            vals=np.array([r["validation_metrics"][key] for r in fold_rows],dtype=float)
            train_vals=np.array([r["train_metrics"][key] for r in fold_rows],dtype=float)
            summary[f"cv_{key}_mean"]=float(np.mean(vals)); summary[f"cv_{key}_std"]=float(np.std(vals,ddof=1))
            summary[f"cv_{key}_min"]=float(np.min(vals)); summary[f"cv_{key}_max"]=float(np.max(vals))
            summary[f"train_{key}_mean"]=float(np.mean(train_vals))
        summary["cv_f1_generalization_gap"]=summary["train_f1_mean"]-summary["cv_f1_mean"]
        results[name]={"summary":summary,"folds":fold_rows}
        oof_by_model[name]=oof.tolist()
    return results,oof_by_model,splits


def profile_model(model, X_train, batch_size=256, warmups=7, repeats=40, fit_repeats=3, artifact_path=None):
    n=len(X_train); rng=np.random.default_rng(42)
    if n>=batch_size: batch=X_train.iloc[:batch_size].copy()
    else: batch=X_train.iloc[rng.choice(n,size=batch_size,replace=True)].copy()
    fit_samples=[]
    for _ in range(max(1,int(fit_repeats))):
        m=clone(model); t0=time.perf_counter(); m.fit(X_train.iloc[:],_PROFILE_Y); fit_samples.append(time.perf_counter()-t0)
    fitted=clone(model); fitted.fit(X_train.iloc[:],_PROFILE_Y)
    for _ in range(int(warmups)): fitted.predict_proba(batch)
    elapsed=[]
    for _ in range(int(repeats)):
        t0=time.perf_counter_ns(); fitted.predict_proba(batch); elapsed.append((time.perf_counter_ns()-t0)/1_000_000)
    med=float(np.median(elapsed)); p95=float(np.quantile(elapsed,.95)); throughput=float(batch_size/(med/1000)) if med>0 else None
    size_mb=None
    if artifact_path:
        joblib.dump(fitted,artifact_path,compress=3); size_mb=Path(artifact_path).stat().st_size/(1024**2)
    return fitted,{"fit_seconds_median":float(np.median(fit_samples)),"fit_seconds_repeats":fit_samples,
        "latency_p50_ms":med,"latency_p95_ms":p95,"latency_measurements_ms":elapsed,"batch_size":int(batch_size),
        "warmup_calls":int(warmups),"timed_calls":int(repeats),"throughput_samples_s":throughput,"model_size_mb":size_mb}

_PROFILE_Y=None


def check_constraints(metrics, constraints):
    violations=[]
    for metric,limits in constraints.items():
        if metric not in metrics or metrics[metric] is None:
            violations.append(f"{metric}: missing value"); continue
        try: value=float(metrics[metric])
        except (TypeError,ValueError): violations.append(f"{metric}: non-numeric value"); continue
        if not np.isfinite(value): violations.append(f"{metric}: non-finite value"); continue
        if "min" in limits and value<float(limits["min"]): violations.append(f"{metric}: {value} < min {limits['min']}")
        if "max" in limits and value>float(limits["max"]): violations.append(f"{metric}: {value} > max {limits['max']}")
    return len(violations)==0,violations


def validate_scenario(config):
    if not config.get("name"): raise ValueError("Scenario must have a name")
    criteria=config.get("criteria",{})
    if not criteria: raise ValueError("Scenario criteria are empty")
    weights=[]
    for name,spec in criteria.items():
        direction=spec.get("direction")
        if direction not in {"maximize","minimize"}: raise ValueError(f"Unknown direction for {name}: {direction}")
        w=float(spec.get("weight",-1)); worst=float(spec["worst"]); best=float(spec["best"])
        if w<0 or not np.isfinite(w): raise ValueError(f"Invalid weight for {name}")
        if direction=="maximize" and not best>worst: raise ValueError(f"For maximize {name}, best must exceed worst")
        if direction=="minimize" and not best<worst: raise ValueError(f"For minimize {name}, best must be less than worst")
        weights.append(w)
    if not np.isclose(sum(weights),1.0,atol=1e-8): raise ValueError(f"Scenario weights must sum to 1.0, got {sum(weights)}")
    return True


def normalize_value(value, specification):
    x=float(value); direction=specification["direction"]; worst=float(specification["worst"]); best=float(specification["best"])
    if not np.isfinite(x): raise ValueError("Cannot normalize non-finite value")
    if direction=="maximize": raw=(x-worst)/(best-worst)
    elif direction=="minimize": raw=(worst-x)/(worst-best)
    else: raise ValueError(f"Unknown direction: {direction}")
    return float(np.clip(raw,0.0,1.0))


def rank_models(results, config):
    validate_scenario(config); allowed=[]; rejected=[]
    for model_name,metrics in results.items():
        ok,violations=check_constraints(metrics,config.get("constraints",{}))
        if not ok:
            rejected.append({"model":model_name,"eligible":False,"violations":violations,"metrics":metrics}); continue
        components={}
        score=0.0
        for metric,spec in config["criteria"].items():
            if metric not in metrics or metrics[metric] is None: raise ValueError(f"{model_name}: ranking metric {metric} is missing")
            component=normalize_value(metrics[metric],spec); components[metric]={"raw":float(metrics[metric]),"normalized":component,"weight":float(spec["weight"])}
            score+=component*float(spec["weight"])
        allowed.append({"model":model_name,"eligible":True,"score":float(score),"components":components,"metrics":metrics})
    allowed.sort(key=lambda x:(-x["score"],x["model"]))
    for i,row in enumerate(allowed,1): row["rank"]=i
    rejected.sort(key=lambda x:x["model"])
    return allowed,rejected


def make_sensitivity_config(base_config, criterion, new_weight):
    c=json.loads(json.dumps(base_config)); specs=c["criteria"]
    if criterion not in specs: raise ValueError(f"Sensitivity criterion {criterion} not in scenario")
    old=float(specs[criterion]["weight"]); remaining=1.0-old; new_remaining=1.0-float(new_weight)
    if not 0<=new_weight<=1: raise ValueError("Sensitivity weight must be between 0 and 1")
    if remaining<=0: raise ValueError("Cannot proportionally rescale other weights")
    specs[criterion]["weight"]=float(new_weight)
    for name,spec in specs.items():
        if name!=criterion: spec["weight"]=float(spec["weight"])*new_remaining/remaining
    validate_scenario(c)
    return c


def select_for_scenario(metric_rows, scenario):
    ranking,rejected=rank_models(metric_rows,scenario)
    return {"scenario":scenario["name"],"description":scenario.get("description"),"ranking":ranking,"rejected":rejected,
        "main_model":ranking[0]["model"] if ranking else None,"reserve_model":ranking[1]["model"] if len(ranking)>1 else None,
        "eligible_count":len(ranking),"rejected_count":len(rejected)}


def run_tests():
    class Tests(unittest.TestCase):
        def test_maximize_norm(self): self.assertAlmostEqual(normalize_value(.75,{"direction":"maximize","worst":.25,"best":.75}),1)
        def test_minimize_norm(self): self.assertAlmostEqual(normalize_value(1,{"direction":"minimize","worst":10,"best":1}),1)
        def test_clip_high(self): self.assertEqual(normalize_value(99,{"direction":"maximize","worst":0,"best":1}),1)
        def test_clip_low(self): self.assertEqual(normalize_value(-99,{"direction":"maximize","worst":0,"best":1}),0)
        def test_exact_constraint_boundary(self): self.assertEqual(check_constraints({"x":.5},{"x":{"min":.5}}),(True,[]))
        def test_constraint_collects_reasons(self):
            ok,v=check_constraints({"a":.1,"b":10},{"a":{"min":.2},"b":{"max":5}}); self.assertFalse(ok); self.assertEqual(len(v),2)
        def test_missing_metric_rejected(self): self.assertFalse(check_constraints({}, {"missing":{"min":0}})[0])
        def test_weights_invalid(self):
            cfg={"name":"x","criteria":{"a":{"direction":"maximize","weight":.8,"worst":0,"best":1}}}
            with self.assertRaises(ValueError): validate_scenario(cfg)
        def test_empty_eligible_set(self):
            ranking,rejected=rank_models({"m":{"x":0.1}},{"name":"x","constraints":{"x":{"min":.9}},"criteria":{"x":{"direction":"maximize","weight":1,"worst":0,"best":1}}})
            self.assertEqual(len(ranking),0); self.assertEqual(rejected[0]["model"],"m")
        def test_tie_break_alphabetical(self):
            cfg={"name":"x","constraints":{},"criteria":{"x":{"direction":"maximize","weight":1,"worst":0,"best":1}}}
            ranking,_=rank_models({"b":{"x":.5},"a":{"x":.5}},cfg); self.assertEqual(ranking[0]["model"],"a")
        def test_invalid_direction(self):
            with self.assertRaises(ValueError): normalize_value(1,{"direction":"sideways","worst":0,"best":1})
        def test_no_eligible_reason_missing(self): self.assertIn("missing value",check_constraints({}, {"x":{"max":2}})[1][0])
        def test_sensitivity_weights_sum(self):
            cfg={"name":"x","criteria":{"a":{"direction":"maximize","weight":.5,"worst":0,"best":1},"b":{"direction":"minimize","weight":.5,"worst":2,"best":0}}}
            new=make_sensitivity_config(cfg,"a",.7); self.assertAlmostEqual(sum(v["weight"] for v in new["criteria"].values()),1)
        def test_probability_threshold_semantics(self): self.assertEqual((np.array([.44,.45,.46])>=.45).astype(int).tolist(),[0,1,1])
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful(): raise SystemExit("PR5 unit tests failed")
    return {"tests_run":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),"successful":result.wasSuccessful()}


def summarize_candidate_cv(cv_result):
    s=cv_result["summary"]
    return {"cv_f1_mean":s["cv_f1_mean"],"cv_f1_std":s["cv_f1_std"],"cv_f1_min":s["cv_f1_min"],
        "cv_recall_mean":s["cv_recall_mean"],"cv_precision_mean":s["cv_precision_mean"],
        "cv_roc_auc_mean":s["cv_roc_auc_mean"],"cv_balanced_accuracy_mean":s["cv_balanced_accuracy_mean"],
        "cv_accuracy_mean":s["cv_accuracy_mean"],"cv_brier_score_mean":s["cv_brier_score_mean"],
        "cv_positive_action_rate_mean":s["cv_positive_action_rate_mean"],"cv_f1_generalization_gap":s["cv_f1_generalization_gap"]}


def run_experiment(data_path, out_dir, config_path=None):
    cfg=json.loads(json.dumps(DEFAULT_CONFIG))
    if config_path:
        user=json.loads(Path(config_path).read_text(encoding="utf-8"));
        for key,val in user.items(): cfg[key]=val
    protocol=cfg["protocol"]; threshold=float(protocol["fixed_threshold"])
    if not 0<threshold<1: raise ValueError("fixed_threshold must be in (0,1)")
    if len(cfg["scenarios"])<2: raise ValueError("At least two engineering scenarios are required")
    for scenario in cfg["scenarios"]: validate_scenario(scenario)
    out=Path(out_dir); tables=out/"tables"; figs=out/"figures"; artifacts=out/"artifacts"; configs=out/"configs"
    for p in (out,tables,figs,artifacts,configs): p.mkdir(parents=True,exist_ok=True)
    Xdf,y,source=read_hill_valley(data_path); ids=np.arange(len(y))
    X_train,X_test,y_train,y_test,ids_train,ids_test=train_test_split(Xdf,y,ids,test_size=float(protocol["test_size"]),stratify=y,random_state=int(protocol["seed"]))
    if set(ids_train)&set(ids_test): raise AssertionError("Train/test overlap")
    candidates=build_candidates(int(protocol["seed"]))
    if len(candidates)<4: raise AssertionError("At least four candidates are required")
    global _PROFILE_Y
    _PROFILE_Y=y_train
    cv_results,oof,splits=evaluate_cv(candidates,X_train,y_train,protocol["cv_splits"],protocol["cv_seed"],threshold)
    metric_rows={}
    fold_records=[]
    for name,item in cv_results.items():
        metric_rows[name]=summarize_candidate_cv(item)
        fold_records.extend([{"model":name,**f} for f in item["folds"]])
    # Resource profiling is performed on the same host and fixed batch for all pipelines.
    fitted_models={}; profiles={}
    for name,est in candidates.items():
        model_path=artifacts/f"candidate_{name}.joblib"
        fitted,profile=profile_model(est,X_train,protocol["batch_size"],protocol["warmup_calls"],protocol["timed_calls"],protocol["fit_repeats"],model_path)
        fitted_models[name]=fitted; profiles[name]=profile
        metric_rows[name].update({"fit_seconds":profile["fit_seconds_median"],"latency_p50_ms":profile["latency_p50_ms"],
            "latency_p95_ms":profile["latency_p95_ms"],"throughput_samples_s":profile["throughput_samples_s"],"model_size_mb":profile["model_size_mb"]})
    # Export CV folds and OOF predictions; no test label has been used so far.
    oof_rows=[]
    for name,probs in oof.items():
        pred=(np.asarray(probs)>=threshold).astype(int)
        for i,(prob,phat) in enumerate(zip(probs,pred)):
            oof_rows.append({"model":name,"train_local_index":i,"original_row_id":int(ids_train[i]),"y_true":int(y_train[i]),"oof_probability":float(prob),"oof_prediction":int(phat),"threshold":threshold})
    pd.DataFrame(fold_records).to_json(tables/"cv_fold_results.json",orient="records",indent=2)
    pd.DataFrame(oof_rows).to_csv(tables/"oof_predictions.csv",index=False)
    pd.DataFrame([{"model":k,**v} for k,v in metric_rows.items()]).to_csv(tables/"benchmark_results.csv",index=False)
    # Two config-driven scenario rankings, with all rejection reasons.
    selections={}; scenario_by_name={s["name"]:s for s in cfg["scenarios"]}
    for scenario in cfg["scenarios"]:
        result=select_for_scenario(metric_rows,scenario); selections[scenario["name"]]=result
        pd.DataFrame(result["ranking"]).to_csv(tables/f"ranking_{scenario['name']}.csv",index=False)
        pd.DataFrame(result["rejected"]).to_json(tables/f"rejected_{scenario['name']}.json",orient="records",indent=2)
        (configs/f"selection_{scenario['name']}.json").write_text(json.dumps(json_safe(result),ensure_ascii=False,indent=2),encoding="utf-8")
    primary_name=protocol["primary_scenario"]
    if primary_name not in selections: raise ValueError(f"primary_scenario {primary_name} is not defined")
    primary=selections[primary_name]
    # Sensitivity: vary only the named weight; all other weights are rescaled proportionally.
    sens_cfg=cfg["sensitivity"]; sens_scenario_name=sens_cfg["scenario"]
    if sens_scenario_name not in scenario_by_name: raise ValueError("Sensitivity scenario not defined")
    base_scenario=scenario_by_name[sens_scenario_name]; sens_rows=[]
    for w in sens_cfg["weights"]:
        sc=make_sensitivity_config(base_scenario,sens_cfg["criterion"],float(w)); ranking,rejected=rank_models(metric_rows,sc)
        sens_rows.append({"weight":float(w),"winner":ranking[0]["model"] if ranking else None,
            "winner_score":ranking[0]["score"] if ranking else None,"ranking":[{"model":x["model"],"score":x["score"],"rank":x["rank"]} for x in ranking],
            "eligible_count":len(ranking),"rejected_count":len(rejected)})
    pd.DataFrame(sens_rows).to_json(tables/"weight_sensitivity.json",orient="records",indent=2)
    winner_counts=pd.Series([x["winner"] for x in sens_rows if x["winner"]]).value_counts().to_dict()
    # Freeze all selections before any test metric is computed.
    primary_expected=None
    if primary.get("main_model"):
        pm=metric_rows[primary["main_model"]]
        primary_expected={"metric":"cv_f1_mean","mean":pm["cv_f1_mean"],"std":pm["cv_f1_std"],
            "mean_plus_minus_one_std":[max(0.0,pm["cv_f1_mean"]-pm["cv_f1_std"]),min(1.0,pm["cv_f1_mean"]+pm["cv_f1_std"])]}
    freeze={"created_at_utc":datetime.now(timezone.utc).isoformat(),"primary_scenario":primary_name,
        "primary_main_model":primary["main_model"],"primary_reserve_model":primary["reserve_model"],
        "scenario_selections":{n:{"main_model":s["main_model"],"reserve_model":s["reserve_model"]} for n,s in selections.items()},
        "expected_primary_metric_range":primary_expected,
        "known_risks":["latency and serialized size depend on the host environment","scenario constraints are illustrative assumptions","probability threshold 0.45 is fixed for this comparison"],
        "reselection_conditions":["change in data distribution or class prevalence","deployment hardware or latency budget changes","hard constraint violation in monitoring","new candidate pipeline or library version is introduced"],
        "fixed_threshold":threshold,"test_set_used_for_selection":False}
    (configs/"frozen_model_selection.json").write_text(json.dumps(json_safe(freeze),ensure_ascii=False,indent=2),encoding="utf-8")
    # Save stable main/reserve named copies before final test access.
    for sn,sel in selections.items():
        for role,key in [("main","main_model"),("reserve","reserve_model")]:
            name=sel.get(key)
            if name:
                joblib.dump(fitted_models[name],artifacts/f"{role}_{sn}_{name}.joblib",compress=3)
    # Test set is opened only now, for the primary scenario's frozen main model.
    final_name=primary.get("main_model"); test_result=None
    if final_name:
        p=fitted_models[final_name].predict_proba(X_test)[:,1]
        test_result={"model":final_name,"metrics":metric_pack(y_test,p,threshold),"probabilities":p.tolist(),
            "y_true":y_test.tolist(),"original_row_ids":ids_test.tolist(),"threshold":threshold,
            "comparison_to_cv":{"cv_f1_mean":metric_rows[final_name]["cv_f1_mean"],"test_f1":float(f1_score(y_test,(p>=threshold).astype(int))),
                "cv_recall_mean":metric_rows[final_name]["cv_recall_mean"],"test_recall":float(recall_score(y_test,(p>=threshold).astype(int),zero_division=0))}}
    # Generate human-readable selection note and visual diagnostics.
    for sn,sel in selections.items():
        lines=[f"# Model selection: {sn}","",f"Main: {sel['main_model']}",f"Reserve: {sel['reserve_model']}","", "## Eligible ranking"]
        lines += [f"{r['rank']}. {r['model']} — {r['score']:.4f}" for r in sel["ranking"]]
        lines += ["", "## Rejected candidates"]
        for r in sel["rejected"]: lines.append(f"- {r['model']}: {'; '.join(r['violations'])}")
        (out/f"MODEL_SELECTION_{sn}.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    names=list(metric_rows)
    fig,ax=plt.subplots(figsize=(7,5))
    for name in names: ax.scatter(metric_rows[name]["cv_f1_mean"],metric_rows[name]["latency_p95_ms"],s=60); ax.annotate(name,(metric_rows[name]["cv_f1_mean"],metric_rows[name]["latency_p95_ms"]),fontsize=7)
    ax.set(xlabel="CV F1 (threshold=0.45)",ylabel="p95 latency (ms / batch)",title="Якість проти затримки"); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(figs/"01_quality_vs_latency.png",dpi=150); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    for name in names: ax.plot(np.arange(1,len(profiles[name]["latency_measurements_ms"])+1),profiles[name]["latency_measurements_ms"],label=name,alpha=.8)
    ax.set(xlabel="Повтор вимірювання",ylabel="Час predict_proba пакета, мс",title="Повторні вимірювання затримки"); ax.legend(fontsize=7); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(figs/"02_latency_repeats.png",dpi=150); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    for sn,sel in selections.items():
        if sel["main_model"]: ax.bar(sn,metric_rows[sel["main_model"]]["cv_f1_mean"],label=sel["main_model"])
    ax.set(ylabel="CV F1",title="Переможці інженерних сценаріїв"); ax.tick_params(axis="x",rotation=10); ax.grid(axis="y",alpha=.25); fig.tight_layout(); fig.savefig(figs/"03_scenario_winners.png",dpi=150); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    for name in names: ax.plot([0,1],[metric_rows[name]["model_size_mb"],metric_rows[name]["model_size_mb"]],alpha=.0)
    ax.bar(names,[metric_rows[n]["model_size_mb"] for n in names]); ax.set(ylabel="Розмір joblib pipeline, МБ",title="Розмір серіалізованих моделей"); ax.tick_params(axis="x",rotation=25); ax.grid(axis="y",alpha=.25); fig.tight_layout(); fig.savefig(figs/"04_model_sizes.png",dpi=150); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    ax.plot([r["weight"] for r in sens_rows],[r["winner_score"] if r["winner_score"] is not None else np.nan for r in sens_rows],marker="o")
    ax.set(xlabel=f"Вага {sens_cfg['criterion']}",ylabel="Балл переможця",title="Чутливість до ваги критерію"); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(figs/"05_weight_sensitivity.png",dpi=150); plt.close(fig)
    test_csv=None
    if test_result:
        test_csv=[{"original_row_id":rid,"y_true":yt,"probability_hill":pr,"prediction_hill":int(pr>=threshold),"threshold":threshold} for rid,yt,pr in zip(ids_test,y_test,test_result["probabilities"])]
        pd.DataFrame(test_csv).to_csv(tables/"final_test_predictions.csv",index=False)
        cm=np.array(test_result["metrics"]["confusion_matrix_tn_fp_fn_tp"])
        fig,ax=plt.subplots(figsize=(4,4)); ax.imshow(cm,cmap="Blues");
        for (i,j),v in np.ndenumerate(cm): ax.text(j,i,str(v),ha="center",va="center")
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=["Valley","Hill"],yticklabels=["Valley","Hill"],xlabel="Прогноз",ylabel="Факт",title="Test confusion matrix"); fig.tight_layout(); fig.savefig(figs/"06_test_confusion_matrix.png",dpi=150); plt.close(fig)
    # Save all measured evidence and the chosen fitted pipelines in a single JSON.
    tests=run_tests()
    reload_check=None
    if final_name:
        frozen_path=artifacts/f"main_{primary_name}_{final_name}.joblib"; reloaded=joblib.load(frozen_path)
        probe=reloaded.predict_proba(X_test.iloc[:min(5,len(X_test))])[:,1]
        reload_check={"artifact":str(frozen_path.relative_to(out)),"loaded_successfully":True,"probe_count":int(len(probe)),"probability_range":[float(np.min(probe)),float(np.max(probe))]}
    result={"metadata":{"script":"run_pr5_hill_valley.py","practical_work":"Практична робота №5 — багатокритеріальний вибір ML-моделі за інженерними обмеженнями",
        "generated_at_utc":datetime.now(timezone.utc).isoformat(),"dataset_source":source,"python_version":sys.version,"platform":platform.platform(),
        "processor":platform.processor(),"numpy_version":np.__version__,"pandas_version":pd.__version__,"scikit_learn_version":sklearn.__version__,"elapsed_seconds":float(time.perf_counter()-START_TIME)},
      "dataset":{"name":"Hill-Valley without noise — Training file","rows":int(len(Xdf)),"feature_count":int(Xdf.shape[1]),"feature_names":list(Xdf.columns),
        "target":"class","class_mapping":{"0":"valley","1":"hill"},"class_counts":pd.Series(y).value_counts().sort_index().to_dict(),
        "missing_values_per_feature":Xdf.isna().sum().to_dict(),"train_count":len(X_train),"test_count":len(X_test),
        "train_class_counts":pd.Series(y_train).value_counts().sort_index().to_dict(),"test_class_counts":pd.Series(y_test).value_counts().sort_index().to_dict(),
        "train_original_row_ids":ids_train.tolist(),"test_original_row_ids":ids_test.tolist(),"split_overlap_count":len(set(ids_train)&set(ids_test))},
      "engineering_context":{"task":"Binary classification of 100-point numeric profiles as Valley (0) or Hill (1).",
        "positive_class":"hill (1)","decision_consumer":"CPU service for profile-shape screening; high-score hill profiles can be prioritized for secondary review.",
        "error_consequences":{"false_positive":"valley profile is routed to additional review unnecessarily","false_negative":"hill profile is not prioritized and can be missed"},
        "resource_constraints_are_educational_assumptions":True,"scenario_definitions":cfg["scenarios"]},
      "protocol":{"test_size":protocol["test_size"],"random_state":protocol["seed"],"stratified":True,"cv":"StratifiedKFold",
        "cv_splits":protocol["cv_splits"],"cv_random_state":protocol["cv_seed"],"fixed_threshold":threshold,
        "threshold_note":"0.45 fixed before comparisons and applied to all models; no threshold optimization in PR5.",
        "preprocessing":"Median imputation within each pipeline; standardization only for logistic regression and calibrated linear SVM.",
        "test_used_for_model_selection":False,"test_access":"one final evaluation after both scenario rankings and main/reserve selections were frozen"},
      "candidate_configurations":{"logistic_regression":"median imputer + StandardScaler + LogisticRegression(C=1,max_iter=3000)",
        "linear_svm_calibrated_C100":"median imputer + sigmoid-Calibrated LinearSVC(C=100, inner cv=3), no scaling (continues prior LinearSVM preprocessing)",
        "controlled_decision_tree":"median imputer + DecisionTree(max_depth=8,min_samples_leaf=4)",
        "random_forest":"median imputer + RandomForest(n_estimators=200,max_features=sqrt,min_samples_leaf=2,class_weight=balanced)",
        "hist_gradient_boosting":"median imputer + HistGradientBoosting(max_iter=150,learning_rate=.06,max_leaf_nodes=15,l2=.5)"},
      "cross_validation":{"candidate_results":cv_results,"summary_metrics":metric_rows,"oof_predictions":oof,"fold_split_indices":[{"fold":i+1,"train_indices":tr.tolist(),"validation_indices":va.tolist()} for i,(tr,va) in enumerate(splits)]},
      "profiling":{"fixed_batch_protocol":{"batch_size":protocol["batch_size"],"warmups":protocol["warmup_calls"],"timed_calls":protocol["timed_calls"],"fit_repeats":protocol["fit_repeats"],"timer":"time.perf_counter_ns; monotonic high-resolution"},"per_model":profiles},
      "scenario_selection":selections,
      "sensitivity_analysis":{"scenario":sens_scenario_name,"criterion":sens_cfg["criterion"],"runs":sens_rows,"winner_counts":winner_counts,
        "winner_changed":len(winner_counts)>1,"weight_range":[min(sens_cfg["weights"]),max(sens_cfg["weights"])]},
      "frozen_decision":freeze,"final_test_evaluation":test_result,"saved_model_reload_test":reload_check,
      "automated_tests":tests,
      "artifacts":{"benchmark_csv":"tables/benchmark_results.csv","fold_json":"tables/cv_fold_results.json","oof_csv":"tables/oof_predictions.csv",
        "scenario_rankings":[f"tables/ranking_{s['name']}.csv" for s in cfg["scenarios"]],"sensitivity_json":"tables/weight_sensitivity.json",
        "main_and_reserve_pipeline_artifacts":True,"figures":[f"figures/{p.name}" for p in sorted(figs.glob("*.png"))]},
      "limitations":["Hill-Valley is a synthetic two-class profile-shape dataset; the engineering costs and service limits are illustrative assumptions, not measured business requirements.",
        "Latency and serialized size depend on this machine, operating system and library versions; repeat profiling on the deployment host.",
        "The probability threshold 0.45 is fixed for the comparison and is not a universal operating point.",
        "Test metrics are reported only for the frozen primary-scenario winner; they were not used to adjust the model or scenario weights."]}
    (out/"pr5_hill_valley_results.json").write_text(json.dumps(json_safe(result),ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    (out/"scenario_config.json").write_text(json.dumps(json_safe(cfg),ensure_ascii=False,indent=2),encoding="utf-8")
    return result

START_TIME=time.perf_counter()

def main():
    ap=argparse.ArgumentParser(description="Run PR5 Hill-Valley multi-criteria model selection")
    ap.add_argument("--data",default=None,help="Local .data/.csv/.txt or UCI ZIP; omitted means download from UCI")
    ap.add_argument("--out",default="pr5_hill_valley_output")
    ap.add_argument("--config",default=None,help="JSON with two or more engineering scenarios")
    ap.add_argument("--test-only",action="store_true")
    args=ap.parse_args()
    if args.test_only: run_tests(); return
    res=run_experiment(args.data,args.out,args.config)
    print(f"PR5 completed: {Path(args.out)/'pr5_hill_valley_results.json'}")
    for s,r in res["scenario_selection"].items(): print(f"{s}: main={r['main_model']}, reserve={r['reserve_model']}")

if __name__=="__main__": main()
