
import re, pickle, numpy as np, pandas as pd, xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from pathlib import Path

CLASS_NAMES = ["too_small", "fit", "too_large"]
LABEL_SHIFT = 1
SHAPE_SHOULDER_FACTOR = {
    "inverted_triangle":0.47,"hourglass":0.44,"rectangle":0.43,
    "apple":0.42,"pear":0.40,"unknown":0.43,
}
CATEGORY_MAP = {
    "dress":"dress","dresses":"dress","top":"top","tops":"top",
    "blouse":"top","shirt":"top","bottom":"bottom","pants":"bottom",
    "skirt":"bottom","jeans":"bottom","jacket":"outerwear",
    "coat":"outerwear","blazer":"outerwear",
    "jumpsuit":"one_piece","romper":"one_piece",
}

class ModelArtifacts:
    def __init__(self):
        self.booster=None; self.feature_names=[]; self.label_encoders={}; self.best_round=0
    def load(self):
        model_path="xgboost_size_model_v2.json" if Path("xgboost_size_model_v2.json").exists() else "xgboost_size_model.json"
        if not Path(model_path).exists():
            raise FileNotFoundError("No model file found.")
        self.booster=xgb.Booster(); self.booster.load_model(model_path)
        self.best_round=getattr(self.booster,"best_iteration",0) or 0
        if Path("feature_names.csv").exists():
            self.feature_names=pd.read_csv("feature_names.csv")["feature"].tolist()
        if Path("label_encoders.pkl").exists():
            with open("label_encoders.pkl","rb") as f: self.label_encoders=pickle.load(f)
        print(f"Model loaded: {model_path} | Features: {len(self.feature_names)}")
        return self

artifacts = ModelArtifacts()

def engineer_features(inp):
    h,w,bu,wa,hi,ag = inp["height"],inp["weight"],inp["bust"],inp["waist"],inp["hips"],inp["age"]
    sw=inp.get("shoulder_width"); cat=inp.get("category","unknown"); src=inp.get("source","unknown")
    bmi=round(w/(h/100)**2,2); whr=round(wa/hi,3); bwr=round(bu/wa,3)
    if whr<0.75 and abs(bu-hi)<5: body_shape="hourglass"
    elif bu>hi+5: body_shape="inverted_triangle"
    elif hi>bu+5: body_shape="pear"
    elif whr>0.85: body_shape="apple"
    else: body_shape="rectangle"
    if sw is None or sw<=0: sw=round(bu*SHAPE_SHOULDER_FACTOR.get(body_shape,0.43),1)
    shr=round(sw/hi,3); sbr=round(sw/bu,3); swr=round(sw/wa,3)
    if abs(sw-hi)<3.5 and whr<0.75: body_shape_v2="hourglass"
    elif sw>hi+3: body_shape_v2="inverted_triangle"
    elif hi>sw+3: body_shape_v2="pear"
    elif whr>0.85: body_shape_v2="apple"
    else: body_shape_v2="rectangle"
    def bucket(val,bins,labels):
        for i,(lo,hb) in enumerate(zip(bins[:-1],bins[1:])):
            if lo<=val<hb: return labels[i]
        return labels[-1]
    height_bucket=bucket(h,[0,155,163,170,178,999],["petite","short","average","tall","extra_tall"])
    weight_bucket=bucket(w,[0,55,65,75,90,999],["very_light","light","medium","heavy","very_heavy"])
    age_group=bucket(ag,[0,24,34,44,54,999],["18-24","25-34","35-44","45-54","55+"])
    shoulder_bucket=bucket(sw,[0,36,39,42,45,99],["narrow","slightly_narrow","average","slightly_wide","wide"])
    cat_grouped=CATEGORY_MAP.get(str(cat).strip().lower(),"other")
    body_type=inp.get("body_type","unknown")
    le=artifacts.label_encoders
    raw_df=pd.DataFrame([{
        "height":h,"weight":w,"bust":bu,"waist":wa,"hips":hi,"age":ag,"shoulder_width":sw,
        "bmi":bmi,"whr":whr,"bwr":bwr,"shr":shr,"sbr":sbr,"swr":swr,
        "_body_shape_v2":body_shape_v2.lower(),"_body_type":str(body_type).lower(),
        "_height_bucket":height_bucket,"_weight_bucket":weight_bucket,
        "_age_group":age_group,"_cat_grouped":cat_grouped,
        "_shoulder_bucket":shoulder_bucket,"_source":str(src).lower(),
    }])
    CAT_MAP={"body_shape_v2":"_body_shape_v2","body_type":"_body_type",
             "height_bucket":"_height_bucket","weight_bucket":"_weight_bucket",
             "age_group":"_age_group","category_grouped":"_cat_grouped",
             "shoulder_bucket":"_shoulder_bucket","source":"_source"}
    for col,raw_col in CAT_MAP.items():
        enc_col=col+"_enc"
        if enc_col not in artifacts.feature_names: continue
        val=raw_df[raw_col].iloc[0]
        if col in le:
            encoder=le[col]; val_clean=str(val).strip().lower()
            raw_df[enc_col]=encoder.transform([val_clean])[0] if val_clean in encoder.classes_ else 0
        else: raw_df[enc_col]=0
    for f in artifacts.feature_names:
        if f not in raw_df.columns: raw_df[f]=0
    return raw_df[artifacts.feature_names]

class MeasurementInput(BaseModel):
    height:float=Field(...,ge=120,le=220)
    weight:float=Field(...,ge=30,le=200)
    bust:float=Field(...,ge=60,le=160)
    waist:float=Field(...,ge=40,le=150)
    hips:float=Field(...,ge=60,le=170)
    age:float=Field(...,ge=13,le=100)
    shoulder_width:Optional[float]=Field(None,ge=28,le=60)
    body_type:Optional[str]=None
    category:Optional[str]="dress"
    source:Optional[str]="unknown"
    @validator("waist")
    def waist_less_than_hips(cls,v,values):
        if "hips" in values and v>=values["hips"]:
            raise ValueError("waist must be less than hips")
        return v
    class Config:
        schema_extra={"example":{"height":165,"weight":62,"bust":90,"waist":72,"hips":96,"age":28,"shoulder_width":38,"category":"dress"}}

class PredictionResponse(BaseModel):
    recommendation:str; confidence:float; confidence_level:str
    probabilities:dict; advice:str; shoulder_used:float; shoulder_estimated:bool

app=FastAPI(title="Clothing Size Recommendation API",version="2.0.0")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])

@app.on_event("startup")
def startup_event():
    print("Loading model..."); artifacts.load(); print("API ready.")

@app.get("/health")
def health():
    return {"status":"ok","model_loaded":artifacts.booster is not None,
            "n_features":len(artifacts.feature_names),"xgboost_version":xgb.__version__}

@app.post("/predict",response_model=PredictionResponse)
def predict(payload:MeasurementInput):
    if artifacts.booster is None:
        raise HTTPException(status_code=503,detail="Model not loaded.")
    inp=payload.dict(); shoulder_estimated=inp.get("shoulder_width") is None
    X=engineer_features(inp)
    dmat=xgb.DMatrix(X)
    ir=(0,artifacts.best_round+1) if artifacts.best_round else (0,0)
    probs=artifacts.booster.predict(dmat,iteration_range=ir).reshape(-1,3)[0]
    pred_idx=int(np.argmax(probs)); pred_label=CLASS_NAMES[pred_idx]; confidence=float(probs[pred_idx])
    conf_level="high" if confidence>=0.75 else "medium" if confidence>=0.55 else "low"
    cat=str(inp.get("category","item")).lower()
    ADVICE={"too_small":f"This {cat} is likely too small. Size up.",
            "fit":f"This {cat} should fit well. ({'High confidence' if conf_level=='high' else 'Moderate confidence'})",
            "too_large":f"This {cat} is likely too large. Size down."}
    shoulder_used=float(X["shoulder_width"].iloc[0]) if "shoulder_width" in X.columns else 0.0
    return PredictionResponse(recommendation=pred_label,confidence=round(confidence,4),
        confidence_level=conf_level,probabilities={c:round(float(p),4) for c,p in zip(CLASS_NAMES,probs)},
        advice=ADVICE[pred_label],shoulder_used=round(shoulder_used,1),shoulder_estimated=shoulder_estimated)

@app.post("/predict/batch")
def predict_batch(payloads:List[MeasurementInput]):
    if len(payloads)>500:
        raise HTTPException(status_code=400,detail="Batch limit is 500.")
    return [predict(p) for p in payloads]

@app.get("/model/info")
def model_info():
    return {"model_version":"2.0","n_features":len(artifacts.feature_names),
            "features":artifacts.feature_names,"classes":CLASS_NAMES,
            "best_round":artifacts.best_round,"xgboost_version":xgb.__version__}

if __name__=="__main__":
    import uvicorn, os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("step8_api:app", host="0.0.0.0", port=port)
