from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import logging
import pandas as pd
import os

# give Heroku the ability to pull in data from DVC upon app start up
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

logging.basicConfig(
    level=logging.INFO,
    filename="logs/app.log",
    format="%(asctime)-15s %(message)s",
)
logger = logging.getLogger()

app = FastAPI()


class Features(BaseModel):
    age: float
    fnlgt: float
    education_num: float
    capital_gain: float
    capital_loss: float
    hours_per_week: float
    workclass_Federal_gov: int
    workclass_Local_gov: int
    workclass_Never_worked: int
    workclass_Private: int
    workclass_Self_emp_inc: int
    workclass_Self_emp_not_inc: int
    workclass_State_gov: int
    workclass_Without_pay: int
    education_11th: int
    education_12th: int
    education_1st_4th: int
    education_5th_6th: int
    education_7th_8th: int
    education_9th: int
    education_Assoc_acdm: int
    education_Assoc_voc: int
    education_Bachelors: int
    education_Doctorate: int
    education_HS_grad: int
    education_Masters: int
    education_Preschool: int
    education_Prof_school: int
    education_Some_college: int
    marital_status_Married_AF_spouse: int
    marital_status_Married_civ_spouse: int
    marital_status_Married_spouse_absent: int
    marital_status_Never_married: int
    marital_status_Separated: int
    marital_status_Widowed: int
    occupation_Adm_clerical: int
    occupation_Armed_Forces: int
    occupation_Craft_repair: int
    occupation_Exec_managerial: int
    occupation_Farming_fishing: int
    occupation_Handlers_cleaners: int
    occupation_Machine_op_inspct: int
    occupation_Other_service: int
    occupation_Priv_house_serv: int
    occupation_Prof_specialty: int
    occupation_Protective_serv: int
    occupation_Sales: int
    occupation_Tech_support: int
    occupation_Transport_moving: int
    relationship_Not_in_family: int
    relationship_Other_relative: int
    relationship_Own_child: int
    relationship_Unmarried: int
    relationship_Wife: int
    race_Asian_Pac_Islander: int
    race_Black: int
    race_Other: int
    race_White: int
    sex_Male: int
    native_country_Cambodia: int
    native_country_Canada: int
    native_country_China: int
    native_country_Columbia: int
    native_country_Cuba: int
    native_country_Dominican_Republic: int
    native_country_Ecuador: int
    native_country_El_Salvador: int
    native_country_England: int
    native_country_France: int
    native_country_Germany: int
    native_country_Greece: int
    native_country_Guatemala: int
    native_country_Haiti: int
    native_country_Holand_Netherlands: int
    native_country_Honduras: int
    native_country_Hong: int
    native_country_Hungary: int
    native_country_India: int
    native_country_Iran: int
    native_country_Ireland: int
    native_country_Italy: int
    native_country_Jamaica: int
    native_country_Japan: int
    native_country_Laos: int
    native_country_Mexico: int
    native_country_Nicaragua: int
    native_country_Outlying_US_Guam_USVI_etc: int
    native_country_Peru: int
    native_country_Philippines: int
    native_country_Poland: int
    native_country_Portugal: int
    native_country_Puerto_Rico: int
    native_country_Scotland: int
    native_country_South: int
    native_country_Taiwan: int
    native_country_Thailand: int
    native_country_Trinadad_Tobago: int
    native_country_United_States: int
    native_country_Vietnam: int
    native_country_Yugoslavia: int


class Config:
    schema_extra = {
        "example": {
            "age": 0.0306705573543917,
            "fnlgt": -1.0636107451560883,
            "education_num": 1.1347387637961643,
            "capital_gain": 0.1484528952174793,
            "capital_loss": -0.2166595270325901,
            "hours_per_week": -0.0354294469727769,
            "workclass_Federal_gov": 0.0,
            "workclass_Local_gov": 0.0,
            "workclass_Never_worked": 0.0,
            "workclass_Private": 0.0,
            "workclass_Self_emp_inc": 0.0,
            "workclass_Self_emp_not_inc": 0.0,
            "workclass_State_gov": 1.0,
            "workclass_Without_pay": 0.0,
            "education_11th": 0.0,
            "education_12th": 0.0,
            "education_1st_4th": 0.0,
            "education_5th_6th": 0.0,
            "education_7th_8th": 0.0,
            "education_9th": 0.0,
            "education_Assoc_acdm": 0.0,
            "education_Assoc_voc": 0.0,
            "education_Bachelors": 1.0,
            "education_Doctorate": 0.0,
            "education_HS_grad": 0.0,
            "education_Masters": 0.0,
            "education_Preschool": 0.0,
            "education_Prof_school": 0.0,
            "education_Some_college": 0.0,
            "marital_status_Married_AF_spouse": 0.0,
            "marital_status_Married_civ_spouse": 0.0,
            "marital_status_Married_spouse_absent": 0.0,
            "marital_status_Never_married": 1.0,
            "marital_status_Separated": 0.0,
            "marital_status_Widowed": 0.0,
            "occupation_Adm_clerical": 1.0,
            "occupation_Armed_Forces": 0.0,
            "occupation_Craft_repair": 0.0,
            "occupation_Exec_managerial": 0.0,
            "occupation_Farming_fishing": 0.0,
            "occupation_Handlers_cleaners": 0.0,
            "occupation_Machine_op_inspct": 0.0,
            "occupation_Other_service": 0.0,
            "occupation_Priv_house_serv": 0.0,
            "occupation_Prof_specialty": 0.0,
            "occupation_Protective_serv": 0.0,
            "occupation_Sales": 0.0,
            "occupation_Tech_support": 0.0,
            "occupation_Transport_moving": 0.0,
            "relationship_Not_in_family": 1.0,
            "relationship_Other_relative": 0.0,
            "relationship_Own_child": 0.0,
            "relationship_Unmarried": 0.0,
            "relationship_Wife": 0.0,
            "race_Asian_Pac_Islander": 0.0,
            "race_Black": 0.0,
            "race_Other": 0.0,
            "race_White": 1.0,
            "sex_Male": 1.0,
            "native_country_Cambodia": 0.0,
            "native_country_Canada": 0.0,
            "native_country_China": 0.0,
            "native_country_Columbia": 0.0,
            "native_country_Cuba": 0.0,
            "native_country_Dominican_Republic": 0.0,
            "native_country_Ecuador": 0.0,
            "native_country_El_Salvador": 0.0,
            "native_country_England": 0.0,
            "native_country_France": 0.0,
            "native_country_Germany": 0.0,
            "native_country_Greece": 0.0,
            "native_country_Guatemala": 0.0,
            "native_country_Haiti": 0.0,
            "native_country_Holand_Netherlands": 0.0,
            "native_country_Honduras": 0.0,
            "native_country_Hong": 0.0,
            "native_country_Hungary": 0.0,
            "native_country_India": 0.0,
            "native_country_Iran": 0.0,
            "native_country_Ireland": 0.0,
            "native_country_Italy": 0.0,
            "native_country_Jamaica": 0.0,
            "native_country_Japan": 0.0,
            "native_country_Laos": 0.0,
            "native_country_Mexico": 0.0,
            "native_country_Nicaragua": 0.0,
            "native_country_Outlying_US_Guam_USVI_etc": 0.0,
            "native_country_Peru": 0.0,
            "native_country_Philippines": 0.0,
            "native_country_Poland": 0.0,
            "native_country_Portugal": 0.0,
            "native_country_Puerto_Rico": 0.0,
            "native_country_Scotland": 0.0,
            "native_country_South": 0.0,
            "native_country_Taiwan": 0.0,
            "native_country_Thailand": 0.0,
            "native_country_Trinadad_Tobago": 0.0,
            "native_country_United_States": 1.0,
            "native_country_Vietnam": 0.0,
            "native_country_Yugoslavia": 0.0,
        }
    }


@app.get("/")
def home():
    return {"Welcome": "Salary Prediction API Home"}


@app.post("/predict")
async def predict(features: Features):
    try:
        clf = joblib.load("models/rfc_model.pkl")
    except Exception as e:
        logger.error(e)
        return "Model not loaded"
    logger.info(f"features: {features}")
    inference_payload = pd.DataFrame(features).iloc[:, 1:].values.transpose()
    logger.info(f"inference payload DataFrame: {inference_payload}")
    prediction = list(clf.predict(inference_payload))
    prediction = [int(x) for x in prediction]
    return {"prediction": prediction}
