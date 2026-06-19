import ee
import json

def main():
    with open("ee-muzzamil12-2f811d1cb2cc.json", "r") as f:
        cred_data = json.load(f)
    credentials = ee.ServiceAccountCredentials(cred_data["client_email"], key_data=json.dumps(cred_data))
    ee.Initialize(credentials=credentials)
    
    try:
        coll = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
        info = coll.first().getInfo()
        print("ECMWF/ERA5_LAND/MONTHLY_AGGR is available!")
        print("Bands:", [b["id"] for b in info["bands"]])
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
