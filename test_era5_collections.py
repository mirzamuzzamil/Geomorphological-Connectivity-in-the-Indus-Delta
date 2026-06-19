import ee
import json

def main():
    with open("ee-muzzamil12-2f811d1cb2cc.json", "r") as f:
        cred_data = json.load(f)
    credentials = ee.ServiceAccountCredentials(cred_data["client_email"], key_data=json.dumps(cred_data))
    ee.Initialize(credentials=credentials)
    
    collections = [
        "ECMWF/ERA5_LAND/MONTHLY_BY_HOUR",
        "ECMWF/ERA5_LAND/MONTHLY",
        "ECMWF/ERA5/MONTHLY"
    ]
    
    for c in collections:
        try:
            coll = ee.ImageCollection(c)
            info = coll.first().getInfo()
            print(f"Collection {c} is available. First ID: {info['id']}")
            print("Bands:", [b["id"] for b in info["bands"]][:5])
        except Exception as e:
            print(f"Collection {c} failed: {e}")

if __name__ == "__main__":
    main()
