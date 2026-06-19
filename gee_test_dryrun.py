import ee
import json

def main():
    try:
        # Load credentials
        with open("ee-muzzamil12-2f811d1cb2cc.json", "r") as f:
            cred_data = json.load(f)
            
        print("Authenticating with Earth Engine...")
        credentials = ee.ServiceAccountCredentials(
            cred_data["client_email"],
            key_data=json.dumps(cred_data)
        )
        
        ee.Initialize(credentials=credentials)
        print("Initialization successful!")
        
        # Test query: Get SRTM DEM metadata
        dem = ee.Image("USGS/SRTMGL1_003")
        info = dem.getInfo()
        print("SRTM DEM Connection Verified. Image ID:", info["id"])
        print("Bands:", [b["id"] for b in info["bands"]])
        
    except Exception as e:
        print("Error during initialization or query:", e)

if __name__ == "__main__":
    main()
