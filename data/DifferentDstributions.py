import csv
import random
import datetime
import numpy as np

# Define the number of rows per distribution
total_rows_per_distribution = 50000

def generate_row(index, distribution):
    """Generate a row of satellite data based on the given distribution."""
    creation_date = datetime.datetime(2021, 11, 1, random.randint(0, 23), random.randint(0, 59), random.randint(0, 59))
    epoch = creation_date - datetime.timedelta(days=random.randint(0, 365))
    
    if distribution == "exponential":
        mean_motion = max(0.1, np.random.exponential(1.5))
        eccentricity = np.clip(np.random.exponential(0.2), 0.0, 1.0)
        inclination = np.clip(np.random.exponential(45.0), 0.0, 180.0)
        semi_major_axis = max(6371.0, 6371.0 + np.random.exponential(500.0))
    elif distribution == "normal":
        mean_motion = max(0.1, np.random.normal(8.0, 2.0))
        eccentricity = np.clip(np.random.normal(0.1, 0.05), 0.0, 1.0)
        inclination = np.clip(np.random.normal(90.0, 20.0), 0.0, 180.0)
        semi_major_axis = max(6371.0, np.random.normal(7371.0, 500.0))
    else:  # Uniform
        mean_motion = round(random.uniform(0.9, 15.0), 8)
        eccentricity = round(random.uniform(0.0, 0.8), 8)
        inclination = round(random.uniform(0.0, 180.0), 4)
        semi_major_axis = round(random.uniform(6371.0, 42164.0), 3)
    
    return [
        "2", "GENERATED VIA SPACE-TRACK.ORG API", creation_date.isoformat(), "18 SPCS", f"OBJECT {index}",
        f"ID-{index}", "EARTH", "TEME", "UTC", "SGP4", epoch.isoformat(), round(mean_motion, 8),
        round(eccentricity, 8), round(inclination, 4), round(random.uniform(0.0, 360.0), 4),
        round(random.uniform(0.0, 360.0), 4), round(random.uniform(0.0, 360.0), 4), "0", "U",
        random.randint(10000, 99999), random.randint(0, 999), random.randint(0, 100000),
        round(random.uniform(0.0, 0.01), 8), round(random.uniform(-0.0001, 0.0001), 8), "0",
        round(semi_major_axis, 3), round(random.uniform(90.0, 1440.0), 3),
        round(random.uniform(0.0, 35786.0), 3), round(random.uniform(0.0, 35786.0), 3),
        random.choice(["DEBRIS", "PAYLOAD", "ROCKET BODY"]), random.choice(["SMALL", "MEDIUM", "LARGE"]),
        random.choice(["US", "CIS", "FR", "PRC", "IND"]),
        f"{random.randint(1950, 2021)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}", "SITE", "",
        random.randint(1000000, 9999999), random.randint(100000, 999999),
        f"0 OBJECT {index}", f"1 {random.randint(10000, 99999)}U {random.randint(10000, 99999)}A   {random.randint(10000, 99999)}",
        f"2 {random.randint(10000, 99999)} {random.uniform(0.0, 360.0):.4f} {random.uniform(0.0, 360.0):.4f}"
    ]

# Generate and save CSV files
for distribution in ["exponential", "normal", "uniform"]:
    output_file = f"{distribution}_generated.csv"
    with open(output_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "CCSDS_OMM_VERS", "COMMENT", "CREATION_DATE", "ORIGINATOR", "OBJECT_NAME", "OBJECT_ID", "CENTER_NAME", 
            "REF_FRAME", "TIME_SYSTEM", "MEAN_ELEMENT_THEORY", "EPOCH", "MEAN_MOTION", "ECCENTRICITY", "INCLINATION", 
            "RA_OF_ASC_NODE", "ARG_OF_PERICENTER", "MEAN_ANOMALY", "EPHEMERIS_TYPE", "CLASSIFICATION_TYPE", 
            "NORAD_CAT_ID", "ELEMENT_SET_NO", "REV_AT_EPOCH", "BSTAR", "MEAN_MOTION_DOT", "MEAN_MOTION_DDOT", 
            "SEMIMAJOR_AXIS", "PERIOD", "APOAPSIS", "PERIAPSIS", "OBJECT_TYPE", "RCS_SIZE", "COUNTRY_CODE", 
            "LAUNCH_DATE", "SITE", "DECAY_DATE", "FILE", "GP_ID", "TLE_LINE0", "TLE_LINE1", "TLE_LINE2"
        ])
        for i in range(1, total_rows_per_distribution + 1):
            writer.writerow(generate_row(i, distribution))
    print(f"Generated {output_file} with {total_rows_per_distribution} rows.")
