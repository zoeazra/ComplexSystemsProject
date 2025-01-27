import csv
import random
import datetime

# Define the file name and the number of rows
output_file = "uniform_generated.csv"
number_of_rows = 50000

# Define column headers
headers = [
    "CCSDS_OMM_VERS",
    "COMMENT",
    "CREATION_DATE",
    "ORIGINATOR",
    "OBJECT_NAME",
    "OBJECT_ID",
    "CENTER_NAME",
    "REF_FRAME",
    "TIME_SYSTEM",
    "MEAN_ELEMENT_THEORY",
    "EPOCH",
    "MEAN_MOTION",
    "ECCENTRICITY",
    "INCLINATION",
    "RA_OF_ASC_NODE",
    "ARG_OF_PERICENTER",
    "MEAN_ANOMALY",
    "EPHEMERIS_TYPE",
    "CLASSIFICATION_TYPE",
    "NORAD_CAT_ID",
    "ELEMENT_SET_NO",
    "REV_AT_EPOCH",
    "BSTAR",
    "MEAN_MOTION_DOT",
    "MEAN_MOTION_DDOT",
    "SEMIMAJOR_AXIS",
    "PERIOD",
    "APOAPSIS",
    "PERIAPSIS",
    "OBJECT_TYPE",
    "RCS_SIZE",
    "COUNTRY_CODE",
    "LAUNCH_DATE",
    "SITE",
    "DECAY_DATE",
    "FILE",
    "GP_ID",
    "TLE_LINE0",
    "TLE_LINE1",
    "TLE_LINE2",
]

# Generate random data for the rows
def generate_row(index):
    creation_date = datetime.datetime(2021, 11, 1, random.randint(0, 23), random.randint(0, 59), random.randint(0, 59))
    epoch = creation_date - datetime.timedelta(days=random.randint(0, 365))
    
    return [
        "2",
        "GENERATED VIA SPACE-TRACK.ORG API",
        creation_date.isoformat(),
        "18 SPCS",
        f"OBJECT {index}",
        f"ID-{index}",
        "EARTH",
        "TEME",
        "UTC",
        "SGP4",
        epoch.isoformat(),
        round(random.uniform(0.9, 15.0), 8),  # MEAN_MOTION
        round(random.uniform(0.0, 0.8), 8),  # ECCENTRICITY
        round(random.uniform(0.0, 180.0), 4),  # INCLINATION
        round(random.uniform(0.0, 360.0), 4),  # RA_OF_ASC_NODE
        round(random.uniform(0.0, 360.0), 4),  # ARG_OF_PERICENTER
        round(random.uniform(0.0, 360.0), 4),  # MEAN_ANOMALY
        "0",
        "U",
        random.randint(10000, 99999),  # NORAD_CAT_ID
        random.randint(0, 999),  # ELEMENT_SET_NO
        random.randint(0, 100000),  # REV_AT_EPOCH
        round(random.uniform(0.0, 0.01), 8),  # BSTAR
        round(random.uniform(-0.0001, 0.0001), 8),  # MEAN_MOTION_DOT
        "0",
        round(random.uniform(6371.0, 42164.0), 3),  # SEMIMAJOR_AXIS
        round(random.uniform(90.0, 1440.0), 3),  # PERIOD
        round(random.uniform(0.0, 35786.0), 3),  # APOAPSIS
        round(random.uniform(0.0, 35786.0), 3),  # PERIAPSIS
        random.choice(["DEBRIS", "PAYLOAD", "ROCKET BODY"]),  # OBJECT_TYPE
        random.choice(["SMALL", "MEDIUM", "LARGE"]),  # RCS_SIZE
        random.choice(["US", "CIS", "FR", "PRC", "IND"]),  # COUNTRY_CODE
        f"{random.randint(1950, 2021)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",  # LAUNCH_DATE
        "SITE",
        "",
        random.randint(1000000, 9999999),
        random.randint(100000, 999999),
        f"0 OBJECT {index}",
        f"1 {random.randint(10000, 99999)}U {random.randint(10000, 99999)}A   {random.randint(10000, 99999)}",  # TLE_LINE1
        f"2 {random.randint(10000, 99999)} {random.uniform(0.0, 360.0):.4f} {random.uniform(0.0, 360.0):.4f}",  # TLE_LINE2
    ]

# Write the CSV file
with open(output_file, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(headers)
    for i in range(1, number_of_rows + 1):
        writer.writerow(generate_row(i))

print(f"Generated {output_file} with {number_of_rows} rows.")
