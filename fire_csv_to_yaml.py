# This script converts the CSV file of fires that we extracted from GlobFire to a YAML file
# that can be used to create the dataset in GEE.

import argparse
import datetime

import pandas as pd
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--csv_path", type=str,
                    help="Path to input CSV file", required=True)
parser.add_argument("--yaml_path", type=str,
                    help="Path to output YAML file", required=True)
parser.add_argument("--year", type=int, help="Year of fires", required=True)
parser.add_argument("--bucket_name", type=str,
                    help="Name of the Google Cloud bucket in which the downloaded files will eventually be saved. "
                    "This does not happen in this script, but the parameter is set in the YAML file "
                    "that we produce here.", required=True)
args = parser.parse_args()

# Read and adjust columns in the CSV file
base_df = pd.read_csv(args.csv_path)
df = base_df[["Id", "lat", "lon", "start_date", "end_date"]].rename(
    columns={"lat": "latitude", "lon": "longitude", "start_date": "start", "end_date": "end"})
df.index = df.Id.map(lambda i: f"fire_{i}")
df["Id"] = "fire_" + (df["Id"].astype(str))
df = df.set_index("Id")
df.start = df.start.map(lambda s: s.split("T")[0])
df.end = df.end.map(lambda s: s.split("T")[0])

# Additional parameters that the next script requires from the YAML file
output_dict = {"output_bucket": args.bucket_name,
               "rectangular_size": 0.5, "year": args.year}
output_dict.update(df.to_dict(orient='index'))

with open(args.yaml_path, 'w') as outfile:
    yaml.dump(
        output_dict,
        outfile,
        sort_keys=False,
        default_style=None
    )

# Remove quotation marks from dates, so the YAML loader interprets the entries as dates.
# There should be a nicer way, but I don't know it, and this works. 

with open(args.yaml_path, "r", encoding="utf8") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if line[0] in [" ", "\t"]:
        new_lines.append(line.replace("'", ""))
    else:
        new_lines.append(line)

with open(args.yaml_path, "w", encoding="utf8") as f:
    f.write("".join(new_lines))
