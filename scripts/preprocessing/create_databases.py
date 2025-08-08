import pandas as pd
import numpy as np
import math
from pathlib import Path
import argparse


### Script to create the different databases created to train the models. 5 databases are created daytime, nighttime and mixedtime databases are for max, min and mean SAT retrieval models. Daytime uses daytime varibles, nighttime uses only nighttime variables and mixed time uses all variables. Inst databases are for instantaneous SAT retrieval models.

# Load MODIS and AEMET preprocessed datafiles
def load_data(base_dir):
    base = Path(base_dir)
    aemet_dir = base / "preprocessing" / "aemet"
    modis_dir = base / "preprocessing" / "modis"

    X = pd.read_csv(aemet_dir / "stations_data.csv", encoding='latin-1')
    T = pd.read_csv(aemet_dir / "daily_sat.csv", encoding='latin-1')
    inst_dia = pd.read_csv(aemet_dir / "interp_sat_day.csv", encoding='latin-1')
    inst_nit = pd.read_csv(aemet_dir / "interp_sat_night.csv", encoding='latin-1')

    LST_D = pd.read_csv(modis_dir  / "day_LST.csv", encoding='latin-1')
    LST_N = pd.read_csv(modis_dir  / "night_LST.csv", encoding='latin-1')
    NDVI = pd.read_csv(modis_dir  / "NDVI.csv", encoding='latin-1')
    AL = pd.read_csv(modis_dir  / "albedo.csv", encoding='latin-1')

    return X, T, inst_dia, inst_nit, LST_D, LST_N, NDVI, AL

# Filling database columns with geographical and topograhical data
def fill_geodata(data, geo):
    for col in ['LATITUD', 'LONGITUD', 'ALTITUD', 'Aspect', 'Slope', 'Dist_coast']:
        data[col] = geo[col].values[0]
    data['Dist_coast'] = data['Dist_coast']
    return data

# Drop those stations without enough data due to statistic reasons
def drop_stations(data, df_merge):

    # If there are less than 50 stations for a 4 years period station is discarted
    if len(data) > 50:
        return pd.concat([df_merge, data]).reset_index(drop=True)
    return df_merge

# Solar coordinate equations published in Valor et al. (2023)
def solar_coords(fecha, day, fi, lon, slope, aspect):

    # Changin angle variables to rad
    fi, lon, slope, aspect = [x * (2 * math.pi) / 360 for x in [fi, lon, slope, aspect]]

    # Date to doy conversion
    dia = fecha.dayofyear

    x = (2 * math.pi * (dia - 1)) / 365

    # Declination
    d = (0.006918 - 0.399912 * math.cos(x) + 0.070257 * math.sin(x) -
         0.006758 * math.cos(2 * x) + 0.000907 * math.sin(2 * x) -
         0.002697 * math.cos(3 * x) + 0.001480 * math.sin(3 * x))
    
    # Time equation
    ET = -0.128 * math.sin((360 * (dia - 1) / 365) - 2.80) - 0.165 * math.sin((2 * 360 * (dia - 1) / 365) - 19.7)

    # Aparent time
    LAT1 = day + (lon * 360 / (2 * math.pi * 15)) + ET

    # Hour solar angle
    omega1 = 15 * (LAT1 - 12) * (2 * math.pi) / 360
    
    # Zenithal solar angle
    zenith1 = math.sin(d) * math.sin(fi) + math.cos(d) * math.cos(fi) * math.cos(omega1)
    zd = math.acos(zenith1)

    # Azimuthal solar angle
    azimut1 = math.cos(d) * math.sin(omega1) / math.sin(zd)
    ad = math.asin(azimut1)

    # Solar inclination angle
    ID = math.acos(math.cos(zd) * math.cos(slope) + math.sin(zd) * math.sin(slope) * math.cos(ad - aspect))
    return zd, ad, ID

# Prepares inputs and merge them deleting days without all variables
def process_station_data(station, X, T, inst_dia, inst_nit, LST_D, LST_N, NDVI, AL):
    geo = X[X['INDICATIVO'] == station].reset_index(drop=True)
    t_data = T[T['INDICATIVO'] == station].reset_index(drop=True)
    lst_d = LST_D[LST_D['INDICATIVO'] == station].reset_index(drop=True)
    lst_n = LST_N[LST_N['INDICATIVO'] == station].reset_index(drop=True)
    ndvi = NDVI[NDVI['INDICATIVO'] == station].reset_index(drop=True)
    albedo = AL[AL['INDICATIVO'] == station].reset_index(drop=True)
    inst_d = inst_dia[inst_dia['INDICATIVO'] == station].reset_index(drop=True)
    inst_n = inst_nit[inst_nit['INDICATIVO'] == station].reset_index(drop=True)

    # Deleting days without all variables
    day = lst_d.merge(ndvi, on=['Time', 'INDICATIVO']).merge(albedo, on=['Time', 'INDICATIVO'])
    night = lst_n.merge(ndvi, on=['Time', 'INDICATIVO']).merge(albedo, on=['Time', 'INDICATIVO'])
    mix = lst_n.merge(lst_d, on=['Time', 'INDICATIVO']).merge(ndvi, on=['Time', 'INDICATIVO']).merge(albedo, on=['Time', 'INDICATIVO'])

    for df in [day, night, mix, inst_d, inst_n, t_data]:
        df['Time'] = pd.to_datetime(df['Time'])

    day = day.merge(t_data, on=['Time', 'INDICATIVO'])
    night = night.merge(t_data, on=['Time', 'INDICATIVO'])
    mix = mix.merge(t_data, on=['Time', 'INDICATIVO'])
    inst_day = day.merge(inst_d, on=['Time', 'INDICATIVO'])
    inst_night = night.merge(inst_n, on=['Time', 'INDICATIVO'])

    for df in [day, night, mix, inst_day, inst_night]:
        fill_geodata(df, geo)

    return day, night, mix, inst_day, inst_night

# Add solar coordinates to daytime databases
def add_solar_coords(df, time_col='Time', view_col='Day_view_time'):
    df['Zenital_angle'], df['Azimutal_angle'], df['Inclinacion_solar'] = zip(*df[[time_col, view_col, 'LATITUD', 'LONGITUD', 'Slope', 'Aspect']].apply(lambda x: solar_coords(*x), axis=1))
    return df



# Main function

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True, help="Base directory containing dades_Modis and dades_AEMET folders")
    args = parser.parse_args()

    # X (stations data), T( max, min and mean SAT)
    X, T, inst_dia, inst_nit, LST_D, LST_N, NDVI, AL = load_data(args.output_dir)
    stations = X['INDICATIVO'].tolist()

    df_day, df_night, df_mix, df_inst_day, df_inst_night = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Loop to select and process valid stations
    for station in stations:
        day, night, mix, inst_day, inst_night = process_station_data(station, X, T, inst_dia, inst_nit, LST_D, LST_N, NDVI, AL)
        df_day = drop_stations(day, df_day)
        df_night = drop_stations(night, df_night)
        df_mix = drop_stations(mix, df_mix)
        df_inst_day = drop_stations(inst_day, df_inst_day)
        df_inst_night = drop_stations(inst_night, df_inst_night)

    output_dir = Path(args.output_dir) / "preprocessing" / "databases"
    output_dir.mkdir(exist_ok=True)

    # Save databases with nighttime inputs
    df_night.dropna().to_csv(output_dir / "database_nighttime.csv", index=False)
    df_inst_night.dropna().to_csv(output_dir / "database_inst_night.csv", index=False)

    # Calculate and add solar coordinates
    df_day = add_solar_coords(df_day)
    df_mix = add_solar_coords(df_mix)
    df_inst_day = add_solar_coords(df_inst_day, view_col='Day_view_time_x')

    # Save databases with daytime inputs
    df_day.dropna().to_csv(output_dir / "database_daytime.csv", index=False)
    df_mix.dropna().to_csv(output_dir / "database_mixtime.csv", index=False)
    df_inst_day.dropna().to_csv(output_dir / "database_inst_day.csv", index=False)

if __name__ == "__main__":

    main()