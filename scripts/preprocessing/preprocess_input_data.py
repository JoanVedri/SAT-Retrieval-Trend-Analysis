import argparse
from pathlib import Path
from glob import glob
import xarray as xr
import numpy as np
import pandas as pd


### Script to preprocess raw data form MODIS and in situ data (AEMET)


## Preprocessing of AEMET data

# Function with includes geographical variables of each station into stations data file
def process_geodata(data_dir, file):
    base_dir = Path(data_dir)
    stations = file['INDICATIVO'].to_list()

    geofile = base_dir / "geo.nc"
    df = pd.DataFrame()

    # Loop for iterating stations
    for station in stations:

        df_st = pd.DataFrame()
        df_st = file.loc[file['INDICATIVO'] == station]
        lat = df_st['LATITUD'].values[0]
        lon = df_st['LONGITUD'].values[0]

        # Open nc file
        with xr.open_dataset(geofile, engine='netcdf4') as xr_data:
                lat_grid = np.tile(xr_data.latitude.values, (xr_data.dims['longitude'], 1)).T
                lon_grid = np.tile(xr_data.longitude.values, (xr_data.dims['latitude'], 1))
                x, y = find_nearest_col_row(lat_grid, lon_grid, [lat, lon])
                
                data = []
                
                asp = xr_data.Aspect[x, y].values
                slp= xr_data.Slope[x, y].values
                dem = xr_data.Dem[x, y].values
                dist_coast = xr_data.Dist_Coast[x, y].values
                data.append([asp, slp, dem, dist_coast])
                df_aux = pd.DataFrame(data, columns=['Aspect', 'Slope', 'DEM_ALT', 'Dist_coast'])
                df = pd.concat([df, df_aux]).reset_index(drop=True)
    
    file['Aspect'] = df['Aspect']
    file['Slope'] = df['Slope']
    file['DEM_ALT'] = df['DEM_ALT']

    # Changing units from m to km
    file['Dist_coast'] = df['Dist_coast']/1000

    return file

# Create a unified stations data file
def create_station_metadata(data_dir,output):

    maestro_dir = Path(data_dir)
    output_dir = Path(output)

    # Files paths
    maestro_dir_in = maestro_dir / "aemet" / "maestro"
    geo_dir_in = maestro_dir / "geodata" 
    maestro_dir_out = output_dir / "preprocessing" / "aemet"
    maestro_files = sorted(glob(str(maestro_dir_in / "Maestro*")))
    maestro_df = pd.DataFrame()

    # Reading files and changing decimal separator
    for file in maestro_files:
        df = pd.read_csv(file, sep=';', encoding='latin-1')
        df['LATITUD'] = df['LATITUD'].str.replace(',', '.').astype(float)
        df['LONGITUD'] = df['LONGITUD'].str.replace(',', '.').astype(float)
        maestro_df = pd.concat([maestro_df, df]).reset_index(drop=True)

    # Dropping unnecessary columns
    maestro_df = maestro_df.drop(columns=['NOM_MUNI','DATUM','TIPO_EMA','RED','IND_SYN','ALTA'])
    maestro_df = process_geodata(geo_dir_in, maestro_df)

    maestro_df.to_csv(maestro_dir_out / "stations_data.csv", index=False)
    return maestro_df

# Create a unified file for AEMET data
def create_sat_database(data_dir, output):

    base_dir = Path(data_dir)
    output_dir = Path(output)

    data_dir_in = base_dir / "aemet" / "data"
    data_dir_out = output_dir / "preprocessing" / "aemet"
    data_files = sorted(glob(str(data_dir_in / "Cuenca*")))
    aemet_df = pd.DataFrame()

    for file in data_files:
        df = pd.read_csv(file, sep=';', encoding='latin-1')
        aemet_df = pd.concat([aemet_df, df]).reset_index(drop=True)

    aemet_df.to_csv(data_dir_out / "raw_sat_data.csv", index=False)
    return aemet_df

# Create a unified file for maximum, minimum and mean SAT data 

def process_daily_sat(data_dir):

    base_dir = Path(data_dir)

    data_dir = base_dir / "preprocessing" / "aemet"

    aemet = pd.read_csv(data_dir / "raw_sat_data.csv")

    # Creating unified tiem column
    time = pd.to_datetime(dict(
        year=aemet['AÑO'],
        month=aemet['MES'],
        day=aemet['DIA'],
        hour=aemet['HORA'],
        minute=aemet['MINUTO']
    ))

    df = pd.DataFrame({
        'INDICATIVO': aemet[' INDICATIVO'].str.strip(),
        'Time': time,
        'TA': aemet['TA'].str.replace(',', '.').astype(float)
    })

    stations = df['INDICATIVO'].unique()
    final_df = pd.DataFrame()

    for station in stations:
        station_df = df[df['INDICATIVO'] == station].copy()
        station_df.set_index('Time', inplace=True)

        # Calculate max, min and mean values per day
        daily_mean = station_df.resample('D')['TA'].mean()
        daily_min = station_df.resample('D')['TA'].min()
        daily_max = station_df.resample('D')['TA'].max()

        # Creating final df and changing units from Celsius to Kelvin
        daily_df = pd.DataFrame({
            'Time': daily_mean.index,
            'INDICATIVO': station,
            'T_mean': daily_mean + 273.15,
            'T_min': daily_min + 273.15,
            'T_max': daily_max + 273.15
        })

        # Remove days with only one value (min == max)
        daily_df.loc[daily_df['T_min'] == daily_df['T_max']] = np.nan
        daily_df = daily_df.dropna().reset_index(drop=True)

        final_df = pd.concat([final_df, daily_df]).reset_index(drop=True)

    final_df.to_csv(data_dir / "daily_sat.csv", index=False)

# Temporal interpolation function
def interpolate_data(dAEMETd, dAEMETn, data_LSTd, data_LSTn):
    s_dia = pd.DataFrame()
    s_nit = pd.DataFrame()

    stations_d = data_LSTd['INDICATIVO'].unique()
    stations_n = data_LSTn['INDICATIVO'].unique()

    # 2 Loops, first for daytime LST and second for nightime LST
    for station in stations_d:
        dAEMETd_aux = dAEMETd[dAEMETd['INDICATIVO'] == station].reset_index(drop=True)
        data_LSTd_aux = data_LSTd[data_LSTd['INDICATIVO'] == station].reset_index(drop=True)

        horas_aemet_dia = dAEMETd_aux["hora_rel"].to_numpy(dtype="f")
        hora_lst_dia = data_LSTd_aux["hora_rel"].to_numpy(dtype="f")
        temp_dia = dAEMETd_aux["TA"].to_numpy(dtype="f")

        # Interpolation and result in Kelvin
        res_Temp_dia = np.interp(hora_lst_dia, horas_aemet_dia, temp_dia) + 273.15
        s_dia_aux = pd.DataFrame({'Temp': res_Temp_dia})
        s_dia = pd.concat([s_dia, s_dia_aux], ignore_index=True)


    for station in stations_n:
        dAEMETn_aux = dAEMETn[dAEMETn['INDICATIVO'] == station].reset_index(drop=True)
        data_LSTn_aux = data_LSTn[data_LSTn['INDICATIVO'] == station].reset_index(drop=True)

        horas_aemet_nit = dAEMETn_aux["hora_rel"].to_numpy(dtype="f")
        hora_lst_nit = data_LSTn_aux["hora_rel"].to_numpy(dtype="f")
        temp_nit = dAEMETn_aux["TA"].to_numpy(dtype="f")

        # Interpolation and result in Kelvin
        res_Temp_nit = np.interp(hora_lst_nit, horas_aemet_nit, temp_nit) + 273.15
        s_nit_aux = pd.DataFrame({'Temp': res_Temp_nit})
        s_nit = pd.concat([s_nit, s_nit_aux], ignore_index=True)


    return s_dia, s_nit

# Function to obtain instanteneous SAT from AEMET data at the satellite surpass time
def process_aemet_satellite_data(output_path,start_date='2019-01-01 00:00:00'):
    

    output_dir = Path(output_path)
    lst_dir = output_dir / "preprocessing" / "modis"
    aemet_dir = output_dir / "preprocessing" / "aemet"
 

    # Load data
    dLSTd = pd.read_csv(lst_dir / "day_LST.csv")
    dLSTn = pd.read_csv(lst_dir / "night_LST.csv")
    aemet = pd.read_csv(aemet_dir / "raw_sat_data.csv")
    maestro = pd.read_csv(aemet_dir / "stations_data.csv", sep=",", encoding="latin-1")
    stations_list = maestro['INDICATIVO'].tolist()

    # Create datetime column
    time = pd.to_datetime(dict(
        year=aemet['AÑO'],
        month=aemet['MES'],
        day=aemet['DIA'],
        hour=aemet['HORA'],
        minute=aemet['MINUTO']
    ))

    dades_aemet = pd.DataFrame({
        'INDICATIVO': aemet[' INDICATIVO'].str.strip(),
        'Time': time,
        'TA': aemet['TA'].str.replace(',', '.').astype(float)
    })

    # Filter and align data
    dAEMETd, dAEMETn = pd.DataFrame(), pd.DataFrame()
    data_LSTd, data_LSTn = pd.DataFrame(), pd.DataFrame()

    for station in stations_list:
        dAEMET_aux = dades_aemet[dades_aemet['INDICATIVO'] == station].reset_index(drop=True)
        dLSTd_aux = dLSTd[dLSTd['INDICATIVO'] == station].reset_index(drop=True)
        dLSTn_aux = dLSTn[dLSTn['INDICATIVO'] == station].reset_index(drop=True)

        diasAEMET = pd.to_datetime(dAEMET_aux['Time']).dt.date
        diasLST = pd.to_datetime(dLSTd_aux['Time']).dt.date
        nitsLST = pd.to_datetime(dLSTn_aux['Time']).dt.date

        dLSTd_aux = dLSTd_aux[diasLST.isin(diasAEMET)]
        dLSTn_aux = dLSTn_aux[nitsLST.isin(diasAEMET)]

        diasLST = pd.to_datetime(dLSTd_aux['Time']).dt.date
        nitsLST = pd.to_datetime(dLSTn_aux['Time']).dt.date

        dAEMETd_aux = dAEMET_aux[diasAEMET.isin(diasLST)]
        dAEMETn_aux = dAEMET_aux[diasAEMET.isin(nitsLST)]

        dAEMETd = pd.concat([dAEMETd, dAEMETd_aux], ignore_index=True)
        dAEMETn = pd.concat([dAEMETn, dAEMETn_aux], ignore_index=True)
        data_LSTd = pd.concat([data_LSTd, dLSTd_aux], ignore_index=True)
        data_LSTn = pd.concat([data_LSTn, dLSTn_aux], ignore_index=True)

    # Compute relative time
    inicio = pd.to_datetime(start_date)
    dAEMETd["hora_rel"] = dAEMETd["Time"].apply(lambda x: (x - inicio).total_seconds() / 3600)
    dAEMETn["hora_rel"] = dAEMETn["Time"].apply(lambda x: (x - inicio).total_seconds() / 3600)
    data_LSTd["hora_rel"] = data_LSTd.apply(lambda x: (pd.to_datetime(x["Time"]) - inicio).total_seconds() / 3600 + x["Day_view_time"], axis=1)
    data_LSTn["hora_rel"] = data_LSTn.apply(lambda x: (pd.to_datetime(x["Time"]) - inicio).total_seconds() / 3600 + x["Night_view_time"], axis=1)

    # Interpolate
    dAEMET_dia, dAEMET_nit = interpolate_data(dAEMETd, dAEMETn, data_LSTd, data_LSTn)

    # Add stations data and save
    dAEMET_dia["Time"] = data_LSTd['Time']
    dAEMET_dia["Day_view_time"] = data_LSTd['Day_view_time']
    dAEMET_dia["INDICATIVO"] = data_LSTd['INDICATIVO']
    dAEMET_dia.to_csv(aemet_dir / "interp_sat_day.csv", index=False)

    dAEMET_nit["Time"] = data_LSTn['Time']
    dAEMET_nit["Night_view_time"] = data_LSTn['Night_view_time']
    dAEMET_nit["INDICATIVO"] = data_LSTn['INDICATIVO']
    dAEMET_nit.to_csv(aemet_dir / "interp_sat_night.csv", index=False)


## Preprocessing of MODIS data

# Find the closest pixel to the station location
def find_nearest_col_row(latitudes, longitudes, match_coords):
    lat_dist2 = (latitudes - match_coords[0])**2
    lon_dist2 = (longitudes - match_coords[1])**2
    geo_dist2 = lat_dist2 + lon_dist2
    row, col = np.where(geo_dist2 == geo_dist2.min())
    return int(row[0]), int(col[0])

# Find the 4 closest pixels to the station location for albedo resampling
def find_2x2_pixel_bounds(latitudes, longitudes, match_coords):
    row, col = find_nearest_col_row(latitudes, longitudes, match_coords)
    prow = min(row + 1, latitudes.shape[0] - 1)
    mrow = max(row - 1, 0)
    pcol = min(col + 1, longitudes.shape[1] - 1)
    mcol = max(col - 1, 0)

    geo_dist2 = [
        (latitudes[mrow, mcol] - match_coords[0])**2 + (longitudes[mrow, mcol] - match_coords[1])**2,
        (latitudes[mrow, pcol] - match_coords[0])**2 + (longitudes[mrow, pcol] - match_coords[1])**2,
        (latitudes[prow, mcol] - match_coords[0])**2 + (longitudes[prow, mcol] - match_coords[1])**2,
        (latitudes[prow, pcol] - match_coords[0])**2 + (longitudes[prow, pcol] - match_coords[1])**2
    ]

    pos = np.argmin(geo_dist2)
    if pos == 0:
        return mrow, row, mcol, col
    elif pos == 1:
        return mrow, row, col, pcol
    elif pos == 2:
        return row, prow, mcol, col
    else:
        return row, prow, col, pcol
    
# Read stations data file
def load_in_situ_master_data(file_path):
    return pd.read_csv(file_path, sep=',', encoding='latin-1').reset_index(drop=True)

# Create daytime and nighttime LST files, they are filtered by view angle and error
def process_lst_data(data_dir, output):
    base_dir = Path(data_dir)
    output_dir = Path(output)

    # Files path
    data_dir_in = base_dir / "modis" / "LST"
    data_dir_out = output_dir / "preprocessing" / "modis"
    aemet_path = output_dir / "preprocessing" / "aemet" / "stations_data.csv"

    # Load stations data
    in_situ_df = load_in_situ_master_data(aemet_path)
    stations = in_situ_df['INDICATIVO'].to_list()

    lst_files = glob(str(data_dir_in / "*.nc"))
    day_LST = pd.DataFrame()
    night_LST = pd.DataFrame()


    for station in stations:

        df_aux = pd.DataFrame()
        df_aux = in_situ_df.loc[in_situ_df['INDICATIVO'] == station]
        lat = df_aux['LATITUD'].values[0]
        lon = df_aux['LONGITUD'].values[0]

        # Extract daytime and nighttime LST
        df_day, df_night = extract_lst_for_station(station, lat, lon, lst_files)
        day_LST = pd.concat([day_LST, df_day]).reset_index(drop=True)
        night_LST = pd.concat([night_LST, df_night]).reset_index(drop=True)

    # Apply filters
    day_LST = filter_lst_data(day_LST, 'QC_Day', 'Day_view_ang')
    night_LST = filter_lst_data(night_LST, 'QC_Night', 'Night_view_ang')

    # Save files
    day_LST.to_csv(data_dir_out / "day_LST.csv", index=False)
    night_LST.to_csv(data_dir_out / "night_LST.csv", index=False)

# Create daily NDVI file
def process_ndvi_data(data_dir, output):
    base_dir = Path(data_dir)
    output_dir = Path(output)

    # Files path
    data_dir_in = base_dir / "modis" / "NDVI"
    data_dir_out = output_dir / "preprocessing" / "modis"
    aemet_path = output_dir / "preprocessing" / "aemet" / "stations_data.csv"

    # Load stations data
    in_situ_df = load_in_situ_master_data(aemet_path)
    stations = in_situ_df['INDICATIVO'].to_list()

    ndvi_files = glob(str(data_dir_in / "*.nc"))
    
    ndvi_data = pd.DataFrame()

    for station in stations:

        df_aux = pd.DataFrame()
        df_aux = in_situ_df.loc[in_situ_df['INDICATIVO'] == station]
        lat = df_aux['LATITUD'].values[0]
        lon = df_aux['LONGITUD'].values[0]

        # Extract 16-days NDVI
        raw_ndvi = extract_ndvi_for_station(station, lat, lon, ndvi_files)

        # Creating daily NDVI
        daily_ndvi = ndvi_daily(raw_ndvi)
        daily_ndvi['INDICATIVO'] = station
        ndvi_data = pd.concat([ndvi_data, daily_ndvi]).reset_index(drop=True)

    # Save file
    ndvi_data.to_csv(data_dir_out / "NDVI.csv", index=False)

# Create albedo file resamplet to 1 km resolution
def process_albedo_data(data_dir, output):
    base_dir = Path(data_dir)
    output_dir = Path(output)

    # Files path
    data_dir_in = base_dir / "modis" / "albedo"
    data_dir_out = output_dir / "preprocessing" / "modis"
    aemet_path = output_dir / "preprocessing" / "aemet" / "stations_data.csv"

    # Load stations data
    in_situ_df = load_in_situ_master_data(aemet_path)
    stations = in_situ_df['INDICATIVO'].to_list()

    albedo_files = glob(str(data_dir_in / "*.nc"))
    albedo_data = pd.DataFrame()

    for station in stations:
        df_aux = pd.DataFrame()
        df_aux = in_situ_df.loc[in_situ_df['INDICATIVO'] == station]
        lat = df_aux['LATITUD'].values[0]
        lon = df_aux['LONGITUD'].values[0]

        # Extract and resample albedo
        alb = extract_albedo_for_station(station, lat, lon, albedo_files)
        albedo_data = pd.concat([albedo_data, alb]).reset_index(drop=True)

    # Save file
    albedo_data.to_csv(data_dir_out / "albedo.csv", index=False)

# Function that extracts LST variables for the different LST files per station and merge them
def extract_lst_for_station(station_id, lat, lon, file_list):
    df_day = pd.DataFrame()
    df_night = pd.DataFrame()

    coord =[lat,lon]

    for file in file_list:

        with xr.open_dataset(file, engine='netcdf4') as xr_data:
            lat_grid = np.tile(xr_data.lat.values, (xr_data.dims['lon'], 1)).T
            lon_grid = np.tile(xr_data.lon.values, (xr_data.dims['lat'], 1))
            datetimeindex = xr_data.indexes['time'].to_datetimeindex()

            x, y = find_nearest_col_row(lat_grid, lon_grid, coord)

            data_day = []
            data_night = []

            # LST variables used
            for ind, time in enumerate(datetimeindex):
                data_day.append([
                    time,
                    xr_data.Day_view_angl[ind, x, y].values,
                    xr_data.Day_view_time[ind, x, y].values,
                    xr_data.QC_Day[ind, x, y].values,
                    xr_data.LST_Day_1km[ind, x, y].values
                ])
                data_night.append([
                    time,
                    xr_data.Night_view_angl[ind, x, y].values,
                    xr_data.Night_view_time[ind, x, y].values,
                    xr_data.QC_Night[ind, x, y].values,
                    xr_data.LST_Night_1km[ind, x, y].values
                ])

        df_day = pd.concat([df_day, pd.DataFrame(data_day, columns=[
            'Time', 'Day_view_ang', 'Day_view_time', 'QC_Day', 'LST_day'])]).reset_index(drop=True)
        df_night = pd.concat([df_night, pd.DataFrame(data_night, columns=[
            'Time', 'Night_view_ang', 'Night_view_time', 'QC_Night', 'LST_night'])]).reset_index(drop=True)

    df_day = df_day.dropna()
    df_night = df_night.dropna()

    df_day['INDICATIVO'] = station_id
    df_night['INDICATIVO'] = station_id

    return df_day, df_night

# Function that extracts NDVI variables for the different files per station and merge them
def extract_ndvi_for_station(station_id, lat, lon, file_list):
    df = pd.DataFrame()

    for file in file_list:
        with xr.open_dataset(file, engine='netcdf4') as xr_data:
            lat_grid = np.tile(xr_data.lat.values, (xr_data.dims['lon'], 1)).T
            lon_grid = np.tile(xr_data.lon.values, (xr_data.dims['lat'], 1))
            datetimeindex = xr_data.indexes['time'].to_datetimeindex()
            x, y = find_nearest_col_row(lat_grid, lon_grid, [lat, lon])

            data = []

            # NDVI variables used
            for ind, time in enumerate(datetimeindex):
                ndvi = xr_data._1_km_16_days_NDVI[ind, x, y].values
                qc = xr_data._1_km_16_days_VI_Quality[ind, x, y].values
                data.append([time, ndvi, qc])

            df = pd.concat([df, pd.DataFrame(data, columns=['Time', 'NDVI', 'QC'])]).reset_index(drop=True)

    return df

# Function that extracts albedo variables for the different files per station, resample it to 1km and merge them
def extract_albedo_for_station(station_id, lat, lon, file_list):
    alb_st = pd.DataFrame()

    for file in file_list:
        with xr.open_dataset(file, engine='netcdf4') as xr_data:
            lat_grid = np.tile(xr_data.lat.values, (xr_data.dims['lon'], 1)).T
            lon_grid = np.tile(xr_data.lon.values, (xr_data.dims['lat'], 1))
            x1, x2, y1, y2 = find_2x2_pixel_bounds(lat_grid, lon_grid, [lat, lon])
            datetimeindex = xr_data.indexes['time'].to_datetimeindex()

            data = []

            # Albedo variables used and resample
            for ind, time in enumerate(datetimeindex):
                pixels = [
                    xr_data.Albedo_WSA_shortwave[ind, x1, y1].values,
                    xr_data.Albedo_WSA_shortwave[ind, x1, y2].values,
                    xr_data.Albedo_WSA_shortwave[ind, x2, y1].values,
                    xr_data.Albedo_WSA_shortwave[ind, x2, y2].values
                ]
                mean = np.nanmean(pixels)
                std = np.nanstd(pixels)
                data.append([time, mean, std])

            df = pd.DataFrame(data, columns=['Time', 'Albedo', 'SD'])
            df['INDICATIVO'] = station_id
            alb_st = pd.concat([alb_st, df]).reset_index(drop=True)

    return alb_st.dropna().reset_index(drop=True)

# Filter for LST data by LST error (> 2K) and view angle (> 45º)
def filter_lst_data(df, qc_col, angle_col):
    df = df[df[qc_col] < 128]
    df = df[abs(df[angle_col]) <= 45]
    return df

# Create daily ndvi values, start date and end date can be modified 
def ndvi_daily(df, start_date='2019-01-01', end_date='2022-12-31'):
    daily_dates = pd.date_range(start=start_date, end=end_date)
    final_data = pd.DataFrame()

    for day in daily_dates:
        closest = df.iloc[(df['Time'] - day).abs().argsort()[:1]]
        final_data = pd.concat([final_data, closest]).reset_index(drop=True)

    final_data['Time'] = daily_dates
    return final_data


## Main function

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Directory containing Maestro station files")
    parser.add_argument("--output_dir", required=True, help="Directory to save output files")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    

    output_dir.mkdir(parents=True, exist_ok=True)

    # Stations data file creation
    create_station_metadata(data_dir, output_dir)
    
    # In situ SAT file creation
    create_sat_database(data_dir, output_dir)

    # Calculate max, min and mean in situ SAT for each day
    process_daily_sat(output_dir)

    # Obtaining LST (daytime and nighttime) for each station
    process_lst_data(data_dir, output_dir)

    # Obtaining NDVI for each station
    process_ndvi_data(data_dir,output_dir)

    # Obtaining albedo for each station
    process_albedo_data(data_dir,output_dir)

    # Create a file for instantaneus SAT interpolated to the satellite's surpass time
    process_aemet_satellite_data(output_dir)


if __name__ == "__main__":
    main()