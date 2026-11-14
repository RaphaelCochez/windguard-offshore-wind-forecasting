"""
Copernicus C3S ERA5 Reanalysis Data Loader (North Sea Offshore Wind Corridors)
Retrieves 100m hub-height horizontal wind vectors (u100, v100), 10m wind vectors (u10, v10),
and surface pressure/thermodynamic fields over North Sea offshore wind corridors.
"""
import cdsapi

# North Sea Offshore Wind Corridor bounding box [North, West, South, East]
NORTH_SEA_BBOX = [56.0, 1.0, 51.0, 5.0]


def fetch_era5_storm_data(output_filename: str, year: int, month: int, days: list[str]):
    client = cdsapi.Client()
    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": [
                "100m_u_component_of_wind",
                "100m_v_component_of_wind",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "mean_sea_level_pressure",
                "surface_pressure",
            ],
            "year": str(year),
            "month": f"{int(month):02d}",
            "day": days,
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": NORTH_SEA_BBOX,
        },
        output_filename,
    )
