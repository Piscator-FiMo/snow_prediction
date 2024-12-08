# SLF Measurement Data

## Terms of Use:

[SLF Data Service - Terms Of Use](https://www.slf.ch/en/services-and-products/slf-data-service)

## Study Plot

Measurements from manual study plots. For further information please visit:
[SLF Description of Manual Measuring Sites](https://www.slf.ch/en/avalanche-bulletin-and-snow-situation/measured-values/description-of-manual-measuring-sites.html)

### Measurands

| Column Name    | Description                                                      | Unit |
|----------------|------------------------------------------------------------------|------|
| station_code   | Code used for measurement station                              	 | 	    |
| measure_date   | timestamp of measurement in UTC                                	 | 	    |
| hyear          | hydrological year [as string/symbol]                           	 | 	    |
| HS             | height of snowpack, total thickness of snowpack, instantaneous 	 | cm   |
| HN_1D          | height of new snow during the last 24 hours                    	 | cm   |
| HNW_1D         | water equivalent of new snow during the last 24h hours         	 | mm   |

## IMIS

Measurements from automated IMIS stations. For further information please visit:
[SLF Description of Automated Stations](https://www.slf.ch/en/avalanche-bulletin-and-snow-situation/measured-values/description-of-automated-stations.html)

### Measurands

**30 Minute Interval**:

| Column Name      | Description                                                       | Unit |
|------------------|-------------------------------------------------------------------|------|
| station_code     | Code used for measurement station                                 |      |
| measure_date     | timestamp of measurement in UTC                                   |      |
| hyear            | hydrological year                                                 |      |
| HS               | height of snowpack, total thickness of snowpack, instantaneous    | cm   |
| TA_30MIN_MEAN    | air temperature, mean over last 30 minutes                        | °C   |
| RH_30MIN_MEAN    | relative humidity, mean over last 30 minutes.                     | %    |
| VW_30MIN_MEAN    | wind speed, vectorial mean over last 30 minutes                   | m/s  |
| VW_30MIN_MAX     | wind speed, max 5sec measurement within last 30 minutes           | m/s  |
| DW_30MIN_MEAN    | wind direction, vectorial mean over last 30 minutes               | °    |
| DW_30MIN_SD      | wind direction, standard diviation over last 30 minutes           | °    |
| TSS_30MIN_MEAN   | snow surface temperature, mean over last 30 minutes               | °C   |
| TS0_30MIN_MEAN   | snow temperature at the ground, mean over last 30 minutes         | °C   |
| TS25_30MIN_MEAN  | snow temperature at 25cm above ground, mean over last 30 minutes  | °C   |
| TS50_30MIN_MEAN  | snow temperature at 50cm above ground, mean over last 30 minutes  | °C   |
| TS100_30MIN_MEAN | snow temperature at 100cm above ground, mean over last 30 minutes | °C   |
| RSWR_30MIN_MEAN  | reflected short wave radiation, mean over last 30 minutes         | W/m2 |

**Daily Interval**:

| Column Name  | Description                                                                    | Unit |
|--------------|--------------------------------------------------------------------------------|------|
| station_code | Code used for measurement station                                              |      |
| measure_date | timestamp of measurement in UTC                                                |      |
| hyear        | hydrological year                                                              |      |
| HS           | height of snowpack, total thickness of snowpack, median over last 24h at 6 UTC | cm   |
| HN_1D        | height of new snow during the last 24 hours modeled by SNOWPACK at 6 UTC       | cm   |
