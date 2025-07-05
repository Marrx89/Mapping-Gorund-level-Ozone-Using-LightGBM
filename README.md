# Mapping-Gorund-level-Ozone-In-Jakarta-Using-LightGBM
to analyze and discover potential for estimates, spatial-temporal distributions, and predictions of GLO using LightGBM machine learning.

# Abstract
Ground-level ozone (GLO) is a harmful air pollutant with significant impacts on human health and the environment. In Indonesia, monitoring remains limited, with only two national and five Jakarta-based stations, underscoring the need for alternative approaches. This study aimed to estimate the spatial-temporal distribution of GLO concentrations in Jakarta from 2022 to 2024 using satellite data and machine learning. We integrated atmospheric, biophysical, and anthropogenic variables into three models: Linear Regression, Random Forest, and Light Gradient Boosting Machine (LightGBM). LightGBM achieved the highest predictive accuracy (R² = 0.73) when spatial variables were included. Key predictors included sulfur dioxide (SO₂), wind patterns, and nighttime light, reflecting the influence of emissions and meteorological conditions. Spatial analysis revealed higher GLO concentrations in industrial and densely built-up areas, especially in North and West Jakarta. Seasonal trends showed peaks during the dry season (74.33 μg/m³) and significant declines in the rainy season (10.16 μg/m³), driven by solar radiation and atmospheric stability. The highest GLO levels were observed in 2023, coinciding with El Niño-related warming. Local Climate Zone (LCZ) analysis further indicated that built-up areas had higher GLO concentrations compared to vegetated zones. This study demonstrates the potential of combining remote sensing and machine learning to estimate GLO in tropical megacities with limited monitoring infrastructure. The findings can support data-driven urban planning and policies aimed at reducing ozone pollution and promoting green urban development

# Objective
1. Analyzing the results of surface ozone concentration estimates in Jakarta from 2022 to 2024 using Light Gradient Boosting Machine.
2. Analyzing the spatial and temporal distribution of surface ozone concentrations in Jakarta during 2022–2024.
3. Projecting surface ozone concentrations in Jakarta in 2025.

# Data
1. The data used from Sentinel-5P includes Total atmospheric O₃ column, Tropospheric NO₂ column density, Integrated vertical CO column density, Vertical SO₂ column density at ground level, and Tropospheric HCHO column density.
2. Data used from ERA5 Meteorological Data includes Total precipitation (TP), Surface net solar radiation (SSR), Wind component U at 10 m altitude (U10) and wind component V at 10 m altitude (V10), Surface pressure (SP), Temperature at 2 m altitude (T2M), and Evaporation (E).
3. Land Surface Temperature (LST) data obtained from the MOD11A112 product.
4. Enhanced Vegetation Index (EVI) data obtained from the MOD13A2 product.
5. Nighttime light data obtained from the VIIRS sensor on the Suomi NPP satellite.
