import arcpy
import pandas as pd
import numpy as np
import scipy.stats as stats

# Set workspace
arcpy.env.workspace = "C:/path/to/your/workspace"
arcpy.env.overwriteOutput = True

# Load public health data
health_data = pd.read_csv("C:/path/to/health_data.csv")

# 1. Data Cleaning and Preprocessing
# Check for missing values in health data
if health_data.isnull().values.any():
    print("Warning: Health data contains missing values. Consider cleaning the data.")
    health_data.fillna(health_data.mean(), inplace=True)  # Simple imputation with mean

# Create a feature class from the health data
arcpy.MakeXYEventLayer_management(health_data, "Longitude", "Latitude", "HealthPoints")

# Save the layer as a feature class
arcpy.CopyFeatures_management("HealthPoints", "HealthHotspots")

# 2. Spatial Analysis: Kernel Density Estimation
output_density = "DensityOutput"
arcpy.KernelDensity_sa("HealthHotspots", output_density, cell_size=100, search_radius=1000)

# Load air quality data
air_quality_data = "C:/path/to/air_quality.shp"  # Ensure this is a shapefile or feature class

# Check if air quality data exists
if arcpy.Exists(air_quality_data):
    air_quality_df = arcpy.FeatureClassToDataFrame(air_quality_data)
else:
    raise FileNotFoundError(f"{air_quality_data} does not exist.")

# 3. Statistical Analysis
# Perform spatial join to analyze the correlation between air quality and health hotspots
output_join = "Health_AirQuality_Join"
arcpy.analysis.SpatialJoin(output_density, air_quality_data, output_join, match_type="INTERSECT")

# Extract air quality and health outcome values
air_quality_values = []
health_outcomes = []

with arcpy.da.SearchCursor(output_join, ['AirQualityValue', 'HealthOutcome']) as cursor:
    for row in cursor:
        air_quality_values.append(row[0])
        health_outcomes.append(row[1])

# Calculate correlation (Pearson's r)
correlation, p_value = stats.pearsonr(health_outcomes, air_quality_values)

print(f"Correlation between health outcomes and air quality: {correlation:.2f}, p-value: {p_value:.3f}")

# Formula for Pearson correlation coefficient:
# r = Σ((x - x̄)(y - ȳ)) / sqrt(Σ(x - x̄)² * Σ(y - ȳ)²)
# where x̄ and ȳ are the means of x and y, respectively.

# 4. Buffer Analysis
# Create buffer around health hotspots to analyze the impact of proximity
buffer_distance = "1000 Meters"  # Define buffer distance
arcpy.Buffer_analysis("HealthHotspots", "HealthHotspotBuffers", buffer_distance)

# 5. Temporal Analysis (if applicable)
# For demonstration, we will assume there's a date column in health data
# health_data['Date'] = pd.to_datetime(health_data['Date'])

# You can analyze changes over time, e.g., by grouping data by year
# health_data['Year'] = health_data['Date'].dt.year
# yearly_health_data = health_data.groupby('Year')['HealthOutcome'].mean().reset_index()

# Create a time series plot (using matplotlib or other libraries)

# Create heat map
mxd_path = "C:/path/to/your/map_document.mxd"  # Specify the path to your MXD file
mxd = arcpy.mapping.MapDocument(mxd_path)
df = arcpy.mapping.ListDataFrames(mxd)[0]

# Add density layer to the map
density_layer = arcpy.mapping.Layer(output_density)
arcpy.mapping.AddLayer(df, density_layer)

# Add buffer layer to the map
buffer_layer = arcpy.mapping.Layer("HealthHotspotBuffers")
arcpy.mapping.AddLayer(df, buffer_layer)

# Save the map
mxd.save()

# Optionally export the map to a PDF
output_pdf = "C:/path/to/your/output_map.pdf"
arcpy.mapping.ExportToPDF(mxd, output_pdf)

print("GIS Data Analysis Complete. Check the map for visualizations.")
