import argparse
import arcpy
import pandas as pd
import numpy as np
import scipy.stats as stats
import logging
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Set up logging to record the process and any warnings/errors
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def set_workspace(workspace_path):
    """Set the ArcGIS workspace to the specified path."""
    arcpy.env.workspace = workspace_path
    arcpy.env.overwriteOutput = True  # Allow overwriting of existing files

def load_health_data(file_path):
    """Load health data from a CSV file into a DataFrame."""
    return pd.read_csv(file_path)

def clean_data(health_data):
    """Clean the health data by handling missing values.
    
    Fills missing values with the mean of their respective columns.
    Logs a warning if any missing values are found.
    """
    if health_data.isnull().values.any():
        logging.warning("Health data contains missing values. Cleaning data...")
        health_data.fillna(health_data.mean(), inplace=True)  # Replace NaNs with column means
    return health_data

def create_health_feature_class(health_data):
    """Create a feature class in ArcGIS from the health data."""
    # Use Longitude and Latitude to create points
    arcpy.MakeXYEventLayer_management(health_data, "Longitude", "Latitude", "HealthPoints")
    arcpy.CopyFeatures_management("HealthPoints", "HealthHotspots")  # Save as a feature class

def perform_kernel_density_estimation():
    """Perform Kernel Density Estimation on health hotspots to visualize density."""
    output_density = "DensityOutput"
    arcpy.KernelDensity_sa("HealthHotspots", output_density, cell_size=100, search_radius=1000)  # Estimate density
    return output_density

def check_air_quality_data(air_quality_data):
    """Check if the air quality data feature class exists and return it as a DataFrame."""
    if arcpy.Exists(air_quality_data):
        return arcpy.FeatureClassToDataFrame(air_quality_data)  # Load feature class into DataFrame
    else:
        raise FileNotFoundError(f"{air_quality_data} does not exist.")  # Raise an error if not found

def spatial_analysis(output_density, air_quality_data):
    """Perform spatial analysis and calculate correlation between health outcomes and air quality."""
    output_join = "Health_AirQuality_Join"
    arcpy.analysis.SpatialJoin(output_density, air_quality_data, output_join, match_type="INTERSECT")  # Join datasets

    air_quality_values = []
    health_outcomes = []

    # Extract air quality and health outcome values from the joined data
    with arcpy.da.SearchCursor(output_join, ['AirQualityValue', 'HealthOutcome']) as cursor:
        for row in cursor:
            air_quality_values.append(row[0])  # Collect air quality values
            health_outcomes.append(row[1])  # Collect health outcomes

    # Calculate correlation using NumPy and SciPy
    correlation, p_value = np.corrcoef(health_outcomes, air_quality_values)[0, 1], stats.pearsonr(health_outcomes, air_quality_values)[1]
    logging.info(f"Correlation between health outcomes and air quality: {correlation:.2f}, p-value: {p_value:.3f}")  # Log results
    return air_quality_values, health_outcomes  # Return data for further analysis

def buffer_analysis():
    """Create a buffer around health hotspots to assess surrounding areas."""
    arcpy.Buffer_analysis("HealthHotspots", "HealthHotspotBuffers", "1000 Meters")  # Create a buffer of 1000 meters

def create_heatmap(output_density):
    """Create and save a heatmap visualization of health hotspots density."""
    plt.figure(figsize=(10, 6))
    plt.title('Health Hotspots Density')
    plt.savefig("C:/path/to/your/heatmap.png")  # Save heatmap as a PNG
    logging.info("Heatmap saved.")

def save_map(mxd_path):
    """Save the ArcGIS map document and export it to a PDF file."""
    mxd = arcpy.mapping.MapDocument(mxd_path)  # Load the MXD file
    df = arcpy.mapping.ListDataFrames(mxd)[0]  # Get the first data frame
    
    # Add layers to the map document
    density_layer = arcpy.mapping.Layer("DensityOutput")
    arcpy.mapping.AddLayer(df, density_layer)

    buffer_layer = arcpy.mapping.Layer("HealthHotspotBuffers")
    arcpy.mapping.AddLayer(df, buffer_layer)

    mxd.save()  # Save the updated map document
    arcpy.mapping.ExportToPDF(mxd, "C:/path/to/your/output_map.pdf")  # Export to PDF
    logging.info("Map saved and exported to PDF.")

def predictive_modeling(air_quality_df, health_outcomes):
    """Build and evaluate a predictive model for health outcomes based on air quality data."""
    X = air_quality_df[['AirQualityValue']]  # Features (input variables)
    y = health_outcomes  # Target variable (output)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Random Forest Regressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)  # Train the model

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Evaluate the model using Mean Squared Error
    mse = mean_squared_error(y_test, predictions)
    logging.info(f"Mean Squared Error of the model: {mse:.2f}")  # Log evaluation results

def main():
    """Main function to execute the GIS analysis workflow."""
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="GIS Data Analysis for Health and Air Quality")
    parser.add_argument("--workspace", required=True, help="Path to the workspace")
    parser.add_argument("--health_data", required=True, help="Path to the health data CSV file")
    parser.add_argument("--air_quality_data", required=True, help="Path to the air quality shapefile")
    parser.add_argument("--mxd_path", required=True, help="Path to the ArcGIS MXD file")

    args = parser.parse_args()  # Parse the command line arguments

    # Execute the GIS analysis steps
    set_workspace(args.workspace)
    health_data = load_health_data(args.health_data)
    health_data = clean_data(health_data)
    create_health_feature_class(health_data)
    output_density = perform_kernel_density_estimation()
    air_quality_data = check_air_quality_data(args.air_quality_data)
    air_quality_values, health_outcomes = spatial_analysis(output_density, air_quality_data)
    buffer_analysis()
    create_heatmap(output_density)
    save_map(args.mxd_path)

    # Prepare data for predictive modeling and call the function
    air_quality_df = pd.DataFrame({'AirQualityValue': air_quality_values})
    predictive_modeling(air_quality_df, health_outcomes)

    logging.info("GIS Data Analysis Complete.")

if __name__ == "__main__":
    main()  # Run the main function
