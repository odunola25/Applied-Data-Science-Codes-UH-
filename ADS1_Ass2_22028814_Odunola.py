
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stats import skew, kurtosis


def read_data(filename, file_type="csv"):
    """
    Load the csv file downloaded from the worldbank database, clean 
    and transpose the data.

    Args:
        filename (string): The name of the csv file, located in the same
        directory.

    Returns:
        data (DataFrame) : A Dataframe of the cleaned original worldbank, with
        years as columns.
        transposed_data (DataFrame) : A DataFrame with the countries as columns
    """

    # Creating file name and reading data
    file = f"{filename}.{file_type}"
    g5_data = pd.read_csv(file)

    # Transposing the original dataframe to have 'YEAR' as column
    Transposed_g5_data = g5_data.T

    # Clean data and restructure data
    # Replacing NA using panda
    g5_data = g5_data.replace('..', 0)

    # Melt g5 data
    g5_data_melt = pd.melt(g5_data,
                           id_vars=['Series Name', 'Series Code', 'Country Name',
                                    'Country Code'],
                           var_name='Year',
                           value_name='Value')

    # Extract year from column names
    g5_data_melt['Year'] = g5_data_melt['Year'].str.extract(r'(\d{4})')

    # Convert 'Year' to numeric
    g5_data_melt['Year'] = pd.to_numeric(g5_data_melt['Year'], errors='coerce')

    # Using Pivot table to group dataframe by country as transposed data
    # for each country
    pivoted_g5_data = g5_data_melt.pivot_table(
        index=['Year', 'Series Name', 'Series Code'],
        columns='Country Name',
        values='Value',
        aggfunc='first')

    # Reset index for a clean structure
    pivoted_g5_data.reset_index(inplace=True)

    # Converting the country columns to float
    column_names = ['France', 'Germany', 'Italy',
                    'United Kingdom', 'United States']
    pivoted_g5_data[column_names] = pivoted_g5_data[column_names].astype(
        float)
    pivoted_g5_data['Year'] = pivoted_g5_data['Year'].astype(
        'object')

    # Return both Dataframes
    return g5_data, pivoted_g5_data, Transposed_g5_data, g5_data_melt


# calling function to read data
g5_data, pivoted_g5_data, Transposed_g5_data, g5_data_melt = read_data(
    "g5_data", 'csv')


# Defining function to explore statistical properties of selected indicators
def explore_indicators(data, countries, series_list):
    """
    Use three statistical Method to explore the data. Describe, Skewness and
    kurtosis.

    Args:
        data (DataFrame): The transposed DataFrame.
        countries (List) : A list of countries to examine.
        series_list (List) : A list of the series names of the indicators to
        examine.

    Returns:
        descriptives (DataFrame) : A DataFrame of the results of the summary
        statistics of the dataframe done using the .describe() function.
        other_stats (DataFrame) : A Dataframe of the results of the skewness 
        and kurtosis of the distribution.
    """
    # Create a new data frame (df) of the country and parsed series
    df = data[data['Series Name'].isin(series_list)]
    df = df.loc[:, countries]

    # Perform descriptive statistics of the numerical variables using describe
    descriptives = df.describe()

    # Calculate the skewness of the distribution
    # Use the skew and kurtosis function from the stats module to calculate
    # the skewness and kurtosis of each country
    country_skewness = list(skew(df[x]) for x
                            in df.columns)
    # Calculate kurtosis
    country_kurtosis = list(kurtosis(df[x]) for x
                            in df.columns)

    # Convert other statistical function results to dataframe
    other_stats = pd.DataFrame({'Country': countries,
                                'Skewness': [round(num, 2) for num in
                                             country_skewness],
                                'Kurtosis': [round(num, 2) for num in
                                             country_kurtosis]})

    return descriptives, other_stats

# calling finction for statstical analysis


selected_countries = ['Germany', 'Italy',
                      'France', 'United States', 'United Kingdom']
selected_indicators = [
    'Rural population (% of total population)',
    'CO2 emissions (metric tons per capita)',
    'Energy use (kg of oil equivalent per capita)',
    'Renewable energy consumption (% of total final energy consumption)',
    'Forest area (% of land area)']

descriptives, other_stats = explore_indicators(
    pivoted_g5_data, selected_countries, selected_indicators)


# Defining function to compare indicators using correlation analysis
def compare_indicators(g5_data_melt, countries, series_list):
    """
    Plot correlation matrix for all selected indicators for each country.

    Args:
        data (DataFrame): The original DataFrame.
        countries (List) : A list of countries to examine.
        series_list (List) : A list of the series names of the indicators to
        examine.

    Returns:
        df (DataFrame) : A Dataframe of the streamlined data that was explored.
        descriptives (DataFrame) : A DataFrame of the results of the summary
        statistics of the dataframe done using the .describe() function.
    """

    # Pivot the above melted data to have indicators as columns
    pivot_g5_data_melt = g5_data_melt.pivot_table(index=['Year', 'Country Name',
                                                         'Country Code'],
                                                  columns='Series Name',
                                                  values='Value',
                                                  aggfunc='first')

    # Reseting index
    pivot_g5_data_melt.reset_index(inplace=True)

    # Convert series name to float and year to object
    pivot_g5_data_melt[series_list] = pivot_g5_data_melt[series_list].astype(
        float)
    pivot_g5_data_melt['Year'] = pivot_g5_data_melt['Year'].astype(
        'object')

    # Plot the correlation for the indicators per country
    plt.figure(figsize=(15, 9))
    for country in countries:

        # Filter based on selected indicator codes and countries
        df = pivot_g5_data_melt[pivot_g5_data_melt['Country Name'] ==
                                country].iloc[:, 3:]

        # Calculate the correlation matrix
        corr_df = df.corr().round(2)

        # visualize the correlation matrix
        sns.heatmap(corr_df, cmap='Spectral', vmin=-1, vmax=1, center=0,
                    square=True, linewidths=.5,
                    cbar_kws={"shrink": .5}, annot=True)
        plt.title(f"Correlation Matrix of indicators for {country}")
        plt.tight_layout()
        plt.show()

    return pivot_g5_data_melt


# calling the function for correlation matrix
pivot_g5_data_melt = compare_indicators(g5_data_melt, selected_countries,
                                        selected_indicators)
print(pivot_g5_data_melt)


# comparing relationships and oatterns using paorplot
# Create a new data frame of the country and parsed series
df_countries = pivot_g5_data_melt[pivot_g5_data_melt['Country Name'].isin(
    selected_countries)]
df = df_countries.loc[:, selected_indicators]

# using pairplot to compare the relationships and patterns that occurs within
# the dataframe
plt.figure(figsize=(15, 9))
sns.pairplot(df)
plt.suptitle(
    f"Pair Plot of {', '.join(selected_indicators)} for {', '.join(selected_countries)}", y=1.02)
plt.tight_layout()
plt.show()

#visualize indicators as it relates to each countries

# create a variable (all indicators)
all_indicators = g5_data['Series Name'].unique()

# Loop through the variables in all indicators and plot
for variable in all_indicators:

    plot_data = g5_data[g5_data['Series Name'] == variable]

    # Drop Uneccessary Columns
    plot_data = plot_data.drop(['Country Code', 'Series Name', 'Series Code'],
                               axis=1)
    # Rename Columns
    plot_data.columns = ['Country Name', '2000', '2004', '2008', '2012',
                         '2016', '2020']
    years = ['2000', '2004', '2008', '2012', '2016', '2020']

    # Convert year columns to float
    plot_data[years] = plot_data[years].astype(float)

    #plot bar chart for plot_data
    plt.figure(figsize=(15, 10))
    plot_data.plot(x='Country Name', kind='bar',
                   stacked=False,
                   title=f'{variable}', fontsize=8.5)

    # Set labels and title
    plt.xlabel('Country')
    plt.ylabel("frequency")
    plt.xticks(rotation='horizontal')
    plt.legend(title='Years', bbox_to_anchor=(
        1, 1), loc='upper left', fontsize=10)

    # Show the plot
    plt.tight_layout()
    plt.show()


# Lineplot of countries grouped by indicators
# Loop variables in selected indicators
for indicator in selected_indicators:
    series_df = pivoted_g5_data[pivoted_g5_data['Series Name'] == indicator]

    # Plot line plot
    plt.figure(figsize=(15, 9))

    # Loop through the variables and plot each one with a different style
    for i, variable in enumerate(selected_countries):
        plt.plot(series_df['Year'], series_df[variable], label=variable)

    # Add labels and title
    plt.xlabel('Year', fontsize=20)
    plt.ylabel('Values', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=15)
    plt.title(
        f"Trend of {indicator} for {', '.join(selected_countries)} over the years",
        fontsize=20)

    # Add a legend
    plt.legend(title='Countries', bbox_to_anchor=(1.05, 1),
               loc='best', fontsize=20)

    plt.tight_layout()
    plt.show()
