# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 14:39:30 2023

@author: HP
"""

import pandas as pd
import matplotlib.pyplot as plt

# QUESTION 1: USING LINE PLOT TO ANALYSE AND VISUALIZE DATA


def plot_drug_usage_by_age(csv_filename, output_filename):
    """
    define a function that shows the line plot of drug usage by age
    """
    # reading my data with panda
    Drug_usage_by_age = pd.read_csv("Drug_use_by_age.csv")

# creating line plot
    x = Drug_usage_by_age["age"]
    y = Drug_usage_by_age["alcohol_use"]
    y1 = Drug_usage_by_age["marijuana_use"]
    y2 = Drug_usage_by_age["cocaine_use"]
    y3 = Drug_usage_by_age["crack_use"]

    plt.figure(figsize=(14,10))
    plt.plot(x, y, '-', label="alcohol")
    plt.plot(x, y1, '--', label="marijuana")
    plt.plot(x, y2, '-,', label="cocaine")
    plt.plot(x, y3, ':', label="crack")

    plt.xlabel("age")
    plt.ylabel("drugs")
    plt.title("Chart showing drugs use by age")
    plt.legend()

# save and show plot
    plt.savefig("drug_usage_by_age.png")
    plt.show()
    return()


# Call the function to create the plot and save it
plot_drug_usage_by_age("Drug_usage_by_age.csv", "drug_usage_by_age.png")


# QUESTION 2: USING BAR CHART (SUB PLOT) TO ANALYSE AND VISUALISE DATA

def analyze_data_and_visualize(csv_filename, output_filename):
    """
    define a function to analyse the vaccination data
    """
    # Read the data from the CSV file
    Vaccination_data = pd.read_csv("vaccination-data.csv")

    # Dropping unwanted columns
    Vaccination_data = Vaccination_data.drop(
        columns=["ISO3", "WHO_REGION", "DATA_SOURCE", "VACCINES_USED"])

    # Creating a new column for total vaccine administered
    Vaccination_data["Total_vaccine_used"] = Vaccination_data["PERSONS_VACCINATED_1PLUS_DOSE"] + \
        Vaccination_data["PERSONS_LAST_DOSE"]

    # Analysing the first five countries represented in the given data
    first_five_countries = Vaccination_data.iloc[0:5]

    # Create a subplot for the visualizations
    plt.figure(figsize=(14, 10))

    # First subplot (country vs Total no of Vaccine allocated)
    plt.subplot(2, 2, 1)
    plt.bar(first_five_countries["COUNTRY"],
            first_five_countries["TOTAL_VACCINATIONS"], color="blue", label="Total Vaccine")
    plt.xlabel("Country")
    plt.ylabel("Total Vaccine")
    plt.title("Chart showing total Vaccines allocated")
    plt.legend()

    # Second subplot (Country vs persons who took more than one dose)
    plt.subplot(2, 2, 2)
    plt.bar(first_five_countries["COUNTRY"], first_five_countries["PERSONS_VACCINATED_1PLUS_DOSE"],
            color="skyblue", label="More than 1 dose taken")
    plt.xlabel("Country")
    plt.ylabel("No of persons")
    plt.title("chart showing persons who took 1dose")
    plt.legend()

    # Third subplot (country vs persons who completed their dose)
    plt.subplot(2, 2, 3)
    plt.bar(first_five_countries["COUNTRY"], first_five_countries["PERSONS_LAST_DOSE"],
            color="green", label="Completed dose")
    plt.xlabel("Country")
    plt.ylabel("Completed dose")
    plt.title("Chart showing persons who completed dose")
    plt.legend()

# save and show plot
    plt.savefig("Vaccination_data.png")
    plt.show()
    return()


# Call the function as the input and the desired output filename as arguments
analyze_data_and_visualize("vaccination-data.csv", "Vaccination_data.png")


# QUESTION 3: USING PIE CHART TO ANALYSE DATA

def visualize_birth_rate_data(csv_filename, output_filename):

    # reading the data
    Birth_rate = pd.read_csv("Birth rate data.csv")

    Birth_rate = Birth_rate.drop(
        columns=["month", "date_of_month", "day_of_week"])

    # grouping births by year to evaluate and analyse total no of births per year (2000-2014)
    Birth_by_year = Birth_rate.groupby("year").sum()["births"]

    """"
    calculate the % of birth for each year in relation to total number of births from
    2000-2014 to detrmine which year contributed the most to the
    total number of births accros the period in consideration
   """

    birth_percentage = Birth_by_year/Birth_by_year.sum()*100

    """
   Using a pie chart function to visualize the information and get insight on the
   years with the highest birth percentage
   """

    plt.figure(figsize=(18, 12))
    plt.pie(birth_percentage, labels=birth_percentage.index,
            autopct='%d%%', startangle=140)
    plt.title("Percentage of Births by Year (2000-2014)")
    plt.axis('equal')

   # save and show plot
    plt.savefig("Birth rate Visuals.png")
    plt.show()
    return()


# Call the function with your CSV file and the desired output filename as arguments
visualize_birth_rate_data("Birth rate data.csv", "Birth_rate_visuals.png")
