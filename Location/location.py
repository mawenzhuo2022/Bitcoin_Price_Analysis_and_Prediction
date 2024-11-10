# -*- coding: utf-8 -*-
# @Author  : Wenzhuo Ma
# @Time    : 2024/11/10 3:40
# @Function:
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize a geolocator with a custom user agent
geolocator = Nominatim(user_agent="YourCustomAppName")

def get_continent(location):
    try:
        logging.info(f"Processing location: {location}")
        time.sleep(1.5)  # Wait for 1.5 seconds between requests to comply with the service's usage policy
        location_result = geolocator.geocode(location, timeout=10)
        if location_result:
            lat, lon = location_result.latitude, location_result.longitude
            logging.info(f"Coordinates for {location}: Latitude = {lat}, Longitude = {lon}")
            location = geolocator.reverse([lat, lon], language='en')
            address = location.raw.get('address', {})
            logging.info(f"Raw address data: {address}")  # Log the raw address data
            country = address.get('country_code', 'Unknown').upper()
            continent_code = country_alpha2_to_continent_code(country)
            continent = continent_code_to_name(continent_code)
            logging.info(f"Continent for {location}: {continent}")  # Log the determined continent
            return continent
        return "Unknown"
    except GeocoderTimedOut:
        logging.error("Geocoding API timed out.")
        return "Timeout"
    except GeocoderServiceError as e:
        logging.error(f"Received an HTTP error (403): {e}")
        return "Error"
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return "Error"

def country_alpha2_to_continent_code(alpha2):
    # 这里你的代码应该是直接返回国家代码对应的大洲名称
    continents = {
        "AF": "Africa", "AS": "Asia", "EU": "Europe", "NA": "North America",
        "SA": "South America", "OC": "Oceania", "AN": "Antarctica"
    }
    # ISO3166-1 alpha-2 国家代码与大洲的映射
    country_to_continent = {
        "US": "NA", "DE": "EU", "TH": "AS", "GB": "EU", "PT": "EU",
        # 添加更多国家代码和大洲的映射
    }
    return continents.get(country_to_continent.get(alpha2.upper(), "Unknown"), "Unknown")

def continent_code_to_name(code):
    # 这个函数现在看起来是多余的，因为我们直接从国家代码转换为大洲名称
    return code  # 这里简化处理，直接返回大洲代码

# 之后的地理编码和处理代码不变


# Read the CSV file
logging.info("Reading CSV file...")
df = pd.read_csv('../data/Sentiment_Analysis/sentiment_analysis.csv')

# Apply the function to get continent names
logging.info("Assigning continent based on location...")
df['Continent'] = df['user_location'].apply(get_continent)

0+df.to_csv('../data/Location/location.csv', index=False)
logging.info("Classification done and new file saved!")
