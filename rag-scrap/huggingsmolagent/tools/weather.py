"""
Weather tool using wttr.in - a free weather service that doesn't require API keys.
"""
import requests
import logging
from smolagents import tool
from typing import Union

logger = logging.getLogger(__name__)

@tool
def get_weather(location: str, format_type: str = "json") -> dict:
    """
    Get current weather information for a location using wttr.in.
    
    Args:
        location (str): City name or location (e.g., "Paris", "London", "New York")
        format_type (str): Format type - "json" for structured data (default), "text" for simple string
        
    Returns:
        dict: Weather information with structured fields when format_type="json", or simple summary when format_type="text"
    Examples:
        >>> get_weather("Paris")
        >>> get_weather("Chatou, France", format_type="text")
        >>> get_weather("London, UK")
    """
    logger.info(f"🌤️ Getting weather for: {location}")
    
    try:
        # wttr.in provides free weather data without API key
        if format_type == "json":
            url = f"https://wttr.in/{location}?format=j1"
        else:
            # Custom format string for readable output
            # %l=location %C=condition %t=temp %w=wind %h=humidity %p=precipitation
            url = f"https://wttr.in/{location}?format=%l:+%C+%t+(feels+like+%f),+Wind:+%w,+Humidity:+%h,+Precipitation:+%p"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9'
        }
        
        response = requests.get(url, headers=headers, timeout=8)
        response.raise_for_status()
        
        if format_type == "json":
            data = response.json()
            
            # Safely extract data with fallbacks
            current = data.get("current_condition", [{}])[0]
            location_info = data.get("nearest_area", [{}])[0]
            weather_desc_list = current.get("weatherDesc", [{}])
            
            # Extract location details
            area_name = location_info.get("areaName", [{}])[0].get("value", location) if location_info.get("areaName") else location
            country = location_info.get("country", [{}])[0].get("value", "") if location_info.get("country") else ""
            
            # Extract weather condition
            condition = weather_desc_list[0].get("value", "Unknown") if weather_desc_list else "Unknown"
            
            # Build structured response
            weather_data = {
                "location": area_name,
                "country": country,
                "temperature_c": current.get("temp_C", "N/A"),
                "temperature_f": current.get("temp_F", "N/A"),
                "feels_like_c": current.get("FeelsLikeC", "N/A"),
                "feels_like_f": current.get("FeelsLikeF", "N/A"),
                "condition": condition,
                "humidity": current.get("humidity", "N/A"),
                "wind_speed_kmph": current.get("windspeedKmph", "N/A"),
                "wind_speed_mph": current.get("windspeedMiles", "N/A"),
                "wind_direction": current.get("winddir16Point", "N/A"),
                "precipitation_mm": current.get("precipMM", "N/A"),
                "pressure_mb": current.get("pressure", "N/A"),
                "visibility_km": current.get("visibility", "N/A"),
                "uv_index": current.get("uvIndex", "N/A"),
                "cloud_cover": current.get("cloudcover", "N/A"),
                "observation_time": current.get("observation_time", "N/A"),
            }
            
            # Create human-readable summary
            full_location = f"{area_name}, {country}" if country else area_name
            summary = (
                f"🌍 **Weather in {full_location}**\n\n"
                f"🌡️ Temperature: {weather_data['temperature_c']}°C ({weather_data['temperature_f']}°F)\n"
                f"🤚 Feels like: {weather_data['feels_like_c']}°C ({weather_data['feels_like_f']}°F)\n"
                f"☁️ Conditions: {condition}\n"
                f"💧 Humidity: {weather_data['humidity']}%\n"
                f"🌬️ Wind: {weather_data['wind_speed_kmph']} km/h ({weather_data['wind_speed_mph']} mph) {weather_data['wind_direction']}\n"
                f"🌧️ Precipitation: {weather_data['precipitation_mm']} mm\n"
                f"☀️ UV Index: {weather_data['uv_index']}\n"
                f"👁️ Visibility: {weather_data['visibility_km']} km"
            )
            
            weather_data["summary"] = summary
            
            logger.info(f"✅ Weather fetched: {weather_data['temperature_c']}°C, {condition} in {full_location}")
            
            return weather_data
            
        else:
            # Simple text format - wttr.in returns pre-formatted string
            weather_text = response.text.strip()
            
            logger.info(f"✅ Weather fetched: {weather_text}")
            
            return {
                "location": location,
                "weather": weather_text,
                "summary": weather_text
            }
            
    except requests.exceptions.Timeout:
        error_msg = f"⏱️ Timeout getting weather for '{location}' - service took too long to respond"
        logger.error(error_msg)
        return {
            "error": "Timeout - weather service is slow or unreachable",
            "location": location,
            "suggestion": "Try again in a moment or check if the location name is correct"
        }
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            error_msg = f"❌ Location '{location}' not found"
            logger.error(error_msg)
            return {
                "error": f"Location '{location}' not recognized",
                "location": location,
                "suggestion": "Try using a different spelling or include the country (e.g., 'Paris, France')"
            }
        else:
            error_msg = f"❌ HTTP error getting weather: {e.response.status_code}"
            logger.error(error_msg)
            return {
                "error": f"Weather service error: {e.response.status_code}",
                "location": location
            }
            
    except requests.exceptions.RequestException as e:
        error_msg = f"❌ Network error getting weather for '{location}': {str(e)}"
        logger.error(error_msg)
        return {
            "error": f"Network error: {str(e)}",
            "location": location,
            "suggestion": "Check your internet connection"
        }
        
    except (KeyError, IndexError, TypeError) as e:
        error_msg = f"❌ Error parsing weather data for '{location}': {str(e)}"
        logger.error(error_msg)
        return {
            "error": f"Data parsing error: {str(e)}",
            "location": location,
            "suggestion": "The weather service returned unexpected data format"
        }
        
    except Exception as e:
        error_msg = f"❌ Unexpected error getting weather for '{location}': {str(e)}"
        logger.error(error_msg)
        return {
            "error": f"Unexpected error: {str(e)}",
            "location": location
        }


# Alternative simple function for quick weather checks
@tool
def get_weather_simple(location: str) -> str:
    """
    Get a quick weather summary in plain text.
    
    Args:
        location: City name or location
        
    Returns:
        str: Simple weather summary
        
    Example:
        >>> get_weather_simple("Tokyo")
        "Tokyo: ☀️ Clear 22°C"
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9'
    }

    try:
        timeout_s = float(__import__("os").getenv("WEATHER_HTTP_TIMEOUT_SECONDS", "8"))
    except Exception:
        timeout_s = 8.0

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            # Ultra-compact format
            url = f"https://wttr.in/{location}?format=3"
            response = requests.get(url, headers=headers, timeout=timeout_s)
            response.raise_for_status()
            return response.text.strip()
        except requests.exceptions.Timeout as e:
            last_error = e
            try:
                __import__("time").sleep(0.25 * (2**attempt))
            except Exception:
                pass
        except Exception as e:
            last_error = e
            break

    logger.error(f"Error in get_weather_simple (wttr.in) for '{location}': {str(last_error)}")
    return (
        f"Could not fetch weather for {location} (wttr.in timeout/unreachable). "
        f"Try again, or ask a different web question if your network blocks wttr.in."
    )