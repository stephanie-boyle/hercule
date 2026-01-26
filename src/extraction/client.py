import requests

def fetch_gho_data(indicator_code):
    """Hits the WHO GHO API for a specific indicator."""
    url = f"https://ghoapi.azureedge.net/api/{indicator_code}"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        return response.json().get('value', [])
    except Exception as e:
        return []