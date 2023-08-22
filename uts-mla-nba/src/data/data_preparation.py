import re

# Dictionary to map month names to corresponding feet values
month_to_feet = {
    'May': 5,
    'Jun': 6,
    'Jul': 7,
    'Apr': 4
}


def transform_height_to_inches(height_str):
    """Transform the height string into a numerical format in inches."""
    # List of placeholders and unexpected values in the height column
    INVALID_HEIGHT_VALUES = ["-", "None", "0", "Jr", "So", "Fr"]

    # Convert input to string to ensure consistent handling
    height_str = str(height_str)

    # Immediately return None for any invalid height values
    if height_str in INVALID_HEIGHT_VALUES:
        return None

    # Handle the format like "6'4" which represents 6 feet 4 inches
    if "'" in height_str:
        feet, inches = height_str.split("'")
        return int(feet) * 12 + int(inches)

    # Handle various formats with a dash, such as "X-Jun", "X-Jul", "X-Apr" and "Jun-00", "Jul-00", "Apr-00"
    if "-" in height_str:
        first_part, second_part = height_str.split("-")

        # Handle cases where the first part is a month (e.g., "Jun-00")
        if first_part in month_to_feet:
            return month_to_feet[first_part] * 12 + int(second_part)

        # Handle cases where the second part is the month (e.g., "11-May")
        if second_part in month_to_feet:
            return month_to_feet[second_part] * 12 + int(first_part)

    # Return None for any unhandled formats
    return None
