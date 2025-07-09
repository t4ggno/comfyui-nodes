from .base_imports import *
import pytz
from typing import Optional, Tuple, Union

class CurrentDateTime:
    """Enhanced datetime utility with timezone support and flexible formatting."""
    
    SUPPORTED_FORMATS = [
        "YYYY-MM-DD_HH-MM-SS",
        "YYYY-MM-DD HH:MM:SS", 
        "YYYY/MM/DD HH:MM:SS",
        "DD-MM-YYYY HH:MM:SS",
        "MM/DD/YYYY HH:MM:SS",
        "ISO 8601",
        "Unix Timestamp",
        "Custom"
    ]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "format": (cls.SUPPORTED_FORMATS, {"default": "YYYY-MM-DD_HH-MM-SS"}),
                "timezone": (["Local", "UTC", "US/Eastern", "US/Central", "US/Mountain", "US/Pacific", "Europe/London", "Europe/Paris", "Asia/Tokyo", "Custom"], {"default": "Local"}),
            },
            "optional": {
                "custom_format": ("STRING", {"default": "%Y-%m-%d_%H-%M-%S", "multiline": False}),
                "custom_timezone": ("STRING", {"default": "UTC", "multiline": False}),
            },
            "hidden": {
                "control_after_generate": (["fixed", "random", "increment"], {"default": "increment"}),
                "value": ("INT", {"default": 0}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("formatted_datetime", "iso_datetime", "timestamp")
    FUNCTION = "get_current_date_time"
    CATEGORY = "t4ggno/utils"
    OUTPUT_NODE = False
    DESCRIPTION = """
Enhanced datetime utility that provides current date and time with timezone support and flexible formatting options.
Returns formatted datetime, ISO format, and Unix timestamp.
"""

    def get_current_date_time(self, format: str, timezone: str, custom_format: str = "", custom_timezone: str = "", **kwargs) -> Tuple[str, str, int]:
        """Get current datetime with enhanced formatting and timezone support."""
        print("=============================")
        print("== Enhanced DateTime Utility")
        
        try:
            # Determine timezone
            tz = self._get_timezone(timezone, custom_timezone)
            
            # Get current datetime
            now = datetime.now(tz) if tz else datetime.now()
            
            # Log current time
            print(f"Current datetime: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            print(f"Timezone: {timezone}")
            print(f"Format: {format}")
            
            # Format datetime based on selection
            formatted_dt = self._format_datetime(now, format, custom_format)
            iso_dt = now.isoformat()
            timestamp = int(now.timestamp())
            
            print(f"Formatted output: {formatted_dt}")
            
            return (formatted_dt, iso_dt, timestamp)
            
        except Exception as e:
            error_msg = f"Error in datetime processing: {str(e)}"
            print(f"ERROR: {error_msg}")
            # Return fallback values
            fallback_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            return (fallback_time, datetime.now().isoformat(), int(time.time()))
    
    def _get_timezone(self, timezone: str, custom_timezone: str) -> Optional[pytz.BaseTzInfo]:
        """Get timezone object based on selection."""
        if timezone == "Local":
            return None
        elif timezone == "UTC":
            return pytz.UTC
        elif timezone == "Custom":
            try:
                return pytz.timezone(custom_timezone)
            except pytz.exceptions.UnknownTimeZoneError:
                print(f"WARNING: Unknown timezone '{custom_timezone}', falling back to UTC")
                return pytz.UTC
        else:
            try:
                return pytz.timezone(timezone)
            except pytz.exceptions.UnknownTimeZoneError:
                print(f"WARNING: Unknown timezone '{timezone}', falling back to local time")
                return None
    
    def _format_datetime(self, dt: datetime, format_type: str, custom_format: str) -> str:
        """Format datetime based on selected format type."""
        format_map = {
            "YYYY-MM-DD_HH-MM-SS": "%Y-%m-%d_%H-%M-%S",
            "YYYY-MM-DD HH:MM:SS": "%Y-%m-%d %H:%M:%S",
            "YYYY/MM/DD HH:MM:SS": "%Y/%m/%d %H:%M:%S",
            "DD-MM-YYYY HH:MM:SS": "%d-%m-%Y %H:%M:%S",
            "MM/DD/YYYY HH:MM:SS": "%m/%d/%Y %H:%M:%S",
            "ISO 8601": None,  # Special case
            "Unix Timestamp": None,  # Special case
            "Custom": custom_format
        }
        
        if format_type == "ISO 8601":
            return dt.isoformat()
        elif format_type == "Unix Timestamp":
            return str(int(dt.timestamp()))
        else:
            format_str = format_map.get(format_type, "%Y-%m-%d_%H-%M-%S")
            return dt.strftime(format_str)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Always return NaN to ensure the node updates on every execution."""
        return float("nan")


class TimestampConverter:
    """Convert between different timestamp formats and timezones."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_type": (["Unix Timestamp", "ISO String", "Formatted String"], {"default": "Unix Timestamp"}),
                "input_value": ("STRING", {"default": "", "multiline": False}),
                "output_format": (CurrentDateTime.SUPPORTED_FORMATS, {"default": "YYYY-MM-DD_HH-MM-SS"}),
                "source_timezone": (["Local", "UTC", "US/Eastern", "US/Central", "US/Mountain", "US/Pacific", "Europe/London", "Europe/Paris", "Asia/Tokyo", "Custom"], {"default": "UTC"}),
                "target_timezone": (["Local", "UTC", "US/Eastern", "US/Central", "US/Mountain", "US/Pacific", "Europe/London", "Europe/Paris", "Asia/Tokyo", "Custom"], {"default": "Local"}),
            },
            "optional": {
                "input_format": ("STRING", {"default": "%Y-%m-%d %H:%M:%S", "multiline": False}),
                "output_custom_format": ("STRING", {"default": "%Y-%m-%d_%H-%M-%S", "multiline": False}),
                "custom_source_tz": ("STRING", {"default": "UTC", "multiline": False}),
                "custom_target_tz": ("STRING", {"default": "UTC", "multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("converted_datetime", "iso_datetime", "timestamp")
    FUNCTION = "convert_timestamp"
    CATEGORY = "t4ggno/utils"
    OUTPUT_NODE = False
    DESCRIPTION = """
Convert timestamps between different formats and timezones.
Supports Unix timestamps, ISO strings, and custom formatted strings.
"""

    def convert_timestamp(self, input_type: str, input_value: str, output_format: str, 
                         source_timezone: str, target_timezone: str,
                         input_format: str = "", output_custom_format: str = "",
                         custom_source_tz: str = "", custom_target_tz: str = "") -> Tuple[str, str, int]:
        """Convert timestamp between formats and timezones."""
        print("=============================")
        print("== Timestamp Converter")
        print(f"Input type: {input_type}")
        print(f"Input value: {input_value}")
        
        try:
            # Parse input
            dt = self._parse_input(input_type, input_value, input_format, source_timezone, custom_source_tz)
            
            # Convert timezone
            target_dt = self._convert_timezone(dt, target_timezone, custom_target_tz)
            
            # Format output
            formatted_dt = self._format_output(target_dt, output_format, output_custom_format)
            iso_dt = target_dt.isoformat()
            timestamp = int(target_dt.timestamp())
            
            print(f"Converted output: {formatted_dt}")
            
            return (formatted_dt, iso_dt, timestamp)
            
        except Exception as e:
            error_msg = f"Error in timestamp conversion: {str(e)}"
            print(f"ERROR: {error_msg}")
            # Return error message and current time as fallback
            now = datetime.now()
            return (error_msg, now.isoformat(), int(now.timestamp()))
    
    def _parse_input(self, input_type: str, input_value: str, input_format: str, 
                    source_timezone: str, custom_source_tz: str) -> datetime:
        """Parse input value based on input type."""
        if input_type == "Unix Timestamp":
            timestamp = float(input_value)
            return datetime.fromtimestamp(timestamp, tz=pytz.UTC)
        elif input_type == "ISO String":
            return datetime.fromisoformat(input_value.replace('Z', '+00:00'))
        else:  # Formatted String
            dt = datetime.strptime(input_value, input_format)
            # Apply source timezone
            if source_timezone != "Local":
                tz = self._get_timezone(source_timezone, custom_source_tz)
                if tz:
                    dt = tz.localize(dt)
            return dt
    
    def _convert_timezone(self, dt: datetime, target_timezone: str, custom_target_tz: str) -> datetime:
        """Convert datetime to target timezone."""
        if target_timezone == "Local":
            return dt.astimezone() if dt.tzinfo else dt
        
        target_tz = self._get_timezone(target_timezone, custom_target_tz)
        if target_tz:
            return dt.astimezone(target_tz) if dt.tzinfo else target_tz.localize(dt)
        return dt
    
    def _get_timezone(self, timezone: str, custom_timezone: str) -> Optional[pytz.BaseTzInfo]:
        """Get timezone object based on selection."""
        if timezone == "UTC":
            return pytz.UTC
        elif timezone == "Custom":
            try:
                return pytz.timezone(custom_timezone)
            except pytz.exceptions.UnknownTimeZoneError:
                print(f"WARNING: Unknown timezone '{custom_timezone}', falling back to UTC")
                return pytz.UTC
        else:
            try:
                return pytz.timezone(timezone)
            except pytz.exceptions.UnknownTimeZoneError:
                print(f"WARNING: Unknown timezone '{timezone}', falling back to UTC")
                return pytz.UTC
    
    def _format_output(self, dt: datetime, format_type: str, custom_format: str) -> str:
        """Format datetime for output."""
        format_map = {
            "YYYY-MM-DD_HH-MM-SS": "%Y-%m-%d_%H-%M-%S",
            "YYYY-MM-DD HH:MM:SS": "%Y-%m-%d %H:%M:%S",
            "YYYY/MM/DD HH:MM:SS": "%Y/%m/%d %H:%M:%S",
            "DD-MM-YYYY HH:MM:SS": "%d-%m-%Y %H:%M:%S",
            "MM/DD/YYYY HH:MM:SS": "%m/%d/%Y %H:%M:%S",
            "ISO 8601": None,
            "Unix Timestamp": None,
            "Custom": custom_format
        }
        
        if format_type == "ISO 8601":
            return dt.isoformat()
        elif format_type == "Unix Timestamp":
            return str(int(dt.timestamp()))
        else:
            format_str = format_map.get(format_type, "%Y-%m-%d_%H-%M-%S")
            return dt.strftime(format_str)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Return hash of input to detect changes."""
        input_str = f"{kwargs.get('input_value', '')}{kwargs.get('output_format', '')}"
        return hashlib.md5(input_str.encode()).hexdigest()


class RandomSeed:
    """Generate random seeds with optional constraints."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["Random", "Time-based", "Custom Range"], {"default": "Random"}),
            },
            "optional": {
                "min_value": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "max_value": ("INT", {"default": 2147483647, "min": 0, "max": 2147483647}),
                "fixed_seed": ("INT", {"default": 42, "min": 0, "max": 2147483647}),
            },
            "hidden": {
                "control_after_generate": (["fixed", "random", "increment"], {"default": "random"}),
                "value": ("INT", {"default": 0}),
            },
        }

    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("seed", "seed_info")
    FUNCTION = "generate_seed"
    CATEGORY = "t4ggno/utils"
    OUTPUT_NODE = False
    DESCRIPTION = """
Generate random seeds with various modes and constraints.
Useful for reproducible randomization in image generation workflows.
"""

    def generate_seed(self, mode: str, min_value: int = 0, max_value: int = 2147483647, 
                     fixed_seed: int = 42, **kwargs) -> Tuple[int, str]:
        """Generate seed based on selected mode."""
        print("=============================")
        print("== Random Seed Generator")
        print(f"Mode: {mode}")
        
        try:
            if mode == "Random":
                seed = random.randint(min_value, max_value)
                info = f"Random seed between {min_value} and {max_value}"
            elif mode == "Time-based":
                seed = int(time.time() * 1000000) % max_value
                if seed < min_value:
                    seed = min_value + (seed % (max_value - min_value))
                info = f"Time-based seed: {seed}"
            else:  # Custom Range
                seed = random.randint(min_value, max_value)
                info = f"Custom range seed: {seed} (range: {min_value}-{max_value})"
            
            print(f"Generated seed: {seed}")
            print(f"Seed info: {info}")
            
            return (seed, info)
            
        except Exception as e:
            error_msg = f"Error generating seed: {str(e)}"
            print(f"ERROR: {error_msg}")
            return (42, error_msg)  # Fallback to fixed seed

    @classmethod
    def IS_CHANGED(cls, mode: str, **kwargs):
        """Return different values based on mode to control caching."""
        if mode == "Random":
            return float("nan")  # Always change for random mode
        elif mode == "Time-based":
            return time.time()  # Change based on time
        else:
            # For custom range, change when range parameters change
            return f"{kwargs.get('min_value', 0)}_{kwargs.get('max_value', 2147483647)}"


NODE_CLASS_MAPPINGS = {
    "CurrentDateTime": CurrentDateTime,
    "TimestampConverter": TimestampConverter,
    "RandomSeed": RandomSeed,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CurrentDateTime": "📅 Current Date Time",
    "TimestampConverter": "🔄 Timestamp Converter", 
    "RandomSeed": "🎲 Random Seed Generator",
}
