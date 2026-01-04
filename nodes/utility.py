from .base_imports import *
import pytz
from typing import Optional, Tuple, Union

class CurrentDateTime(comfy_io.ComfyNode):
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
    def define_schema(cls) -> comfy_io.Schema:
        return comfy_io.Schema(
            node_id="CurrentDateTime",
            display_name="📅 Current Date Time",
            category="t4ggno/utils",
            description="""
            Enhanced datetime utility that provides current date and time with timezone support and flexible formatting options.
            Returns formatted datetime, ISO format, and Unix timestamp.
            """,
            inputs=[
                comfy_io.Combo.Input("format", options=cls.SUPPORTED_FORMATS, default="YYYY-MM-DD_HH-MM-SS"),
                comfy_io.Combo.Input("timezone", options=["Local", "UTC", "US/Eastern", "US/Central", "US/Mountain", "US/Pacific", "Europe/London", "Europe/Paris", "Asia/Tokyo", "Custom"], default="Local"),
                comfy_io.String.Input("custom_format", default="%Y-%m-%d_%H-%M-%S", multiline=False, lazy=True),
                comfy_io.String.Input("custom_timezone", default="UTC", multiline=False, lazy=True),
            ],
            outputs=[
                comfy_io.String.Output(display_name="formatted_datetime"),
                comfy_io.String.Output(display_name="iso_datetime"),
                comfy_io.Int.Output(display_name="timestamp"),
            ]
        )

    @classmethod
    def check_lazy_status(cls, format, timezone, custom_format, custom_timezone, **kwargs):
        needed = []
        if format == "Custom" and custom_format is None:
            needed.append("custom_format")
        if timezone == "Custom" and custom_timezone is None:
            needed.append("custom_timezone")
        return needed

    @classmethod
    def execute(cls, format: str, timezone: str, custom_format: str, custom_timezone: str, **kwargs) -> comfy_io.NodeOutput:
        """Get current datetime with enhanced formatting and timezone support."""
        print("=============================")
        print("== Enhanced DateTime Utility")
        
        try:
            # Determine timezone
            tz = cls._get_timezone(timezone, custom_timezone)
            
            # Get current datetime
            now = datetime.now(tz) if tz else datetime.now()
            
            # Log current time
            print(f"Current datetime: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            print(f"Timezone: {timezone}")
            print(f"Format: {format}")
            
            # Format datetime based on selection
            formatted_dt = cls._format_datetime(now, format, custom_format)
            iso_dt = now.isoformat()
            timestamp = int(now.timestamp())
            
            print(f"Formatted output: {formatted_dt}")
            
            return comfy_io.NodeOutput(formatted_dt, iso_dt, timestamp)
            
        except Exception as e:
            error_msg = f"Error in datetime processing: {str(e)}"
            print(f"ERROR: {error_msg}")
            # Return fallback values
            fallback_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            return comfy_io.NodeOutput(fallback_time, datetime.now().isoformat(), int(time.time()))
    
    @classmethod
    def _get_timezone(cls, timezone: str, custom_timezone: str) -> Optional[pytz.BaseTzInfo]:
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
    
    @classmethod
    def _format_datetime(cls, dt: datetime, format_type: str, custom_format: str) -> str:
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
    def fingerprint_inputs(cls, **kwargs):
        """Always return NaN to ensure the node updates on every execution."""
        return float("nan")


class TimestampConverter(comfy_io.ComfyNode):
    """Convert between different timestamp formats and timezones."""
    
    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        return comfy_io.Schema(
            node_id="TimestampConverter",
            display_name="🔄 Timestamp Converter",
            category="t4ggno/utils",
            description="""
            Convert timestamps between different formats and timezones.
            Supports Unix timestamps, ISO strings, and custom formatted strings.
            """,
            inputs=[
                comfy_io.Combo.Input("input_type", options=["Unix Timestamp", "ISO String", "Formatted String"], default="Unix Timestamp"),
                comfy_io.String.Input("input_value", default="", multiline=False),
                comfy_io.Combo.Input("output_format", options=CurrentDateTime.SUPPORTED_FORMATS, default="YYYY-MM-DD_HH-MM-SS"),
                comfy_io.Combo.Input("source_timezone", options=["Local", "UTC", "US/Eastern", "US/Central", "US/Mountain", "US/Pacific", "Europe/London", "Europe/Paris", "Asia/Tokyo", "Custom"], default="UTC"),
                comfy_io.Combo.Input("target_timezone", options=["Local", "UTC", "US/Eastern", "US/Central", "US/Mountain", "US/Pacific", "Europe/London", "Europe/Paris", "Asia/Tokyo", "Custom"], default="Local"),
                comfy_io.String.Input("input_format", default="%Y-%m-%d %H:%M:%S", multiline=False, lazy=True),
                comfy_io.String.Input("output_custom_format", default="%Y-%m-%d_%H-%M-%S", multiline=False, lazy=True),
                comfy_io.String.Input("custom_source_tz", default="UTC", multiline=False, lazy=True),
                comfy_io.String.Input("custom_target_tz", default="UTC", multiline=False, lazy=True),
            ],
            outputs=[
                comfy_io.String.Output(display_name="converted_datetime"),
                comfy_io.String.Output(display_name="iso_datetime"),
                comfy_io.Int.Output(display_name="timestamp"),
            ]
        )

    @classmethod
    def check_lazy_status(cls, input_type, input_value, output_format, source_timezone, target_timezone, input_format, output_custom_format, custom_source_tz, custom_target_tz, **kwargs):
        needed = []
        if input_type == "Formatted String" and input_format is None:
            needed.append("input_format")
        
        if output_format == "Custom" and output_custom_format is None:
            needed.append("output_custom_format")
            
        if source_timezone == "Custom" and custom_source_tz is None:
            needed.append("custom_source_tz")
            
        if target_timezone == "Custom" and custom_target_tz is None:
            needed.append("custom_target_tz")
            
        return needed

    @classmethod
    def execute(cls, input_type: str, input_value: str, output_format: str, 
                         source_timezone: str, target_timezone: str,
                         input_format: str, output_custom_format: str,
                         custom_source_tz: str, custom_target_tz: str, **kwargs) -> comfy_io.NodeOutput:
        """Convert timestamp between formats and timezones."""
        print("=============================")
        print("== Timestamp Converter")
        print(f"Input type: {input_type}")
        print(f"Input value: {input_value}")
        
        try:
            # Parse input
            dt = cls._parse_input(input_type, input_value, input_format, source_timezone, custom_source_tz)
            
            # Convert timezone
            target_dt = cls._convert_timezone(dt, target_timezone, custom_target_tz)
            
            # Format output
            formatted_dt = cls._format_output(target_dt, output_format, output_custom_format)
            iso_dt = target_dt.isoformat()
            timestamp = int(target_dt.timestamp())
            
            print(f"Converted output: {formatted_dt}")
            
            return comfy_io.NodeOutput(formatted_dt, iso_dt, timestamp)
            
        except Exception as e:
            error_msg = f"Error in timestamp conversion: {str(e)}"
            print(f"ERROR: {error_msg}")
            # Return error message and current time as fallback
            now = datetime.now()
            return comfy_io.NodeOutput(error_msg, now.isoformat(), int(now.timestamp()))
    
    @classmethod
    def _parse_input(cls, input_type: str, input_value: str, input_format: str, 
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
                tz = cls._get_timezone(source_timezone, custom_source_tz)
                if tz:
                    dt = tz.localize(dt)
            return dt
    
    @classmethod
    def _convert_timezone(cls, dt: datetime, target_timezone: str, custom_target_tz: str) -> datetime:
        """Convert datetime to target timezone."""
        if target_timezone == "Local":
            return dt.astimezone() if dt.tzinfo else dt
        
        target_tz = cls._get_timezone(target_timezone, custom_target_tz)
        if target_tz:
            return dt.astimezone(target_tz) if dt.tzinfo else target_tz.localize(dt)
        return dt
    
    @classmethod
    def _get_timezone(cls, timezone: str, custom_timezone: str) -> Optional[pytz.BaseTzInfo]:
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
    
    @classmethod
    def _format_output(cls, dt: datetime, format_type: str, custom_format: str) -> str:
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
    def fingerprint_inputs(cls, input_value: str, output_format: str, **kwargs):
        """Return hash of input to detect changes."""
        input_str = f"{input_value}{output_format}"
        return hashlib.md5(input_str.encode()).hexdigest()


class RandomSeed(comfy_io.ComfyNode):
    """Generate random seeds with optional constraints."""
    
    @classmethod
    def define_schema(cls) -> comfy_io.Schema:
        return comfy_io.Schema(
            node_id="RandomSeed",
            display_name="🎲 Random Seed Generator",
            category="t4ggno/utils",
            description="""
            Generate random seeds with various modes and constraints.
            Useful for reproducible randomization in image generation workflows.
            """,
            inputs=[
                comfy_io.Combo.Input("mode", options=["Random", "Time-based", "Custom Range"], default="Random"),
                comfy_io.Int.Input("min_value", default=0, min=0, max=2147483647),
                comfy_io.Int.Input("max_value", default=2147483647, min=0, max=2147483647),
                comfy_io.Int.Input("fixed_seed", default=42, min=0, max=2147483647),
                comfy_io.Combo.Input("control_after_generate", options=["fixed", "random", "increment"], default="random"),
                comfy_io.Int.Input("value", default=0),
            ],
            outputs=[
                comfy_io.Int.Output(display_name="seed"),
                comfy_io.String.Output(display_name="seed_info"),
            ]
        )

    @classmethod
    def execute(cls, mode: str, min_value: int, max_value: int, fixed_seed: int, **kwargs) -> comfy_io.NodeOutput:
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
            
            return comfy_io.NodeOutput(seed, info)
            
        except Exception as e:
            error_msg = f"Error generating seed: {str(e)}"
            print(f"ERROR: {error_msg}")
            return comfy_io.NodeOutput(42, error_msg)  # Fallback to fixed seed

    @classmethod
    def fingerprint_inputs(cls, mode: str, min_value: int, max_value: int, **kwargs):
        """Return different values based on mode to control caching."""
        if mode == "Random":
            return float("nan")  # Always change for random mode
        elif mode == "Time-based":
            return time.time()  # Change based on time
        else:
            # For custom range, change when range parameters change
            return f"{min_value}_{max_value}"
