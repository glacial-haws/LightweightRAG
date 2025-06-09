
import logging
import time


class LogConfig:

    RED = "\033[38;5;160m"  # A bright red
    AMBER = "\033[38;5;214m"  # An orange/amber color
    GREEN = "\033[38;5;46m"   # A bright green
    GRAY = "\033[90m"
    HI = "\033[38;5;117m"  # Light blue
    RESET = "\033[0m"

    log_level: int = logging.INFO

    @classmethod
    def setup_logging(cls, log_level: int = logging.ERROR) -> None:
        """Setup logging configuration.
        
        Sets up console logging with colored output and also writes error logs 
        to a temporary file along with command line arguments, but only when 
        an actual error occurs.
        
        Args:
            log_level: The logging level to use
        """
        cls.log_level = log_level  # Store for worker processes
        
        class ColoredFormatter(logging.Formatter):
            converter = time.localtime
            
            def formatTime(self, record, datefmt=None):
                ct = self.converter(record.created)
                if datefmt:
                    s = time.strftime(datefmt, ct)
                    return f"\033[90m{s}.{int(record.msecs):03d}\033[0m"
                else:
                    return time.strftime("%Y-%m-%d %H:%M:%S", ct)
            
            def _format_module_name(self, name: str, max_length: int = 16) -> str:
                """Format module name to fit within max_length.
                If name contains dots, takes the rightmost part. Otherwise truncates from the end."""
                if len(name) <= max_length:
                    return name
                
                if '.' in name:
                    # Take rightmost part after dot
                    parts = name.split('.')
                    result = parts[-1]
                    if len(result) > max_length:
                        result = result[:max_length]
                else:
                    # Just truncate from end
                    result = name[:max_length]
                    
                return result
            
            def format(self, record):
                # Format and color module name in gray
                record.name = f'\033[90m{self._format_module_name(record.name)}\033[0m'
                
                if record.levelno == logging.DEBUG:
                    # For debug, color both level name and message in gray
                    record.levelname = f'\033[90m{record.levelname}\033[0m'
                    record.msg = f'\033[90m{record.msg}\033[0m'
                elif record.levelno == logging.INFO:
                    record.levelname = f'\033[32m{record.levelname}\033[0m'  # green
                elif record.levelno == logging.WARNING:
                    record.levelname = f'\033[33m{record.levelname}\033[0m'  # yellow
                elif record.levelno == logging.ERROR:
                    record.levelname = f'\033[31m{record.levelname}\033[0m'  # red
                
                return super().format(record)
        
        # Create console handler with colored formatter
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(ColoredFormatter(cls.get_log_str(), datefmt=cls.get_datefmt()))
        
        # Configure root logger with just the console handler initially
        logging.basicConfig(
            level=log_level,
            handlers=[console_handler]
        )

    @classmethod
    def get_log_str(cls) -> str:
        """Return log format string."""
        return '%(asctime)s %(name)s %(levelname)s: %(message)s'

    @classmethod
    def get_datefmt(cls) -> str:
        """Return date format string that shows only time."""
        return '%H:%M:%S'

