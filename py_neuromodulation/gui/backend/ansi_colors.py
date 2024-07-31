from enum import StrEnum


class ANSI_COLORS(StrEnum):
    """Utility class for ANSI escape codes generate by Claude.ai"""

    # Reset
    RESET = "\033[0m"

    # Regular Colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bold Colors
    BLACK_BOLD = "\033[1;30m"
    RED_BOLD = "\033[1;31m"
    GREEN_BOLD = "\033[1;32m"
    YELLOW_BOLD = "\033[1;33m"
    BLUE_BOLD = "\033[1;34m"
    MAGENTA_BOLD = "\033[1;35m"
    CYAN_BOLD = "\033[1;36m"
    WHITE_BOLD = "\033[1;37m"

    # Underline Colors
    BLACK_UNDERLINED = "\033[4;30m"
    RED_UNDERLINED = "\033[4;31m"
    GREEN_UNDERLINED = "\033[4;32m"
    YELLOW_UNDERLINED = "\033[4;33m"
    BLUE_UNDERLINED = "\033[4;34m"
    MAGENTA_UNDERLINED = "\033[4;35m"
    CYAN_UNDERLINED = "\033[4;36m"
    WHITE_UNDERLINED = "\033[4;37m"

    # Background Colors
    BLACK_BACKGROUND = "\033[40m"
    RED_BACKGROUND = "\033[41m"
    GREEN_BACKGROUND = "\033[42m"
    YELLOW_BACKGROUND = "\033[43m"
    BLUE_BACKGROUND = "\033[44m"
    MAGENTA_BACKGROUND = "\033[45m"
    CYAN_BACKGROUND = "\033[46m"
    WHITE_BACKGROUND = "\033[47m"

    # High Intensity Colors
    BLACK_BRIGHT = "\033[90m"
    RED_BRIGHT = "\033[91m"
    GREEN_BRIGHT = "\033[92m"
    YELLOW_BRIGHT = "\033[93m"
    BLUE_BRIGHT = "\033[94m"
    MAGENTA_BRIGHT = "\033[95m"
    CYAN_BRIGHT = "\033[96m"
    WHITE_BRIGHT = "\033[97m"

    # High Intensity Bold Colors
    BLACK_BOLD_BRIGHT = "\033[1;90m"
    RED_BOLD_BRIGHT = "\033[1;91m"
    GREEN_BOLD_BRIGHT = "\033[1;92m"
    YELLOW_BOLD_BRIGHT = "\033[1;93m"
    BLUE_BOLD_BRIGHT = "\033[1;94m"
    MAGENTA_BOLD_BRIGHT = "\033[1;95m"
    CYAN_BOLD_BRIGHT = "\033[1;96m"
    WHITE_BOLD_BRIGHT = "\033[1;97m"

    # High Intensity Backgrounds
    BLACK_BACKGROUND_BRIGHT = "\033[100m"
    RED_BACKGROUND_BRIGHT = "\033[101m"
    GREEN_BACKGROUND_BRIGHT = "\033[102m"
    YELLOW_BACKGROUND_BRIGHT = "\033[103m"
    BLUE_BACKGROUND_BRIGHT = "\033[104m"
    MAGENTA_BACKGROUND_BRIGHT = "\033[105m"
    CYAN_BACKGROUND_BRIGHT = "\033[106m"
    WHITE_BACKGROUND_BRIGHT = "\033[107m"

    # Styles
    BOLD = "\033[1m"
    FAINT = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    BLINK = "\033[5m"
    NEGATIVE = "\033[7m"
    CROSSED = "\033[9m"

    @staticmethod
    def color_text(text, color):
        return f"{color}{text}{ANSI_COLORS.RESET}"

    @staticmethod
    def combine_styles(*styles):
        return "".join(styles)


# Usage examples:
# print(ANSI_COLORS.color_text("This is red bold text", ANSI_COLORS.combine_styles(ANSI_COLORS.RED, ANSI_COLORS.BOLD)))
# print(ANSI_COLORS.color_text("This is blue underlined text", ANSI_COLORS.combine_styles(ANSI_COLORS.BLUE, ANSI_COLORS.UNDERLINE)))
