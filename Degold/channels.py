"""
Channel configuration for AI Lipsync Generator.

Import and use:
    from channels import CHANNELS, get_channel

    config = get_channel("RRU")
    print(config.drive_folder)
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ChannelConfig:
    """Configuration for a lipsync channel."""
    code: str           # Channel code (e.g., "RRU", "JDRP", "DSR")
    drive_folder: str   # Output Google Drive folder ID
    avatar_folder: str  # Avatar image Google Drive folder ID
    name: str = ""      # Optional display name


# Channel configurations
CHANNELS = {
    "RRU": ChannelConfig(
        code="RRU",
        drive_folder="1XJY8HUEWvFH0cI68tvyrPEyU7bTpeNLQ",
        avatar_folder="16cHw8fgefC89zelexv_OSQNoqzhKekwO",
        name="RennReports",
    ),
    "DSR": ChannelConfig(
        code="DSR",
        drive_folder="1pawcev4vFELwEl80GyRejKDbYP_vfK2B",
        avatar_folder="1tK3bR2IjTOt2fJoJXKNc-kPphxHyJM9o",
        name="DeepSeaReports",
    ),
    # Add more channels here:
    # "JDRP": ChannelConfig(
    #     code="JDRP",
    #     drive_folder="...",
    #     avatar_folder="...",
    #     name="Journal of Drunk People",
    # ),
}


def get_channel(code: str) -> Optional[ChannelConfig]:
    """Get channel config by code (case-insensitive)."""
    return CHANNELS.get(code.upper())


def list_channels() -> list[ChannelConfig]:
    """List all configured channels."""
    return list(CHANNELS.values())
