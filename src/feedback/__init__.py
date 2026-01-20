"""Feedback and rejection learning system.

This module provides:
- RejectionDatabase: Track rejected videos and channels
- ChannelScorer: Calculate channel reputation scores
- DaVinciImporter: Parse DaVinci Resolve marker exports
"""

from .rejections import (
    RejectedVideo,
    RejectionDatabase,
    ChannelStats,
    load_rejection_database,
    get_global_rejection_db_path,
    create_project_rejections_template,
)

from .davinci_import import (
    # Classes
    MarkerAction,
    MarkerFeedback,
    ImportResult,
    CSVValidationError,
    # Functions
    import_davinci_markers,
    import_davinci_markers_full,
    apply_approvals_to_database,
    export_rejections_to_csv,
    generate_feedback_report,
    validate_csv_structure,
    detect_fps_from_segments,
    check_segments_freshness,
    parse_edl_markers,
    detect_marker_file_type,
)

from .channel_scorer import (
    ChannelScorer,
    ChannelMetadata,
    ChannelCategory,
    create_channel_scorer,
    load_channel_categories_from_config,
)

from .client_profiles import (
    ClientProfile,
    ContentPreferences,
    QualityThresholds,
    EvolvedPreset,
    get_or_create_client_profile,
    list_client_profiles,
    evolve_preset_from_history,
    apply_client_profile_to_config,
    get_client_rejections_path,
)

__all__ = [
    # Rejection database
    'RejectedVideo',
    'RejectionDatabase',
    'ChannelStats',
    'load_rejection_database',
    'get_global_rejection_db_path',
    'create_project_rejections_template',
    # DaVinci import
    'MarkerAction',
    'MarkerFeedback',
    'ImportResult',
    'CSVValidationError',
    'import_davinci_markers',
    'import_davinci_markers_full',
    'apply_approvals_to_database',
    'export_rejections_to_csv',
    'generate_feedback_report',
    'validate_csv_structure',
    'detect_fps_from_segments',
    'check_segments_freshness',
    'parse_edl_markers',
    'detect_marker_file_type',
    # Channel scoring
    'ChannelScorer',
    'ChannelMetadata',
    'ChannelCategory',
    'create_channel_scorer',
    'load_channel_categories_from_config',
    # Client profiles
    'ClientProfile',
    'ContentPreferences',
    'QualityThresholds',
    'EvolvedPreset',
    'get_or_create_client_profile',
    'list_client_profiles',
    'evolve_preset_from_history',
    'apply_client_profile_to_config',
    'get_client_rejections_path',
]
