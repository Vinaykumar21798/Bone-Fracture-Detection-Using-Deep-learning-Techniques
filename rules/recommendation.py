"""
Rule-based recommendation system for fracture severity.
Provides BOTH detailed reasoning and clean clinical summaries.
"""

from typing import Dict


SEVERITY_RECOMMENDATIONS = {
    "Low": {
        "treatment": "Conservative treatment recommended",
        "surgery_required": "No",
        "actions": [
            "Rest and immobilization",
            "Pain management",
            "Gradual return to activity",
            "Follow-up if pain persists"
        ]
    },
    "Medium": {
        "treatment": "Orthopedic consultation required",
        "surgery_required": "Possibly",
        "actions": [
            "Immobilization using splint or cast",
            "Consult orthopedic specialist",
            "Avoid heavy activity",
            "Follow-up X-ray in 1–2 weeks"
        ]
    },
    "High": {
        "treatment": "Emergency medical care required",
        "surgery_required": "Likely Yes",
        "actions": [
            "Emergency medical care required",
            "Consult orthopedic specialist",
            "Avoid movement and weight bearing",
            "Follow-up X-ray after stabilization"
        ]
    }
}


def get_recommendation(severity_level: str) -> Dict:
    """
    Returns structured recommendation dictionary.
    """
    if severity_level not in SEVERITY_RECOMMENDATIONS:
        severity_level = "Medium"

    return SEVERITY_RECOMMENDATIONS[severity_level]


def format_clinical_summary(severity_level: str) -> str:
    """
    CLEAN clinical summary for GUI output (NO emojis, NO paragraphs).
    """
    rec = get_recommendation(severity_level)

    lines = [
        rec["treatment"],
        f"Surgery Required: {rec['surgery_required']}"
    ]

    for action in rec["actions"][1:]:
        lines.append(action)

    return "\n".join(lines)
