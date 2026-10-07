"""
sms_alert.py — builds and (once a provider is configured) sends the
emergency SMS for a confirmed fire detection.

No SMS gateway is wired in yet. send_fire_alert() always builds the message
and logs what it *would* send, so the phone-number UI, the cooldown, and the
Alert page's history all work end-to-end right now. Once there's an account
with a provider (Fast2SMS, Twilio, MSG91, ...), fill in _dispatch()'s "sent"
branch with that provider's HTTP call — nothing else needs to change.

No Flask/SQLAlchemy imports here on purpose: rtsp_stream.py (a separate
process from the dashboard Flask app) imports this module directly.
"""
import os

ALERT_MESSAGE_TEMPLATE = (
    "EMERGENCY ALERT !! Fire detected at {location} ({camera}). Please check immediately."
)


def build_message(location, camera):
    return ALERT_MESSAGE_TEMPLATE.format(location=location, camera=camera)


def _dispatch(phone_number, message):
    api_key = os.environ.get("SMS_PROVIDER_API_KEY")
    if not api_key:
        print(f"[sms-alert] SIMULATED (no SMS_PROVIDER_API_KEY set) -> {phone_number}: {message}")
        return "simulated"

    # TODO: call your chosen SMS provider here once you have an account,
    # e.g. Fast2SMS:
    #   import requests
    #   requests.post(
    #       "https://www.fast2sms.com/dev/bulkV2",
    #       headers={"authorization": api_key},
    #       data={"route": "q", "message": message, "numbers": phone_number.lstrip("+91")},
    #       timeout=10,
    #   )
    print(f"[sms-alert] sending via configured provider -> {phone_number}")
    return "sent"


def send_fire_alert(phone_number, location, camera):
    """Returns (message, status) where status is 'sent' | 'simulated' | 'failed'."""
    message = build_message(location, camera)
    try:
        status = _dispatch(phone_number, message)
    except Exception as e:
        print(f"[sms-alert] failed: {e}")
        status = "failed"
    return message, status
