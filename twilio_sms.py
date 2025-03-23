
import os
from twilio.rest import Client
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def send_sms(body, to_phone_number):
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    from_phone = os.getenv("TWILIO_PHONE_NUMBER")

    if not account_sid or not auth_token or not from_phone:
        raise ValueError("Twilio credentials are missing. Set environment variables.")

    client = Client(account_sid, auth_token)

    message = client.messages.create(
        body=body,
        from_=from_phone,
        to=to_phone_number
    )
    return f"SMS sent: {message.sid}"

# Example usage:
# send_sms("Reminder: Meeting at 5 PM today!", "+19876543210")
