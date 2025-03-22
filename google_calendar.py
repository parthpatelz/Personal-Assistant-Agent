from google.oauth2 import service_account
from googleapiclient.discovery import build

SCOPES = ['https://www.googleapis.com/auth/calendar']

def add_event(summary, start_time, end_time, timezone='UTC'):
    creds = service_account.Credentials.from_service_account_file('credentials.json', scopes=SCOPES)
    service = build('calendar', 'v3', credentials=creds)

    event = {
        'summary': summary,
        'start': {'dateTime': start_time, 'timeZone': timezone},
        'end': {'dateTime': end_time, 'timeZone': timezone},
    }
    service.events().insert(calendarId='primary', body=event).execute()
    return "Event added successfully!"

# Example usage
# add_event("Meeting with Team", "2025-02-05T09:00:00", "2025-02-05T10:00:00")