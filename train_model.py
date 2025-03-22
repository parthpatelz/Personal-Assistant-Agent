import spacy
from spacy.training import Example

# Initialize a blank model with textcat
nlp = spacy.blank("en")
config = {
    "model": {
        "@architectures": "spacy.TextCatBOW.v3",
        "exclusive_classes": True,
        "ngram_size": 2,  # Use bigrams for better context
        "no_output_layer": False
    }
}
textcat = nlp.add_pipe("textcat", config=config)
textcat.add_label("SET_REMINDER")
textcat.add_label("SCHEDULE_MEETING")

# Training data: intent + example sentences
TRAIN_DATA = [
("Remind me to water the plants at 8 AM", {"cats": {"SET_REMINDER": 1.0, "SCHEDULE_MEETING": 0.0}}),
("Set a reminder to take the dog for a walk at 7 PM", {"cats": {"SET_REMINDER": 1.0, "SCHEDULE_MEETING": 0.0}}),
("Alert me to submit the report by 5 PM tomorrow", {"cats": {"SET_REMINDER": 1.0, "SCHEDULE_MEETING": 0.0}}),
("Notify me to call the dentist on Thursday at 10 AM", {"cats": {"SET_REMINDER": 1.0, "SCHEDULE_MEETING": 0.0}}),
("Remind me to pick up groceries at 6 PM today", {"cats": {"SET_REMINDER": 1.0, "SCHEDULE_MEETING": 0.0}}),
("Set a reminder to pay the electricity bill by Friday", {"cats": {"SET_REMINDER": 1.0, "SCHEDULE_MEETING": 0.0}}),
("Alert me to renew my gym membership next Monday", {"cats": {"SET_REMINDER": 1.0, "SCHEDULE_MEETING": 0.0}}),
("Remind me to check the mailbox at 4 PM", {"cats": {"SET_REMINDER": 1.0, "SCHEDULE_MEETING": 0.0}}),
("Set a reminder to take medication at 9 AM daily", {"cats": {"SET_REMINDER": 1.0, "SCHEDULE_MEETING": 0.0}}),
("Notify me to email the client by noon tomorrow", {"cats": {"SET_REMINDER": 1.0, "SCHEDULE_MEETING": 0.0}}),
("Remind me to book flight tickets next week", {"cats": {"SET_REMINDER": 1.0, "SCHEDULE_MEETING": 0.0}}),
("Alert me to finish the presentation by 3 PM today", {"cats": {"SET_REMINDER": 1.0, "SCHEDULE_MEETING": 0.0}}),
("Set a reminder to return library books on Saturday", {"cats": {"SET_REMINDER": 1.0, "SCHEDULE_MEETING": 0.0}}),
("Remind me to wish Sarah happy birthday tomorrow", {"cats": {"SET_REMINDER": 1.0, "SCHEDULE_MEETING": 0.0}}),
("Notify me to charge my phone at 10 PM", {"cats": {"SET_REMINDER": 1.0, "SCHEDULE_MEETING": 0.0}}),

("Schedule a team sync-up for Monday at 11 AM", {"cats": {"SET_REMINDER": 0.0, "SCHEDULE_MEETING": 1.0}}),
("Arrange a client call next Wednesday at 2 PM", {"cats": {"SET_REMINDER": 0.0, "SCHEDULE_MEETING": 1.0}}),
("Plan a project kickoff meeting on March 15th", {"cats": {"SET_REMINDER": 0.0, "SCHEDULE_MEETING": 1.0}}),
("Set up a brainstorming session for Friday afternoon", {"cats": {"SET_REMINDER": 0.0, "SCHEDULE_MEETING": 1.0}}),
("Book a demo with the sales team tomorrow at 9:30 AM", {"cats": {"SET_REMINDER": 0.0, "SCHEDULE_MEETING": 1.0}}),
("Organize a parent-teacher conference next week", {"cats": {"SET_REMINDER": 0.0, "SCHEDULE_MEETING": 1.0}}),
("Schedule a doctor's appointment for Thursday at 3 PM", {"cats": {"SET_REMINDER": 0.0, "SCHEDULE_MEETING": 1.0}}),
("Arrange a lunch meeting with Alex on Friday", {"cats": {"SET_REMINDER": 0.0, "SCHEDULE_MEETING": 1.0}}),
("Plan a Zoom call with the remote team next Monday", {"cats": {"SET_REMINDER": 0.0, "SCHEDULE_MEETING": 1.0}}),
("Set up a review meeting for the Q4 budget tomorrow", {"cats": {"SET_REMINDER": 0.0, "SCHEDULE_MEETING": 1.0}}),
("Book a conference room for a strategy session at 4 PM", {"cats": {"SET_REMINDER": 0.0, "SCHEDULE_MEETING": 1.0}}),
("Schedule a training workshop for new hires next month", {"cats": {"SET_REMINDER": 0.0, "SCHEDULE_MEETING": 1.0}}),
("Organize a catch-up call with the marketing team", {"cats": {"SET_REMINDER": 0.0, "SCHEDULE_MEETING": 1.0}}),
("Arrange a quarterly review meeting on April 10th", {"cats": {"SET_REMINDER": 0.0, "SCHEDULE_MEETING": 1.0}}),
("Plan a coffee chat with the CEO next Tuesday", {"cats": {"SET_REMINDER": 0.0, "SCHEDULE_MEETING": 1.0}}),

("Please remind me to take out the trash at 7 PM", {"cats": {"SET_REMINDER": 1.0, "SCHEDULE_MEETING": 0.0}}),
("Alert me to feed the cat tomorrow morning at 8 AM", {"cats": {"SET_REMINDER": 1.0, "SCHEDULE_MEETING": 0.0}}),
("Set a reminder for my dentist appointment on Thursday at 2:30 PM", {"cats": {"SET_REMINDER": 1.0, "SCHEDULE_MEETING": 0.0}}),
("Notify me to check the oven in 30 minutes", {"cats": {"SET_REMINDER": 1.0, "SCHEDULE_MEETING": 0.0}}),
("Remind me to wish Dad happy birthday at 9 AM tomorrow", {"cats": {"SET_REMINDER": 1.0, "SCHEDULE_MEETING": 0.0}}),
("Can you remind me to water the plants at 6 PM today?", {"cats": {"SET_REMINDER": 1.0, "SCHEDULE_MEETING": 0.0}}),
("Set a reminder to pick up the dry cleaning by Friday", {"cats": {"SET_REMINDER": 1.0, "SCHEDULE_MEETING": 0.0}}),
("Alert me to submit the project report by 5 PM sharp", {"cats": {"SET_REMINDER": 1.0, "SCHEDULE_MEETING": 0.0}}),
("Remind me to call the bank tomorrow at 11:15 AM", {"cats": {"SET_REMINDER": 1.0, "SCHEDULE_MEETING": 0.0}}),
("Notify me to take my vitamins every day at 8 AM", {"cats": {"SET_REMINDER": 1.0, "SCHEDULE_MEETING": 0.0}}),
("Set a reminder to renew my passport next week", {"cats": {"SET_REMINDER": 1.0, "SCHEDULE_MEETING": 0.0}}),
("Remind me to backup my laptop files tonight at 10 PM", {"cats": {"SET_REMINDER": 1.0, "SCHEDULE_MEETING": 0.0}}),
("Alert me to charge my phone before bed at 11 PM", {"cats": {"SET_REMINDER": 1.0, "SCHEDULE_MEETING": 0.0}}),
("Notify me to lock the doors at midnight", {"cats": {"SET_REMINDER": 1.0, "SCHEDULE_MEETING": 0.0}}),
("Set a reminder to email the client by noon tomorrow", {"cats": {"SET_REMINDER": 1.0, "SCHEDULE_MEETING": 0.0}}),

("Schedule a team standup every weekday at 9 AM", {"cats": {"SET_REMINDER": 0.0, "SCHEDULE_MEETING": 1.0}}),
("Arrange a budget review meeting for next Monday at 3 PM", {"cats": {"SET_REMINDER": 0.0, "SCHEDULE_MEETING": 1.0}}),
("Book a conference call with the Tokyo office tomorrow at 8 AM GMT", {"cats": {"SET_REMINDER": 0.0, "SCHEDULE_MEETING": 1.0}}),
("Plan a product launch meeting for April 20th at 10:30 AM", {"cats": {"SET_REMINDER": 0.0, "SCHEDULE_MEETING": 1.0}}),
("Set up a client onboarding session next Wednesday at 2 PM", {"cats": {"SET_REMINDER": 0.0, "SCHEDULE_MEETING": 1.0}}),
("Organize a brainstorming session with the design team on Friday", {"cats": {"SET_REMINDER": 0.0, "SCHEDULE_MEETING": 1.0}}),
("Schedule a 1:1 with my manager tomorrow at 4 PM", {"cats": {"SET_REMINDER": 0.0, "SCHEDULE_MEETING": 1.0}}),
("Arrange a project kickoff meeting for March 15th at 9 AM", {"cats": {"SET_REMINDER": 0.0, "SCHEDULE_MEETING": 1.0}}),
("Plan a quarterly review meeting with stakeholders next month", {"cats": {"SET_REMINDER": 0.0, "SCHEDULE_MEETING": 1.0}}),
("Book a demo with the engineering team for Tuesday at 11 AM", {"cats": {"SET_REMINDER": 0.0, "SCHEDULE_MEETING": 1.0}}),
("Schedule a training workshop for new hires on June 5th", {"cats": {"SET_REMINDER": 0.0, "SCHEDULE_MEETING": 1.0}}),
("Organize a strategy meeting with the board next Friday", {"cats": {"SET_REMINDER": 0.0, "SCHEDULE_MEETING": 1.0}}),
("Set up a lunch meeting with the marketing team tomorrow at 12:30 PM", {"cats": {"SET_REMINDER": 0.0, "SCHEDULE_MEETING": 1.0}}),
("Plan a Zoom call with remote developers on Thursday at 3 PM", {"cats": {"SET_REMINDER": 0.0, "SCHEDULE_MEETING": 1.0}}),
("Arrange a sprint planning meeting for next Monday morning", {"cats": {"SET_REMINDER": 0.0, "SCHEDULE_MEETING": 1.0}}),
("Alert me to take medication at 8 AM daily", {"cats": {"SET_REMINDER": 1.0, "SCHEDULE_MEETING": 0.0}}),
("Notify me to water the plants at 7 PM today", {"cats": {"SET_REMINDER": 1.0, "SCHEDULE_MEETING": 0.0}}),
("Remind me to submit the report by 5 PM tomorrow", {"cats": {"SET_REMINDER": 1.0, "SCHEDULE_MEETING": 0.0}}),
("Arrange a client call next Wednesday at 3 PM", {"cats": {"SET_REMINDER": 0.0, "SCHEDULE_MEETING": 1.0}}),
("Plan a team sync-up on Friday at 10 AM", {"cats": {"SET_REMINDER": 0.0, "SCHEDULE_MEETING": 1.0}}),
("Book a demo with the engineering team tomorrow", {"cats": {"SET_REMINDER": 0.0, "SCHEDULE_MEETING": 1.0}}),


# Ambiguous cases (teach the model to say "no")
("Schedule a reminder to call Mom", {"cats": {"SET_REMINDER": 0.0, "SCHEDULE_MEETING": 0.0}}),  # Invalid command
("Remind me to schedule a meeting tomorrow", {"cats": {"SET_REMINDER": 1.0, "SCHEDULE_MEETING": 0.0}}),  # "schedule" in task
("Set up a meeting reminder for Friday", {"cats": {"SET_REMINDER": 1.0, "SCHEDULE_MEETING": 0.0}}),  # Mixed keywords

# Negative examples for robustness
("What’s the weather like today?", {"cats": {"SET_REMINDER": 0.0, "SCHEDULE_MEETING": 0.0}}),
("Tell me a joke", {"cats": {"SET_REMINDER": 0.0, "SCHEDULE_MEETING": 0.0}}),
]

# Train the model
optimizer = nlp.initialize()
for _ in range(200):
    for text, annotations in TRAIN_DATA:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        nlp.update([example], sgd=optimizer)

# Save the model
nlp.to_disk("intent_classifier")