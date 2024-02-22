import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any

from utils.constants import EventType


@dataclass
class LogEvent:
    type: EventType
    timestamp: datetime
    test_id: Optional[str] = None
    data: Optional[Dict[str, str]] = field(default_factory=dict)

    def to_json(self):
        return {"type": self.type.value, "timestamp": self.timestamp.timestamp(), "test_id": self.test_id, "data": self.json_data()}

    def json_data(self):
        ret = {}
        for k, v in self.data.items():
            if isinstance(v, datetime):
                v = v.timestamp()

            ret[k] = v
        return ret

    @classmethod
    def from_json(cls, json_event):
        json_event["timestamp"] = datetime.fromtimestamp(json_event["timestamp"])
        json_event["type"] = EventType(json_event["type"])

        if json_event["data"].get("time", None) and json_event["data"]["time"] > 0:
            json_event["data"]["time"] = datetime.fromtimestamp(json_event["data"]["time"])

        return cls(**json_event)


class MasterLog:

    def __init__(self, save_file: Path):
        self.log: List[LogEvent] = []
        self.save_file = save_file

    def add_send_message(self, message: str, timestamp: datetime, test_id: str = "", is_question: bool = False):
        event_type = EventType.SEND_MESSAGE if test_id != "" else EventType.SEND_FILL
        event = LogEvent(event_type, timestamp, test_id, {"message": message, "is_question": is_question})
        self.add_event(event)

    def add_response_message(self, message: str, timestamp: datetime, test_id: str = "", is_question: bool = False):
        event_type = EventType.RESPONSE_MESSAGE if test_id != "" else EventType.RESPONSE_FILL
        event = LogEvent(event_type, timestamp, test_id, {"message": message, "is_question": is_question})
        self.add_event(event)

    def add_wait_event(self, test_id: str, timestamp: datetime, tokens=0, time=0):
        event = LogEvent(EventType.WAIT, timestamp=timestamp, test_id=test_id, data={"tokens": tokens, "time": time})
        self.add_event(event)

    def begin_test(self, test_id, timestamp):
        event = LogEvent(EventType.BEGIN, timestamp=timestamp, test_id=test_id)
        self.add_event(event)

    def end_test(self, test_id: str, timestamp: datetime):
        event = LogEvent(EventType.END, timestamp=timestamp, test_id=test_id)
        self.add_event(event)

    def add_event(self, event: LogEvent):
        self.log.append(event)
        self.save_event(event)

    def save_event(self, event: LogEvent):
        with open(self.save_file, "a") as fd:
            fd.write(json.dumps(event.to_json()) + "\n")

    def human_readable_full_log(self, test_id: str, message: str) -> List[str]:
        # Collate all the messages from the index point
        messages = []
        index = self.find_message(test_id, message)

        for event in self.log[index:]:
            if event.type in [EventType.SEND_MESSAGE, EventType.RESPONSE_MESSAGE]:
                sender = "Test" if event.type == EventType.SEND_MESSAGE else "Agent"
                messages.append(f"{sender} ({event.timestamp}): {event.data['message']}")
            elif event.type in [ EventType.SEND_FILL, EventType.RESPONSE_FILL]:
                sender = "System" if event.type == EventType.SEND_FILL else "Agent"
                messages.append(f"{sender} ({event.timestamp}): {event.data['message']}")
            elif event.type == EventType.WAIT:
                messages.append(f"SYSTEM ({event.timestamp}): Test '{event.test_id}' WAITING for {event.data['tokens']} tokens until {event.data['time']}")
            elif event.type == EventType.BEGIN:
                messages.append(f"SYSTEM ({event.timestamp}): Test '{event.test_id}' BEGINS")
            elif event.type == EventType.END:
                messages.append(f"SYSTEM ({event.timestamp}): Test '{event.test_id}' ENDS")
            else:
                raise ValueError("Unknown event found")

        return messages

    def find_message(self, test_id: str, message: str) -> int:
        # Find the index of the sent message
        for idx, event in enumerate(self.log):
            if event.test_id == test_id and event.type == EventType.SEND_MESSAGE and event.data["message"] == message:
                return idx

        raise ValueError(f"Message {message} for test: {test_id} not found in log {self.human_readable_full_log()}")

    def load(self):
        self.log = []
        with open(self.save_file, "r") as fd:
            events_list = [json.loads(x) for x in fd.readlines()]

        for event in events_list:
            self.log.append(LogEvent.from_json(event))

    def get_tests_in_progress(self) -> Dict[str, int]:
        running_tests = []
        actions_taken = {}

        # Pass 1: Get the list of tests that are currently running
        for event in self.log:
            if event.type == EventType.BEGIN:
                running_tests.append(event.test_id)

            if event.type == EventType.END:
                assert event.test_id in running_tests, "Test has ended, but not begun!"
                running_tests.remove(event.test_id)

        # Pass 2: How many actions has the script taken?
        for test_id in running_tests:
            num_actions = 0
            for event in self.log:
                # The only two scripted events are sending a message and waiting.
                if event.test_id == test_id and event.type in [EventType.SEND_MESSAGE, EventType.WAIT]:
                    num_actions += 1

            actions_taken[test_id] = num_actions
        return actions_taken

    def messages(self, test_id: str = "") -> list[str]:
        messages = []
        for event in self.log:
            if event.type in [EventType.SEND_MESSAGE, EventType.RESPONSE_MESSAGE, EventType.SEND_FILL, EventType.RESPONSE_FILL]:
                if test_id == "" or event.test_id == test_id:
                    sender = "Test" if event.type == EventType.SEND_MESSAGE else "System" if event.type == EventType.SEND_FILL else "Agent"
                    messages.append(f"{sender} ({event.timestamp}): {event.data['message']}")

        return messages

    def get_test_events(self, test_id: str) -> list[LogEvent]:
        events = []
        for event in self.log:
            if event.test_id == test_id:
                events.append(event)
        return events

    def as_context(self, test_id: str = ""):
        context = []

        for event in self.log:
            if test_id == "" or event.test_id == test_id:
                if event.type in [EventType.SEND_MESSAGE, EventType.SEND_FILL]:
                    context.append({"role": "user", "content": event.data["message"], "timestamp": event.timestamp})
                elif event.type in [EventType.RESPONSE_MESSAGE, EventType.RESPONSE_FILL]:
                    context.append({"role": "assistant", "content": event.data["message"], "timestamp": event.timestamp})

        return context

    def get_questions_and_responses(self, test_id: str):
        questions = []
        responses = []

        for event in self.get_test_events(test_id):
            if event.type == EventType.SEND_MESSAGE and event.data["is_question"]:
                questions.append(event.data["message"])

            elif event.type == EventType.RESPONSE_MESSAGE and event.data["is_question"]:
                responses.append(event.data["message"])

        return questions, responses
