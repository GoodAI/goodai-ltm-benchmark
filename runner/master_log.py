from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List

from utils.constants import EventType


@dataclass
class LogEvent:
    type: EventType
    timestamp: datetime
    test_id: Optional[str] = None
    data: Optional[Dict[str, str]] = field(default_factory=dict)


class MasterLog:

    def __init__(self):
        self.log: List[LogEvent] = []

    def add_send_message(self, message: str, timestamp: datetime, test_id: Optional[str] = None, is_question: Optional[bool] = False):
        if test_id:
            event = LogEvent(EventType.SEND_MESSAGE, timestamp, test_id,  {"message": message, "is_question": is_question})
        else:
            event = LogEvent(EventType.SEND_FILL, timestamp, data={"message": message, "is_question": False})

        self.add_event(event)

    def add_response_message(self, message: str, timestamp: datetime, test_id: Optional[str] = None, is_question: Optional[bool] = False):
        if test_id:
            event = LogEvent(EventType.RESPONSE_MESSAGE, timestamp, test_id,  {"message": message, "is_question": is_question})
        else:
            event = LogEvent(EventType.RESPONSE_FILL, timestamp, data={"message": message, "is_question": False})

        self.add_event(event)

    def add_wait_event(self, test_id: str, timestamp: datetime, tokens=0, time=0):
        event = LogEvent(EventType.WAIT, timestamp=timestamp, test_id=test_id, data={"tokens":tokens, "time":time})
        self.add_event(event)

    def begin_test(self, test_id, timestamp):
        event = LogEvent(EventType.BEGIN, timestamp=timestamp, test_id=test_id)
        self.add_event(event)

    def end_test(self, test_id: str, timestamp: datetime):
        event = LogEvent(EventType.END, timestamp=timestamp, test_id=test_id)
        self.add_event(event)

    def add_event(self, event: LogEvent):
        self.log.append(event)

    def human_readable_full_log(self, test_id: str, message: str) -> List[str]:
        # Collate all the messages from the index point
        messages = []
        index = self.find_message(test_id, message)

        for event in self.log[index:]:
            if event.type in [EventType.SEND_MESSAGE, EventType.RESPONSE_MESSAGE, EventType.SEND_FILL, EventType.RESPONSE_FILL]:
                sender = "Test" if event.type == EventType.SEND_MESSAGE else "Agent"
                messages.append(f"{sender} ({event.timestamp}): {event.data['message']}")
            elif event.type == EventType.WAIT:
                messages.append(f"SYSTEM ({event.timestamp}): Test '{event.test_id}' WAITING for {event.data['tokens']} tokens until {event.data['time']}")
            elif event.type == EventType.BEGIN:
                messages.append(f"SYSTEM ({event.timestamp}): Test '{event.test_id}' BEGINS")
            elif event.type == EventType.END:
                messages.append(f"SYSTEM ({event.timestamp}): Test '{event.test_id}' ENDS")
            else:
                messages.append(f"SYSTEM ({event.timestamp}):  UNKNOWN EVENT")

        return messages

    def find_message(self, test_id: str, message: str) -> int:
        index = 0
        # Find the index of the sent message
        for event in self.log:
            if event.test_id == test_id and event.type == EventType.SEND_MESSAGE and event.data["message"] == message:
                break
            index += 1

        return index

    def from_json(self, json_obj):
        for log_event in json_obj:
            log_event["timestamp"] = datetime.fromtimestamp(log_event["timestamp"])
            log_event["type"] = EventType(log_event["type"])

            if log_event["data"].get("time") and log_event["data"]["time"] > 0:
                log_event["data"]["time"] = datetime.fromtimestamp(log_event["data"]["time"])

            self.log.append(LogEvent(**log_event))

    def to_json(self):
        ret = []
        for x in self.log:
            ret.append({"type": x.type.value, "timestamp": x.timestamp.timestamp(), "test_id": x.test_id, "data": self._to_json_data(x.data)})

        return ret

    def _to_json_data(self, data):
        ret = {}
        for k, v in data.items():
            if isinstance(v, datetime):
                v = v.timestamp()

            ret[k] = v
        return ret

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

    def messages(self, test_id: Optional[str] = "") -> list[str]:
        messages = []
        for event in self.log:
            if event.type in [EventType.SEND_MESSAGE, EventType.RESPONSE_MESSAGE, EventType.SEND_FILL, EventType.RESPONSE_FILL]:
                if test_id == "" or event.test_id == test_id:
                    sender = "Test" if event.type == EventType.SEND_MESSAGE else "Agent"
                    messages.append(f"{sender} ({event.timestamp}): {event.data['message']}")

        return messages

    def get_test_events(self, test_id: str) -> list[LogEvent]:
        events = []
        for event in self.log:
            if event.test_id == test_id:
                events.append(event)
        return events

    def as_context(self, test_id: Optional[str] = ""):
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
