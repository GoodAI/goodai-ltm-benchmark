import json
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any, Callable, Iterator

from utils.constants import EventType, EVENT_SENDER, ResetPolicy


@dataclass
class LogEvent:
    type: EventType
    timestamp: datetime
    test_id: Optional[str] = None
    data: Optional[Dict[str, Any]] = field(default_factory=dict)

    def to_json(self):
        return {
            "type": self.type.value,
            "timestamp": self.timestamp.timestamp(),
            "test_id": self.test_id,
            "data": self.json_data()
        }

    def json_data(self) -> dict:
        ret = {}
        for k, v in self.data.items():
            if isinstance(v, timedelta):
                v = v.seconds
            if isinstance(v, ResetPolicy):
                v = v.value
            ret[k] = v
        return ret

    @classmethod
    def from_json(cls, json_event: dict) -> "LogEvent":
        kwargs = deepcopy(json_event)
        kwargs["type"] = EventType(kwargs["type"])
        kwargs["timestamp"] = datetime.fromtimestamp(kwargs["timestamp"])
        if "time" in kwargs["data"]:
            kwargs["data"]["time"] = timedelta(seconds=kwargs["data"]["time"])
        if "policy" in kwargs["data"]:
            kwargs["data"]["policy"] = ResetPolicy(kwargs["data"]["policy"])
        return cls(**kwargs)


class MasterLog:

    def __init__(self, save_file: Path):
        self.log: List[LogEvent] = []
        self.save_file = save_file

    def add_send_message(self, message: str, timestamp: datetime, test_id: str = "", is_question: bool = False):
        event_type = EventType.SEND_MESSAGE if test_id != "" else EventType.SEND_FILL
        assert not (is_question and event_type == EventType.SEND_FILL), "Filler is tagged as a question."
        event = LogEvent(event_type, timestamp, test_id, {"message": message, "is_question": is_question})
        self.add_event(event)

    def add_response_message(self, message: str, timestamp: datetime, test_id: str = "", is_question: bool = False):
        event_type = EventType.RESPONSE_MESSAGE if test_id != "" else EventType.RESPONSE_FILL
        assert not (is_question and event_type == EventType.RESPONSE_FILL), "Response to filler is tagged as a question."
        event = LogEvent(event_type, timestamp, test_id, {"message": message, "is_question": is_question})
        self.add_event(event)

    def add_wait_event(
        self,
        test_id: str,
        timestamp: datetime,
        tokens: int = 0,
        time: timedelta = timedelta(seconds=0),
        percentage_finished: float = 0.0
    ):
        event = LogEvent(
            EventType.WAIT,
            timestamp=timestamp,
            test_id=test_id,
            data={"tokens": tokens, "time": time, "percentage_finished": percentage_finished},
        )
        self.add_event(event)

    def begin_test(self, test_id, timestamp):
        event = LogEvent(EventType.BEGIN, timestamp=timestamp, test_id=test_id)
        self.add_event(event)

    def end_test(self, test_id: str, timestamp: datetime):
        event = LogEvent(EventType.END, timestamp=timestamp, test_id=test_id)
        self.add_event(event)

    def add_reset_event(self, policy: ResetPolicy, timestamp: datetime):
        event = LogEvent(EventType.SUITE_RESET, timestamp=timestamp, test_id="", data={"policy": policy})
        self.add_event(event)

    def add_llm_call(self, test_id: str, timestamp: datetime, response: str):
        event = LogEvent(EventType.LLM_CALL, test_id=test_id, timestamp=timestamp, data={"response": response})
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
            if event.type in [EventType.SEND_MESSAGE, EventType.RESPONSE_MESSAGE, EventType.SEND_FILL, EventType.RESPONSE_FILL]:
                sender = EVENT_SENDER[event.type]
                messages.append(f"{sender} ({event.timestamp}): {event.data['message']}")
            elif event.type == EventType.WAIT:

                wait_cond = []
                if event.data['tokens'] > 0:
                    wait_cond.append(f"{event.data['tokens']} TOKENS")
                if event.data['time'].seconds > 0:
                    wait_cond.append(f"{event.data['time']} TIME")
                if event.data['percentage_finished'] > 0:
                    wait_cond.append(f"{event.data['percentage_finished']}% TESTS FINISHED")
                wait_cond = ", ".join(wait_cond)

                messages.append(f"SYSTEM ({event.timestamp}): Test '{event.test_id}' WAITING for {wait_cond}.")
            elif event.type == EventType.BEGIN:
                messages.append(f"SYSTEM ({event.timestamp}): Test '{event.test_id}' BEGINS")
            elif event.type == EventType.END:
                messages.append(f"SYSTEM ({event.timestamp}): Test '{event.test_id}' ENDS")
            elif event.type == EventType.SUITE_RESET:
                messages.append(f"SYSTEM ({event.timestamp}):  Suite was RESET with policy {event.data['policy']}")
            elif event.type == EventType.LLM_CALL:
                res = event.data["response"]
                messages.append(f"SYSTEM ({event.timestamp}): Test '{event.test_id}' CALLS an LLM. Response:\n{res}")
            else:
                raise ValueError("Unknown event found")

        return messages

    def find_message(self, test_id: str, message: str) -> int:
        # Find the index of the sent message
        for idx, event in enumerate(self.log):
            if event.test_id == test_id and event.type == EventType.SEND_MESSAGE and event.data["message"] == message:
                return idx
        raise ValueError(f"Message {repr(message)} for test {repr(test_id)} not found in log.")

    def load(self):
        self.log = []
        with open(self.save_file, "r") as fd:
            events_list = [json.loads(x) for x in fd.readlines()]

        for event in events_list:
            self.log.append(LogEvent.from_json(event))

    def messages(self, test_id: str = "") -> list[str]:
        messages = []
        for event in self.log:
            if event.type in [EventType.SEND_MESSAGE, EventType.RESPONSE_MESSAGE, EventType.SEND_FILL, EventType.RESPONSE_FILL]:
                if test_id == "" or event.test_id == test_id:
                    sender = "Test" if event.type == EventType.SEND_MESSAGE else "System" if event.type == EventType.SEND_FILL else "Agent"
                    messages.append(f"{sender} ({event.timestamp}): {event.data['message']}")

        return messages

    def test_events(
        self, test_id: str, event_type: EventType | set[EventType] = None, filter_fn: Callable[[LogEvent], bool] = None
    ) -> Iterator[LogEvent]:
        for event in self.log:
            if event.test_id != test_id:
                continue
            if event_type is not None:
                if isinstance(event_type, EventType):
                    if event.type != event_type:
                        continue
                elif isinstance(event_type, set):
                    if event.type not in event_type:
                        continue
            if filter_fn is not None:
                if not filter_fn(event):
                    continue
            yield event

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

        for event in self.test_events(
            test_id,
            event_type={EventType.SEND_MESSAGE, EventType.RESPONSE_MESSAGE},
            filter_fn=lambda e: e.data["is_question"],
        ):
            (questions if event.type == EventType.SEND_MESSAGE else responses).append(event.data["message"])

        return questions, responses

    def get_cached_response(self, test_id: str, llm_call_idx: int) -> str | None:
        idx = 0
        for event in self.test_events(test_id, event_type=EventType.LLM_CALL):
            if idx < llm_call_idx:
                idx += 1
                continue
            return event.data["response"]
