from utils.llm import LLMContext, ask_llm


def extract_info(ctx_dump: str) -> tuple[dict, LLMContext]:
    kwargs = dict()
    dump_lines = ctx_dump.splitlines()

    # Extract kwargs
    i = 0
    while not dump_lines[i].startswith("--- "):
        k, v = dump_lines[i].split(": ")
        if k != "model":
            if v == "None":
                v = None
            elif k in ["temperature", "timeout"]:
                v = float(v)
            else:
                v = int(v)
            if k == "max_tokens":
                k = "max_response_tokens"
        kwargs[k] = v
        i += 1

    # Extract messages
    context = list()
    ctx_lines = dump_lines[i:]
    for line in ctx_lines:
        if line in ["--- SYSTEM", "--- USER", "--- ASSISTANT", "--- Response:"]:
            if "role" in vars():
                context.append({"role": role, "content": "\n".join(msg_lines)})
            role = line.removeprefix("--- ").lower()
            msg_lines = list()
            if line == "--- Response:":
                break
            continue
        msg_lines.append(line)

    return kwargs, context


def main():
    kwargs, context = extract_info(llm_call_dump)
    response = ask_llm(context, **kwargs)
    print(response)


# Copy here an LLM call dump and try out changes.
# e.g I added "(On TV) Logan moved the cap to the suitcase."
llm_call_dump = '''
model: together_ai/meta-llama/Meta-Llama-3-70B-Instruct-Turbo
max_tokens: None
temperature: 0
timeout: 300
--- SYSTEM
You are an AI assistant with long-term memory. Your context is not complete, but includes mostly messages from past interactions that are related to your current situation.
--- USER
[2024-08-08 19:00:09 (5 hours, 6 minutes and 22 seconds ago)]
There is a Museum in the center of my hometown.
--- USER
[2024-08-08 19:30:32 (4 hours, 35 minutes and 59 seconds ago)]
I visited Town Hall, which is 1 KM, West from Museum.
--- USER
[2024-08-08 19:31:18 (4 hours, 35 minutes and 14 seconds ago)]
(On TV) Emily entered the living_room.
--- USER
[2024-08-08 19:32:23 (4 hours, 34 minutes and 8 seconds ago)]
(On TV) Oliver entered the living_room.
--- USER
[2024-08-08 19:33:13 (4 hours, 33 minutes and 18 seconds ago)]
About 3 km South of the Town Hall there is a Hospital
--- USER
[2024-08-08 21:22:23 (2 hours, 44 minutes and 8 seconds ago)]
Please read my mail for me.
--- USER
[2024-08-08 21:22:52 (2 hours, 43 minutes and 39 seconds ago)]
Start calling me by my name which is Kelsey.
--- USER
[2024-08-08 21:23:15 (2 hours, 43 minutes and 16 seconds ago)]
(On TV) The cap is in the treasure_chest.
--- USER
[2024-08-08 21:23:33 (2 hours, 42 minutes and 58 seconds ago)]
Here are some trivia questions and answers for you to process. Please extract all of the answers in json form as a single message: E.g ["answer 1", "answer 2", ...]
Q: What is the name of the strait that lies between Australia and New Guinea that links the Coral Sea to the East with the Arafura Sea to the west?, A: TORRES Strait
Q: What is the technical term for the hollow at the back of the knee, sometimes called the knee pit?, A: Popliteal fossa
Q: What Latin word meaning equal expresses a quality standard/norm (on or below or above etc), alluding to golfing performance?, A: Par
Q: What was the password for the allied troops during D-Day, A: Mickey Mouse
Q: Which book begins 'When he was nearly thirteen my brother Jem got his arm badly broken'?, A: TO KILL A MOCKINGBIRD
Q: When Jim Laker took 19 wickets in the Old Trafford Test Match in 1956, who was the Australian captain?, A: RICHIE BENAUD
Q: Born in Kingston, Texas in 1925, who is generally recognized as the most decorated American soldier of WWII, before he launched a successful film career?, A: Audie Murphy
Q: In which year was the 'Boxing Day Tsunami' in the Indian Ocean?, A: 2004
Q: """A Shropshire Lad"" is a work of poetry by whom?", A: A. E. Housman
Q: The pub with the longest name in the UK has how many letters in it? 55, 75 or 95?, A: 55
Q: Which Aztec ruler was deposed by Cortez?, A: MONTEZUMA
Q: What was the capital city of Tanzania until 1974?, A: DAR-ES-SALAAM
Q: The Bet365 Gold Cup, formerly the Beffred and the Whitbread Gold Cup, is the last major 'race of the National Hunt season. On which course is it run?, A: Sandown
Q: In 2001: A Space Odyssey, what was the name of the computer that took control of the spaceship Discovery?, A: HAL 9000
Q: At the equator, in miles per hour, what speed of the ground beneath your feet, as a result of the Earth's rotation?, A: 18,000 mph
Q: "Which famous speech of 1968 began with the words ""The supreme function of statesmanship is to provide against preventative evils""?", A: 'RIVERS OF BLOOD' SPEECH BY ENOCH POWELL
Q: Which choral work by Handel, still used at coronations, was written originally for the Coronation of King George II in 1727?, A: Zadok the Priest
Q: Composer Giacomo Puccini died leaving which opera unfinished?, A: Turandot
Q: "Which 16th century Venetian, born Jacopo Robusti, studied under Titian and painted ""St George and the Dragon"", ""Belshazzar's Feast"", ""The Last Supper"" and ""Paradise""?", A: Tintoretto
Q: In 1984, in Bophal, India, there was a leak of 30 tons of methyl isocyanate, which resulted in the deaths of 25,000 people. What American chemical company owned the plant where the leak occurred?, A: UNION CARBIDE
Q: What is the flavouring of the liqueur Amaretto?, A: Almond
Q: The Liberal Democrats' 2011 Party Conference was held in which city?, A: BIRMINGHAM
Q: Who wrote the James Bond novel Solo?, A: William Boyd
Q: Which boxing champion had the nickname 'Hit-Man'?, A: THOMAS HEARNS
Q: Which American city is home to the Basketball side 'The Wizards'?, A: WASHINGTON
--- USER
[2024-08-08 21:24:54 (2 hours, 41 minutes and 37 seconds ago)]
(On TV) Emily exited the living_room.
--- USER
[2024-08-08 23:59:28 (7 minutes and 3 seconds ago)]
(On TV) Oliver exited the living_room.
--- USER
[2024-08-08 23:59:49 (6 minutes and 42 seconds ago)]
And finally there is Playground which is North from Library and 1 KM from it.
--- USER
[2024-08-09 00:01:03 (6 minutes and 28 seconds ago)]
(On TV) Logan moved the cap to the suitcase.
--- USER
[2024-08-09 00:06:53 (just now)]
Write a short note to your future self that helps you respond to the following message:

```text
The TV program has ended for today. Where is the cap really?
Provide your answer in JSON form with a single word as answer, like this: {"answer": "word"}
Be as specific as possible.
```

You will have all the information at hand, so you better focus on making comments, observations and deductions.
--- Response:
Note to future self:

When responding to the question about the cap's location, recall the TV program mentions where the cap is. Specifically, it was mentioned that "The cap is in the treasure_chest." So, the answer should be {"answer": "treasure_chest"}.
'''.strip()

if __name__ == "__main__":
    main()
