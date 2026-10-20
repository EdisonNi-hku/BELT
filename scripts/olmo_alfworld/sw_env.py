from __future__ import annotations

import re

SPLIT_DIRS = {"test": "test", "dev": "dev"}
VARIATIONS_PER_TASK = 3
EXPECTED_GAMES = {"test": 90, "dev": 89}
SIMPLIFICATION = "easy"
STEP_LIMIT = 100_000
BLOCKED = {"reset task": "This command is not available in this episode."}
_ENV = None


JAVA_OPTS = ["-XX:+UnlockExperimentalVMOptions", "-XX:hashCode=2", "-XX:-UsePerfData", "-Xlog:disable"]


def _env():
    global _ENV
    if _ENV is None:
        import scienceworld.scienceworld as SWM
        from scienceworld import ScienceWorldEnv

        launch = SWM.launch_gateway

        def launch_with_opts(*a, **k):
            k["javaopts"] = list(k.get("javaopts") or []) + JAVA_OPTS
            return launch(*a, **k)
        SWM.launch_gateway = launch_with_opts
        try:
            for attempt in range(5):
                try:
                    _ENV = ScienceWorldEnv("", envStepLimit=STEP_LIMIT)
                    break
                except ValueError:
                    if attempt == 4:
                        raise
                    import time

                    time.sleep(1 + attempt)
        finally:
            SWM.launch_gateway = launch
    return _ENV


def game_list(split: str) -> list[str]:
    env = _env()
    out = []
    for task in sorted(env.get_task_names()):
        env.load(task, 0, SIMPLIFICATION)
        vs = sorted(env.get_variations_test() if SPLIT_DIRS[split] == "test" else env.get_variations_dev())
        if len(vs) < VARIATIONS_PER_TASK and split == "test":
            raise SystemExit(f"{task}: only {len(vs)} {split} variations")
        out += [f"{task}::{v}" for v in vs[:VARIATIONS_PER_TASK]]
    if len(out) != EXPECTED_GAMES[split]:
        raise SystemExit(f"{split}: {len(out)} games, expected {EXPECTED_GAMES[split]}")
    return out


def game_id(game: str) -> str:
    return game


DOORS = re.compile(r"^(\s*You also see: )(A door to .*)$")
AMBIGUOUS = "Ambiguous request"
OPTION = re.compile(r"^(\d+):\s*(.+?)\s*$", re.M)


def canon(text: str) -> str:
    out, run = [], []
    for line in str(text).split("\n"):
        if line.startswith("\t"):
            run.append(line)
            continue
        if run:
            out += sorted(run)
            run = []
        m = DOORS.match(line)
        if m:
            body, end = (m.group(2)[:-1], ".") if m.group(2).endswith(".") else (m.group(2), "")
            items = body.split(", ")
            if all(x.startswith("A door to ") for x in items):
                line = m.group(1) + ", ".join(sorted(items)) + end
        out.append(line)
    out += sorted(run)
    return "\n".join(out)


def _step(env, action: str):
    obs, _, done, info = env.step(action)
    obs = str(obs)
    if obs.startswith(AMBIGUOUS) and not done:
        opts = OPTION.findall(obs)
        if opts:
            k, txt = min(opts, key=lambda kv: kv[1])
            obs2, _, done, info = env.step(k)
            obs = f"(Ambiguous command; the simulator took: {txt})\n{obs2}"
    return canon(obs), done, info


def state_after(game: str, history: list[str]) -> dict:
    env = _env()
    task, var = game.rsplit("::", 1)
    env.load(task, int(var), SIMPLIFICATION)
    obs, info = env.reset()
    score, done = float(info["score"]), False
    observations = []
    for action in history:
        if action in BLOCKED:
            observations.append(BLOCKED[action])
            continue
        obs_a, done, info = _step(env, action)
        observations.append(obs_a)
        score = float(info["score"])
        if done:
            break
    return {"initial": canon(obs), "task": str(env.get_task_description()), "observations": observations,
            "templates": [t for t in env.get_possible_actions() if t not in BLOCKED],
            "objects": sorted(str(o) for o in env.get_possible_objects()),
            "admissible": None, "won": score >= 100.0, "done": bool(done), "score": max(0.0, score),
            "raw_score": score, "simplification": SIMPLIFICATION}


def determinism(game: str, history: list[str]) -> bool:
    return state_after(game, history) == state_after(game, history)


PROMPT_TEMPLATE = (
    "You are an agent in a simulated world with rooms, objects, living things and scientific instruments. You interact "
    "with a text-based environment by issuing exactly one command per turn, and you must complete the task described "
    "below.\n\n"
    "Task: {task}\n\n"
    "Initial observation:\n{initial}\n\n"
    "Previous turns (your command, then what you observed):\n{transcript}\n\n"
    "Action templates (replace OBJ with the name of an object):\n{templates}\n\n"
    "Objects you can currently see or use:\n{objects}\n\n"
    "Choose your next command. After thinking, reply with exactly one line of the form:\n"
    "Action: <command>"
)


def build_user_turn(state: dict, history: list[str]) -> str:
    if history:
        transcript = "\n".join(f"> {a}\n{o.strip()}" for a, o in zip(history, state["observations"]))
    else:
        transcript = "(none)"
    return PROMPT_TEMPLATE.format(task=state["task"].strip(), initial=state["initial"].strip(), transcript=transcript,
                                  templates="\n".join(state["templates"]), objects=", ".join(state["objects"]))


def prompt_ids(tokenizer, user_turn: str) -> list[int]:
    from alf_env import prompt_ids as _p

    return _p(tokenizer, user_turn)


from alf_env import ACTION_RE, CLOSE, FORCED_SUFFIX, normalise, parse_forced, parse_natural
