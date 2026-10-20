from __future__ import annotations

import glob
import json
import os
import random
import re

SPLIT_DIRS = {"train": "train", "eval_in_distribution": "valid_seen", "eval_out_of_distribution": "valid_unseen"}
EXPECTED_GAMES = {"train": 3553, "eval_in_distribution": 140, "eval_out_of_distribution": 134}
BANNER = "-= Welcome to TextWorld, ALFRED! =-"
EXPERT_RANDOM_SEED = 0


def data_root() -> str:
    return os.environ["ALFWORLD_DATA"]


def game_list(split: str) -> list[str]:
    base = os.path.join(data_root(), "json_2.1.1", SPLIT_DIRS[split])
    out = []
    for path in sorted(glob.glob(os.path.join(base, "**", "game.tw-pddl"), recursive=True)):
        root = os.path.dirname(path)
        if "movable" in root or "Sliced" in root or not os.path.exists(os.path.join(root, "traj_data.json")):
            continue
        with open(path) as f:
            if not json.load(f).get("solvable"):
                continue
        out.append(path)
    if len(out) != EXPECTED_GAMES[split]:
        raise SystemExit(f"{split}: {len(out)} games, expected {EXPECTED_GAMES[split]}")
    return out


def game_id(path: str) -> str:
    return os.path.relpath(os.path.dirname(path), os.path.join(data_root(), "json_2.1.1"))


def make_env(path: str):
    import textworld
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredExpert, AlfredInfos

    infos = textworld.EnvInfos(won=True, admissible_commands=True, facts=True, extras=["gamefile", "expert_plan"])
    return textworld.start(path, infos, wrappers=[AlfredDemangler(shuffle=False), AlfredInfos, AlfredExpert])


def _view(state) -> dict:
    return {"feedback": str(state["feedback"]), "admissible": [str(c) for c in state["admissible_commands"]],
            "expert_plan": [str(c) for c in state["extra.expert_plan"]], "won": bool(state["won"])}


def state_after(path: str, history: list[str]) -> dict:
    random.seed(EXPERT_RANDOM_SEED)
    env = make_env(path)
    try:
        st = env.reset()
        first = _view(st)
        observations = []
        for action in history:
            st, _, _ = env.step(action)
            observations.append(str(st["feedback"]))
        cur = _view(st)
    finally:
        env.close()
    return {"initial": first["feedback"], "observations": observations, "admissible": cur["admissible"],
            "expert_plan": cur["expert_plan"], "won": cur["won"]}


def determinism(path: str, history: list[str]) -> bool:
    a, b = state_after(path, history), state_after(path, history)
    return a == b


PROMPT_TEMPLATE = (
    "You are an agent in a household. You interact with a text-based environment by issuing exactly one command "
    "per turn, and you must complete the task described below.\n\n"
    "{initial}\n\n"
    "Previous turns (your command, then what you observed):\n{transcript}\n\n"
    "Current admissible commands:\n{admissible}\n\n"
    "Choose your next command. After thinking, reply with exactly one line of the form:\n"
    "Action: <command>"
)


def initial_text(feedback: str) -> str:
    text = feedback.replace(BANNER, "").strip()
    return text


def build_user_turn(state: dict, history: list[str]) -> str:
    if history:
        lines = []
        for action, obs in zip(history, state["observations"]):
            lines.append(f"> {action}\n{obs.strip()}")
        transcript = "\n".join(lines)
    else:
        transcript = "(none)"
    return PROMPT_TEMPLATE.format(initial=initial_text(state["initial"]), transcript=transcript,
                                  admissible="\n".join(state["admissible"]))


def prompt_ids(tokenizer, user_turn: str) -> list[int]:
    from rl_controller import olmo3_think_contract as o

    o.assert_tokenizer(tokenizer)
    return list(tokenizer.apply_chat_template([{"role": "user", "content": user_turn}], tokenize=True,
                                              add_generation_prompt=True))


CLOSE = "</think>"
FORCED_SUFFIX = "</think>\n\nAction: "
ACTION_RE = re.compile(r"^\s*\**\s*action\s*\**\s*:\s*\**\s*(.*?)\s*$", re.IGNORECASE)


def normalise(cmd: str) -> str:
    cmd = re.sub(r"\s+", " ", cmd.strip().strip("`*\"'").strip()).lower()
    return cmd.rstrip(".").strip()


def parse_forced(text: str) -> str:
    return normalise(text.split("\n", 1)[0])


def parse_natural(text_after_close: str) -> tuple[str, bool]:
    lines = text_after_close.split("\n")
    for line in lines:
        m = ACTION_RE.match(line)
        if m:
            return normalise(m.group(1)), True
    for line in lines:
        if line.strip():
            return normalise(line), False
    return "", False
