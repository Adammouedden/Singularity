import arc_agi
from arcengine import FrameDataRaw#, GameAction

arc = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE)

env = arc.make("ls20", render_mode="human")

observation = env.get_observation()

print(observation)

def my_renderer(steps: int, frame_data: FrameDataRaw) -> None:
    print(f"Step {steps}: {frame_data.state.name}")
    print(f"Frame: {frame_data}")

#env = arc.make("ls20", render_mode="human")

# See available actions
#print("Action Space:", env.action_space)

# Take an action
#jobs = env.step(GameAction.ACTION1)

# Check your scorecard
#print(arc.get_scorecard())

def extract_frame():
    pass