import arc_agi

arc = arc_agi.Arcade()
games = arc.get_environments()

for game in games:
    print(f"{game.game_id}: {game.title}")
    
if games:
    game_id = games[0].game_id
    env = arc.make(game_id, render_mode="terminal")