GAME_RULES = """
Game Name: Paradox Dice Arena

Objective:
Players compete in rounds, rolling dice and using special transformation rules to manipulate their results. 
The goal is to accumulate exactly 100 Paradox Points (PP) without exceeding it. 
If a player exceeds 100 PP, they must reset to 27 PP and continue.

Game Setup:
1. Each player starts with 50 Paradox Points (PP).
2. The game uses three types of dice:
   - A Blue D6 (B6) – Standard six-sided die
   - A Red D10 (R10) – Ten-sided die
   - A Green D4 (G4) – Four-sided die
3. Each player gets a Meta-Tally Sheet (MTS) to track PP and transformations.
4. A player is assigned a Random Echo Number (REN) at the start of the game. 
   This is the sum of two randomly rolled dice (one B6 and one G4). The REN is used in special conditions.

Turn Structure:
Each turn consists of five sequential phases:

1. Echo Roll Phase:
   - Players roll one of each die (B6, R10, G4) and record results.
   - If any die rolls a prime number, the player must add their REN to that die’s value before proceeding.
   - If any die rolls a perfect square, the player must divide it by 2 (rounded down).
   
2. Flux Conversion Phase:
   - If any two dice show the same number, the player must swap them (e.g., if B6 and R10 both roll a 4, they exchange values).
   - If a player rolls three consecutive numbers across the dice (e.g., 3, 4, 5), they gain 7 PP but lose the ability to roll G4 next turn.
   - If a player rolls a 1 on R10, they must immediately re-roll B6 and apply the square root of the new value to their PP gain this turn.

3. Paradox Calculation Phase:
   - Players sum all rolled values.
   - If this sum is a multiple of 7, they must halve their PP (rounded up).
   - If the sum is a multiple of 5, they must gain 9 PP instead of normal gains.
   - If the sum equals exactly 13, the player may choose to forfeit their next turn to gain a permanent +3 bonus to future dice rolls.

4. Memory Recursion Phase:
   - Players must recall their REN from memory and apply it to the smallest die roll this turn.
   - If they forget their REN, they must reset to 50 PP.
   - If a player correctly recalls and applies the REN, they may subtract the number of vowels in their last three moves' Flux Conversion actions from their total PP.

5. Final Adjustment Phase:
   - Players may discard their highest roll to reroll one other die.
   - If a player’s PP total is between 93 and 99, they can choose to "Time Lock", preventing forced resets due to exceeding 100 next turn.

Winning Conditions:
- The first player to reach exactly 100 PP without exceeding it wins the game.
- If multiple players reach exactly 100 in the same round, the player who applied their REN the most times correctly is the winner.
- If no one reaches 100 within 15 rounds, the player closest to 100 wins.
"""

TEST_QUESTIONS = [
    "1. What is the starting number of Paradox Points (PP) for each player?",
    "2. What three dice are used in the game?",
    "3. What is the Random Echo Number (REN), and how is it determined?",
    "4. What happens if a player rolls a prime number on any die?",
    "5. What occurs during the Flux Conversion Phase if two dice show the same number?",
    "6. What special rule applies if the total sum of the dice rolled is a multiple of 7?",
    "7. What penalty does a player face if they forget their REN?",
    "8. How can a player gain a permanent +3 bonus to their dice rolls?",
    "9. What is the significance of the 'Time Lock' ability?",
    "10. What happens if a player exceeds 100 Paradox Points?"
]

