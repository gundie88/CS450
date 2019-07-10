from __future__ import division
from math import sqrt
import random as rnd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import json
import pandas as pd

def check_game(a, b):
    if(a == '0' and b == '1' or a == '1' and b == '2' or a == '2' and b == '0'):
        return -1
    elif a == b:
        return 0
    else:
        return 1

def append_to_csv(wins, losses, ties):
    string = (str(wins) + ", " + str(losses) + ", " + str(ties) + "\n")
    f = open("demo.csv", "a")
    f.write(string)
    f.close()

def graph_computer_win_percentage(rounds, win_percent_at_each_round, wins_losses_ties = -1):
    df = pd.DataFrame(win_percent_at_each_round)
    ax = df.plot()

    ax.set_ylim(bottom=0, top=1)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda win_percent_at_each_round, _: '{:.0%}'.format(win_percent_at_each_round)))
    ax.get_legend().remove()
    plt.plot(x=rounds)
    
    plt.ylabel("Win percentage")
    plt.xlabel("Round number")
    
#    print("\n\n    Final win percentage by the computer: {}%".format(round(win_percent_at_each_round[len(win_percent_at_each_round)-1]*100, 2)), end="")
    if(wins_losses_ties != -1):
        print("Wins: {}, Losses: {}, Ties: {}".format(wins_losses_ties[0], wins_losses_ties[1], wins_losses_ties[2]))
    
    plt.show()


def game():   
    RPS_disp  = {'0' : 'rock', '1' : 'paper', '2' : 'scissor'}
    
    ##############
    # Gets all possible hand combinations over the course of n (previous) rounds
    num_remembered_rounds = 3 # alias for n
    json_string = "{ "
    
    for i in range(len(RPS_disp)**num_remembered_rounds):
        current_array = []
        current_value = i
        if(i > 0):
            json_string += ","
        for j in range(num_remembered_rounds):
            current_array.append(current_value % len(RPS_disp))
            current_value = current_value // len(RPS_disp)
        json_string += '"'
        for k in range(num_remembered_rounds):
            json_string += str(current_array[num_remembered_rounds-1-k])
        json_string += '":' + str(len(RPS_disp))
    json_string += " }"
    RPS_count = json.loads(json_string)
    ##############

    # For the graph use after the loop breaks
    win_percent_at_each_round = [] 
    win_percent_at_each_roundv2 = [] 
    round_number = 1
    round_numberv2 = 1
    rounds = []
    roundsv2 = []

    wins_losses_ties = [0, 0, 0]
    
    wins = 0
    ties = 0
    losses = 0
    
    last2 = '33'
    #T-1, T

    # Create header for csv
    string = (str(wins) + ", " + str(losses) + ", " + str(ties))
    f = open("demo.csv", "w")
    f.write("Wins, Losses, Ties\n")
    f.close()
    
    
    #Loops until user presses q
    while(1):
        roll = input('Please choose Rock(r), Paper(p), Scissors(s) or Quit(q)\n')
        
        while(roll not in ['r', 'p', 's', 'q']):
            roll = input("Please type Rock(r), Paper(p), Scissors(s) or Quit(q)\n")
    
        if roll == 'r':
            x = '0'
        elif roll == 'p':
            x = '1'
        elif roll == 's':
            x = '2'
        elif roll == 'q':
            return wins_losses_ties, rounds, win_percent_at_each_round, roundsv2, win_percent_at_each_roundv2
            break
    
        if(last2[0] == '3'):
            y = str( rnd.randint(0,2) )
        else:
            r_count = RPS_count[last2 + '0']
            p_count = RPS_count[last2 + '1']
            s_count = RPS_count[last2 + '2']
    
            tot_count = r_count + p_count + s_count
    
            q_dist = [ r_count/tot_count, p_count/tot_count, 1- (r_count/tot_count) - (p_count/tot_count) ]
            
            result = [ max(q_dist[2]-q_dist[1],0), max(q_dist[0]-q_dist[2],0), max(q_dist[1]-q_dist[0],0) ]
            resultnorm = sqrt(result[0]*result[0] + result[1]*result[1] + result[2]*result[2])
            result = [result[0]/resultnorm, result[1]/resultnorm, 1 - result[0]/resultnorm - result[1]/resultnorm]
    
            y = rnd.uniform(0,1)
    
            if y <= result[0]:
                y = '0'
            elif y <= result[0] + result[1]:
                y = '1'
            else:
                y = '2'
    
            #update dictionary
            RPS_count[last2+x] += 1
    
        last2 = last2[1] + x
    
        print('You played: ' + RPS_disp[x] + '\nComputer played: ' + RPS_disp[y])
    
        if check_game(x,y) == -1:
            losses += 1
            wins_losses_ties[1] += 1
            print("You lost!\n")
            roundsv2.append(round_numberv2)
            round_numberv2 += 1
            win_percent_at_each_roundv2.append(losses/(wins+losses))
        elif check_game(x,y) == 0:
            ties   += 1
            wins_losses_ties[2] += 1
            print("You tied!\n")
        elif check_game(x,y) == 1:
            wins   += 1
            wins_losses_ties[0] += 1
            print("You won!\n")
            roundsv2.append(round_numberv2)
            round_numberv2 += 1
            win_percent_at_each_roundv2.append(losses/(wins+losses))
    
        print('Wins:', wins, 'Losses:', losses, 'Ties:', ties)
        
        append_to_csv(wins, losses, ties)
        win_percent_at_each_round.append(losses/(wins+losses+ties)) # Append current win percentage to this array (for graphing later)
        rounds.append(round_number)
        round_number += 1
        
        
def main():
    
    num_wins_losses_ties, rounds, win_percent_at_each_round, roundsv2, win_percent_at_each_roundv2 = game()
    
    print("\n\n    Final win percentage by the computer: {}%".format(round(
            win_percent_at_each_round[len(win_percent_at_each_round)-1]*100, 2)))
    graph_computer_win_percentage(rounds, win_percent_at_each_round, wins_losses_ties = num_wins_losses_ties)
    
    print("\n\n    Final win percentage by the computer (not accounting for ties): {}%".format(round(win_percent_at_each_roundv2[len(win_percent_at_each_roundv2)-1]*100, 2)))
    graph_computer_win_percentage(roundsv2, win_percent_at_each_roundv2)


main()
