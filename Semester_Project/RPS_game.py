from random import randint
    
def play():
    player_score = 0
    cpu_score = 0

   
    continue_playing = True
    while continue_playing:
        p_choice = input("Please choose Rock, Paper, or Scissors\n")
        p_choice = p_choice.lower()
        while p_choice not in ("rock", "paper", "scissors"):
                print("Please try again")
                p_choice = input("Please choose Rock, Paper, or Scissors\n")         
        RPS = ["rock", "paper", "scissors"]
        cpu_choice = RPS[randint(0,2)]  

        #Tie outcome
        if p_choice == cpu_choice:
            player_score += 0.5
            cpu_score += 0.5
            print(player_score, cpu_score)
            print("Tie!")           
            
        #Rock outcome
        elif p_choice == "rock":
            if cpu_choice == "paper":
                cpu_score += 1
                print("You Lose!")
            elif cpu_choice == "scissors":
                player_score += 1
                print("You Win!")                 
            
        #Paper outcome
        elif p_choice == "paper":
            if cpu_choice == "scissors":
                cpu_score += 1
                print("You Lose!")
            elif cpu_choice == "rock":
                player_score += 1
                print("You Win!")
    
        #Scissors outcome                
        elif p_choice == "scissors":
            if cpu_choice == "rock":
                cpu_score += 1
                print("You Lose!")
            elif cpu_choice == "paper":
                player_score += 1
                print("You Win!")
                
        print("Score:\nPlayer: {} \nComputer:{}" .format(player_score,cpu_score))
        play_again = input("Play again? Y/N\n")
        
        while play_again not in ("Y", "y", "N", "n"):
            print("Please try again")
            play_again = input("Play again? Y/N\n")
            
        if play_again == "N" or play_again == "n":
            print("Game Over")
            continue_playing = False
                                  
def game_start():
    
    while True:
        begin = input("Would you like to play Rock, Paper, Scissors? (Y/N)\n")
        if begin == "Y" or begin == "y":
            play()
            break
        elif begin == "N" or begin == "n":
            print("Game Over")
            break
        else:
            print("Please try again")

if __name__ == "__main__":
    game_start()
    
