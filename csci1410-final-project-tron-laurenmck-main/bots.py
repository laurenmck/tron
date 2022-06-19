
#!/usr/bin/python

from io import BufferedReader
import numpy as np
from tronproblem import *
from trontypes import CellType, PowerupType
import random, math
from collections import deque

# Throughout this file, ASP means adversarial search problem.

#if no path between the players 

class StudentBot:
    """ Write your student bot here"""
    turn = 0 
    def __init__(self):
            order = ["U", "D", "L", "R"]
            random.shuffle(order)
            self.order = order
    def decide(self, asp):
        """
        Input: asp, a TronProblem
        Output: A direction in {'U','D','L','R'}

        To get started, you can get the current
        state by calling asp.get_start_state()
        """
        endgame_check = 0
        while self.turn < 3:
            self.turn += 1
            state = asp.get_start_state()
            locs = state.player_locs
            board = state.board
            ptm = state.ptm
            loc = locs[ptm]
            possibilities = list(TronProblem.get_safe_actions(board,loc))
            if not possibilities: 
                return "U"
            decision = possibilities[0]
            for move in self.order:
                if move not in possibilities:
                    continue
                next_loc = TronProblem.move(loc, move)
                if len(TronProblem.get_safe_actions(board, next_loc)) < 4:
                    decision = move
                    break
            return decision

        state = asp.get_start_state()
        locs = state.player_locs
        board = state.board
        ptm = state.ptm
        #locs = locs[ptm]
        if endgame_check == 0:
            if self.endgame_check(ptm, locs, board):
                #print("entered")
                endgame_check = 1
            else:

        #7 for all maps but 
                direction = self.alpha_beta_cutoff(asp, state, ptm, 6, self.heuristic_func)
        #print("hello")
        if endgame_check == 1:
            direction = self.endgame_bot(state, locs, board, ptm)
        return direction


    def alpha_beta_cutoff(self, asp, state, player, cutoff_ply, heuristic_func):
        
        depth = 0
        #x is the move which should be made 
        v, x = self.Max_Value_AB_Cutoff(asp, state, player, -math.inf, math.inf, cutoff_ply, depth, heuristic_func)
        
        return x


    def Max_Value_AB_Cutoff(self, asp, state, player, alpha, beta, cutoff, depth, heuristic_func): 
        
        #Set random value for move for final return 
        move = 'D'
       # print("max_value ")
        #Evaluate terminal if terminal state
        if asp.is_terminal_state(state):
          #  print("terminal")
            cost_tuple = asp.evaluate_state(state)
            num_to_return = cost_tuple[player]
          #  print(num_to_return)
            return num_to_return, None
    
        #Evaluate using heuristic if depth has reached cutoff
        if depth == cutoff:
           # print("heuristic")
            board = state.board
            locs = state.player_locs
            num_to_return = self.voronoi(player, locs, board)
            #num_to_return = heuristic_func(board, loc)
          #  print(num_to_return)
            return num_to_return, None
    
        #Increment the depth 
        depth = depth + 1

        v = -math.inf

        board = state.board
        loc = state.player_locs
        player_loc = loc[state.ptm]
        
        #Get the safe actions to loop through
        available_actions = TronProblem.get_safe_actions(board, player_loc)

        #Loop through the available actions
        for x in available_actions: 
            #Transition to new state
            curr_state = asp.transition(state, x)

            #Call AB cutoff min value on new transitioned state
            v_2, a = self.Min_Value_AB_Cutoff(asp, curr_state, player, alpha, beta, cutoff, depth, heuristic_func)
    
            if v_2 > v:
                v = v_2
                move = x
                alpha = max(alpha, v)

            if v >= beta:
          
                return v, move
    
        return v, move

    def Min_Value_AB_Cutoff(self, asp, state, player, alpha, beta, cutoff, depth, heuristic_func): 
        #Set move
        move = 'D'
      #  print("min_value_called")
       #Evaluate if terminal state
        if asp.is_terminal_state(state):
        #    print("terminal")
            cost_tuple = asp.evaluate_state(state)
            num_to_return = cost_tuple[player]
            return num_to_return, None

        #Evaluate if reached cutoff
        if depth == cutoff:
         #   print("heuristic")
            board = state.board
            locs = state.player_locs
            num_to_return = self.voronoi(player, locs, board)
            #num_to_return = heuristic_func(board, loc)
            return num_to_return, None
    
        depth = depth + 1
        v = math.inf
        board = state.board
        loc = state.player_locs
        player_loc = loc[state.ptm]
        #available_actions = asp.get_available_actions(state)
        available_actions = TronProblem.get_safe_actions(board, player_loc)
       
       #Loop through available actions
        for x in available_actions: 
            #Transition to new state
            curr_state = asp.transition(state, x)

            #Call max value cutoff 
            v_2, a = self.Max_Value_AB_Cutoff(asp, curr_state, player, alpha, beta, cutoff, depth, heuristic_func)

            if v_2 < v:
                v = v_2
                move = x
                beta = min(beta, v)

            if v <= alpha:
                return v, move
                
        return v, move    


    def heuristic_func(self, board, start):
        return 0
        
    def voronoi(self, player, locs, board):
           # print("voronoi")
           # print(player)
            playerloc = locs[player]
           # print(playerloc)
            if player == 0: 
                opposingplayerloc = locs[1]
            else: 
                opposingplayerloc = locs[0]

            playerWins = self.djikstra_attempt(board, playerloc)
            #print(playerWins)
            #print("hello")
            opponentWins = self.djikstra_attempt(board, opposingplayerloc)
           # print(opponentWins)
            #print(board)
            #print(playerWins)
            #print(opponentWins)
            playercount = 0
            opponentcount = 0

            for r in range(len(board)):
                for c in range(len(board[0])):
                    if playerWins[r, c] < opponentWins[r, c]:
                        playercount +=1
                    if opponentWins[r][c] < playerWins[r][c]:
                        opponentcount +=1 
           # print("total player count")
           # print(playercount)
           # print("opponent count")
          #  print(opponentcount)

            a = -len(board)* len(board[0])
            b = len(board)* len(board[0])
            #outcome = (playercount - opponentcount)/(len(board)*len(board[0]))
            outcome = ((playercount - opponentcount) - a)/(b-a)
          #  print(outcome)
            #print("voroni", outcome, locs[player])
            return outcome

    def djikstra_attempt(self, board, start):
        #manhatan distance - move that generated most possible moves 
        #start = loc (r, c)
        #muta
        start_r, start_c = start
        #holds distances from loc to each open square - initialized as inf
        dists = np.full((len(board), len(board[0])), np.inf)
        visited = np.zeros((len(board), len(board[0])))
        dists[start_r,start_c] = 0.0
       # print("distance setup")
       # print(dists)
    
        #safe neighbors from starting point
        safeneighbors = TronProblem.get_safe_actions(board, start)

        moves = []
        for neighbor in safeneighbors: 
                #neighbor row, neighbor col 
            moves.append(TronProblem.move(start, neighbor))
            neighborr, neighborc = TronProblem.move(start, neighbor) 
            dists[neighborr, neighborc] = 1
        queue = deque(moves)

        while len(queue) != 0:
            #print(queue)
            currentr, currentc = queue.popleft()
            newdistance = dists[currentr, currentc] + 1
            safeneighbors = TronProblem.get_safe_actions(board, (currentr, currentc))
            #print((currentr, currentc), safeneighbors)
            for neighbor in safeneighbors: 
                move = TronProblem.move((currentr, currentc), neighbor) 
                newrow, newcol = move
                if(newdistance < dists[newrow, newcol]):
                    dists[newrow, newcol] = newdistance
                if visited[newrow, newcol] == 0:
                    queue.append(move)
                    visited[newrow, newcol] = 1
        return dists

    def endgame_check(self, player, locs, board):
        playerloc = locs[player]
        #print(playerloc)
        #print("player location")
        if player == 0: 
            opposingplayerloc = locs[1]
        else: 
            opposingplayerloc = locs[0]
        playerWins = self.djikstra_attempt(board, playerloc)
        #print(playerloc)
        #print(opposingplayerloc)
        #print(playerWins)
        x, y = opposingplayerloc
        #print(playerWins[x, y])

        if playerWins[x+1, y] !=math.inf:
            return False
        if playerWins[x-1, y] !=math.inf:
            return False
        if playerWins[x, y+1] !=math.inf:
            return False
        if playerWins[x, y-1] !=math.inf:
            return False
        if playerWins[x+1, y-1] !=math.inf:
            return False
        if playerWins[x+1, y+1] !=math.inf:
            return False
        if playerWins[x-1, y-1] !=math.inf:
            return False
        if playerWins[x-1, y+1] !=math.inf:
            return False
        else:
            return True

    def endgame_bot(self, state, locs, board, ptm):
        #print("Using endgame bot")
        playerloc = locs[ptm]

        num_moves = -math.inf
        move = "U"
        for action in TronProblem.get_safe_actions(board, playerloc):
            new_player_loc = TronProblem.move(playerloc, action) 
            dists = self.djikstra_attempt(board, new_player_loc)
            counter = 0
            for i in range(len(board)):
                for j in range(len(board[0])):
                    if dists[i][j] != math.inf:
                        counter = counter + 1
            if counter > num_moves:
                num_moves = counter
                move = action


        return move
        
    def cleanup(self):
        """opposingplayerloc
        Input: None
        Output: None

        This function will be called in between
        games during grading. You can use it
        to reset any variables your bot uses during the game
        (for example, you could use this function to reset a
        turns_elapsed counter to zero). If you don't need it,
        feel free to leave it as "pass"
        """
        
        pass
class RandBot:
    """Moves in a random (safe) direction"""

    def decide(self, asp):
        """
        Input: asp, a TronProblem
        Output: A direction in {'U','D','L','R'}
        """
        state = asp.get_start_state()
        locs = state.player_locs
        board = state.board
        ptm = state.ptm
        loc = locs[ptm]
        possibilities = list(TronProblem.get_safe_actions(board, loc))
        if possibilities:
            return random.choice(possibilities)
        return "U"
    
    def cleanup(self):
        pass


class WallBot:
    """Hugs the wall"""

    def __init__(self):
        order = ["U", "D", "L", "R"]
        random.shuffle(order)
        self.order = order

    def cleanup(self):
        order = ["U", "D", "L", "R"]
        random.shuffle(order)
        self.order = order

    def decide(self, asp):
        """
        Input: asp, a TronProblem
        Output: A direction in {'U','D','L','R'}
        """
        state = asp.get_start_state()
        locs = state.player_locs
        board = state.board
        ptm = state.ptm
        loc = locs[ptm]
        possibilities = list(TronProblem.get_safe_actions(board, loc))
        if not possibilities:
            return "U"
        decision = possibilities[0]
        for move in self.order:
            if move not in possibilities:
                continue
            next_loc = TronProblem.move(loc, move)
            if len(TronProblem.get_safe_actions(board, next_loc)) < 3:
                decision = move
                break
        return decision
