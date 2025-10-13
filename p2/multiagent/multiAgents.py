# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        if successorGameState.isWin():
            return float('inf')
        if successorGameState.isLose():
            return float('-inf')

        score = 0.0
        if action == Directions.STOP:
            score -= 10.0 #es imito ro ar gacherdes

        food_list = newFood.asList()
        if len(food_list):
            dfood = min([manhattanDistance(newPos, f) for f in food_list])
            if dfood != 0:
                score += 1.0 / dfood
           

        ghost_pos = [g.getPosition() for g in newGhostStates]
        ns = [t for t in newScaredTimes]
        if len(ghost_pos):
            close = []
            for i in range(len(ghost_pos)):
                if ns[i] == 0:
                    close.append(manhattanDistance(newPos, ghost_pos[i]))
            if len(close):
                mg = min(close)
                if mg == 0:
                    return float('-inf')
                
        
        return successorGameState.getScore() + score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        def val(st, d, a):
            if d == self.depth or st.isWin() or st.isLose():
                return self.evaluationFunction(st)
            if a == 0:
                return mx(st, d, a)
            return mn(st, d, a)

        def mx(st, d, a):
            best = float('-inf')
            acts = st.getLegalActions(0)
            if not acts:
                return self.evaluationFunction(st)
            for ac in acts:
                ns = st.generateSuccessor(0, ac)
                v = val(ns, d, 1)
                if v > best:
                    best = v
            return best

        def mn(st, d, a):
            best = float('inf')
            acts = st.getLegalActions(a)
            if not acts:
                return self.evaluationFunction(st)
            na = a + 1
            nd = d
            if na == st.getNumAgents():
                na = 0
                nd = d + 1
            for ac in acts:
                ns = st.generateSuccessor(a, ac)
                v = val(ns, nd, na)
                if v < best:
                    best = v
            return best

        best_val = float('-inf')
        best_act = Directions.STOP
        for ac in gameState.getLegalActions(0):
            ns = gameState.generateSuccessor(0, ac)
            v = val(ns, 0, 1)
            if v > best_val:
                best_val = v
                best_act = ac
        return best_act

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        def val(st, d, a, al, be):
            if d == self.depth or st.isWin() or st.isLose():
                return self.evaluationFunction(st)
            if a == 0:
                return mx(st, d, al, be)
            return mn(st, d, a, al, be)

        def mx(st, d, al, be):
            v = float('-inf')
            acts = st.getLegalActions(0)
            if not acts:
                return self.evaluationFunction(st)
            for ac in acts:
                ns = st.generateSuccessor(0, ac)
                v = max(v, val(ns, d, 1, al, be))
                if v > be:
                    return v
                if v > al:
                    al = v
            return v

        def mn(st, d, a, al, be):
            v = float('inf')
            acts = st.getLegalActions(a)
            if not acts:
                return self.evaluationFunction(st)
            na = a + 1
            nd = d
            if na == st.getNumAgents():
                na = 0
                nd = d + 1
            for ac in acts:
                ns = st.generateSuccessor(a, ac)
                v = min(v, val(ns, nd, na, al, be))
                if v < al:
                    return v
                if v < be:
                    be = v
            return v

        al = float('-inf')
        be = float('inf')
        best = float('-inf')
        act = Directions.STOP
        for ac in gameState.getLegalActions(0):
            ns = gameState.generateSuccessor(0, ac)
            v = val(ns, 0, 1, al, be)
            if v > best:
                best = v
                act = ac
            if best > al:
                al = best
        return act

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        def val(st, d, a):
            if d == self.depth or st.isWin() or st.isLose():
                return self.evaluationFunction(st)
            if a == 0:
                return mx(st, d)
            return ex(st, d, a)

        def mx(st, d):
            best = float('-inf')
            acts = st.getLegalActions(0)
            if not acts:
                return self.evaluationFunction(st)
            for ac in acts:
                ns = st.generateSuccessor(0, ac)
                v = val(ns, d, 1)
                if v > best:
                    best = v
            return best

        def ex(st, d, a):
            acts = st.getLegalActions(a)
            if not acts:
                return self.evaluationFunction(st)
            na = a + 1
            nd = d
            if na == st.getNumAgents():
                na = 0
                nd = d + 1
            p = 1.0 / float(len(acts))
            s = 0.0
            for ac in acts:
                ns = st.generateSuccessor(a, ac)
                s += p * val(ns, nd, na)
            return s

        best_val = float('-inf')
        best_act = Directions.STOP
        for ac in gameState.getLegalActions(0):
            ns = gameState.generateSuccessor(0, ac)
            v = val(ns, 0, 1)
            if v > best_val:
                best_val = v
                best_act = ac
        return best_act

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    if currentGameState.isWin():
        return float('inf')
    if currentGameState.isLose():
        return float('-inf')

    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    food_list = food.asList()
    ghosts = currentGameState.getGhostStates()
    scared = [g.scaredTimer for g in ghosts]
    caps = currentGameState.getCapsules()

    v = 0.0
    v += 1.0 * currentGameState.getScore()

    if len(food_list):
        dmin = min([manhattanDistance(pos, f) for f in food_list])
        if dmin != 0:
            v += 2.0 / dmin
        

    if len(caps):
        dcap = min([manhattanDistance(pos, c) for c in caps])
        if dcap != 0:
            v += 1.0 / dcap
        v -= 2.0 * len(caps) 

    gpos = [g.getPosition() for g in ghosts]
    if len(gpos):
        near = []
        for i in range(len(gpos)):
            d = manhattanDistance(pos, gpos[i])
            if scared[i] == 0:
                near.append(d)
        if len(near):
            md = min(near)
            if md == 0:
                return float('-inf')
            if md <= 1:
                v -= 2.0
            else:
                v -= 0.5 / md

    return v

# Abbreviation
better = betterEvaluationFunction
