import multiprocessing
import time
import os
import random
import numpy as np
import math
from random import choice
import statistics

# --- Sudoku Puzzle Definition ---
# I've cleaned up the string to make it easier to parse.

'''
Brother We Cooked

He do it in 11 sec
---
090600000
000831000
160000200
070090100
000057900
006000003
038005000
600000704
000000010
---

Another one
That tuf one he did it on 52 sec
---
900020010
006800070
000000000
000000503
009005001
004100080
300500000
700600090
200008730
---
'''

startingSudoku = """
900020010
006800070
000000000
000000503
009005001
004100080
300500000
700600090
200008730
"""

# --- Core Sudoku Logic (from your original code) ---
# These functions are used by each individual solver process.

def PrintSudoku(sudoku):
    """Prints the Sudoku board in a readable format."""
    print("\n")
    for i in range(len(sudoku)):
        line = ""
        if i == 3 or i == 6:
            print("---------------------")
        for j in range(len(sudoku[i])):
            if j == 3 or j == 6:
                line += "| "
            line += str(sudoku[i, j]) + " "
        print(line)

def FixSudokuValues(fixed_sudoku):
    """Creates a mask of fixed (non-zero) cells."""
    for i in range(0, 9):
        for j in range(0, 9):
            if fixed_sudoku[i, j] != 0:
                fixed_sudoku[i, j] = 1
    return fixed_sudoku

def CalculateNumberOfErrors(sudoku):
    """Calculates the total number of errors (cost function)."""
    numberOfErrors = 0
    for i in range(0, 9):
        numberOfErrors += CalculateNumberOfErrorsRowColumn(i, i, sudoku)
    return numberOfErrors

def CalculateNumberOfErrorsRowColumn(row, column, sudoku):
    """Calculates errors in a specific row and column."""
    numberOfErrors = (9 - len(np.unique(sudoku[:, column]))) + (9 - len(np.unique(sudoku[row, :])))
    return numberOfErrors

def CreateList3x3Blocks():
    """Generates a list of cell coordinates for each 3x3 block."""
    finalListOfBlocks = []
    for r in range(0, 9):
        tmpList = []
        block1 = [i + 3 * ((r) % 3) for i in range(0, 3)]
        block2 = [i + 3 * math.trunc((r) / 3) for i in range(0, 3)]
        for x in block1:
            for y in block2:
                tmpList.append([x, y])
        finalListOfBlocks.append(tmpList)
    return finalListOfBlocks

def RandomlyFill3x3Blocks(sudoku, listOfBlocks):
    """Fills the empty cells of the Sudoku with random valid numbers within each 3x3 block."""
    for block in listOfBlocks:
        for box in block:
            if sudoku[box[0], box[1]] == 0:
                currentBlock = sudoku[block[0][0]:(block[-1][0] + 1), block[0][1]:(block[-1][1] + 1)]
                sudoku[box[0], box[1]] = choice([i for i in range(1, 10) if i not in currentBlock])
    return sudoku

def SumOfOneBlock(sudoku, oneBlock):
    """Calculates the sum of values in a single 3x3 block."""
    finalSum = 0
    for box in oneBlock:
        finalSum += sudoku[box[0], box[1]]
    return finalSum

def TwoRandomBoxesWithinBlock(fixedSudoku, block):
    """Selects two random, non-fixed boxes within a given block."""
    while True:
        firstBox = random.choice(block)
        secondBox = choice([box for box in block if box is not firstBox])
        if fixedSudoku[firstBox[0], firstBox[1]] != 1 and fixedSudoku[secondBox[0], secondBox[1]] != 1:
            return [firstBox, secondBox]

def FlipBoxes(sudoku, boxesToFlip):
    """Swaps the values of the two given boxes."""
    proposedSudoku = np.copy(sudoku)
    placeHolder = proposedSudoku[boxesToFlip[0][0], boxesToFlip[0][1]]
    proposedSudoku[boxesToFlip[0][0], boxesToFlip[0][1]] = proposedSudoku[boxesToFlip[1][0], boxesToFlip[1][1]]
    proposedSudoku[boxesToFlip[1][0], boxesToFlip[1][1]] = placeHolder
    return proposedSudoku

def ProposedState(sudoku, fixedSudoku, listOfBlocks):
    """Proposes a new state by flipping two boxes in a random block."""
    randomBlock = random.choice(listOfBlocks)
    if SumOfOneBlock(fixedSudoku, randomBlock) > 6:
        return (sudoku, 1, 1)
    boxesToFlip = TwoRandomBoxesWithinBlock(fixedSudoku, randomBlock)
    proposedSudoku = FlipBoxes(sudoku, boxesToFlip)
    return [proposedSudoku, boxesToFlip]

def ChooseNewState(currentSudoku, fixedSudoku, listOfBlocks, sigma):
    """Decides whether to accept the new proposed state based on simulated annealing."""
    proposal = ProposedState(currentSudoku, fixedSudoku, listOfBlocks)
    newSudoku = proposal[0]
    boxesToCheck = proposal[1]
    currentCost = CalculateNumberOfErrorsRowColumn(boxesToCheck[0][0], boxesToCheck[0][1], currentSudoku) + CalculateNumberOfErrorsRowColumn(boxesToCheck[1][0], boxesToCheck[1][1], currentSudoku)
    newCost = CalculateNumberOfErrorsRowColumn(boxesToCheck[0][0], boxesToCheck[0][1], newSudoku) + CalculateNumberOfErrorsRowColumn(boxesToCheck[1][0], boxesToCheck[1][1], newSudoku)
    costDifference = newCost - currentCost
    rho = math.exp(-costDifference / sigma)
    if np.random.uniform(1, 0, 1) < rho:
        return [newSudoku, costDifference]
    return [currentSudoku, 0]

def ChooseNumberOfIterations(fixed_sudoku):
    """Determines the number of iterations based on the number of fixed values."""
    numberOfIterations = 0
    for i in range(0, 9):
        for j in range(0, 9):
            if fixed_sudoku[i, j] != 0:
                numberOfIterations += 1
    return numberOfIterations

def CalculateInitialSigma(sudoku, fixedSudoku, listOfBlocks):
    """Calculates an initial temperature (sigma) for the annealing process."""
    listOfDifferences = []
    tmpSudoku = sudoku
    for i in range(1, 10):
        tmpSudoku = ProposedState(tmpSudoku, fixedSudoku, listOfBlocks)[0]
        listOfDifferences.append(CalculateNumberOfErrors(tmpSudoku))
    if len(listOfDifferences) > 1:
        return statistics.pstdev(listOfDifferences)
    return 1 # Default sigma if standard deviation can't be calculated

# --- Parallel Worker Function ---
def solve_worker(initial_sudoku, solution_queue):
    """
    This function represents a single, independent attempt to solve the Sudoku.
    It's designed to be run in its own process.
    """
    # Each worker gets its own unique random seed
    random.seed()

    decreaseFactor = 0.99
    stuckCount = 0

    fixedSudoku = np.copy(initial_sudoku)
    FixSudokuValues(fixedSudoku)
    
    listOfBlocks = CreateList3x3Blocks()
    
    # Each worker starts with its own random guess
    tmpSudoku = RandomlyFill3x3Blocks(np.copy(initial_sudoku), listOfBlocks)
    
    sigma = CalculateInitialSigma(tmpSudoku, fixedSudoku, listOfBlocks)
    score = CalculateNumberOfErrors(tmpSudoku)
    iterations = ChooseNumberOfIterations(fixedSudoku)

    if score <= 0:
        solution_queue.put(tmpSudoku)
        return

    # This is the main solving loop for this worker
    while True:
        previousScore = score
        for i in range(0, iterations):
            newState = ChooseNewState(tmpSudoku, fixedSudoku, listOfBlocks, sigma)
            tmpSudoku = newState[0]
            scoreDiff = newState[1]
            score += scoreDiff
            if score <= 0:
                # Found a solution! Put it in the queue for the main process.
                solution_queue.put(tmpSudoku)
                return

        sigma *= decreaseFactor
        if score >= previousScore:
            stuckCount += 1
        else:
            stuckCount = 0
        
        # If the solver is stuck, give it a kick or reset.
        if stuckCount > 100:
            # Resetting this worker to try again with a new random board
            stuckCount = 0
            tmpSudoku = RandomlyFill3x3Blocks(np.copy(initial_sudoku), listOfBlocks)
            score = CalculateNumberOfErrors(tmpSudoku)
            sigma = CalculateInitialSigma(tmpSudoku, fixedSudoku, listOfBlocks)


# --- Main Execution Block ---
if __name__ == "__main__":
    # Parse the initial Sudoku board
    sudoku_board = np.array([[int(i) for i in line] for line in startingSudoku.split()])
    print("--- Initial Sudoku ---")
    PrintSudoku(sudoku_board)
    
    # Determine the number of CPU cores to use
    cpu_count = os.cpu_count()
    print(f"\n--- Starting parallel solver on {cpu_count} cores ---")
    start_time = time.time()

    # A Manager allows sharing Python objects (like a Queue) between processes
    with multiprocessing.Manager() as manager:
        # This queue is used for a worker process to send the solution back
        solution_queue = manager.Queue()

        # A Pool creates a number of worker processes
        with multiprocessing.Pool(processes=cpu_count) as pool:
            # Start one 'solve_worker' task on each core.
            # Each worker will run independently.
            for _ in range(cpu_count):
                pool.apply_async(solve_worker, args=(sudoku_board, solution_queue))
            
            # --- Wait for the first solution ---
            # The .get() call will block here and wait until one of the workers
            # puts a solution into the queue.
            print("\nSearching for a solution...")
            solution = solution_queue.get()
            
            # --- Terminate other workers ---
            # Once we have a solution, we don't need the other workers anymore.
            # .terminate() immediately stops all processes in the pool.
            pool.terminate()
            pool.join()

    end_time = time.time()
    
    print("\n--- Solution Found! ---")
    PrintSudoku(solution)
    print(f"\nTotal errors in solution: {CalculateNumberOfErrors(solution)}")
    print(f"Solved in {end_time - start_time:.4f} seconds.")





'''
----
No Using Of MultiCores
----
import random
import numpy as np
import math 
from random import choice
import statistics 
"""
ease one
---
                    024007000
                    600000000
                    003680415
                    431005000
                    500000032
                    790000060
                    209710800
                    040093000
                    310004750
---
Hard one
---
090600000
000831000
160000200
070090100
000057900
006000003
038005000
600000704
000000010
---


"""
startingSudoku = """
                    090600000
                    000831000
                    160000200
                    070090100
                    000057900
                    006000003
                    038005000
                    600000704
                    000000010
                """

sudoku = np.array([[int(i) for i in line] for line in startingSudoku.split()])

def PrintSudoku(sudoku):
    print("\n")
    for i in range(len(sudoku)):
        line = ""
        if i == 3 or i == 6:
            print("---------------------")
        for j in range(len(sudoku[i])):
            if j == 3 or j == 6:
                line += "| "
            line += str(sudoku[i,j])+" "
        print(line)

def FixSudokuValues(fixed_sudoku):
    for i in range (0,9):
        for j in range (0,9):
            if fixed_sudoku[i,j] != 0:
                fixed_sudoku[i,j] = 1
    
    return(fixed_sudoku)

# Cost Function    
def CalculateNumberOfErrors(sudoku):
    numberOfErrors = 0 
    for i in range (0,9):
        numberOfErrors += CalculateNumberOfErrorsRowColumn(i ,i ,sudoku)
    return(numberOfErrors)

def CalculateNumberOfErrorsRowColumn(row, column, sudoku):
    numberOfErrors = (9 - len(np.unique(sudoku[:,column]))) + (9 - len(np.unique(sudoku[row,:])))
    return(numberOfErrors)


def CreateList3x3Blocks ():
    finalListOfBlocks = []
    for r in range (0,9):
        tmpList = []
        block1 = [i + 3*((r)%3) for i in range(0,3)]
        block2 = [i + 3*math.trunc((r)/3) for i in range(0,3)]
        for x in block1:
            for y in block2:
                tmpList.append([x,y])
        finalListOfBlocks.append(tmpList)
    return(finalListOfBlocks)

def RandomlyFill3x3Blocks(sudoku, listOfBlocks):
    for block in listOfBlocks:
        for box in block:
            if sudoku[box[0],box[1]] == 0:
                currentBlock = sudoku[block[0][0]:(block[-1][0]+1),block[0][1]:(block[-1][1]+1)]
                sudoku[box[0],box[1]] = choice([i for i in range(1,10) if i not in currentBlock])
    return sudoku

def SumOfOneBlock (sudoku, oneBlock):
    finalSum = 0
    for box in oneBlock:
        finalSum += sudoku[box[0], box[1]]
    return(finalSum)

def TwoRandomBoxesWithinBlock(fixedSudoku, block):
    while (1):
        firstBox = random.choice(block)
        secondBox = choice([box for box in block if box is not firstBox ])

        if fixedSudoku[firstBox[0], firstBox[1]] != 1 and fixedSudoku[secondBox[0], secondBox[1]] != 1:
            return([firstBox, secondBox])

def FlipBoxes(sudoku, boxesToFlip):
    proposedSudoku = np.copy(sudoku)
    placeHolder = proposedSudoku[boxesToFlip[0][0], boxesToFlip[0][1]]
    proposedSudoku[boxesToFlip[0][0], boxesToFlip[0][1]] = proposedSudoku[boxesToFlip[1][0], boxesToFlip[1][1]]
    proposedSudoku[boxesToFlip[1][0], boxesToFlip[1][1]] = placeHolder
    return (proposedSudoku)

def ProposedState (sudoku, fixedSudoku, listOfBlocks):
    randomBlock = random.choice(listOfBlocks)

    if SumOfOneBlock(fixedSudoku, randomBlock) > 6:  
        return(sudoku, 1, 1)
    boxesToFlip = TwoRandomBoxesWithinBlock(fixedSudoku, randomBlock)
    proposedSudoku = FlipBoxes(sudoku,  boxesToFlip)
    return([proposedSudoku, boxesToFlip])

def ChooseNewState (currentSudoku, fixedSudoku, listOfBlocks, sigma):
    proposal = ProposedState(currentSudoku, fixedSudoku, listOfBlocks)
    newSudoku = proposal[0]
    boxesToCheck = proposal[1]
    currentCost = CalculateNumberOfErrorsRowColumn(boxesToCheck[0][0], boxesToCheck[0][1], currentSudoku) + CalculateNumberOfErrorsRowColumn(boxesToCheck[1][0], boxesToCheck[1][1], currentSudoku)
    newCost = CalculateNumberOfErrorsRowColumn(boxesToCheck[0][0], boxesToCheck[0][1], newSudoku) + CalculateNumberOfErrorsRowColumn(boxesToCheck[1][0], boxesToCheck[1][1], newSudoku)
    # currentCost = CalculateNumberOfErrors(currentSudoku)
    # newCost = CalculateNumberOfErrors(newSudoku)
    costDifference = newCost - currentCost
    rho = math.exp(-costDifference/sigma)
    if(np.random.uniform(1,0,1) < rho):
        return([newSudoku, costDifference])
    return([currentSudoku, 0])


def ChooseNumberOfItterations(fixed_sudoku):
    numberOfItterations = 0
    for i in range (0,9):
        for j in range (0,9):
            if fixed_sudoku[i,j] != 0:
                numberOfItterations += 1
    return numberOfItterations

def CalculateInitialSigma (sudoku, fixedSudoku, listOfBlocks):
    listOfDifferences = []
    tmpSudoku = sudoku
    for i in range(1,10):
        tmpSudoku = ProposedState(tmpSudoku, fixedSudoku, listOfBlocks)[0]
        listOfDifferences.append(CalculateNumberOfErrors(tmpSudoku))
    return (statistics.pstdev(listOfDifferences))


def solveSudoku (sudoku):
    numberOfTryinig=0
    f = open("demofile2.txt", "a")
    solutionFound = 0
    while (solutionFound == 0):
        decreaseFactor = 0.99
        stuckCount = 0
        fixedSudoku = np.copy(sudoku)
        PrintSudoku(sudoku)
        FixSudokuValues(fixedSudoku)
        listOfBlocks = CreateList3x3Blocks()
        tmpSudoku = RandomlyFill3x3Blocks(sudoku, listOfBlocks)
        sigma = CalculateInitialSigma(sudoku, fixedSudoku, listOfBlocks)
        score = CalculateNumberOfErrors(tmpSudoku)
        itterations = ChooseNumberOfItterations(fixedSudoku)
        if score <= 0:
            solutionFound = 1

        while solutionFound == 0:
            numberOfTryinig=numberOfTryinig+1
            previousScore = score
            for i in range (0, itterations):
                newState = ChooseNewState(tmpSudoku, fixedSudoku, listOfBlocks, sigma)
                tmpSudoku = newState[0]
                scoreDiff = newState[1]
                score += scoreDiff
                print(score)
                f.write(str(score) + '\n')
                if score <= 0:
                    solutionFound = 1
                    break

            sigma *= decreaseFactor
            if score <= 0:
                solutionFound = 1
                break
            if score >= previousScore:
                stuckCount += 1
            else:
                stuckCount = 0
            if (stuckCount > 80):
                sigma += 2
            if(CalculateNumberOfErrors(tmpSudoku)==0):
                PrintSudoku(tmpSudoku)
                break
    f.close()
    print("THis Try Numbers"+str(numberOfTryinig))
    return(tmpSudoku)

solution = solveSudoku(sudoku)
print(CalculateNumberOfErrors(solution))
PrintSudoku(solution)
'''