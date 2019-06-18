package main

import (
	"fmt"
	"log"
	"math/rand"
	"time"

	"gorgonia.org/gorgonia"
)

var cardinals = [4]Vector{
	Vector{0, 1},  // E
	Vector{1, 0},  // N
	Vector{-1, 0}, // S
	Vector{0, -1}, // W
}

type DQN struct {
	*NN
	gorgonia.VM
	gorgonia.Solver
	Memories []Memory // The Q-Table - stores State/Action/Reward/NextState/NextMoves/IsDone - added to each train x times per episode

	gamma       float32
	epsilon     float32
	epsDecayMin float32
	decay       float32
}

func (m *DQN) init() {
	if _, err := m.NN.cons(); err != nil {
		panic(err)
	}
	m.VM = gorgonia.NewTapeMachine(m.NN.g)
	m.Solver = gorgonia.NewRMSPropSolver()
}

func (m *DQN) replay(batchsize int) error {
	var N int
	if batchsize < len(m.Memories) {
		N = batchsize
	} else {
		N = len(m.Memories)
	}
	Xs := make([]input, 0, N)
	Ys := make([]float32, 0, N)
	mems := make([]Memory, N)
	copy(mems, m.Memories)
	rand.Shuffle(len(mems), func(i, j int) {
		mems[i], mems[j] = mems[j], mems[i]
	})

	for b := 0; b < batchsize; b++ {
		mem := mems[b]

		var y float32
		if mem.isDone {
			y = mem.Reward
		} else {
			var nextRewards []float32
			for _, next := range mem.NextMovables {
				nextReward, err := m.predict(mem.NextState, next)
				if err != nil {
					return err
				}
				nextRewards = append(nextRewards, nextReward)
			}
			reward := max(nextRewards)
			y = mem.Reward + m.gamma*reward
		}
		Xs = append(Xs, input{mem.State, mem.Action})
		Ys = append(Ys, y)
		if err := m.VM.RunAll(); err != nil {
			return err
		}
		m.VM.Reset()
		if err := m.Solver.Step(m.model()); err != nil {
			return err
		}
		if m.epsilon > m.epsDecayMin {
			m.epsilon *= m.decay
		}
	}
	return nil
}

func (m *DQN) predict(player Point, action Vector) (float32, error) {
	x := input{State: player, Action: action}
	m.Let1(x)
	if err := m.VM.RunAll(); err != nil {
		return 0, err
	}
	m.VM.Reset()
	retVal := m.predVal.Data().([]float32)[0]
	return retVal, nil
}

func (m *DQN) train(mz *Maze) (err error) {
	var episodes = 20000
	var times = 1000
	var score float32

	for e := 0; e < episodes; e++ {
		for t := 0; t < times; t++ {
			if e%100 == 0 && t%999 == 1 {
				log.Printf("episode %d, %dst loop", e, t)
			}

			moves := getPossibleActions(mz)
			action := m.bestAction(mz, moves)
			reward, isDone := mz.Value(action)
			score = score + reward
			player := mz.player
			mz.Move(action)
			nextMoves := getPossibleActions(mz)
			mem := Memory{State: player, Action: action, Reward: reward, NextState: mz.player, NextMovables: nextMoves, isDone: isDone}
			m.Memories = append(m.Memories, mem)
		}
	}
	return nil
}

func (m *DQN) bestAction(state *Maze, moves []Vector) (bestAction Vector) {
	var bestActions []Vector
	var maxActValue float32 = -100
	for _, a := range moves {
		actionValue, err := m.predict(state.player, a)
		if err != nil {
			// DO SOMETHING
		}
		if actionValue > maxActValue {
			maxActValue = actionValue
			bestActions = append(bestActions, a)
		} else if actionValue == maxActValue {
			bestActions = append(bestActions, a)
		}
	}
	// shuffle bestActions
	rand.Shuffle(len(bestActions), func(i, j int) {
		bestActions[i], bestActions[j] = bestActions[j], bestActions[i]
	})
	return bestActions[0]
}

func getPossibleActions(m *Maze) (retVal []Vector) {
	for i := range cardinals {
		if m.CanMoveTo(m.player, cardinals[i]) {
			retVal = append(retVal, cardinals[i])
		}
	}
	return retVal
}

func max(a []float32) float32 {
	var m float32 = -999999999
	for i := range a {
		if a[i] > m {
			m = a[i]
		}
	}
	return m
}

func main() {
	// DQN vars

	// var times int = 1000
	var gamma float32 = 0.95  // discount factor
	var epsilon float32 = 1.0 // exploration/exploitation bias, set to 1.0/exploration by default
	var epsilonDecayMin float32 = 0.01
	var epsilonDecay float32 = 0.995

	rand.Seed(time.Now().UTC().UnixNano())
	dqn := &DQN{
		NN:          NewNN(32),
		gamma:       gamma,
		epsilon:     epsilon,
		epsDecayMin: epsilonDecayMin,
		decay:       epsilonDecay,
	}
	dqn.init()

	m := NewMaze(5, 10)
	fmt.Printf("%+#v", m.repr)
	fmt.Printf("%v %v\n", m.start, m.goal)

	fmt.Printf("%v\n", m.CanMoveTo(m.start, Vector{0, 1}))
	fmt.Printf("%v\n", m.CanMoveTo(m.start, Vector{1, 0}))
	fmt.Printf("%v\n", m.CanMoveTo(m.start, Vector{0, -1}))
	fmt.Printf("%v\n", m.CanMoveTo(m.start, Vector{-1, 0}))

	if err := dqn.train(m); err != nil {
		panic(err)
	}

	m.Reset()
	for {
		moves := getPossibleActions(m)
		best := dqn.bestAction(m, moves)
		reward, isDone := m.Value(best)
		log.Printf("\n%#v", m.repr)
		log.Printf("player at: %v best: %v", m.player, best)
		log.Printf("reward %v, done %v", reward, isDone)
		m.Move(best)
	}
}
