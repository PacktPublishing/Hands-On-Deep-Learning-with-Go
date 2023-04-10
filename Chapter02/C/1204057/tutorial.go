package main

import (
	"fmt"
	"log"
	"math/rand"

	. "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

var err error

type nn struct {
	g          *ExprGraph
	w0, w1, w2 *Node

	pred    *Node
	predVal Value
}

func newNN(g *ExprGraph) *nn {
	// Create node for w/weight
	wB := tensor.Random(tensor.Float64, 3)
	wB1 := tensor.Random(tensor.Float64, 3)
	wB2 := tensor.Random(tensor.Float64, 3)

	wT := tensor.New(tensor.WithBacking(wB), tensor.WithShape(3, 1))
	wT1 := tensor.New(tensor.WithBacking(wB1), tensor.WithShape(1, 3))
	wT2 := tensor.New(tensor.WithBacking(wB2), tensor.WithShape(3, 1))

	w0 := NewMatrix(g,
		tensor.Float64,
		WithName("w"),
		WithShape(3, 1),
		WithValue(wT),
	)
	w1 := NewMatrix(g,
		tensor.Float64,
		WithName("w"),
		WithShape(1, 3),
		WithValue(wT1),
	)
	w2 := NewMatrix(g,
		tensor.Float64,
		WithName("w"),
		WithShape(3, 1),
		WithValue(wT2),
	)
	return &nn{
		g:  g,
		w0: w0,
		w1: w1,
		w2: w2,
	}
}

func (m *nn) learnables() Nodes {
	return Nodes{m.w0, m.w1, m.w2}
}

func (m *nn) fwd(x *Node) (err error) {
	var l0, l1, l2 *Node

	// Set first layer to be copy of input
	l0 = x
	fmt.Printf("\nL0: %+v\n", l0.Shape())

	// Dot product of l0 and w0, use as input for Sigmoid
	l0dot := Must(Mul(l0, m.w0))
	fmt.Printf("\nL0dot: %+v\n", l0dot.Shape())

	// Build hidden layer out of result
	l1 = Must(Sigmoid(l0dot))
	fmt.Printf("\nL1dot: %+v\n", l1.Shape())
	l1dot := Must(Mul(l1, m.w1))
	fmt.Printf("\nL1dot: %+v\n", l1dot.Shape())

	l2 = Must(Tanh(l1dot))
	l2dot := Must(Mul(l2, m.w2))
	fmt.Printf("\nL2dot: %+v\n", l2dot.Shape())

	l3 := Must(LeakyRelu(l2dot, 0.1))

	m.pred = l3
	fmt.Println("Pred: ", m.pred.Shape())
	Read(m.pred, &m.predVal)
	return nil

}

func main() {

	rand.Seed(31337)

	// Create graph and network
	g := NewGraph()
	m := newNN(g)

	// Set input x to network
	xB := []float64{10, 5, 15, 5, 7, 8, 9, 10, 3, 2, 2, 1}
	xT := tensor.New(tensor.WithBacking(xB), tensor.WithShape(4, 3))
	x := NewMatrix(g,
		tensor.Float64,
		WithName("X"),
		WithShape(4, 3),
		WithValue(xT),
	)

	// Define validation data set
	yB := []float64{3, 2, 2, 1}
	yT := tensor.New(tensor.WithBacking(yB), tensor.WithShape(4, 1))
	y := NewMatrix(g,
		tensor.Float64,
		WithName("y"),
		WithShape(4, 1),
		WithValue(yT),
	)

	// Run forward pass
	if err := m.fwd(x); err != nil {
		log.Fatalf("%+v", err)
	}

	// Calculate Cost w/MSE
	losses := Must(Sub(y, m.pred))
	square := Must(Square(losses))
	cost := Must(Mean(square))

	// Do Gradient updates
	if _, err = Grad(cost, m.learnables()...); err != nil {
		log.Fatal(err)
	}

	// Instantiate VM and Solver
	vm := NewTapeMachine(g, BindDualValues(m.learnables()...))
	solver := NewVanillaSolver(WithLearnRate(1.0))

	for i := 0; i < 10000; i++ {
		fmt.Printf("\nIteration: %d\n", i)
		vm.Reset()
		if err = vm.RunAll(); err != nil {
			log.Fatalf("Failed at inter  %d: %v", i, err)
		}
		solver.Step(NodesToValueGrads(m.learnables()))
		vm.Reset()
	}
	fmt.Println("\n\nOutput after Training: \n", m.predVal)
}
