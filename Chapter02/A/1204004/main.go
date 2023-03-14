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
	g                                  *ExprGraph
	w0, w1, w2, w3, w4, w5, w6, w7, w8 *Node

	pred    *Node
	predVal Value
}

func newNN(g *ExprGraph) *nn {
	// Create node for w/weight
	wB := tensor.Random(tensor.Float64, 3)
	wB1 := tensor.Random(tensor.Float64, 3)
	wB2 := tensor.Random(tensor.Float64, 3)
	wB3 := tensor.Random(tensor.Float64, 3)
	wB4 := tensor.Random(tensor.Float64, 3)
	wB5 := tensor.Random(tensor.Float64, 3)
	wB6 := tensor.Random(tensor.Float64, 3)
	wB7 := tensor.Random(tensor.Float64, 3)
	wB8 := tensor.Random(tensor.Float64, 3)

	wT := tensor.New(tensor.WithBacking(wB), tensor.WithShape(3, 1))
	wT1 := tensor.New(tensor.WithBacking(wB1), tensor.WithShape(1, 3))
	wT2 := tensor.New(tensor.WithBacking(wB2), tensor.WithShape(3, 1))
	wT3 := tensor.New(tensor.WithBacking(wB3), tensor.WithShape(1, 3))
	wT4 := tensor.New(tensor.WithBacking(wB4), tensor.WithShape(3, 1))
	wT5 := tensor.New(tensor.WithBacking(wB5), tensor.WithShape(1, 3))
	wT6 := tensor.New(tensor.WithBacking(wB6), tensor.WithShape(3, 1))
	wT7 := tensor.New(tensor.WithBacking(wB7), tensor.WithShape(1, 3))
	wT8 := tensor.New(tensor.WithBacking(wB8), tensor.WithShape(3, 1))

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
	w3 := NewMatrix(g,
		tensor.Float64,
		WithName("w"),
		WithShape(1, 3),
		WithValue(wT3),
	)
	w4 := NewMatrix(g,
		tensor.Float64,
		WithName("w"),
		WithShape(3, 1),
		WithValue(wT4),
	)
	w5 := NewMatrix(g,
		tensor.Float64,
		WithName("w"),
		WithShape(1, 3),
		WithValue(wT5),
	)
	w6 := NewMatrix(g,
		tensor.Float64,
		WithName("w"),
		WithShape(3, 1),
		WithValue(wT6),
	)
	w7 := NewMatrix(g,
		tensor.Float64,
		WithName("w"),
		WithShape(1, 3),
		WithValue(wT7),
	)
	w8 := NewMatrix(g,
		tensor.Float64,
		WithName("w"),
		WithShape(3, 1),
		WithValue(wT8),
	)
	return &nn{
		g:  g,
		w0: w0,
		w1: w1,
		w2: w2,
		w3: w3,
		w4: w4,
		w5: w5,
		w6: w6,
		w7: w7,
		w8: w8,
	}
}

func (m *nn) learnables() Nodes {
	return Nodes{m.w8}
}

func (m *nn) fwd(x *Node) (err error) {
	var l0, l1, l2, l3, l4, l5, l6, l7, l8 *Node

	// Set first layer to be copy of input
	l0 = x

	// Dot product of l0 and w0, use as input for Sigmoid
	l0dot := Must(Mul(l0, m.w0))

	// Build hidden layer out of result
	l1 = Must(Tanh(l0dot))
	l1dot := Must(Mul(l1, m.w1))
	l2 = Must(LeakyRelu(l1dot, 0.1))
	l2dot := Must(Mul(l2, m.w2))
	l3 = Must(Log1p(l2dot))
	l3dot := Must(Mul(l3, m.w3))
	l4 = Must(Sin(l3dot))
	l4dot := Must(Mul(l4, m.w4))
	l5 = Must(Cos(l4dot))
	l5dot := Must(Mul(l5, m.w5))
	l6 = Must(Exp(l5dot))
	l6dot := Must(Mul(l6, m.w6))
	l7 = Must(Cube(l6dot))
	l7dot := Must(Mul(l7, m.w7))
	l8 = Must(Sigmoid(l7dot))
	l8dot := Must(Mul(l8, m.w8))

	m.pred = l8dot
	Read(m.pred, &m.predVal)
	return nil

}

func main() {

	rand.Seed(31337)

	// Create graph and network
	g := NewGraph()
	m := newNN(g)

	// Set input x to network
	xB := []float64{0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1}
	xT := tensor.New(tensor.WithBacking(xB), tensor.WithShape(4, 3))
	x := NewMatrix(g,
		tensor.Float64,
		WithName("X"),
		WithShape(4, 3),
		WithValue(xT),
	)

	// Define validation data set
	yB := []float64{0, 0, 1, 1}
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
	fmt.Println("Output before Training: \n", m.pred.Shape())
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
		vm.Reset()
		fmt.Printf("\nStep %d\n", i)
		if err = vm.RunAll(); err != nil {
			log.Fatalf("Failed at inter  %d: %v", i, err)
		}
		solver.Step(NodesToValueGrads(m.learnables()))
		vm.Reset()
	}
	fmt.Println("\n\nOutput after Training: \n", m.predVal)
}
