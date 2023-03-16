package main

import (
	"fmt"
	"log"
	"math/rand"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

var err error

type nn struct {
	g  *gorgonia.ExprGraph
	w0 *gorgonia.Node

	pred    *gorgonia.Node
	predVal gorgonia.Value
}

func newNN(g *gorgonia.ExprGraph) *nn {
	// Create node for w/weight
	wB := tensor.Random(tensor.Float64, 3)
	wT := tensor.New(tensor.WithBacking(wB), tensor.WithShape(3, 1))
	w0 := gorgonia.NewMatrix(g,
		tensor.Float64,
		gorgonia.WithName("w"),
		gorgonia.WithShape(3, 1),
		gorgonia.WithValue(wT),
	)
	return &nn{
		g:  g,
		w0: w0,
	}
}

func (m *nn) learnables() gorgonia.Nodes {
	return gorgonia.Nodes{m.w0}
}

func (m *nn) fwd(x *gorgonia.Node) (err error) {
	var l0, l1 *gorgonia.Node

	// Set first layer to be copy of input
	l0 = x

	// Dot product of l0 and w0, use as input for Sigmoid
	l0dot := gorgonia.Must(gorgonia.Mul(l0, m.w0))

	// Build hidden layer out of result
	l1 = gorgonia.Must(gorgonia.Sigmoid(l0dot))

	m.pred = l1
	gorgonia.Read(m.pred, &m.predVal)
	return nil

}

func main() {

	rand.Seed(31337)
	//NewRand(NewSource(31337))

	// Create graph and network
	g := gorgonia.NewGraph()
	m := newNN(g)

	// Set input x to network
	xB := []float64{0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1}
	xT := tensor.New(tensor.WithBacking(xB), tensor.WithShape(4, 3))
	x := gorgonia.NewMatrix(g,
		tensor.Float64,
		gorgonia.WithName("X"),
		gorgonia.WithShape(4, 3),
		gorgonia.WithValue(xT),
	)

	// Define validation data set
	yB := []float64{0, 0, 1, 1}
	yT := tensor.New(tensor.WithBacking(yB), tensor.WithShape(4, 1))
	y := gorgonia.NewMatrix(g,
		tensor.Float64,
		gorgonia.WithName("y"),
		gorgonia.WithShape(4, 1),
		gorgonia.WithValue(yT),
	)

	// Run forward pass
	if err := m.fwd(x); err != nil {
		log.Fatalf("%+v", err)
	}

	// Calculate Cost w/MSE
	losses := gorgonia.Must(gorgonia.Sub(y, m.pred))
	square := gorgonia.Must(gorgonia.Square(losses))
	cost := gorgonia.Must(gorgonia.Mean(square))

	// Do Gradient updates
	if _, err = gorgonia.Grad(cost, m.learnables()...); err != nil {
		log.Fatal(err)
	}

	// Instantiate VM and Solver
	vm := gorgonia.NewTapeMachine(g, gorgonia.BindDualValues(m.learnables()...))
	solver := gorgonia.NewVanillaSolver(gorgonia.WithLearnRate(1.0))

	for i := 0; i < 10000; i++ {
		vm.Reset()
		if err = vm.RunAll(); err != nil {
			log.Fatalf("Failed at inter  %d: %v", i, err)
		}
		solver.Step(gorgonia.NodesToValueGrads(m.learnables()))
		vm.Reset()
	}
	fmt.Println("\n\nOutput after Training: \n", m.predVal)
}
