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
	g       *gorgonia.ExprGraph
	ws      []*gorgonia.Node
	pred    *gorgonia.Node
	predVal gorgonia.Value
}

func newNN(g *gorgonia.ExprGraph) *nn {
	// Create nodes for weights
	l1 := gorgonia.NewMatrix(g,
		tensor.Float64,
		gorgonia.WithName("l1"),
		gorgonia.WithShape(3, 4),
		gorgonia.WithInit(gorgonia.GlorotN(1.0)),
	)
	l2 := gorgonia.NewMatrix(g,
		tensor.Float64,
		gorgonia.WithName("l2"),
		gorgonia.WithShape(4, 5),
		gorgonia.WithInit(gorgonia.GlorotN(1.0)),
	)
	l3 := gorgonia.NewMatrix(g,
		tensor.Float64,
		gorgonia.WithName("l3"),
		gorgonia.WithShape(5, 6),
		gorgonia.WithInit(gorgonia.GlorotN(1.0)),
	)
	l4 := gorgonia.NewMatrix(g,
		tensor.Float64,
		gorgonia.WithName("l4"),
		gorgonia.WithShape(6, 7),
		gorgonia.WithInit(gorgonia.GlorotN(1.0)),
	)
	l5 := gorgonia.NewMatrix(g,
		tensor.Float64,
		gorgonia.WithName("l5"),
		gorgonia.WithShape(7, 8),
		gorgonia.WithInit(gorgonia.GlorotN(1.0)),
	)
	// Create a slice of weight nodes
	ws := []*gorgonia.Node{l1, l2, l3, l4, l5}

	return &nn{
		g:  g,
		ws: ws,
	}
}

func (m *nn) learnables() gorgonia.Nodes {
	// Return the slice of weight nodes
	return gorgonia.Nodes(m.ws)
}

func (m *nn) fwd(x *gorgonia.Node) (err error) {
	var l *gorgonia.Node = x

	// Build network
	for i := 0; i < len(m.ws)-1; i++ {
		l = gorgonia.Must(gorgonia.Mul(l, m.ws[i]))
		l = gorgonia.Must(gorgonia.Sigmoid(l))
	}

	// Last layer without activation function
	l = gorgonia.Must(gorgonia.Mul(l, m.ws[len(m.ws)-1]))

	m.pred = l
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
	solver := gorgonia.NewAdamSolver(gorgonia.WithLearnRate(0.001), gorgonia.WithBeta1(0.9), gorgonia.WithBeta2(0.999))

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
