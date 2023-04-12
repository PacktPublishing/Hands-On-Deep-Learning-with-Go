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
	w1 := gorgonia.NewMatrix(g,
		tensor.Float64,
		gorgonia.WithName("w1"),
		gorgonia.WithShape(3, 4),
		gorgonia.WithInit(gorgonia.GlorotN(1.0)),
	)
	w2 := gorgonia.NewMatrix(g,
		tensor.Float64,
		gorgonia.WithName("w2"),
		gorgonia.WithShape(4, 5),
		gorgonia.WithInit(gorgonia.GlorotN(1.0)),
	)
	w3 := gorgonia.NewMatrix(g,
		tensor.Float64,
		gorgonia.WithName("w3"),
		gorgonia.WithShape(5, 6),
		gorgonia.WithInit(gorgonia.GlorotN(1.0)),
	)
	w4 := gorgonia.NewMatrix(g,
		tensor.Float64,
		gorgonia.WithName("w4"),
		gorgonia.WithShape(6, 7),
		gorgonia.WithInit(gorgonia.GlorotN(1.0)),
	)
	w5 := gorgonia.NewMatrix(g,
		tensor.Float64,
		gorgonia.WithName("w5"),
		gorgonia.WithShape(7, 1),
		gorgonia.WithInit(gorgonia.GlorotN(1.0)),
	)

	// Create a slice of weight nodes
	ws := []*gorgonia.Node{w1, w2, w3, w4, w5}

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
	for _, w := range m.ws {
		l = gorgonia.Must(gorgonia.Mul(l, w))
		l = gorgonia.Must(gorgonia.Sigmoid(l))
	}

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

	// Do Gradient updates
 if _, err = gorgonia.Grad(cost, m.learnables()...); err != nil {
	log.Fatal(err)
}

// Instantiate VM and Solver
vm := gorgonia.NewTapeMachine(g, gorgonia.BindDualValues(m.learnables()...))
solver := gorgonia.NewVanillaSolver(gorgonia.WithLearnRate(0.1))

// Train
for i := 0; i < 1000; i++ {
	if err = vm.RunAll(); err != nil {
		log.Fatal(err)
	}

	if err = solver.Step(gorgonia.NodesToValueGrads(m.learnables())); err != nil {
		log.Fatal(err)
	}
	vm.Reset()
}

// Predict
if err = m.fwd(x); err != nil {
	log.Fatalf("fwd failed: %+v", err)
}

fmt.Printf("Prediction: %v\n", m.predVal)
