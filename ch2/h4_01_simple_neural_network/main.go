package main

import (
	"fmt"
	"log"
	"math/rand"

	gg "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

var err error

type nn struct {
	g      *gg.ExprGraph
	w0, w1 *gg.Node

	pred    *gg.Node
	predVal gg.Value
}

func newNN(g *gg.ExprGraph) *nn {
	// Create node for w/weight
	wB := tensor.Random(tensor.Float64, 3)
	wT := tensor.New(tensor.WithBacking(wB), tensor.WithShape(3, 1))
	w0 := gg.NewMatrix(g,
		tensor.Float64,
		gg.WithName("w"),
		gg.WithShape(3, 1),
		gg.WithValue(wT),
	)
	return &nn{
		g:  g,
		w0: w0,
	}
}

func (m *nn) learnables() gg.Nodes {
	return gg.Nodes{m.w0}
}

func (m *nn) fwd(x *gg.Node) (err error) {
	var l0, l1 *gg.Node

	// Set first layer to be copy of input
	l0 = x

	// Dot product of l0 and w0, use as input for Sigmoid
	l0dot := gg.Must(gg.Mul(l0, m.w0))

	// Build hidden layer out of result
	l1 = gg.Must(gg.Sigmoid(l0dot))

	m.pred = l1
	gg.Read(m.pred, &m.predVal)
	return nil

}

func main() {

	rand.Seed(31337)

	// Create graph and network
	g := gg.NewGraph()
	m := newNN(g)

	// Set input x to network
	// xB := []float64{0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1}
	// xT := tensor.New(tensor.WithBacking(gg.WithInit(gg.GlorotN(1.0))), tensor.WithShape(4, 3))
	x := gg.NewMatrix(g,
		tensor.Float64,
		gg.WithName("X"),
		gg.WithShape(4, 3),
		gg.WithInit(gg.GlorotN(1.0)),
	)

	// Define validation data set
	yB := []float64{0, 0, 1, 1}
	yT := tensor.New(tensor.WithBacking(yB), tensor.WithShape(4, 1))
	y := gg.NewMatrix(g,
		tensor.Float64,
		gg.WithName("y"),
		gg.WithShape(4, 1),
		gg.WithValue(yT),
	)

	// Run forward pass
	if err := m.fwd(x); err != nil {
		log.Fatalf("%+v", err)
	}

	// Calculate Cost w/MSE
	losses := gg.Must(gg.Sub(y, m.pred))
	square := gg.Must(gg.Square(losses))
	cost := gg.Must(gg.Mean(square))

	// Do Gradient updates
	if _, err = gg.Grad(cost, m.learnables()...); err != nil {
		log.Fatal(err)
	}

	// Instantiate VM and Solver
	vm := gg.NewTapeMachine(g, gg.BindDualValues(m.learnables()...))
	//defer vm.Close()
	solver := gg.NewLossGradSolver(gg.WithLearnRate(1.0))

	// for i := 0; i < 10000; i++ {
	//	vm.Reset()
	// solver := NewVanillaSolver(WithLearnRate(1.0))

	for i := 0; i < 100000; i++ {
		vm.Reset()
		if err = vm.RunAll(); err != nil {
			log.Fatalf("Failed at inter  %d: %v", i, err)
		}
		solver.Step(gg.NodesToValueGrads(m.learnables()))

		// vm.Reset()
		if i == 1 {
			fmt.Printf("\n\nOutput at step %v: %v \n", i, m.predVal)
			fmt.Println("Cost: ", cost.Value())
		}
		if i == 5000 {
			fmt.Printf("\n\nOutput at step %v: %v \n", i, m.predVal)
			fmt.Println("Cost: ", cost.Value())
		}
	}
	fmt.Println("\n\nOutput after Training: \n", m.predVal)
	fmt.Println("Final weights: ", m.w0.Value())
}
