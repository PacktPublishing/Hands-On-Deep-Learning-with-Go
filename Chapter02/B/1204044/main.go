package main

import (
	. "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type nn struct {
	g      *ExprGraph
	w0, w1 *Node

	pred *Node
}

func newNN(g *ExprGraph) *nn {
	// Create node for w/weight (needs fixed values replaced with random values w/mean 0)
	wB := []float64{-0.167855599, 0.44064899, -0.99977125}
	wT := tensor.New(tensor.WithBacking(wB), tensor.WithShape(3, 1))
	w0 := NewMatrix(g,
		tensor.Float64,
		WithName("w"),
		WithShape(3, 1),
		WithValue(wT),
	)
	return &nn{
		g:  g,
		w0: w0,
	}
}

func (m *nn) learnables() Nodes {
	return Nodes{m.w0}
}

func (m *nn) fwd(x *Node) (err error) {
	var l0, l1 *Node

	// Set first layer to be copy of input
	l0 = x

	// Dot product of l0 and w0, use as input for Sigmoid
	l0dot := Must(Mul(l0, m.w0))

	// Build hidden layer out of result
	l1 = Must(Sigmoid(l0dot))
	// fmt.Println("l1: \n", l1.Value())

	m.pred = l1
	return

}
