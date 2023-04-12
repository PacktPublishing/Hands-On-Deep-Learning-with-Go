package main

import (
	"fmt"
	"math/rand"
	"time"

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
		gorgonia.WithShape(7, 8),
		gorgonia.WithInit(gorgonia.GlorotN(1.0)),
	)
	w6 := gorgonia.NewMatrix(g,
		tensor.Float64,
		gorgonia.WithName("w6"),
		gorgonia.WithShape(8, 9),
		gorgonia.WithInit(gorgonia.GlorotN(1.0)),
	)
	w7 := gorgonia.NewMatrix(g,
		tensor.Float64,
		gorgonia.WithName("w7"),
		gorgonia.WithShape(9, 1),
		gorgonia.WithInit(gorgonia.GlorotN(1.0)),
	)

	// Create a slice of weight nodes
	ws := []*gorgonia.Node{w1, w2, w3, w4, w5, w6, w7}

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
	// Use the current time as the random seed
	rand.Seed(time.Now().UnixNano())

	// Generate a random number between 1 and 100
	randomNumber := rand.Intn(100) + 1

	fmt.Printf("Nomor Acak dari 1 sampai 100 adalah : %d\n", randomNumber)
}
