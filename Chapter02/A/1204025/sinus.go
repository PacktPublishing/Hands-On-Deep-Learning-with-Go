package main

import (
	"fmt"

	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func main() {
	g := G.NewGraph()
	var z *G.Node

	// declare x with weight initialization from vecB
	vecB := []float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	vecT := tensor.New(tensor.WithBacking(vecB), tensor.WithShape(10))
	x := G.NewVector(g,
		tensor.Float64,
		G.WithName("x"),
		G.WithShape(10),
		G.WithValue(vecT),
	)

	// declare y = x^2
	y := G.Must(G.Square(x))

	// execute the graph and get the result
	machine := G.NewTapeMachine(g)
	defer machine.Close()
	if err := machine.RunAll(); err != nil {
		panic(err)
	}
	fmt.Println(y.Value())
}
