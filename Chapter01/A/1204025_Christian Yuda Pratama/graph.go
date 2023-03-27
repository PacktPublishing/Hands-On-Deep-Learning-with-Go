package main

import (
	"fmt"

	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func main() {
	g := G.NewGraph()
	var z *G.Node

	// declare mat, with weight initialization from matB
	matB := []float64{0.9, 0.7, 0.4, 0.2}
	matT := tensor.New(tensor.WithBacking(matB), tensor.WithShape(2, 2))
	mat := G.NewMatrix(g,
		tensor.Float64,
		G.WithName("W"),
		G.WithShape(2, 2),
		G.WithValue(matT),
	)

	// declare x with weight initialization from vecB
	vecB := []float64{5, 7}
	vecT := tensor.New(tensor.WithBacking(vecB), tensor.WithShape(2))
	vec := G.NewVector(g,
		tensor.Float64,
		G.WithName("x"),
		G.WithShape(2),
		G.WithValue(vecT),
	)

	// declare b
	b := G.NewScalar(g,
		tensor.Float64,
		G.WithName("b"),
		G.WithValue(3.0),
	)

	a, err := G.Mul(mat, vec)
	if err != nil {
		panic(err)
	}
	z, err = G.Add(a, b)
	if err != nil {
		panic(err)
	}

	// execute the graph and get the result
	machine := G.NewTapeMachine(g)
	defer machine.Close()
	if err = machine.RunAll(); err != nil {
		panic(err)
	}
	fmt.Println(z.Value())
}
