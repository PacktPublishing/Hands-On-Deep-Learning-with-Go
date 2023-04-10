package main

import (
	"fmt"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func main() {
	// Define the matrix W, vector x, and bias b
	W := tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{1.0, 2.0, 3.0, 4.0}))
	x := tensor.New(tensor.WithShape(2), tensor.WithBacking([]float64{2.0, 3.0}))
	b := tensor.New(tensor.WithShape(2), tensor.WithBacking([]float64{1.0, 1.0}))

	// Define the symbolic variables and expressions
	g := gorgonia.NewGraph()
	Wsym := gorgonia.NodeFromAny(g, W, gorgonia.WithName("W"))
	xsym := gorgonia.NodeFromAny(g, x, gorgonia.WithName("x"))
	bsym := gorgonia.NodeFromAny(g, b, gorgonia.WithName("b"))
	var z *gorgonia.Node
	z, _ = gorgonia.Mul(Wsym, xsym)
	z, _ = gorgonia.Add(z, bsym)

	// Define the machine and run the computations
	machine := gorgonia.NewTapeMachine(g)
	defer machine.Close()

	if err := machine.RunAll(); err != nil {
		fmt.Printf("Error running computations: %v", err)
		return
	}

	// Retrieve the results and print them
	zval, _ := z.Value().Data().([]float64)
	fmt.Printf("The result z is: %v", zval)
}
