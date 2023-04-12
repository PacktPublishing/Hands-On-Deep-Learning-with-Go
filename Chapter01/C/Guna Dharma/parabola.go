package main

import (
	"fmt"
	"log"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func main() {
	g := gorgonia.NewGraph()

	// Create variables for the parameters of the parabola equation
	a := gorgonia.NewScalar(g, tensor.Float64, gorgonia.WithName("a"))
	b := gorgonia.NewScalar(g, tensor.Float64, gorgonia.WithName("b"))
	c := gorgonia.NewScalar(g, tensor.Float64, gorgonia.WithName("c"))

	// Create a variable for the input value of x
	x := gorgonia.NewScalar(g, tensor.Float64, gorgonia.WithName("x"))

	// Compute the value of y using the parabola equation
	y := gorgonia.Must(gorgonia.Add(
		gorgonia.Must(gorgonia.Mul(a, gorgonia.Must(gorgonia.Pow(x, gorgonia.NewConstant(2.0, gorgonia.WithName("2")))))),

		gorgonia.Must(gorgonia.Mul(b, x)),
	))

	// Define the input and output nodes of the computation graph
	m := gorgonia.NewTapeMachine(g)
	defer m.Close()

	// Set the values of the parameters a, b, and c
	gorgonia.Let(a, 1.0)
	gorgonia.Let(b, 2.0)
	gorgonia.Let(c, 3.0)

	// Compute the value of y for x=4.0
	gorgonia.Let(x, 4.0)
	if err := m.RunAll(); err != nil {
		log.Fatal(err)
	}

	// Print the result
	fmt.Println(y.Value())
}
