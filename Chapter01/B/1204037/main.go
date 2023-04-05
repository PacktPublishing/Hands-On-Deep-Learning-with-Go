package main

import (
	"fmt"

	"gorgonia.org/gorgonia"
)

//Persamaan Mutlak

func main() {
	g := gorgonia.NewGraph()

	// Input nodes
	x := gorgonia.NewScalar(g, gorgonia.Float64, gorgonia.WithName("x"))
	y := gorgonia.NewScalar(g, gorgonia.Float64, gorgonia.WithName("y"))

	// Absolute value expression
	absExpr, err := gorgonia.Abs(x)
	if err != nil {
		panic(err)
	}

	// Output node
	output := gorgonia.Must(gorgonia.Sub(absExpr, y))

	// Define a machine to execute the graph
	machine := gorgonia.NewTapeMachine(g)

	// Bind input values
	gorgonia.Let(x, 2.0)
	gorgonia.Let(y, 1.5)

	// Run the graph
	if err = machine.RunAll(); err != nil {
		panic(err)
	}

	// Get the output value
	result := output.Value().Data().(float64)

	fmt.Println(result) // Output: 0.5
}
